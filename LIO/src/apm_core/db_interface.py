from __future__ import annotations

from datetime import datetime
from typing import List, Dict, Any, Optional

import pandas as pd
from psycopg2 import pool
import pymssql

from .settings import IniSettings


class DBInterface:
    """
    One shared DB interface for ALL sensors.

    - Raw pull from Postgres sri_get_tag_data(...)
    - Write scores into MSSQL dbo.<mssql_table> (APM_Scalability / APM_Scalability_Dev)
    - Write events into MSSQL dbo.<mssql_events_table> (NewApmModelEvents / NewApmModelEvents_Dev)

    IMPORTANT:
    Idempotent write (DELETE then INSERT) for each score row to prevent duplicates.
    """

    def __init__(self, settings: IniSettings, logger):
        self.settings = settings  # <- expose settings for notifier
        self.s = settings
        self.logger = logger

        # Tables (driven by config.ini section)
        self.score_table = str(self.s.mssql_table)
        self.events_table = str(getattr(self.s, "mssql_events_table", "NewApmModelEvents"))

        # -----------------------------
        # Postgres pool (with timeouts)
        # -----------------------------
        self.pg_pool = pool.SimpleConnectionPool(
            1,
            10,
            host=self.s.pg_host,
            database=self.s.pg_dbname,
            user=self.s.pg_user,
            password=self.s.pg_password,
            port=self.s.pg_port,
            connect_timeout=10,  # ✅ prevent hanging connects
            keepalives=1,
            keepalives_idle=30,
            keepalives_interval=10,
            keepalives_count=5,
        )
        self.logger.info("PostgreSQL pool initialised")

        # -----------------------------
        # MSSQL sink (with timeouts)
        # -----------------------------
        self.mssql_conn = self._connect_mssql()
        self.logger.info(f"MSSQL connected | score_table={self.score_table} | events_table={self.events_table}")

    def _connect_mssql(self):
        # pymssql supports login_timeout; timeout support depends on build, but safe to pass in most envs
        return pymssql.connect(
            server=self.s.mssql_host,
            user=self.s.mssql_user,
            password=self.s.mssql_password,
            database=self.s.mssql_dbname,
            port=self.s.mssql_port,
            as_dict=True,
            login_timeout=10,
            timeout=15,
        )

    def _reconnect_mssql(self) -> None:
        try:
            if self.mssql_conn:
                self.mssql_conn.close()
        except Exception:
            pass
        self.mssql_conn = self._connect_mssql()
        self.logger.warning(f"MSSQL reconnected | score_table={self.score_table} | events_table={self.events_table}")

    def close(self) -> None:
        try:
            if self.pg_pool:
                self.pg_pool.closeall()
        except Exception:
            pass
        try:
            if self.mssql_conn:
                self.mssql_conn.close()
        except Exception:
            pass

    # -----------------------------
    # Postgres helper
    # -----------------------------
    @staticmethod
    def _dt_to_pg_string(dt: datetime) -> str:
        return dt.strftime("%Y-%m-%d %H:%M:%S")

    def pull_raw_tags_postgres(
        self,
        *,
        tags: List[str],
        start: datetime,
        end: datetime,
        interval_str: str,
        statement_timeout_ms: int = 15000,  # ✅ 15s query timeout
        max_retries: int = 3,
    ) -> pd.DataFrame:
        """
        Pull tag data from sri_get_tag_data with:
          - statement_timeout to prevent hanging queries
          - retries/backoff
          - pool conn eviction on failure
        """
        if not tags:
            return pd.DataFrame()

        tags_pipe = "|".join(tags)
        start_s = self._dt_to_pg_string(start)
        end_s = self._dt_to_pg_string(end)

        # ✅ only fetch columns we actually use (smaller payload)
        sql = """
            SELECT rt_timestamp, sourceidentifier, rt_value
            FROM sri_get_tag_data(%s, %s, %s, %s::interval);
        """

        last_err = None

        for attempt in range(1, max_retries + 1):
            conn = None
            try:
                conn = self.pg_pool.getconn()

                # ✅ Apply statement timeout to this connection/session
                with conn.cursor() as cur:
                    cur.execute("SET statement_timeout = %s;", (int(statement_timeout_ms),))

                with conn.cursor() as cur:
                    cur.execute(sql, (tags_pipe, start_s, end_s, interval_str))
                    rows = cur.fetchall()
                    cols = [d[0] for d in cur.description]

                df = pd.DataFrame(rows, columns=cols)
                if df.empty:
                    self.logger.warning(f"Postgres sri_get_tag_data returned 0 rows | tags={len(tags)}")
                else:
                    self.logger.info(f"Postgres raw rows={len(df)} cols={list(df.columns)}")

                return df

            except Exception as e:
                last_err = e
                self.logger.warning(f"Postgres pull failed (attempt {attempt}/{max_retries}) | err={e}")

                # Evict broken conn from pool
                try:
                    if conn is not None:
                        self.pg_pool.putconn(conn, close=True)
                        conn = None
                except Exception:
                    pass

                # Backoff
                if attempt < max_retries:
                    import time
                    time.sleep(min(2 ** (attempt - 1), 5))

            finally:
                if conn is not None:
                    try:
                        self.pg_pool.putconn(conn)
                    except Exception:
                        pass

        # All retries failed
        raise last_err

    # -----------------------------
    # MSSQL write (IDEMPOTENT) - scores
    # -----------------------------
    def _delete_existing_row(self, *, cur, site: str, displayname: str, ts: datetime) -> None:
        table = self.score_table
        del_sql = f"""
        DELETE FROM dbo.{table}
        WHERE CAST([site] AS VARCHAR(50)) = %s
          AND CAST([displayname] AS VARCHAR(255)) = %s
          AND [timestamp] = %s
        """
        cur.execute(del_sql, (site, displayname, ts))

    def write_rows_mssql(self, rows: List[Dict[str, Any]]) -> None:
        """
        Write rows with a single retry if MSSQL connection drops.
        """
        if not rows:
            return

        def _do_write():
            table = self.score_table
            ins_sql = f"""
            INSERT INTO dbo.{table} ([displayname],[value],[granularity],[site],[timestamp],[meta])
            VALUES (%s,%s,%s,%s,%s,%s)
            """
            with self.mssql_conn.cursor() as cur:
                for r in rows:
                    site = r["site"]
                    disp = r["displayname"]
                    ts = r["timestamp"]

                    self._delete_existing_row(cur=cur, site=site, displayname=disp, ts=ts)

                    cur.execute(
                        ins_sql,
                        (
                            disp,
                            float(r["value"]) if r["value"] is not None else None,
                            int(r["granularity"]),
                            site,
                            ts,
                            r.get("meta", None),
                        ),
                    )
                self.mssql_conn.commit()

        try:
            _do_write()
        except Exception as e:
            self.logger.warning(f"MSSQL score write failed, retrying once | err={e}")
            self._reconnect_mssql()
            _do_write()

    def write_series_mssql(
        self,
        *,
        displayname: str,
        series: pd.Series,
        site: str,
        granularity: int,
        meta: Optional[str] = None,
    ) -> None:
        if series is None or series.empty:
            self.logger.warning(f"write_series_mssql: empty series for {displayname}")
            return

        rows = []
        for ts, val in series.items():
            if pd.isna(ts):
                continue

            ts_dt = pd.to_datetime(ts).to_pydatetime()
            if ts_dt.tzinfo is not None:
                ts_dt = ts_dt.replace(tzinfo=None)

            rows.append(
                {
                    "displayname": displayname,
                    "value": None if pd.isna(val) else float(val),
                    "granularity": granularity,
                    "site": site,
                    "timestamp": ts_dt,
                    "meta": meta,
                }
            )

        self.write_rows_mssql(rows)

    # -----------------------------
    # REQUIRED WRAPPERS
    # -----------------------------
    def write_series_mssql_idempotent(
        self,
        displayname: str,
        series: pd.Series,
        site: str,
        granularity: int,
        meta=None,
    ):
        """Wrapper name used by pipeline/alerts code."""
        self.write_series_mssql(displayname=displayname, series=series, site=site, granularity=granularity, meta=meta)

    # -----------------------------
    # MSSQL write - events (config-driven table)
    # -----------------------------
    def save_model_event_data(
        self,
        model_name: str,
        model_type: str,
        trigger_time: datetime,
        score: float,
        level: int,
        event_details: str,
        site: str,
    ) -> None:
        """
        Insert into dbo.<events_table>, retry once on MSSQL reconnect.
        """
        table = self.events_table
        sql = f"""
        INSERT INTO dbo.{table}
          ([model_name],[model_type],[trigger_time],[score],[level],[event_details],[site],[created_at])
        VALUES
          (%s,%s,%s,%s,%s,%s,%s,GETUTCDATE())
        """

        def _do():
            with self.mssql_conn.cursor() as cur:
                cur.execute(sql, (model_name, model_type, trigger_time, float(score), int(level), event_details, site))
                self.mssql_conn.commit()

        try:
            _do()
        except Exception as e:
            self.logger.warning(f"MSSQL event insert failed, retrying once | err={e}")
            self._reconnect_mssql()
            _do()