# APM/LIO/src/apm_core/db_interface.py
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

    - Raw pull from Minocore API (handled elsewhere)
    - Optional Postgres support remains if needed
    - Write into MSSQL APM_Scalability(_Dev) with columns:
        displayname, value, granularity, site, timestamp, meta

    IMPORTANT:
    Idempotent write (DELETE then INSERT) for each row to prevent duplicates.
    """

    def __init__(self, settings: IniSettings, logger):
        self.settings = settings  # <- expose settings for notifier
        self.s = settings
        self.logger = logger

        # Postgres pool (optional)
        self.pg_pool = pool.SimpleConnectionPool(
            1,
            10,
            host=self.s.pg_host,
            database=self.s.pg_dbname,
            user=self.s.pg_user,
            password=self.s.pg_password,
            port=self.s.pg_port,
        )
        self.logger.info("PostgreSQL pool initialised")

        # MSSQL sink
        self.mssql_conn = pymssql.connect(
            server=self.s.mssql_host,
            user=self.s.mssql_user,
            password=self.s.mssql_password,
            database=self.s.mssql_dbname,
            port=self.s.mssql_port,
            as_dict=True,
        )
        self.logger.info(f"MSSQL connected | table={self.s.mssql_table}")

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
    # Postgres helper (optional)
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
    ) -> pd.DataFrame:
        if not tags:
            return pd.DataFrame()

        tags_pipe = "|".join(tags)
        start_s = self._dt_to_pg_string(start)
        end_s = self._dt_to_pg_string(end)

        sql = """
            SELECT *
            FROM sri_get_tag_data(%s, %s, %s, %s::interval);
        """

        conn = self.pg_pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, (tags_pipe, start_s, end_s, interval_str))
                rows = cur.fetchall()
                cols = [d[0] for d in cur.description]
        finally:
            self.pg_pool.putconn(conn)

        df = pd.DataFrame(rows, columns=cols)
        if df.empty:
            self.logger.warning(f"Postgres sri_get_tag_data returned 0 rows | tags={len(tags)}")
            return df

        self.logger.info(f"Postgres raw rows={len(df)} cols={list(df.columns)}")
        return df

    # -----------------------------
    # MSSQL write (IDEMPOTENT)
    # -----------------------------
    def _delete_existing_row(self, *, cur, site: str, displayname: str, ts: datetime) -> None:
        table = self.s.mssql_table
        del_sql = f"""
        DELETE FROM dbo.{table}
        WHERE CAST([site] AS VARCHAR(50)) = %s
          AND CAST([displayname] AS VARCHAR(255)) = %s
          AND [timestamp] = %s
        """
        cur.execute(del_sql, (site, displayname, ts))

    def write_rows_mssql(self, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return

        table = self.s.mssql_table

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
    # REQUIRED WRAPPERS (so other modules don't break)
    # -----------------------------
    def write_series_mssql_idempotent(self, displayname: str, series: pd.Series, site: str, granularity: int, meta=None):
        """Wrapper name used by pipeline/alerts code."""
        self.write_series_mssql(displayname=displayname, series=series, site=site, granularity=granularity, meta=meta)

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
        Insert into dbo.NewApmModelEvents (simple insert).
        """
        sql = """
        INSERT INTO dbo.NewApmModelEvents
          ([model_name],[model_type],[trigger_time],[score],[level],[event_details],[site],[created_at])
        VALUES
          (%s,%s,%s,%s,%s,%s,%s,GETUTCDATE())
        """
        with self.mssql_conn.cursor() as cur:
            cur.execute(sql, (model_name, model_type, trigger_time, float(score), int(level), event_details, site))
            self.mssql_conn.commit()