from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from psycopg2 import pool


class DBInterface:
    """
    Flask DB adapter.

    - MSSQL: read model scores + section status + feature contributions (from APM_Scalability_Dev or APM_Scalability)
    - MSSQL: read events (NewApmModelEvents)
    - Postgres (DATABASE): raw tag plots via sri_get_tag_data() for legacy-like charts (Option B)
    """

    def __init__(
        self,
        *,
        config,
        logger,
        mssql_pool,
        mssql_table: str,
        entity_id_dict: Optional[dict] = None,
    ):
        self.config = config
        self.logger = logger
        self.pool = mssql_pool
        self.table = mssql_table

        self.entity_id_dict = entity_id_dict or {}
        self.reverse_entity_id_dict = {v: k for k, v in self.entity_id_dict.items()} if self.entity_id_dict else {}

        self.mssql_conn = None  # acquired per-request, returned to pool
        self.pg_pool = None     # persistent pool created on-demand

    # ------------------------------------------------------------------
    # MSSQL (pool) helpers
    # ------------------------------------------------------------------
    def _ensure_mssql_conn(self):
        """Acquire a pooled MSSQL connection if needed."""
        if self.mssql_conn is None:
            self.mssql_conn = self.pool.get_connection()
            try:
                self.logger.info("MSSQL connection acquired (Flask)")
            except Exception:
                pass

    def release_connections(self):
        """Release MSSQL connection back to the pool (do NOT close pg pool here)."""
        if self.mssql_conn is not None:
            try:
                self.pool.release_connection(self.mssql_conn)
                try:
                    self.logger.info("MSSQL connection released (Flask)")
                except Exception:
                    pass
            except Exception:
                pass
            finally:
                self.mssql_conn = None

    def cleanup(self):
        """Close pools (call on Flask shutdown)."""
        try:
            self.release_connections()
        except Exception:
            pass

        try:
            if self.pg_pool is not None:
                self.pg_pool.closeall()
                self.pg_pool = None
        except Exception:
            pass

        try:
            if self.pool is not None:
                self.pool.close_all_connections()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # MSSQL reads
    # ------------------------------------------------------------------
    def get_neuro_displayname_data(
        self,
        *,
        date_from: datetime,
        date_to: datetime,
        model_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Read from MSSQL (APM_Scalability[_Dev]) and return wide DF:
          index=timestamp, columns=displayname, values=value
        """
        self._ensure_mssql_conn()

        required_cols = {"displayname", "value", "timestamp"}

        if model_cols is None or len(model_cols) == 0:
            query = f"""
                SELECT displayname, value, CAST([timestamp] AS DATETIME) AS [timestamp]
                FROM dbo.{self.table}
                WHERE [timestamp] BETWEEN %s AND %s
            """
            params = (date_from, date_to)
        else:
            placeholders = ",".join(["%s"] * len(model_cols))
            query = f"""
                SELECT displayname, value, CAST([timestamp] AS DATETIME) AS [timestamp]
                FROM dbo.{self.table}
                WHERE [timestamp] BETWEEN %s AND %s
                  AND displayname IN ({placeholders})
            """
            params = tuple([date_from, date_to] + list(model_cols))

        try:
            # IMPORTANT: use cursor(as_dict=True) because we set as_dict in pool connection
            with self.mssql_conn.cursor(as_dict=True) as cur:
                cur.execute(query, params)
                rows = cur.fetchall()

            if not rows:
                return pd.DataFrame()

            df = pd.DataFrame(rows)
            if not required_cols.issubset(df.columns):
                self.logger.error(f"MSSQL returned unexpected columns: {list(df.columns)}")
                return pd.DataFrame()

            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.dropna(subset=["timestamp"])

            # keep last per timestamp/displayname
            df = df.drop_duplicates(subset=["timestamp", "displayname"], keep="last")
            wide = df.pivot(index="timestamp", columns="displayname", values="value").sort_index()
            return wide

        except Exception as e:
            self.logger.error(f"get_neuro_displayname_data failed: {e}", exc_info=True)
            return pd.DataFrame()

    def get_model_event_data(
        self,
        *,
        model_name: Optional[List[str]] = None,
        model_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        level: Optional[int] = 1,
    ) -> List[Dict[str, Any]]:
        """
        Reads events from dbo.NewApmModelEvents (reworked APM).
        Returns list of dicts compatible with legacy frontend fields.
        """
        self._ensure_mssql_conn()

        query = """
            SELECT model_name, model_type, trigger_time, score, level, event_details, site
            FROM dbo.NewApmModelEvents
            WHERE 1=1
        """
        params: List[Any] = []

        if level is not None:
            query += " AND level = %s"
            params.append(int(level))

        if model_name:
            placeholders = ",".join(["%s"] * len(model_name))
            query += f" AND model_name IN ({placeholders})"
            params.extend(model_name)

        if model_type:
            query += " AND model_type = %s"
            params.append(model_type)

        if start_time:
            query += " AND trigger_time >= %s"
            params.append(start_time)

        if end_time:
            query += " AND trigger_time <= %s"
            params.append(end_time)

        query += " ORDER BY trigger_time DESC"

        try:
            with self.mssql_conn.cursor(as_dict=True) as cur:
                cur.execute(query, tuple(params))
                rows = cur.fetchall()

            out: List[Dict[str, Any]] = []
            for r in rows or []:
                details = r.get("event_details")
                try:
                    details_obj = json.loads(details) if isinstance(details, str) else details
                except Exception:
                    details_obj = details

                out.append(
                    {
                        "model_name": r.get("model_name"),
                        "model_type": r.get("model_type"),
                        "trigger_time": r.get("trigger_time"),
                        "score": float(r.get("score", 0.0)) * 100.0,  # legacy displayed %
                        "level": r.get("level"),
                        "event_details": details_obj,
                        "site": r.get("site"),
                    }
                )
            return out

        except Exception as e:
            self.logger.error(f"get_model_event_data failed: {e}", exc_info=True)
            return []

    # ------------------------------------------------------------------
    # Postgres raw pull (plots) using [DATABASE]
    # ------------------------------------------------------------------
    def _ensure_pg_pool(self):
        """Creates Postgres pool on-demand using [DATABASE] in config.ini."""
        if self.pg_pool is None:
            self.pg_pool = pool.SimpleConnectionPool(
                1,
                10,
                host=self.config["DATABASE"]["host"],
                database=self.config["DATABASE"]["dbname"],
                user=self.config["DATABASE"]["user"],
                password=self.config["DATABASE"]["password"],
                port=self.config["DATABASE"]["port"],
            )
            self.logger.info("PostgreSQL pool initialised (Flask raw plots)")

    @staticmethod
    def _dt_to_pg_string(dt: datetime) -> str:
        return pd.to_datetime(dt).strftime("%Y-%m-%d %H:%M:%S")

    def pull_raw_tags_postgres(
        self,
        *,
        tags: List[str],
        start: datetime,
        end: datetime,
        interval_str: str,
    ) -> pd.DataFrame:
        """
        Calls sri_get_tag_data(tags_pipe, start, end, interval::interval).
        Returns long DF with columns like:
          rt_timestamp, sourceidentifier, rt_value
        """
        if not tags:
            return pd.DataFrame(index=pd.DatetimeIndex([]))

        self._ensure_pg_pool()

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

        return pd.DataFrame(rows, columns=cols)

    def build_wide_from_postgres(
        self,
        *,
        tags: List[str],
        start: datetime,
        end: datetime,
        interval_str: str,
    ) -> pd.DataFrame:
        """
        Returns wide DF indexed by timestamp, columns=tag strings, values=rt_value.
        Always returns a DatetimeIndex (even if empty).
        """
        raw = self.pull_raw_tags_postgres(tags=tags, start=start, end=end, interval_str=interval_str)
        if raw is None or raw.empty:
            return pd.DataFrame(index=pd.DatetimeIndex([]))

        needed = {"rt_timestamp", "sourceidentifier", "rt_value"}
        if not needed.issubset(set(raw.columns)):
            self.logger.error(f"Postgres sri_get_tag_data returned unexpected cols: {list(raw.columns)}")
            return pd.DataFrame(index=pd.DatetimeIndex([]))

        df_long = raw[["rt_timestamp", "sourceidentifier", "rt_value"]].copy()
        df_long.rename(
            columns={"rt_timestamp": "timestamp", "sourceidentifier": "tag", "rt_value": "value"},
            inplace=True,
        )
        df_long["timestamp"] = pd.to_datetime(df_long["timestamp"], errors="coerce")
        df_long = df_long.dropna(subset=["timestamp"])
        if df_long.empty:
            return pd.DataFrame(index=pd.DatetimeIndex([]))

        df_wide = (
            df_long.pivot_table(index="timestamp", columns="tag", values="value", aggfunc="last")
            .sort_index()
        )

        df_wide.index = pd.to_datetime(df_wide.index, errors="coerce")
        df_wide = df_wide[~df_wide.index.isna()]
        return df_wide