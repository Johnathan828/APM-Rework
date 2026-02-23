# APM/LIO/src/apm_core/raw_pull.py
from __future__ import annotations

from datetime import datetime
from typing import Tuple, Dict

import pandas as pd

from .ssot import SensorSpec
from .db_interface import DBInterface


def _safe_eval(expr: str, locals_dict: Dict[str, float]) -> bool:
    return bool(eval(expr, {"__builtins__": {}}, locals_dict))


def build_wide_frame(
    *,
    db: DBInterface,
    sensor: SensorSpec,
    start: datetime,
    end: datetime,
) -> Tuple[pd.DataFrame, pd.Series]:
    feature_tags = sensor.feature_tags

    shutdown = sensor.shutdown_rules
    shutdown_tag_map = shutdown.get("tags", {}) or {}
    shutdown_tags = list(shutdown_tag_map.values())

    tags = list(dict.fromkeys(feature_tags + shutdown_tags))
    interval_str = sensor.get_interval_string()

    raw = db.pull_raw_tags_postgres(tags=tags, start=start, end=end, interval_str=interval_str)
    if raw is None or raw.empty:
        return pd.DataFrame(), pd.Series(dtype=float)

    # LOCKED schema (confirmed by your log)
    df_long = raw[["rt_timestamp", "sourceidentifier", "rt_value"]].copy()
    df_long.rename(columns={"rt_timestamp": "timestamp", "sourceidentifier": "tag", "rt_value": "value"}, inplace=True)

    df_long["timestamp"] = pd.to_datetime(df_long["timestamp"])
    df_wide = df_long.pivot_table(index="timestamp", columns="tag", values="value", aggfunc="last").sort_index()

    # Shutdown filter
    shutdown_expr = (shutdown.get("shutdown_rules") or "").strip()
    if not shutdown_expr or not shutdown_tag_map:
        return df_wide, pd.Series(1.0, index=df_wide.index, name="Filter_Tag")

    filter_vals = []
    for _, row in df_wide.iterrows():
        locals_dict = {}
        for var_name, tag_id in shutdown_tag_map.items():
            v = row.get(tag_id)
            locals_dict[var_name] = float(v) if pd.notna(v) else float("nan")

        try:
            is_shutdown = _safe_eval(shutdown_expr, locals_dict)
            filter_vals.append(0.0 if is_shutdown else 1.0)
        except Exception:
            filter_vals.append(0.0)

    filter_series = pd.Series(filter_vals, index=df_wide.index, name="Filter_Tag")
    return df_wide, filter_series
