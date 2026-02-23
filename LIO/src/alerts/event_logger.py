# APM/LIO/src/alerts/event_logger.py
from __future__ import annotations

import json
from datetime import datetime
from typing import Dict, Any, Optional

import pandas as pd


class EventLogger:
    def __init__(self, *, ssot: Dict[str, Any], sensor_name: str, logger, db):
        self.ssot = ssot
        self.sensor_name = sensor_name
        self.logger = logger
        self.db = db

    def log_event(
        self,
        *,
        score: float,
        level: int,
        trigger_time: datetime,
        details: Dict[str, Any],
        feature_contrib: Optional[pd.DataFrame] = None,
        latest_ts: Optional[datetime] = None,
    ) -> None:
        model_type = details.get("method", "unknown")

        payload = {
            "sensor": self.sensor_name,
            "score": score,
            "latest_ts": latest_ts.isoformat() if latest_ts else None,
            "details": details,
        }

        if feature_contrib is not None and not feature_contrib.empty and latest_ts is not None:
            try:
                row = feature_contrib.loc[latest_ts]
                payload["top_contributors"] = (
                    row.sort_values(ascending=False).head(10).to_dict()
                    if hasattr(row, "sort_values")
                    else {}
                )
            except Exception:
                payload["top_contributors"] = {}

        self.db.save_model_event_data(
            model_name=self.sensor_name,
            model_type=str(model_type),
            trigger_time=trigger_time,
            score=float(score),
            level=int(level),
            event_details=json.dumps(payload),
            site=str(self.ssot.get("site", "SITE")),
        )

        self.logger.info(f"{self.sensor_name}: event logged to NewApmModelEvents")