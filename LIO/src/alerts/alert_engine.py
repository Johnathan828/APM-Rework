# APM/LIO/src/alerts/alert_engine.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

import pandas as pd


def _to_utc_naive_index(s: pd.Series) -> pd.Series:
    """Ensure datetime index is UTC-naive to avoid tz compare crashes."""
    if s is None or s.empty:
        return s
    idx = pd.to_datetime(s.index, utc=True).tz_convert(None)
    s2 = s.copy()
    s2.index = idx
    return s2.sort_index()


@dataclass
class AlertDecision:
    should_alert: bool
    reason: str
    latest_score: float
    latest_section_status: float
    alarm_thresh: float
    filter_value: float
    anomaly_started_at: Optional[datetime]


class AlertEngine:
    """
    OLD-APM style:
      - Gate must be OPEN (section_status >= filter_value)
      - Alarm when score <= alarm_thresh
      - Holding rule: must be continuously anomalous for alert_holding_minutes
    """

    def __init__(self, *, ssot: Dict[str, Any], sensor_name: str, logger):
        self.ssot = ssot
        self.sensor_name = sensor_name
        self.logger = logger

        cfg = ssot.get(sensor_name, {}) or {}
        other = cfg.get("Other", {}) or {}
        ft = cfg.get("filter_tag", {}) or {}

        self.alarm_thresh = float(other.get("alarm_thresh", 0.75))
        self.filter_value = float(ft.get("filter_value", other.get("filter_value", 0.9)))
        self.alert_holding_minutes = int(other.get("alert_holding_minutes", 15))

    def evaluate(self, *, score_series: pd.Series, section_status_series: pd.Series, now: datetime) -> AlertDecision:
        score_series = _to_utc_naive_index(score_series)
        section_status_series = _to_utc_naive_index(section_status_series)

        if score_series is None or score_series.empty:
            return AlertDecision(
                should_alert=False,
                reason="no_score",
                latest_score=float("nan"),
                latest_section_status=float("nan"),
                alarm_thresh=self.alarm_thresh,
                filter_value=self.filter_value,
                anomaly_started_at=None,
            )

        latest_ts = score_series.index.max()
        latest_score = float(score_series.loc[latest_ts])
        latest_sec = float(section_status_series.loc[latest_ts]) if (section_status_series is not None and not section_status_series.empty) else 0.0

        gate_open = latest_sec >= self.filter_value
        if not gate_open:
            return AlertDecision(
                should_alert=False,
                reason="gate_closed",
                latest_score=latest_score,
                latest_section_status=latest_sec,
                alarm_thresh=self.alarm_thresh,
                filter_value=self.filter_value,
                anomaly_started_at=None,
            )

        # Anomaly definition for this system:
        # score <= alarm_thresh triggers alarm
        is_anom_now = latest_score <= self.alarm_thresh
        if not is_anom_now:
            return AlertDecision(
                should_alert=False,
                reason="score_normal",
                latest_score=latest_score,
                latest_section_status=latest_sec,
                alarm_thresh=self.alarm_thresh,
                filter_value=self.filter_value,
                anomaly_started_at=None,
            )

        # Holding window: continuously anomalous for last N minutes
        window_start = (pd.to_datetime(latest_ts) - timedelta(minutes=self.alert_holding_minutes)).to_pydatetime()
        window_scores = score_series.loc[score_series.index >= window_start]

        if window_scores.empty:
            return AlertDecision(
                should_alert=False,
                reason="holding_not_met",
                latest_score=latest_score,
                latest_section_status=latest_sec,
                alarm_thresh=self.alarm_thresh,
                filter_value=self.filter_value,
                anomaly_started_at=None,
            )

        all_anom = (window_scores <= self.alarm_thresh).all()
        if not all_anom:
            # find most recent non-anom -> anomaly start is next point
            non_anom_idx = window_scores[window_scores > self.alarm_thresh].index
            if len(non_anom_idx) > 0:
                last_ok = non_anom_idx.max()
                # anomaly started after last_ok
                after = window_scores.loc[window_scores.index > last_ok]
                start_at = after.index.min().to_pydatetime() if not after.empty else latest_ts.to_pydatetime()
            else:
                start_at = window_scores.index.min().to_pydatetime()

            return AlertDecision(
                should_alert=False,
                reason="holding_not_met",
                latest_score=latest_score,
                latest_section_status=latest_sec,
                alarm_thresh=self.alarm_thresh,
                filter_value=self.filter_value,
                anomaly_started_at=start_at,
            )

        anomaly_started_at = window_scores.index.min().to_pydatetime()

        return AlertDecision(
            should_alert=True,
            reason="alarm",
            latest_score=latest_score,
            latest_section_status=latest_sec,
            alarm_thresh=self.alarm_thresh,
            filter_value=self.filter_value,
            anomaly_started_at=anomaly_started_at,
        )