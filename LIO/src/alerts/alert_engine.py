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
    Legacy-like decision logic (NO "holding window" inside engine):

      - Gate must be OPEN (section_status >= filter_value)
      - Startup suppression: after a gate-open transition, ignore alerts for startup_period minutes
      - Alarm when score <= alarm_thresh

    IMPORTANT:
      - Email spam prevention (cooldown) belongs in run_once.py state logic,
        NOT inside the engine.
    """

    def __init__(self, *, ssot: Dict[str, Any], sensor_name: str, logger):
        self.ssot = ssot
        self.sensor_name = sensor_name
        self.logger = logger

        cfg = ssot.get(sensor_name, {}) or {}
        other = cfg.get("Other", {}) or {}
        ft = cfg.get("filter_tag", {}) or {}

        # Prefer filter_tag.alarm_thresh if present, else fall back to Other.alarm_thresh
        self.alarm_thresh = float(ft.get("alarm_thresh", other.get("alarm_thresh", 0.75)))
        self.filter_value = float(ft.get("filter_value", other.get("filter_value", 0.9)))

        # Startup suppression window (minutes) – legacy behaviour
        self.startup_period_minutes = int(other.get("startup_period", 0))

    def _in_startup_period(self, *, section_status_series: pd.Series, latest_ts: pd.Timestamp) -> bool:
        """
        Startup means: the gate has recently transitioned from CLOSED -> OPEN,
        and we are within startup_period_minutes of that transition.
        """
        if self.startup_period_minutes <= 0:
            return False
        if section_status_series is None or section_status_series.empty:
            return False

        sec = section_status_series.copy().dropna()
        if sec.empty:
            return False

        # Consider CLOSED where section_status < filter_value
        closed_mask = sec < self.filter_value
        if not closed_mask.any():
            # gate has been open for entire available history; can't detect a "recent open"
            return False

        last_closed_ts = closed_mask[closed_mask].index.max()

        if pd.to_datetime(latest_ts) <= pd.to_datetime(last_closed_ts):
            return False

        delta = pd.to_datetime(latest_ts) - pd.to_datetime(last_closed_ts)
        return delta < timedelta(minutes=self.startup_period_minutes)

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

        latest_sec = (
            float(section_status_series.loc[latest_ts])
            if (section_status_series is not None and not section_status_series.empty and latest_ts in section_status_series.index)
            else 0.0
        )

        # Gate must be open
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

        # Startup suppression
        if self._in_startup_period(section_status_series=section_status_series, latest_ts=latest_ts):
            return AlertDecision(
                should_alert=False,
                reason="startup_period",
                latest_score=latest_score,
                latest_section_status=latest_sec,
                alarm_thresh=self.alarm_thresh,
                filter_value=self.filter_value,
                anomaly_started_at=None,
            )

        # Alarm condition: score <= alarm_thresh
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

        # Immediate alert (cooldown handled elsewhere)
        return AlertDecision(
            should_alert=True,
            reason="alarm",
            latest_score=latest_score,
            latest_section_status=latest_sec,
            alarm_thresh=self.alarm_thresh,
            filter_value=self.filter_value,
            anomaly_started_at=latest_ts.to_pydatetime() if hasattr(latest_ts, "to_pydatetime") else None,
        )