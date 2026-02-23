# APM/LIO/src/pipeline/run_once.py
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any

import pandas as pd

from ..apm_core.ssot import get_sensor
from ..apm_core.raw_pull import build_wide_frame
from ..models.factory import build_model

from ..alerts.alert_engine import AlertEngine, AlertDecision
from ..alerts.notifier import Notifier
from ..alerts.event_logger import EventLogger
from ..alerts.state_store import StateStore


def _utc_now_naive() -> datetime:
    return datetime.utcnow()


def run_sensor_once(
    *,
    ssot: Dict[str, Any],
    sensor_name: str,
    db,
    logger,
    global_debug: bool,
    sensor_debug: bool,
    site_root: Path,
) -> None:
    sensor = get_sensor(ssot, sensor_name)
    sensor_cfg = sensor.cfg

    site = ssot.get("site", "SITE")
    gran = int(sensor.other.get("granularity", 15))

    # --- window (old APM short window) ---
    end = _utc_now_naive()
    start = end - timedelta(minutes=15)

    df_wide, filter_tag_series = build_wide_frame(db=db, sensor=sensor, start=start, end=end)

    if df_wide is None or df_wide.empty:
        logger.warning(f"{sensor_name}: no raw data returned, skipping")
        return

    # displayname -> raw tag string
    feature_displaynames_to_tags: Dict[str, str] = {}
    for disp, meta in (sensor_cfg.get("Features", {}) or {}).items():
        if isinstance(meta, dict) and meta.get("tag"):
            feature_displaynames_to_tags[disp] = meta["tag"]

    # --- MODEL (runtime rollback: FixedThresholds only) ---
    model, method_name, method_cfg = build_model(sensor_cfg=sensor_cfg, sensor_name=sensor_name)
    logger.info(f"{sensor_name}: active_method={method_name}")

    result = model.score(df_wide=df_wide, feature_displaynames_to_tags=feature_displaynames_to_tags)

    score_series = result.score
    if score_series is None or score_series.empty:
        logger.warning(f"{sensor_name}: model returned empty score series, skipping")
        return

    # Standardise index to UTC naive (prevents tz crash)
    score_series = score_series.copy()
    score_series.index = pd.to_datetime(score_series.index, utc=True).tz_convert(None)
    score_series = score_series.sort_index()

    # --- Section status aligned to score timestamps (FIX: align tz/index first) ---
    ft_name = sensor.filter_tag_displayname or f"{sensor_name}_Section_Status"

    # Standardise filter_tag_series index to UTC naive FIRST
    filter_tag_series = filter_tag_series.copy()
    filter_tag_series.index = pd.to_datetime(filter_tag_series.index, utc=True).tz_convert(None)
    filter_tag_series = filter_tag_series.sort_index()

    # Now reindex safely onto score index
    section_series = filter_tag_series.reindex(score_series.index).ffill().fillna(0.0)

    # --- Write ONLY latest point per run ---
    latest_ts = score_series.index.max()
    logger.info(
        f"{sensor_name}: gate_debug | latest_ts={latest_ts} | "
        f"raw_filter_latest={float(filter_tag_series.loc[latest_ts]) if latest_ts in filter_tag_series.index else 'MISSING'} | "
        f"section_latest={float(section_series.loc[latest_ts])}"
    )

    # Score
    db.write_series_mssql_idempotent(
        displayname=sensor_name,
        series=score_series.loc[[latest_ts]],
        site=site,
        granularity=gran,
        meta=None,
    )

    # Section status
    db.write_series_mssql_idempotent(
        displayname=ft_name,
        series=section_series.loc[[latest_ts]],
        site=site,
        granularity=gran,
        meta=None,
    )

    # Feature contributions (latest only)
    feature_contrib = result.feature_contrib if result.feature_contrib is not None else pd.DataFrame()
    if not feature_contrib.empty:
        feature_contrib = feature_contrib.copy()
        feature_contrib.index = pd.to_datetime(feature_contrib.index, utc=True).tz_convert(None)
        for disp in feature_contrib.columns:
            s = feature_contrib[disp]
            if latest_ts in s.index:
                db.write_series_mssql_idempotent(
                    displayname=disp,
                    series=s.loc[[latest_ts]],
                    site=site,
                    granularity=gran,
                    meta=None,
                )

    logger.info(f"{sensor_name}: wrote latest score + section status + feature contributions to MSSQL (idempotent)")

    # --- Alert decision ---
    engine = AlertEngine(ssot=ssot, sensor_name=sensor_name, logger=logger)
    decision: AlertDecision = engine.evaluate(
        score_series=score_series,
        section_status_series=section_series,
        now=end,
    )

    notifier = Notifier(ssot=ssot, sensor_name=sensor_name, logger=logger, db=db, global_debug=global_debug)
    event_logger = EventLogger(ssot=ssot, sensor_name=sensor_name, logger=logger, db=db)

    # SENSOR DEBUG: send test email every run
    if sensor_debug:
        logger.warning(f"{sensor_name}: SENSOR DEBUG ENABLED -> sending TEST email every run")
        notifier.send_test_email(
            score=float(decision.latest_score),
            section_status=float(decision.latest_section_status),
            latest_ts=latest_ts.to_pydatetime() if hasattr(latest_ts, "to_pydatetime") else latest_ts,
            feature_contrib=feature_contrib,
            reason=decision.reason,
        )
        return

    # Option B: once per episode
    store = StateStore(base_dir=site_root / "etc" / "state")
    state = store.load(sensor_name)

    if decision.should_alert:
        if not state.get("episode_active", False):
            state["episode_active"] = True
            state["episode_started_at"] = decision.anomaly_started_at.isoformat() if decision.anomaly_started_at else None
            store.save(sensor_name, state)

            event_logger.log_event(
                score=float(decision.latest_score),
                level=1,
                trigger_time=_utc_now_naive(),
                latest_ts=latest_ts.to_pydatetime() if hasattr(latest_ts, "to_pydatetime") else None,
                feature_contrib=feature_contrib,
                details={
                    "reason": decision.reason,
                    "method": method_name,
                    "alarm_thresh": decision.alarm_thresh,
                    "section_status": float(decision.latest_section_status),
                },
            )

            notifier.send_alert(
                score=float(decision.latest_score),
                section_status=float(decision.latest_section_status),
                latest_ts=latest_ts.to_pydatetime() if hasattr(latest_ts, "to_pydatetime") else latest_ts,
                feature_contrib=feature_contrib,
                reason=decision.reason,
            )
            logger.warning(f"{sensor_name}: ALERT SENT (episode start) | score={decision.latest_score:.3f}")
        else:
            logger.info(f"{sensor_name}: alert suppressed (episode already active) | score={decision.latest_score:.3f}")
    else:
        # Recovery closes episode
        if state.get("episode_active", False) and decision.reason in ("score_normal", "gate_closed"):
            state["episode_active"] = False
            state["episode_started_at"] = None
            store.save(sensor_name, state)
            logger.info(f"{sensor_name}: episode closed (recovered)")

        logger.info(
            f"{sensor_name}: no alert | reason={decision.reason} | score={decision.latest_score:.3f} | sec={decision.latest_section_status:.3f}"
        )