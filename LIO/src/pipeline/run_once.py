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

    # --- MODEL ---
    model, method_name, method_cfg = build_model(
        sensor_cfg=sensor_cfg, sensor_name=sensor_name, site_root=site_root
    )
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

    # --- Section status aligned to score timestamps ---
    ft_name = sensor.filter_tag_displayname or f"{sensor_name}_Section_Status"

    filter_tag_series = filter_tag_series.copy()
    filter_tag_series.index = pd.to_datetime(filter_tag_series.index, utc=True).tz_convert(None)
    filter_tag_series = filter_tag_series.sort_index()

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

    # Feature contributions (latest only to MSSQL)
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

    # ------------------------------------------------------------------
    # Build a "legacy-like" email feature_contrib window:
    # Use last alert_holding_minutes so likely-cause uses max() over window.
    # ------------------------------------------------------------------
    holding_minutes_for_table = int(sensor.other.get("alert_holding_minutes", 15))
    window_start_ts = latest_ts - pd.Timedelta(minutes=holding_minutes_for_table)

    if feature_contrib is not None and not feature_contrib.empty:
        feature_contrib_email = feature_contrib.loc[feature_contrib.index >= window_start_ts].copy()
    else:
        feature_contrib_email = feature_contrib

    # ------------------------------------------------------------------
    # Email context for Detection Mode + Trigger column (WINDOW-BASED)
    # ------------------------------------------------------------------
    pretty_method = {
        "FixedThresholds": "Fixed Thresholds",
        "StatisticalThresholds": "Statistical Thresholds"
    }.get(method_name, method_name)

    # Build a windowed df_wide aligned to UTC-naive + same window as likely-cause table
    df_wide_window = None
    try:
        df_wide2 = df_wide.copy()
        df_wide2.index = pd.to_datetime(df_wide2.index, utc=True).tz_convert(None)
        df_wide2 = df_wide2.sort_index()
        df_wide_window = df_wide2.loc[df_wide2.index >= window_start_ts].copy()
    except Exception:
        df_wide_window = None

    # Threshold maps (for trigger text)
    thr_high = {}
    thr_low = {}
    if method_name == "FixedThresholds":
        thr_high = (method_cfg.get("high", {}) or {})
        thr_low = (method_cfg.get("low", {}) or {})
    elif method_name == "StatisticalThresholds":
        thr_high = getattr(model, "high_map", {}) or {}
        thr_low = getattr(model, "low_map", {}) or {}

    trigger_context = {
        "pretty_method": pretty_method,
        # raw values WINDOW (tag -> pd.Series)
        "raw_window_by_tag": {
            tag: pd.to_numeric(df_wide_window[tag], errors="coerce")
            for tag in feature_displaynames_to_tags.values()
            if df_wide_window is not None and tag in df_wide_window.columns
        },
        "thresholds": {"high": thr_high, "low": thr_low},
    }

    # SENSOR DEBUG: send test email every run
    if sensor_debug:
        logger.warning(f"{sensor_name}: SENSOR DEBUG ENABLED -> sending TEST email every run")
        notifier.send_test_email(
            score=float(decision.latest_score),
            section_status=float(decision.latest_section_status),
            latest_ts=latest_ts.to_pydatetime() if hasattr(latest_ts, "to_pydatetime") else latest_ts,
            feature_contrib=feature_contrib_email,  # <-- windowed
            reason=decision.reason,
            alarm_thresh=float(decision.alarm_thresh),
            filter_value=float(decision.filter_value),
            method_name=method_name,
            trigger_context=trigger_context,
        )
        return

    # ------------------------------------------------------------------
    # LEGACY-LIKE ALERTING: cooldown repeats + per-sensor "email in flight" guard
    # - send immediately when decision.should_alert=True
    # - repeat at most once per alert_holding_minutes while anomaly persists
    # - prevent overlapping emails for this sensor using state lock
    # ------------------------------------------------------------------
    store = StateStore(base_dir=site_root / "etc" / "state")
    state = store.load(sensor_name)

    holding_minutes = int(sensor.other.get("alert_holding_minutes", 15))
    email_inflight_timeout_minutes = int(sensor.other.get("email_inflight_timeout_minutes", 30))
    now_utc = _utc_now_naive()

    def _parse_iso_dt(s: str):
        try:
            return datetime.fromisoformat(s.replace("Z", ""))
        except Exception:
            return None

    if decision.should_alert:
        # Cooldown gate (legacy-like)
        last_sent_at = _parse_iso_dt(str(state.get("last_sent_at", "")))
        if last_sent_at is not None:
            mins_since = (now_utc - last_sent_at).total_seconds() / 60.0
            if mins_since < holding_minutes:
                logger.info(
                    f"{sensor_name}: alert suppressed (cooldown) | "
                    f"mins_since_last={mins_since:.2f} < holding={holding_minutes}"
                )
                return

        # In-flight guard (prevents overlapping email generation/sends for this sensor)
        in_flight = bool(state.get("email_in_flight", False))
        in_flight_since_raw = str(state.get("email_in_flight_since", "")) if state.get("email_in_flight_since") else ""
        in_flight_since = _parse_iso_dt(in_flight_since_raw) if in_flight_since_raw else None

        if in_flight:
            if in_flight_since is None:
                # malformed timestamp -> reset
                state["email_in_flight"] = False
                state["email_in_flight_since"] = None
                store.save(sensor_name, state)
            else:
                mins_inflight = (now_utc - in_flight_since).total_seconds() / 60.0
                if mins_inflight < email_inflight_timeout_minutes:
                    logger.warning(
                        f"{sensor_name}: email suppressed (in_flight) | "
                        f"mins_inflight={mins_inflight:.2f} < timeout={email_inflight_timeout_minutes}"
                    )
                    return
                else:
                    logger.warning(
                        f"{sensor_name}: email_in_flight stale -> resetting | "
                        f"mins_inflight={mins_inflight:.2f} >= timeout={email_inflight_timeout_minutes}"
                    )
                    state["email_in_flight"] = False
                    state["email_in_flight_since"] = None
                    store.save(sensor_name, state)

        # Log event (optional)
        try:
            event_logger.log_event(
                score=float(decision.latest_score),
                level=1,
                trigger_time=now_utc,
                latest_ts=latest_ts.to_pydatetime() if hasattr(latest_ts, "to_pydatetime") else None,
                feature_contrib=feature_contrib_email,  # <-- windowed
                details={
                    "reason": decision.reason,
                    "method": method_name,
                    "alarm_thresh": float(decision.alarm_thresh),
                    "section_status": float(decision.latest_section_status),
                    "cooldown_minutes": holding_minutes,
                },
            )
            logger.info(f"{sensor_name}: event logged to NewApmModelEvents")
        except Exception as e:
            logger.exception(f"{sensor_name}: event logging failed (continuing to email) | err={e}")

        # Set in-flight lock BEFORE heavy work (plots + SMTP)
        state["email_in_flight"] = True
        state["email_in_flight_since"] = now_utc.isoformat()
        store.save(sensor_name, state)

        # Send email; on failure, do NOT update last_sent_at (so we retry next tick)
        try:
            notifier.send_alert(
                score=float(decision.latest_score),
                section_status=float(decision.latest_section_status),
                latest_ts=latest_ts.to_pydatetime() if hasattr(latest_ts, "to_pydatetime") else latest_ts,
                feature_contrib=feature_contrib_email,  # <-- windowed
                reason=decision.reason,
                alarm_thresh=float(decision.alarm_thresh),
                filter_value=float(decision.filter_value),
                method_name=method_name,
                trigger_context=trigger_context,
            )
        except Exception as e:
            logger.exception(f"{sensor_name}: ALERT FAILED (email) | err={e}")
            return
        finally:
            # Always clear in-flight lock
            state["email_in_flight"] = False
            state["email_in_flight_since"] = None
            store.save(sensor_name, state)

        # Update cooldown timestamp AFTER success
        state["last_sent_at"] = now_utc.isoformat()
        store.save(sensor_name, state)

        logger.warning(f"{sensor_name}: ALERT SENT (cooldown) | score={decision.latest_score:.3f}")

    else:
        logger.info(
            f"{sensor_name}: no alert | reason={decision.reason} | score={decision.latest_score:.3f} | sec={decision.latest_section_status:.3f}"
        )