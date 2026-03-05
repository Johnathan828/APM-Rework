# flask_app/app.py
from __future__ import annotations

import configparser
import datetime
import json
import logging
import os
import signal
from datetime import timedelta
from pathlib import Path
from threading import Lock
from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.io as pio
import pytz
from flask import Blueprint, Flask, jsonify, render_template, request, send_file, url_for
from werkzeug.utils import secure_filename

from modules.DBmanager import DBInterface
from modules.heartbeat_reader import read_heartbeat, status_from_heartbeat
from modules.pooling import MSSQLConnectionPool
from modules.process_control import (
    start_sensor,
    stop_sensor,
    supervisor_status,
    start_supervisor,
    stop_supervisor,
)
from modules.ssot_adapter import build_name_maps, load_ssot, ssot_as_data

pio.renderers.default = "browser"

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
FLASK_ROOT = Path(__file__).resolve().parent
SITE_ROOT = FLASK_ROOT.parent
STATIC_DATA = FLASK_ROOT / "static" / "data"

# -----------------------------------------------------------------------------
# config.ini
# -----------------------------------------------------------------------------
config = configparser.ConfigParser()
ini_path = SITE_ROOT / "etc" / "config.ini"
config.read(str(ini_path))

flask_host = config.get("FLASK", "host", fallback="0.0.0.0")
flask_port = int(config.get("FLASK", "port", fallback="8871"))
url_prefix = config.get("FLASK", "url_prefix", fallback="/Neuromine/LIO/APM/")
tz_name = config.get("FLASK", "timezone", fallback="Africa/Johannesburg")
os.environ["TZ"] = tz_name

# -----------------------------------------------------------------------------
# Flask app + blueprint
# -----------------------------------------------------------------------------
app = Flask(__name__, static_folder="static")
neuromine_bp = Blueprint("neuromine", __name__, url_prefix=url_prefix)

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logger = logging.getLogger("flask_apm")
logger.setLevel(logging.INFO)

log_path = SITE_ROOT / "etc" / "logs" / "flask_app.log"
log_path.parent.mkdir(parents=True, exist_ok=True)

handler = logging.FileHandler(str(log_path))
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S%z",
)
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)

# -----------------------------------------------------------------------------
# SSOT (config.json)
# -----------------------------------------------------------------------------
ssot = load_ssot(SITE_ROOT)
data = ssot_as_data(ssot)  # {sensor_key: cfg}
desc_to_key, key_to_desc = build_name_maps(data)

# -----------------------------------------------------------------------------
# MSSQL pool + DB layer
# -----------------------------------------------------------------------------
def _mssql_section_name() -> str:
    return "SQLALCHEMY_DEV" if bool(ssot.get("debug", False)) else "SQLALCHEMY"


SQL_SECTION = _mssql_section_name()
MSSQL_TABLE = config[SQL_SECTION]["table"]

mssql_pool = MSSQLConnectionPool(
    server=config[SQL_SECTION]["host"],
    database=config[SQL_SECTION]["dbname"],
    user=config[SQL_SECTION]["user"],
    password=config[SQL_SECTION]["password"],
    port=config[SQL_SECTION]["port"],
    max_connections=3,
)

entity_id_dict_path = STATIC_DATA / "Gams_key_value.json"
try:
    with open(entity_id_dict_path, "r") as f:
        entity_id_dict = json.load(f)
except Exception:
    entity_id_dict = {}
    logger.warning("Gams_key_value.json missing/unreadable (ok).")

data_collector = DBInterface(
    config=config,
    logger=logger,
    mssql_pool=mssql_pool,
    mssql_table=MSSQL_TABLE,
    entity_id_dict=entity_id_dict,
)

# -----------------------------------------------------------------------------
# modelsRunStatus.json (UI state)
# -----------------------------------------------------------------------------
models_status_path = STATIC_DATA / "modelsRunStatus.json"


def init_models_status() -> dict:
    models_status_path.parent.mkdir(parents=True, exist_ok=True)
    if models_status_path.exists():
        with open(models_status_path, "r") as f:
            ms = json.load(f)
    else:
        ms = {}

    for sensor_key, cfg in data.items():
        pretty = str(cfg.get("Model_Description", sensor_key))
        if pretty not in ms:
            ms[pretty] = {"sensor_key": sensor_key, "status": "Stopped", "PID": None, "enabled": False}

        ms[pretty]["sensor_key"] = sensor_key
        ms[pretty].setdefault("enabled", False)
        ms[pretty].setdefault("status", "Stopped")
        ms[pretty].setdefault("PID", None)
        ms[pretty].pop("path", None)

    with open(models_status_path, "w") as f:
        json.dump(ms, f, indent=4)

    return ms


models_status = init_models_status()

# -----------------------------------------------------------------------------
# Cache
# -----------------------------------------------------------------------------
cache_lock = Lock()
CACHE = {
    "model_df": {"ts": None, "ttl_s": 300, "value": pd.DataFrame()},  # 5 min
    "events": {"ts": None, "ttl_s": 300, "value": []},  # 5 min
    "model_df_larger": {"ts": None, "ttl_s": 900, "value": pd.DataFrame()},  # 15 min
}


def _now_utc():
    return datetime.datetime.utcnow()


def _cache_fresh(entry) -> bool:
    ts = entry["ts"]
    if ts is None:
        return False
    return (_now_utc() - ts).total_seconds() < float(entry["ttl_s"])


def get_latest_events_per_model(events, N=1):
    model_events = {}
    for event in events:
        mn = event.get("model_name")
        model_events.setdefault(mn, []).append(event)

    out = []
    for mn, lst in model_events.items():
        lst_sorted = sorted(lst, key=lambda x: x.get("trigger_time"), reverse=True)
        out.extend(lst_sorted[:N])
    return out


def get_cached_model_df(minutes: int = 15) -> pd.DataFrame:
    with cache_lock:
        if _cache_fresh(CACHE["model_df"]):
            return CACHE["model_df"]["value"]

    date_to = datetime.datetime.utcnow()
    date_from = date_to - datetime.timedelta(minutes=minutes)

    model_cols = list(data.keys())
    section_cols = [
        (data[k].get("filter_tag", {}) or {}).get("FilterTagName", f"{k}_Section_Status")
        for k in data.keys()
    ]
    cols = list(dict.fromkeys(model_cols + section_cols))

    try:
        df = data_collector.get_neuro_displayname_data(date_from=date_from, date_to=date_to, model_cols=cols)
    except Exception as e:
        logger.error(f"get_cached_model_df failed: {e}", exc_info=True)
        df = pd.DataFrame()
    finally:
        data_collector.release_connections()

    with cache_lock:
        CACHE["model_df"]["value"] = df
        CACHE["model_df"]["ts"] = _now_utc()
    return df


def get_cached_model_df_larger(hours: int = 6, cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Cached MSSQL pull for larger window (used by plots).
    If cols is None: defaults to only SSOT sensor keys.
    """
    cache_key = "model_df_larger"

    with cache_lock:
        if _cache_fresh(CACHE[cache_key]):
            df_cached = CACHE[cache_key]["value"]
            if cols is None:
                return df_cached
            if isinstance(df_cached, pd.DataFrame) and not df_cached.empty:
                missing = [c for c in cols if c not in df_cached.columns]
                if not missing:
                    return df_cached

    date_to = datetime.datetime.utcnow()
    date_from = date_to - datetime.timedelta(hours=hours)

    if cols is None:
        cols = list(data.keys())

    try:
        df = data_collector.get_neuro_displayname_data(date_from=date_from, date_to=date_to, model_cols=cols)
    except Exception as e:
        logger.error(f"get_cached_model_df_larger failed: {e}", exc_info=True)
        df = pd.DataFrame()
    finally:
        data_collector.release_connections()

    with cache_lock:
        CACHE[cache_key]["value"] = df
        CACHE[cache_key]["ts"] = _now_utc()
    return df


def get_cached_events(days: int = 1):
    with cache_lock:
        if _cache_fresh(CACHE["events"]):
            return CACHE["events"]["value"]

    date_to = datetime.datetime.utcnow()
    date_from = date_to - datetime.timedelta(days=days)

    try:
        ev = data_collector.get_model_event_data(
            model_name=list(data.keys()),
            start_time=date_from,
            end_time=date_to,
            level=1,
        ) or []
    except Exception as e:
        logger.error(f"get_cached_events failed: {e}", exc_info=True)
        ev = []
    finally:
        data_collector.release_connections()

    with cache_lock:
        CACHE["events"]["value"] = ev
        CACHE["events"]["ts"] = _now_utc()
    return ev


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@neuromine_bp.route("/")
def home():
    models_info = []
    for sensor_key, cfg in data.items():
        models_info.append(
            {
                "name": sensor_key,
                "description": cfg.get("Model_Description", sensor_key),
                "num_features": len(cfg.get("Features", {}) or {}),
            }
        )
    return render_template("home.html", models_info=models_info)


@neuromine_bp.route("/lives_agents")
def lives_agents():
    # IMPORTANT: pass base_url so JS hits the correct prefixed routes
    return render_template("live_agents.html", base_url=url_for("neuromine.home"))


@neuromine_bp.route("/lives_agents_data", methods=["GET"])
def lives_agents_data():
    date_to = datetime.datetime.now()
    live_update = []
    summary_stats = {}

    try:
        model_df = get_cached_model_df(minutes=15)
        events = get_cached_events(days=1)
        filtered_events = get_latest_events_per_model(events, N=2) if events else []

        if not model_df.empty:
            for sensor_key, cfg in data.items():
                status_tag = (cfg.get("filter_tag", {}) or {}).get("FilterTagName", f"{sensor_key}_Section_Status")

                if sensor_key in model_df.columns and not model_df[sensor_key].dropna().empty:
                    health_score = float(model_df[sensor_key].dropna().iloc[-1]) * 100.0
                else:
                    health_score = 0.0
                health_score = round(health_score, 1)

                if status_tag in model_df.columns and not model_df[status_tag].dropna().empty:
                    on_off_status = float(model_df[status_tag].dropna().iloc[-1])
                else:
                    on_off_status = 0.0

                alarm_thresh = float((cfg.get("Other", {}) or {}).get("alarm_thresh", 0.75)) * 100.0

                if on_off_status == 0:
                    latest_result = "Section Off"
                elif health_score < 0:
                    latest_result = "No Data"
                    health_score = 0
                elif (health_score > alarm_thresh) and (health_score < alarm_thresh + (100 - alarm_thresh) * 0.5):
                    latest_result = "Warning"
                elif health_score > alarm_thresh:
                    latest_result = "Normal"
                else:
                    latest_result = "Triggered"

                live_update.append(
                    {
                        "agent_name": cfg.get("Model_Description", sensor_key),
                        "plant_section": cfg.get("Plant Section", "Unknown"),
                        "type": "Machine Learning",
                        "latest_result": latest_result,
                        "probability_date": date_to,
                        "alert_probability": health_score,
                        "events": [ev for ev in filtered_events if ev.get("model_name") == sensor_key],
                    }
                )

        total_events = len(filtered_events)
        average_health = sum(ev.get("score", 0) for ev in filtered_events) / total_events if total_events > 0 else 0

        summary_stats = {
            "Total_events": total_events,
            "average_health": round(average_health, 2),
            "Total_monitored_sections": 1,
            "total_tag_inputs": 0,
        }

    except Exception:
        logger.error("lives_agents_data exception", exc_info=True)

    return jsonify(models=live_update, summary_stats=summary_stats)


@neuromine_bp.route("/get-image/<image_name>")
def serve_dynamic_image(image_name):
    """
    Legacy-compatible: Live Agents groups try to load /get-image/<PlantSection>.gif
    """
    filename = secure_filename(image_name)
    image_path = FLASK_ROOT / "static" / "images" / filename

    if image_path.exists():
        return send_file(str(image_path), mimetype="image/gif")
    return jsonify({"error": "Image not found"}), 404


@neuromine_bp.route("/get_model_health")
def get_model_health():
    """
    SSOT + legacy-compatible endpoint used by Live Agents click.

    Health score:
      - MSSQL APM_Scalability(_Dev): displayname=sensor score (sensor_key)

    Feature plots:
      - Postgres sri_get_tag_data via build_wide_from_postgres (raw tag trends)
    """
    agent_name = (request.args.get("agent_name") or "").strip()
    if not agent_name:
        return jsonify({"error": "agent_name is required"}), 400

    # Resolve sensor_key robustly:
    description_map = {str(cfg.get("Model_Description", k)): k for k, cfg in data.items()}
    if agent_name in data:
        sensor_key = agent_name
    elif agent_name in description_map:
        sensor_key = description_map[agent_name]
    else:
        return jsonify({"error": f"Unknown agent_name: {agent_name}"}), 400

    sensor_cfg = data.get(sensor_key, {}) or {}
    features_cfg = sensor_cfg.get("Features", {}) or {}

    # Raw tags to plot (SSOT-driven)
    tags = []
    tag_to_feature = {}
    for feat_name, feat_cfg in features_cfg.items():
        tag = (feat_cfg or {}).get("tag")
        if tag:
            tags.append(tag)
            tag_to_feature[tag] = feat_name

    # Score column candidates (robust): sensor_key, and (optionally) FilterTagName
    score_displaynames = [sensor_key]
    status_tag = (sensor_cfg.get("filter_tag", {}) or {}).get("FilterTagName")
    if status_tag and status_tag not in score_displaynames:
        score_displaynames.append(status_tag)

    try:
        now_utc = pd.Timestamp.utcnow()
        start_utc = now_utc - pd.Timedelta(hours=6)

        # -------------------------
        # 1) Health score from MSSQL
        # -------------------------
        model_df = get_cached_model_df_larger(hours=6, cols=score_displaynames)
        if model_df is None:
            model_df = pd.DataFrame()

        timestamps = []
        health_scores = []

        if not model_df.empty:
            model_df = model_df.sort_index()
            model_df.replace(-1, np.nan, inplace=True)
            model_df.ffill(inplace=True)
            model_df.bfill(inplace=True)

            score_col = sensor_key if sensor_key in model_df.columns else None
            if score_col is not None and not model_df[score_col].dropna().empty:
                timestamps = model_df.index.strftime("%Y-%m-%d %H:%M:%S").tolist()
                health_scores = (model_df[score_col].astype(float) * 100.0).tolist()

        logger.info(
            "GET_MODEL_HEALTH | agent=%s sensor_key=%s | model_df.empty=%s cols=%s | points=%s",
            agent_name,
            sensor_key,
            model_df.empty,
            list(model_df.columns)[:20] if isinstance(model_df, pd.DataFrame) and not model_df.empty else [],
            0 if not timestamps else len(timestamps),
        )

        # -------------------------
        # 2) Raw feature tags from Postgres
        # -------------------------
        feature_timestamps = []
        feature_df_payload = {}

        # Postgres interval literal
        interval_str = "1 minute"

        if tags:
            raw_wide = data_collector.build_wide_from_postgres(
                tags=tags,
                start=start_utc.to_pydatetime(),
                end=now_utc.to_pydatetime(),
                interval_str=interval_str,
            )

            if raw_wide is None:
                raw_wide = pd.DataFrame(index=pd.DatetimeIndex([]))

            if not raw_wide.empty:
                raw_wide = raw_wide.sort_index()
                raw_wide.replace(-1, np.nan, inplace=True)
                raw_wide.ffill(inplace=True)
                raw_wide.bfill(inplace=True)

                feature_timestamps = raw_wide.index.strftime("%Y-%m-%d %H:%M:%S").tolist()

                for tag_col in raw_wide.columns:
                    feat_name = tag_to_feature.get(tag_col)
                    if not feat_name:
                        continue
                    vals = raw_wide[tag_col].astype(float).where(pd.notnull(raw_wide[tag_col]), None).tolist()
                    feature_df_payload[feat_name] = vals

        # -------------------------
        # 3) Feature state highlighting
        # If you later wire contributions into MSSQL, compute it here.
        # For now keep stable.
        # -------------------------
        feature_state = {feat: 0.0 for feat in features_cfg.keys()}

        payload = {
            "timestamps": timestamps,
            "health_scores": health_scores,
            "feature_timestamps": feature_timestamps,
            "feature_df": feature_df_payload,
            "health_score": health_scores[-1] if health_scores else 0,
            "last_check": timestamps[-1] if timestamps else None,
            "feature_state": feature_state,
            "feature_details": sensor_cfg,
        }
        return jsonify(payload)

    except Exception:
        logger.error("get_model_health exception", exc_info=True)
        return jsonify({"error": "Error fetching model health"}), 500

    finally:
        try:
            data_collector.release_connections()
        except Exception:
            pass


@neuromine_bp.route("/get_alert_detail", methods=["GET"])
def get_alert_detail():
    """
    Legacy-compatible endpoint used by Recent Alert Events View.

    IMPORTANT FIX:
    Use Postgres raw tag pull for feature contributions (no get_timescale_data dependency).
    """
    agent_name = request.args.get("agent_name")
    trigger_time = request.args.get("trigger_time")

    if not agent_name or not trigger_time:
        return jsonify({"error": "agent_name and trigger_time are required"}), 400

    sensor_key = agent_name
    sensor_cfg = data.get(sensor_key, {}) or {}
    features_cfg = sensor_cfg.get("Features", {}) or {}

    try:
        date_to = pd.to_datetime(trigger_time, utc=True, errors="coerce")
        if pd.isna(date_to):
            date_to = pd.to_datetime(trigger_time, errors="coerce")
        if pd.isna(date_to):
            return jsonify({"error": f"Could not parse trigger_time: {trigger_time}"}), 400

        date_to_dt = date_to.to_pydatetime()
        start_time = date_to_dt - datetime.timedelta(seconds=5)
        pull_start = date_to_dt - datetime.timedelta(hours=5)

        # Get event (with contributions)
        events = data_collector.get_model_event_data(
            model_name=[sensor_key],
            start_time=start_time,
            end_time=date_to_dt,
        ) or []

        if not events:
            return jsonify({"error": "No event found in window"}), 404

        event0 = events[0]
        event_details = event0.get("event_details", {}) or {}
        contrib = (event_details.get("Feature") or event_details.get("feature") or {}) or {}

        triggered_tags = []
        for feat_name, v in contrib.items():
            if feat_name in features_cfg:
                triggered_tags.append({"feature": feat_name, "value": float(v)})

        # Model score history from MSSQL
        model_df = data_collector.get_neuro_displayname_data(
            date_from=pull_start,
            date_to=date_to_dt,
            model_cols=[sensor_key],
        )
        if model_df is None:
            model_df = pd.DataFrame()

        if not model_df.empty:
            model_df.replace(-1, np.nan, inplace=True)
            model_df.ffill(inplace=True)
            model_df = model_df.where(pd.notnull(model_df), None)

        model_df = model_df.reset_index()
        model_health_data = model_df.rename(columns={"index": "timestamp"}).to_dict(orient="records")

        # Feature history from Postgres raw tags (SSOT tags)
        tags = []
        tag_to_feature = {}
        for feat_name, feat_cfg in features_cfg.items():
            tag = (feat_cfg or {}).get("tag")
            if tag:
                tags.append(tag)
                tag_to_feature[tag] = feat_name

        feature_contributions = {}
        if tags:
            raw_wide = data_collector.build_wide_from_postgres(
                tags=tags,
                start=pull_start,
                end=date_to_dt,
                interval_str="1 minute",
            )
            if raw_wide is None:
                raw_wide = pd.DataFrame(index=pd.DatetimeIndex([]))

            if not raw_wide.empty:
                raw_wide = raw_wide.sort_index()
                raw_wide.replace(-1, np.nan, inplace=True)
                raw_wide.ffill(inplace=True)
                raw_wide.bfill(inplace=True)

                # Build legacy-like per-feature series: [{timestamp, value}, ...]
                for tag_col in raw_wide.columns:
                    feat_name = tag_to_feature.get(tag_col, tag_col)
                    series_df = pd.DataFrame(
                        {"timestamp": raw_wide.index.strftime("%Y-%m-%d %H:%M:%S"), "value": raw_wide[tag_col].tolist()}
                    )
                    feature_contributions[feat_name] = series_df.to_dict(orient="records")

        return jsonify(
            {
                "triggered_tags": triggered_tags,
                "model_health_data": model_health_data,
                "feature_contributions": feature_contributions,
                "model_details": sensor_cfg,
            }
        )

    except Exception:
        logger.error("get_alert_detail exception", exc_info=True)
        return jsonify({"error": "Error fetching alert detail"}), 500

    finally:
        try:
            data_collector.release_connections()
        except Exception:
            pass


@neuromine_bp.route("/startstop", methods=["GET"])
def show_start_stop_page():
    """
    Recompute UI status from heartbeat + enabled switch.
    Keeps UI consistent with backend after supervisor stop.
    """
    allowed_pretty_names = set(desc_to_key.keys())

    for pretty, info in models_status.items():
        if pretty not in allowed_pretty_names:
            continue

        sensor_key = info.get("sensor_key")
        enabled = bool(info.get("enabled", False))

        if not enabled or not sensor_key:
            models_status[pretty]["status"] = "Stopped"
            models_status[pretty]["PID"] = None
            continue

        hb = read_heartbeat(SITE_ROOT, sensor_key)
        hb_status = status_from_heartbeat(hb)

        if hb_status in ("Running", "Stale"):
            models_status[pretty]["status"] = "Running"
        else:
            models_status[pretty]["status"] = "Stopped"
            models_status[pretty]["PID"] = None

    with open(models_status_path, "w") as f:
        json.dump(models_status, f, indent=4)

    agents = []
    stopped_count = 0
    running_count = 0

    for pretty, details in models_status.items():
        if pretty not in allowed_pretty_names:
            continue

        st = details.get("status", "Stopped")
        if st == "Stopped":
            stopped_count += 1
        else:
            running_count += 1

        agents.append(
            {
                "name": pretty,
                "status": st,
                "action": "Start" if st == "Stopped" else "Stop",
            }
        )

    sup_status, _ = supervisor_status(SITE_ROOT)

    return render_template(
        "start_stop_agents.html",
        agents=agents,
        stopped_count=stopped_count,
        running_count=running_count,
        supervisor_status=sup_status,
    )


@neuromine_bp.route("/supervisor_action", methods=["POST"])
def supervisor_action():
    payload = request.get_json() or {}
    action = (payload.get("action") or "").strip().lower()

    if action == "start":
        pid = start_supervisor(SITE_ROOT)
        st, _ = supervisor_status(SITE_ROOT)
        return jsonify({"status": st, "pid": pid, "action": "Stop"})

    if action == "stop":
        killed = stop_supervisor(SITE_ROOT)

        # disable all sensors in UI state
        for pretty, info in models_status.items():
            if pretty in desc_to_key:
                info["enabled"] = False
                info["status"] = "Stopped"
                info["PID"] = None

        with open(models_status_path, "w") as f:
            json.dump(models_status, f, indent=4)

        st, _ = supervisor_status(SITE_ROOT)
        return jsonify({"status": st, "killed": killed, "action": "Start"})

    st, pid = supervisor_status(SITE_ROOT)
    return jsonify({"status": st, "pid": pid, "action": "Stop" if st in ("Running", "Stale") else "Start"})


@neuromine_bp.route("/startstop_action", methods=["POST"])
def start_stop_agent():
    payload = request.get_json() or {}
    action = (payload.get("action") or "").strip()
    agent_name = payload.get("agent_name")

    now = datetime.datetime.now(pytz.timezone(tz_name))
    updated_agents = []

    if action == "StartAll":
        start_supervisor(SITE_ROOT)

        for pretty, info in models_status.items():
            sensor_key = info.get("sensor_key")
            if not sensor_key or sensor_key not in data:
                continue

            try:
                models_status[pretty]["enabled"] = True
                start_sensor(SITE_ROOT, sensor_key)
                models_status[pretty]["status"] = "Running"  # optimistic
                models_status[pretty]["PID"] = None
                updated_agents.append({"name": pretty, "status": "Running"})
                logger.info(f"Started {pretty} ({sensor_key}) at {now.isoformat()}")
            except Exception as e:
                logger.error(f"Error starting {pretty}: {e}", exc_info=True)

        with open(models_status_path, "w") as f:
            json.dump(models_status, f, indent=4)

        return jsonify(
            {
                "status": "Success",
                "updated_agents": updated_agents,
                "running_count": sum(1 for a in models_status.values() if a.get("status") == "Running"),
                "stopped_count": sum(1 for a in models_status.values() if a.get("status") == "Stopped"),
            }
        )

    if not agent_name or agent_name not in models_status:
        return jsonify({"status": "Error", "message": "Agent not found"}), 404

    sensor_key = models_status[agent_name].get("sensor_key")
    if not sensor_key or sensor_key not in data:
        return jsonify({"status": "Error", "message": "Not a reworked SSOT sensor"}), 400

    if action.lower() == "start":
        try:
            models_status[agent_name]["enabled"] = True
            start_sensor(SITE_ROOT, sensor_key)
            models_status[agent_name]["status"] = "Running"
            models_status[agent_name]["PID"] = None
            logger.info(f"Started {agent_name} ({sensor_key}) at {now.isoformat()}")
        except Exception as e:
            logger.error(f"Error starting {agent_name}: {e}", exc_info=True)

    elif action.lower() == "stop":
        try:
            stop_sensor(SITE_ROOT, sensor_key)
            models_status[agent_name]["enabled"] = False
            models_status[agent_name]["status"] = "Stopped"
            models_status[agent_name]["PID"] = None
            logger.info(f"Stopped {agent_name} ({sensor_key}) at {now.isoformat()}")
        except Exception as e:
            logger.error(f"Error stopping {agent_name}: {e}", exc_info=True)

    with open(models_status_path, "w") as f:
        json.dump(models_status, f, indent=4)

    return jsonify(
        {
            "status": models_status[agent_name].get("status"),
            "enabled": bool(models_status[agent_name].get("enabled", False)),
            "action": "Stop" if models_status[agent_name].get("status") == "Running" else "Start",
            "running_count": sum(1 for a in models_status.values() if a.get("status") == "Running"),
            "stopped_count": sum(1 for a in models_status.values() if a.get("status") == "Stopped"),
        }
    )


@neuromine_bp.route("/proxy-banner")
def proxy_banner():
    image_path = FLASK_ROOT / "static" / "images" / "MinopexBanner.png"
    return send_file(str(image_path), mimetype="image/png")


def shutdown_handler(signum, frame):
    logger.info("Flask shutdown signal received")
    try:
        stop_supervisor(SITE_ROOT)
    except Exception:
        pass
    os._exit(0)


signal.signal(signal.SIGINT, shutdown_handler)
signal.signal(signal.SIGTERM, shutdown_handler)

app.register_blueprint(neuromine_bp)

if __name__ == "__main__":
    print(f"Flask PID={os.getpid()} WERKZEUG_RUN_MAIN={os.environ.get('WERKZEUG_RUN_MAIN')}")
    print(f"Flask app running on {flask_host}:{flask_port} prefix={url_prefix}")
    app.run(host=flask_host, port=flask_port, debug=False, use_reloader=False)