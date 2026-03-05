# flask_app/app.py
from __future__ import annotations

import configparser
import datetime
import json
import logging
import os
import signal
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional

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

# This influences time.* functions on many linux systems,
# but NOT pandas UTC calls. We still explicitly convert where needed.
os.environ["TZ"] = tz_name
TZ = pytz.timezone(tz_name)

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


def invalidate_cache(*keys: str):
    """Force cache refresh (used when starting/stopping sensors)."""
    with cache_lock:
        for k in keys:
            if k in CACHE:
                CACHE[k]["ts"] = None


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
# Runtime truth overlay (supervisor + enabled + heartbeat)
# -----------------------------------------------------------------------------
def runtime_state_by_sensor() -> Dict[str, str]:
    """
    Returns {sensor_key: "Running"|"Stopped"} based on:
      - supervisor status
      - enabled flag
      - heartbeat freshness
    """
    sup_st, _ = supervisor_status(SITE_ROOT)
    supervisor_ok = sup_st in ("Running", "Stale")  # your existing semantics

    out: Dict[str, str] = {}

    # map pretty -> sensor_key + enabled
    for pretty, info in models_status.items():
        sensor_key = info.get("sensor_key")
        if not sensor_key or sensor_key not in data:
            continue

        enabled = bool(info.get("enabled", False))
        if not enabled or not supervisor_ok:
            out[sensor_key] = "Stopped"
            continue

        hb = read_heartbeat(SITE_ROOT, sensor_key)
        hb_st = status_from_heartbeat(hb)

        # IMPORTANT: for LIVE MONITORING, treat "Stale" as STOPPED,
        # otherwise a dead worker looks "running" for too long.
        if hb_st == "Running":
            out[sensor_key] = "Running"
        else:
            out[sensor_key] = "Stopped"

    # ensure all sensors exist in dict
    for sk in data.keys():
        out.setdefault(sk, "Stopped")

    return out


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
    # Use SAST for the displayed timestamp in UI
    date_to = datetime.datetime.now(TZ)

    live_update = []
    summary_stats = {}

    try:
        # Runtime truth overlay
        rt = runtime_state_by_sensor()

        model_df = get_cached_model_df(minutes=15)
        events = get_cached_events(days=1)
        filtered_events = get_latest_events_per_model(events, N=2) if events else []

        # Compute overall average health excluding Section Off / Stopped
        health_for_avg: List[float] = []

        for sensor_key, cfg in data.items():
            status_tag = (cfg.get("filter_tag", {}) or {}).get("FilterTagName", f"{sensor_key}_Section_Status")

            # If runtime says Stopped, override immediately (even if MSSQL has old "normal" rows)
            runtime_status = rt.get(sensor_key, "Stopped")
            if runtime_status != "Running":
                live_update.append(
                    {
                        "agent_name": cfg.get("Model_Description", sensor_key),
                        "plant_section": cfg.get("Plant Section", "Unknown"),
                        "type": "Machine Learning",
                        "latest_result": "Stopped",
                        "probability_date": date_to.strftime("%a, %d %b %Y %H:%M:%S %Z"),
                        "alert_probability": 0.0,
                        "events": [ev for ev in filtered_events if ev.get("model_name") == sensor_key],
                    }
                )
                continue

            # Otherwise, use DB data for score + section status
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

            # For overall health average: exclude Section Off
            if latest_result != "Section Off":
                health_for_avg.append(float(health_score))

            live_update.append(
                {
                    "agent_name": cfg.get("Model_Description", sensor_key),
                    "plant_section": cfg.get("Plant Section", "Unknown"),
                    "type": "Machine Learning",
                    "latest_result": latest_result,
                    "probability_date": date_to.strftime("%a, %d %b %Y %H:%M:%S %Z"),
                    "alert_probability": health_score,
                    "events": [ev for ev in filtered_events if ev.get("model_name") == sensor_key],
                }
            )

        total_events = len(filtered_events)
        average_health_events = (
            sum(ev.get("score", 0) for ev in filtered_events) / total_events if total_events > 0 else 0
        )

        overall_avg = (sum(health_for_avg) / len(health_for_avg)) if health_for_avg else 0.0

        summary_stats = {
            "Total_events": total_events,
            "average_health": round(average_health_events, 2),
            "overall_average_health": round(overall_avg, 2),  # ✅ used by JS Average Health tile
            "Total_monitored_sections": 1,
            "total_tag_inputs": 0,
        }

    except Exception:
        logger.error("lives_agents_data exception", exc_info=True)

    return jsonify(models=live_update, summary_stats=summary_stats)


@neuromine_bp.route("/get-image/<image_name>")
def serve_dynamic_image(image_name):
    filename = secure_filename(image_name)
    image_path = FLASK_ROOT / "static" / "images" / filename

    if image_path.exists():
        return send_file(str(image_path), mimetype="image/gif")
    return jsonify({"error": "Image not found"}), 404


# (get_model_health / get_alert_detail left as you already have them working)
# -----------------------------------------------------------------------------


@neuromine_bp.route("/startstop", methods=["GET"])
def show_start_stop_page():
    """
    Recompute UI status from heartbeat + enabled switch.
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
        invalidate_cache("model_df", "events", "model_df_larger")
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

        invalidate_cache("model_df", "events", "model_df_larger")
        st, _ = supervisor_status(SITE_ROOT)
        return jsonify({"status": st, "killed": killed, "action": "Start"})

    st, pid = supervisor_status(SITE_ROOT)
    return jsonify({"status": st, "pid": pid, "action": "Stop" if st in ("Running", "Stale") else "Start"})


@neuromine_bp.route("/startstop_action", methods=["POST"])
def start_stop_agent():
    payload = request.get_json() or {}
    action = (payload.get("action") or "").strip()
    agent_name = payload.get("agent_name")

    now = datetime.datetime.now(TZ)
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
                models_status[pretty]["status"] = "Running"
                models_status[pretty]["PID"] = None
                updated_agents.append({"name": pretty, "status": "Running"})
                logger.info(f"Started {pretty} ({sensor_key}) at {now.isoformat()}")
            except Exception as e:
                logger.error(f"Error starting {pretty}: {e}", exc_info=True)

        with open(models_status_path, "w") as f:
            json.dump(models_status, f, indent=4)

        invalidate_cache("model_df", "events", "model_df_larger")

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

    invalidate_cache("model_df", "events", "model_df_larger")

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