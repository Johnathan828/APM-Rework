from __future__ import annotations

from flask import Flask, render_template, request, jsonify, Blueprint, send_file
from threading import Lock
import os
import signal
import json
import logging
import datetime
import pytz
import configparser
from pathlib import Path

import psutil
import pandas as pd
import plotly.io as pio

from modules.pooling import MSSQLConnectionPool
from modules.DBmanager import DBInterface
from modules.ssot_adapter import load_ssot, ssot_as_data, build_name_maps
from modules.process_control import start_sensor, stop_pid
from modules.heartbeat_reader import read_heartbeat, status_from_heartbeat

pio.renderers.default = "browser"

app = Flask(__name__, static_folder="static")
neuromine_bp = Blueprint("neuromine", __name__, url_prefix="/Neuromine/LIO/APM/")

FLASK_ROOT = Path(__file__).resolve().parent
SITE_ROOT = FLASK_ROOT.parent
STATIC_DATA = FLASK_ROOT / "static" / "data"

os.environ["TZ"] = "Africa/Johannesburg"

# -----------------------
# Logging
# -----------------------
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

# -----------------------
# config.ini (secrets/env)
# -----------------------
config = configparser.ConfigParser()
ini_path = SITE_ROOT / "etc" / "config.ini"
config.read(str(ini_path))

# -----------------------
# config.json (SSOT)
# -----------------------
ssot = load_ssot(SITE_ROOT)
data = ssot_as_data(ssot)  # {sensor_key: cfg}
desc_to_key, key_to_desc = build_name_maps(data)


def _mssql_section_name() -> str:
    return "SQLALCHEMY_DEV" if bool(ssot.get("debug", False)) else "SQLALCHEMY"


SQL_SECTION = _mssql_section_name()
MSSQL_TABLE = config[SQL_SECTION]["table"]

# -----------------------
# MSSQL pool + DB layer
# -----------------------
mssql_pool = MSSQLConnectionPool(
    server=config[SQL_SECTION]["host"],
    database=config[SQL_SECTION]["dbname"],
    user=config[SQL_SECTION]["user"],
    password=config[SQL_SECTION]["password"],
    port=config[SQL_SECTION]["port"],
    max_connections=3,
)

# legacy/optional mapping file
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

# -----------------------
# modelsRunStatus.json
# -----------------------
models_status_path = STATIC_DATA / "modelsRunStatus.json"


def init_models_status() -> dict:
    """
    Ensure all SSOT sensors exist in file.

    Adds a manual-only switch:
      enabled: False by default

    UI rules:
      - enabled=False => always show Stopped
      - enabled=True  => show Running only if heartbeat says Running/Stale
    """
    models_status_path.parent.mkdir(parents=True, exist_ok=True)
    if models_status_path.exists():
        with open(models_status_path, "r") as f:
            ms = json.load(f)
    else:
        ms = {}

    for sensor_key, cfg in data.items():
        pretty = str(cfg.get("Model_Description", sensor_key))
        if pretty not in ms:
            ms[pretty] = {
                "sensor_key": sensor_key,
                "status": "Stopped",
                "PID": None,
                "enabled": False,
            }

        ms[pretty]["sensor_key"] = sensor_key
        ms[pretty].setdefault("enabled", False)
        ms[pretty].setdefault("status", "Stopped")
        ms[pretty].setdefault("PID", None)

        # remove legacy fields
        ms[pretty].pop("path", None)

        # IMPORTANT: do not auto-enable or auto-run at init
        ms[pretty]["status"] = "Stopped"
        ms[pretty]["PID"] = None

    with open(models_status_path, "w") as f:
        json.dump(ms, f, indent=4)

    return ms


models_status = init_models_status()

# -----------------------
# In-process caching
# -----------------------
cache_lock = Lock()
CACHE = {
    "model_df": {"ts": None, "ttl_s": 300, "value": pd.DataFrame()},         # 5 min
    "events": {"ts": None, "ttl_s": 300, "value": []},                       # 5 min
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
        if mn not in model_events:
            model_events[mn] = []
        model_events[mn].append(event)

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


def get_cached_model_df_larger(hours: int = 6) -> pd.DataFrame:
    with cache_lock:
        if _cache_fresh(CACHE["model_df_larger"]):
            return CACHE["model_df_larger"]["value"]

    date_to = datetime.datetime.utcnow()
    date_from = date_to - datetime.timedelta(hours=hours)

    cols = list(data.keys())
    try:
        df = data_collector.get_neuro_displayname_data(date_from=date_from, date_to=date_to, model_cols=cols)
    except Exception as e:
        logger.error(f"get_cached_model_df_larger failed: {e}", exc_info=True)
        df = pd.DataFrame()
    finally:
        data_collector.release_connections()

    with cache_lock:
        CACHE["model_df_larger"]["value"] = df
        CACHE["model_df_larger"]["ts"] = _now_utc()
    return df


# -----------------------
# Routes
# -----------------------
@neuromine_bp.route("/")
def home():
    """
    Manual-only control:
      - if enabled=False => always show Stopped
      - if enabled=True  => show Running only if heartbeat Running/Stale
    """
    for pretty, info in models_status.items():
        sensor_key = info.get("sensor_key")
        enabled = bool(info.get("enabled", False))

        if not enabled:
            models_status[pretty]["status"] = "Stopped"
            models_status[pretty]["PID"] = None
            continue

        hb_status = "Down"
        if sensor_key:
            hb = read_heartbeat(SITE_ROOT, sensor_key)
            hb_status = status_from_heartbeat(hb)

        if hb_status in ("Running", "Stale"):
            models_status[pretty]["status"] = "Running"
        else:
            models_status[pretty]["status"] = "Stopped"
            models_status[pretty]["PID"] = None

    with open(models_status_path, "w") as f:
        json.dump(models_status, f, indent=4)

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


@neuromine_bp.route("/home_reload")
def home_reload():
    return render_template("home_reload.html")


@neuromine_bp.route("/lives_agents")
def lives_agents():
    return render_template("live_agents.html")


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

                # latest score
                if sensor_key in model_df.columns and not model_df[sensor_key].dropna().empty:
                    health_score = float(model_df[sensor_key].dropna().iloc[-1]) * 100.0
                else:
                    health_score = 0.0
                health_score = round(health_score, 1)

                # latest section status
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
        average_health = sum(ev["score"] for ev in filtered_events) / total_events if total_events > 0 else 0

        summary_stats = {
            "Total_events": total_events,
            "average_health": round(average_health, 2),
            "Total_monitored_sections": 1,
            "total_tag_inputs": 0,
        }
    except Exception:
        logger.error("lives_agents_data exception", exc_info=True)

    return jsonify(models=live_update, summary_stats=summary_stats)


@neuromine_bp.route("/get_model_health")
def get_model_health():
    agent_name = request.args.get("agent_name")
    if not agent_name:
        return jsonify({"error": "agent_name required"}), 400

    desc_map = {data[k].get("Model_Description", k): k for k in data.keys()}
    if agent_name not in desc_map:
        return jsonify({"error": "unknown agent_name"}), 404

    sensor_key = desc_map[agent_name]
    cfg = data[sensor_key]

    try:
        model_df = get_cached_model_df_larger(hours=6)
        model_df = model_df.copy()
        model_df.bfill(inplace=True)
        model_df.ffill(inplace=True)

        timestamps = model_df.index.strftime("%Y-%m-%d %H:%M:%S").tolist() if not model_df.empty else []
        health_scores = (model_df[sensor_key] * 100.0).tolist() if (not model_df.empty and sensor_key in model_df.columns) else []

        now = pd.Timestamp.utcnow().to_pydatetime()
        cutoff = now - datetime.timedelta(hours=6)

        feature_tags = [
            meta.get("tag")
            for meta in (cfg.get("Features", {}) or {}).values()
            if isinstance(meta, dict) and meta.get("tag")
        ]

        interval_str = "2 minutes"
        feature_df = data_collector.build_wide_from_postgres(
            tags=feature_tags,
            start=cutoff,
            end=now,
            interval_str=interval_str,
        )

        if feature_df is None or feature_df.empty:
            feature_df = pd.DataFrame(index=pd.DatetimeIndex([]))
        else:
            feature_df.index = pd.to_datetime(feature_df.index, errors="coerce")
            feature_df = feature_df[~feature_df.index.isna()]
            feature_df = feature_df.sort_index()
            feature_df.bfill(inplace=True)
            feature_df.ffill(inplace=True)

        tag_to_feature_map = {}
        for feat_name, meta in (cfg.get("Features", {}) or {}).items():
            if isinstance(meta, dict):
                tag_to_feature_map[meta.get("tag")] = feat_name

        feature_data = {}
        for col in feature_df.columns:
            if col in tag_to_feature_map:
                feature_data[tag_to_feature_map[col]] = feature_df[col].tolist()

        feature_timestamps = feature_df.index.strftime("%Y-%m-%d %H:%M:%S").tolist()
        feature_state = {f: 0 for f in (cfg.get("Features", {}) or {}).keys()}

        health_data = {
            "timestamps": timestamps,
            "health_scores": health_scores,
            "feature_timestamps": feature_timestamps,
            "feature_df": feature_data,
            "health_score": health_scores[-1] if health_scores else 0,
            "last_check": timestamps[-1] if timestamps else "",
            "feature_state": feature_state,
            "feature_details": cfg,
        }
        return jsonify(health_data)

    except Exception:
        logger.error("get_model_health exception", exc_info=True)
        return jsonify({"error": "Error fetching model health"}), 500
    finally:
        data_collector.release_connections()


@neuromine_bp.route("/startstop", methods=["GET"])
def show_start_stop_page():
    agents = []
    stopped_count = 0
    running_count = 0

    allowed_pretty_names = set(desc_to_key.keys())

    for pretty, details in models_status.items():
        if pretty not in allowed_pretty_names:
            continue

        st = details.get("status", "Stopped")
        enabled = bool(details.get("enabled", False))

        if st == "Stopped":
            stopped_count += 1
        elif st == "Running":
            running_count += 1

        agents.append(
            {
                "name": pretty,
                "status": st,
                "enabled": enabled,
                "path": "",
                "action": "Start" if st == "Stopped" else "Stop",
            }
        )

    return render_template(
        "start_stop_agents.html",
        agents=agents,
        stopped_count=stopped_count,
        running_count=running_count,
    )


@neuromine_bp.route("/startstop_action", methods=["POST"])
def start_stop_agent():
    payload = request.get_json() or {}
    action = (payload.get("action") or "").strip()
    agent_name = payload.get("agent_name")

    now = datetime.datetime.now(pytz.timezone("Africa/Johannesburg"))
    updated_agents = []

    if action == "StartAll":
        for pretty, info in models_status.items():
            sensor_key = info.get("sensor_key")
            if not sensor_key or sensor_key not in data:
                continue

            # enable + start
            if info.get("status") != "Running":
                try:
                    models_status[pretty]["enabled"] = True
                    pid = start_sensor(SITE_ROOT, sensor_key)
                    models_status[pretty]["status"] = "Running"
                    models_status[pretty]["PID"] = pid
                    updated_agents.append({"name": pretty, "status": "Running"})
                    logger.info(f"Started {pretty} ({sensor_key}) pid={pid} at {now.isoformat()}")
                except Exception as e:
                    logger.error(f"Error starting {pretty}: {e}")

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

    if action.lower() == "start" and models_status[agent_name].get("status") == "Stopped":
        try:
            models_status[agent_name]["enabled"] = True
            pid = start_sensor(SITE_ROOT, sensor_key)
            models_status[agent_name]["status"] = "Running"
            models_status[agent_name]["PID"] = pid
            logger.info(f"Started {agent_name} ({sensor_key}) pid={pid} at {now.isoformat()}")
        except Exception as e:
            logger.error(f"Error starting {agent_name}: {e}")

    elif action.lower() == "stop":
        try:
            models_status[agent_name]["enabled"] = False
            pid = models_status[agent_name].get("PID")
            if pid:
                stop_pid(pid)
            models_status[agent_name]["status"] = "Stopped"
            models_status[agent_name]["PID"] = None
            logger.info(f"Stopped {agent_name} at {now.isoformat()}")
        except Exception as e:
            logger.error(f"Error stopping {agent_name}: {e}")

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

    for pretty, info in models_status.items():
        try:
            pid = info.get("PID")
            if info.get("status") == "Running" and pid:
                stop_pid(pid)
        except Exception:
            pass

        models_status[pretty]["enabled"] = False
        models_status[pretty]["status"] = "Stopped"
        models_status[pretty]["PID"] = None

    with open(models_status_path, "w") as f:
        json.dump(models_status, f, indent=4)

    try:
        data_collector.release_connections()
    except Exception:
        pass

    os._exit(0)


signal.signal(signal.SIGINT, shutdown_handler)
signal.signal(signal.SIGTERM, shutdown_handler)

app.register_blueprint(neuromine_bp)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8871))
    print(f"Flask app running on port {port}")
    app.run(host="0.0.0.0", port=port)