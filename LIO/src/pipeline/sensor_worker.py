# APM/LIO/src/pipeline/sensor_worker.py
from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

import pandas as pd

from src.apm_core.ssot import load_ssot
from src.apm_core.settings import load_ini
from src.apm_core.db_interface import DBInterface
from src.pipeline.run_once import run_sensor_once


def _configure_sensor_logger(site_root: Path, sensor_name: str, level: int = logging.INFO) -> logging.Logger:
    """
    IMPORTANT: per-process log file to avoid multi-process file handler collisions.
    """
    log_dir = site_root / "etc" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"apm_{sensor_name}.log"

    logger = logging.getLogger(f"apm.{sensor_name}")
    logger.setLevel(level)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = RotatingFileHandler(str(log_path), maxBytes=5_000_000, backupCount=5)
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def _utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _write_heartbeat(
    *,
    site_root: Path,
    sensor_name: str,
    every_seconds: int,
    latest_score_ts: object = None,
) -> None:
    """
    Writes a small heartbeat file used by Flask to show Running/Stale/Down.

    Path:
      LIO/etc/state/heartbeat_<sensor>.json

    Atomic write (tmp -> replace) so Flask never reads partial JSON.
    """
    state_dir = site_root / "etc" / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    hb_path = state_dir / f"heartbeat_{sensor_name}.json"

    latest_score_ts_utc = None
    if latest_score_ts is not None:
        try:
            latest_score_ts_utc = pd.to_datetime(latest_score_ts, utc=True).strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            latest_score_ts_utc = str(latest_score_ts)

    payload = {
        "sensor": sensor_name,
        "worker_pid": int(os.getpid()),
        "tick_ok_at_utc": _utc_now_iso(),
        "latest_score_ts_utc": latest_score_ts_utc,
        "every_seconds": int(every_seconds),
    }

    tmp = hb_path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(hb_path)


def run_sensor_process(site_root: Path, sensor_name: str, every_seconds: int) -> None:
    """
    Single sensor worker loop. This runs in its OWN PROCESS.

    Legacy-like behavior:
      - Keep ONE persistent DBInterface connection per worker process
      - Recreate DBInterface only when it breaks or when debug changes
      - Health-check DB connections each tick (prevents stale pool/conn issues)
      - Write a heartbeat file after each successful tick (for Flask status)
    """
    # Set a clear process title (so ps/top doesn't confuse this worker with supervisor)
    try:
        from setproctitle import setproctitle  # type: ignore

        setproctitle(f"apm_worker:{sensor_name}")
    except Exception:
        pass

    logger = _configure_sensor_logger(site_root, sensor_name, level=logging.INFO)

    db = None
    last_global_debug = None

    def _make_db(global_debug: bool):
        settings = load_ini(site_root, debug=global_debug)
        return DBInterface(settings=settings, logger=logger), settings

    def _health_check(db_obj) -> None:
        # Postgres pool closed?
        pool_closed = getattr(getattr(db_obj, "pg_pool", None), "closed", True)
        if pool_closed:
            raise RuntimeError("Postgres pool is closed")

        # MSSQL connection usable?
        try:
            conn = getattr(db_obj, "mssql_conn", None)
            if conn is None:
                raise RuntimeError("MSSQL connection is None")
            with conn.cursor() as cur:
                cur.execute("SELECT 1 AS ok")
                _ = cur.fetchone()
        except Exception as e:
            raise RuntimeError(f"MSSQL ping failed: {e}")

    while True:
        t0 = time.time()
        latest_ts = None

        try:
            ssot = load_ssot(site_root)
            global_debug = bool(ssot.get("debug", False))
            sensor_cfg = ssot.get(sensor_name, {}) or {}
            sensor_debug = bool(sensor_cfg.get("debug", False))

            # (Re)create DB if first run or global debug mode changed
            if db is None or (last_global_debug is not None and global_debug != last_global_debug):
                if db is not None:
                    try:
                        db.close()
                    except Exception:
                        pass

                db, settings = _make_db(global_debug)
                last_global_debug = global_debug
                logger.warning(
                    f"DB_READY | sensor={sensor_name} | global_debug={global_debug} | mssql_table={settings.mssql_table}"
                )

            _health_check(db)

            latest_ts = run_sensor_once(
                ssot=ssot,
                sensor_name=sensor_name,
                db=db,
                logger=logger,
                global_debug=global_debug,
                sensor_debug=sensor_debug,
                site_root=site_root,
            )

            try:
                _write_heartbeat(
                    site_root=site_root,
                    sensor_name=sensor_name,
                    every_seconds=every_seconds,
                    latest_score_ts=latest_ts,
                )
            except Exception as hb_e:
                logger.warning(f"HEARTBEAT_WRITE_FAIL | err={hb_e}")

            dt = time.time() - t0
            logger.info(f"WORKER_TICK_OK | seconds={dt:.2f}")

        except Exception as e:
            logger.exception(f"WORKER_TICK_FAIL | err={e}")

            if db is not None:
                try:
                    db.close()
                except Exception:
                    pass
            db = None

        time.sleep(max(1, int(every_seconds)))