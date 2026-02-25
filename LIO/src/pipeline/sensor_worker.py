# APM/LIO/src/pipeline/sensor_worker.py
from __future__ import annotations

import logging
import sys
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

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


def run_sensor_process(site_root: Path, sensor_name: str, every_seconds: int) -> None:
    """
    Single sensor worker loop. This runs in its OWN PROCESS.
    """
    logger = _configure_sensor_logger(site_root, sensor_name, level=logging.INFO)

    while True:
        try:
            ssot = load_ssot(site_root)
            global_debug = bool(ssot.get("debug", False))
            sensor_cfg = ssot.get(sensor_name, {}) or {}
            sensor_debug = bool(sensor_cfg.get("debug", False))

            settings = load_ini(site_root, debug=global_debug)
            db = DBInterface(settings=settings, logger=logger)

            logger.warning(
                f"WORKER_START | sensor={sensor_name} | every={every_seconds}s | "
                f"global_debug={global_debug} | sensor_debug={sensor_debug} | mssql_table={settings.mssql_table}"
            )

            try:
                t0 = time.time()
                run_sensor_once(
                    ssot=ssot,
                    sensor_name=sensor_name,
                    db=db,
                    logger=logger,
                    global_debug=global_debug,
                    sensor_debug=sensor_debug,
                    site_root=site_root,
                )
                dt = time.time() - t0
                logger.info(f"WORKER_TICK_OK | seconds={dt:.2f}")
            finally:
                db.close()

        except Exception as e:
            logger.exception(f"WORKER_TICK_FAIL | err={e}")

        time.sleep(max(1, int(every_seconds)))