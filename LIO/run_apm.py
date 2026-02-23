# APM/LIO/run_apm.py
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.apm_core.ssot import load_ssot
from src.apm_core.settings import load_ini
from src.apm_core.db_interface import DBInterface
from src.pipeline.run_loop import run_all_sensors_once, run_forever


def _get_bool(v, default=False) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "y", "on")
    return default


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--loop", action="store_true", help="Run continuously (legacy-like scheduler)")
    parser.add_argument("--sensors", nargs="*", default=None, help="Optional list of sensors to run")
    parser.add_argument("--tick", type=int, default=5, help="Loop tick seconds")
    args = parser.parse_args()

    site_root = Path(__file__).resolve().parent
    ssot = load_ssot(site_root)

    global_debug = _get_bool(ssot.get("debug", False), False)

    # Important: global debug decides DEV/PRD table + Email section via load_ini
    settings = load_ini(site_root, debug=global_debug)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("apm")

    db = DBInterface(settings=settings, logger=logger)
    try:
        if args.loop:
            run_forever(ssot=ssot, db=db, logger=logger, site_root=site_root, sensor_names=args.sensors, tick_seconds=args.tick)
        else:
            run_all_sensors_once(ssot=ssot, db=db, logger=logger, site_root=site_root, sensor_names=args.sensors)
    finally:
        db.close()


if __name__ == "__main__":
    main()