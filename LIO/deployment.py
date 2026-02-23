# APM/LIO/deployment.py
from __future__ import annotations

import argparse
import logging
import signal
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from src.apm_core.ssot import load_ssot
from src.apm_core.settings import load_ini
from src.apm_core.db_interface import DBInterface
from src.pipeline.run_loop import run_forever


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


class _Stopper:
    stop = False


def _install_signal_handlers(logger: logging.Logger) -> _Stopper:
    stopper = _Stopper()

    def _handler(signum, frame):
        stopper.stop = True
        logger.warning(f"Shutdown requested (signal={signum})")

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)
    return stopper


def _configure_logging(site_root: Path, level: int = logging.INFO) -> logging.Logger:
    log_dir = site_root / "etc" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "apm.log"

    logger = logging.getLogger("apm")
    logger.setLevel(level)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # Rotating file
    fh = RotatingFileHandler(str(log_path), maxBytes=5_000_000, backupCount=5)
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def main() -> None:
    parser = argparse.ArgumentParser(description="NewAPM deployment runner (dev server, prod-like)")
    parser.add_argument("--sensors", nargs="*", default=None, help="Optional sensors subset")
    parser.add_argument("--tick", type=int, default=5, help="Scheduler tick seconds")
    args = parser.parse_args()

    site_root = Path(__file__).resolve().parent
    logger = _configure_logging(site_root, level=logging.INFO)

    stopper = _install_signal_handlers(logger)

    ssot = load_ssot(site_root)
    global_debug = _get_bool(ssot.get("debug", False), False)

    # global_debug routes DEV/PRD table + Email section
    settings = load_ini(site_root, debug=global_debug)

    db = DBInterface(settings=settings, logger=logger)

    logger.warning(
        f"DEPLOYMENT START | site={ssot.get('site')} | global_debug={global_debug} | mssql_table={settings.mssql_table}"
    )

    try:
        # We run our loop manually so we can honor stop flag cleanly.
        # run_forever itself is infinite; so here we just rely on SIGINT/SIGTERM to exit the process.
        # If you want a hard stop flag inside run_forever, we can add that next.
        run_forever(ssot=ssot, db=db, logger=logger, site_root=site_root, sensor_names=args.sensors, tick_seconds=args.tick)
    finally:
        logger.warning("DEPLOYMENT STOP | closing DB connections")
        db.close()


if __name__ == "__main__":
    main()