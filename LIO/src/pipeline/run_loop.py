# APM/LIO/src/pipeline/run_loop.py
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

from ..apm_core.ssot import list_sensors
from .run_once import run_sensor_once


@dataclass
class SensorRunPlan:
    sensor_name: str
    run_every_seconds: int
    global_debug: bool
    sensor_debug: bool


def _get_bool(v: Any, default: bool = False) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "y", "on")
    return default


def _sensor_interval_seconds(ssot: Dict[str, Any], sensor_name: str) -> int:
    """
    Determine how often to run a sensor.
    Priority:
      1) Sensor Other.run_every_seconds
      2) Sensor Other.run_every_minutes
      3) Sensor granularity * 60 (minutes granularity assumed)
      4) Default 60 seconds
    """
    cfg = ssot.get(sensor_name, {}) or {}
    other = (cfg.get("Other", {}) or {})
    # explicit overrides
    if "run_every_seconds" in other:
        try:
            return max(5, int(other["run_every_seconds"]))
        except Exception:
            pass
    if "run_every_minutes" in other:
        try:
            return max(1, int(other["run_every_minutes"])) * 60
        except Exception:
            pass

    # fallback: granularity
    try:
        gran = int(other.get("granularity", 15))
        gran_type = str(other.get("granularity_type", "minutes")).strip().lower()
        if gran_type in ("sec", "second", "seconds"):
            return max(5, gran)
        # assume minutes for legacy scoring cadence
        return max(60, gran * 60)
    except Exception:
        return 60


def build_run_plans(ssot: Dict[str, Any]) -> List[SensorRunPlan]:
    global_debug = _get_bool(ssot.get("debug", False), False)

    plans: List[SensorRunPlan] = []
    for sensor_name in list_sensors(ssot):
        sensor_cfg = ssot.get(sensor_name, {}) or {}
        sensor_debug = _get_bool(sensor_cfg.get("debug", False), False)

        plans.append(
            SensorRunPlan(
                sensor_name=sensor_name,
                run_every_seconds=_sensor_interval_seconds(ssot, sensor_name),
                global_debug=global_debug,
                sensor_debug=sensor_debug,
            )
        )

    # stable order
    plans.sort(key=lambda p: p.sensor_name.lower())
    return plans


def run_all_sensors_once(
    *,
    ssot: Dict[str, Any],
    db,
    logger,
    site_root: Path,
    sensor_names: Optional[List[str]] = None,
) -> None:
    """
    Run one cycle across sensors.
    - isolates exceptions per sensor
    - uses global_debug from ssot + sensor_debug per sensor
    """
    global_debug = _get_bool(ssot.get("debug", False), False)
    names = sensor_names or list_sensors(ssot)

    for sensor_name in names:
        sensor_cfg = ssot.get(sensor_name, {}) or {}
        sensor_debug = _get_bool(sensor_cfg.get("debug", False), False)

        t0 = time.time()
        try:
            logger.info(f"RUN_START | {sensor_name}")
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
            logger.info(f"RUN_OK | {sensor_name} | seconds={dt:.2f}")
        except Exception as e:
            dt = time.time() - t0
            logger.exception(f"RUN_FAIL | {sensor_name} | seconds={dt:.2f} | err={e}")


def run_forever(
    *,
    ssot: Dict[str, Any],
    db,
    logger,
    site_root: Path,
    sensor_names: Optional[List[str]] = None,
    tick_seconds: int = 5,
) -> None:
    """
    Lightweight scheduler loop (legacy-like):
    - each sensor has its own run cadence
    - loop ticks every tick_seconds and runs due sensors
    - keeps running even if some sensors fail
    """
    plans = build_run_plans(ssot)

    if sensor_names:
        wanted = set(sensor_names)
        plans = [p for p in plans if p.sensor_name in wanted]

    if not plans:
        logger.warning("No sensors found to run.")
        return

    next_run: Dict[str, float] = {}
    now = time.time()
    for p in plans:
        next_run[p.sensor_name] = now  # run immediately on startup
        logger.info(f"SCHEDULE | {p.sensor_name} | every={p.run_every_seconds}s")

    logger.info("Entering run loop...")
    while True:
        now = time.time()

        for p in plans:
            due_at = next_run.get(p.sensor_name, now)
            if now >= due_at:
                t0 = time.time()
                try:
                    logger.info(f"RUN_START | {p.sensor_name}")
                    run_sensor_once(
                        ssot=ssot,
                        sensor_name=p.sensor_name,
                        db=db,
                        logger=logger,
                        global_debug=p.global_debug,
                        sensor_debug=p.sensor_debug,
                        site_root=site_root,
                    )
                    dt = time.time() - t0
                    logger.info(f"RUN_OK | {p.sensor_name} | seconds={dt:.2f}")
                except Exception as e:
                    dt = time.time() - t0
                    logger.exception(f"RUN_FAIL | {p.sensor_name} | seconds={dt:.2f} | err={e}")
                finally:
                    next_run[p.sensor_name] = time.time() + p.run_every_seconds

        time.sleep(max(1, int(tick_seconds)))