# APM/LIO/src/apm_core/debug.py
from __future__ import annotations
from typing import Any, Dict


def sensor_debug(ssot: Dict[str, Any], sensor_cfg: Dict[str, Any]) -> bool:
    """
    Single, predictable debug value.

    Priority:
      1) sensor_cfg["debug"] if present
      2) ssot["debug"] global fallback
      3) False
    """
    if "debug" in sensor_cfg:
        return bool(sensor_cfg.get("debug"))
    return bool(ssot.get("debug", False))
