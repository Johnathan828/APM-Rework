# APM/LIO/src/models/factory.py
from __future__ import annotations

from typing import Dict, Any, Tuple

from .fixed_thresholds import FixedThresholdsModel


def build_model(*, sensor_cfg: Dict[str, Any], sensor_name: str) -> Tuple[object, str, Dict[str, Any]]:
    method_block = sensor_cfg.get("Method", {}) or {}
    active = []
    for method_name, cfg in method_block.items():
        if isinstance(cfg, dict) and bool(cfg.get("Active", False)):
            active.append((method_name, cfg))

    if len(active) != 1:
        raise ValueError(f"Must have exactly one Active model method. Found {len(active)}")

    method_name, method_cfg = active[0]

    if method_name == "FixedThresholds":
        return (
            FixedThresholdsModel(sensor_cfg=sensor_cfg, method_cfg=method_cfg, sensor_name=sensor_name),
            method_name,
            method_cfg,
        )

    raise NotImplementedError(
        f"Model method '{method_name}' not supported right now (runtime rollback). "
        f"Set Method.FixedThresholds.Active=true in config.json."
    )