# APM/LIO/src/models/factory.py
from __future__ import annotations

from typing import Dict, Any, Tuple
from pathlib import Path

from .fixed_thresholds import FixedThresholdsModel
from .statistical_thresholds import StatisticalThresholdsModel


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

    if method_name == "StatisticalThresholds":
        # site_root = .../APM/LIO (because this file is .../APM/LIO/src/models/factory.py)
        site_root = Path(__file__).resolve().parents[2]
        return (
            StatisticalThresholdsModel(sensor_cfg=sensor_cfg, method_cfg=method_cfg, sensor_name=sensor_name, site_root=site_root),
            method_name,
            method_cfg,
        )

    raise NotImplementedError(
        f"Model method '{method_name}' not supported yet. "
        f"Set Method.FixedThresholds.Active=true or Method.StatisticalThresholds.Active=true in config.json."
    )