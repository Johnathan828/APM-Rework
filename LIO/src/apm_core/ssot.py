# APM/LIO/src/apm_core/ssot.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple


@dataclass(frozen=True)
class SensorSpec:
    name: str
    cfg: Dict[str, Any]

    @property
    def features(self) -> Dict[str, Dict[str, Any]]:
        return self.cfg.get("Features", {}) or {}

    @property
    def feature_displaynames(self) -> List[str]:
        return list(self.features.keys())

    @property
    def feature_tags(self) -> List[str]:
        tags = []
        for _, meta in self.features.items():
            if isinstance(meta, dict) and meta.get("tag"):
                tags.append(meta["tag"])
        return tags

    @property
    def shutdown_rules(self) -> Dict[str, Any]:
        return self.cfg.get("shutdown_rules", {}) or {}

    @property
    def filter_tag_displayname(self) -> str | None:
        return (self.cfg.get("filter_tag", {}) or {}).get("FilterTagName")

    @property
    def other(self) -> Dict[str, Any]:
        return self.cfg.get("Other", {}) or {}

    @property
    def debug(self) -> bool:
        return bool(self.cfg.get("debug", False))

    def get_interval_string(self) -> str:
        """
        Old APM passes '<granularity> <unit>'::interval into sri_get_tag_data.
        We derive it from SSOT Other.granularity/granularity_type.
        """
        other = self.other
        n = int(other.get("granularity", 15))
        unit = str(other.get("granularity_type", "seconds")).strip().lower()

        # normalize to Postgres interval units
        if unit in ("sec", "second", "seconds"):
            unit = "seconds"
        elif unit in ("min", "minute", "minutes"):
            unit = "minutes"
        elif unit in ("hr", "hour", "hours"):
            unit = "hours"
        elif unit in ("day", "days"):
            unit = "days"

        return f"{n} {unit}"

    def get_active_method(self) -> Tuple[str, Dict[str, Any]]:
        method_block = self.cfg.get("Method", {}) or {}
        active = []
        for method_name, params in method_block.items():
            if isinstance(params, dict) and bool(params.get("Active", False)):
                active.append((method_name, params))
        if len(active) != 1:
            raise ValueError(
                f"Sensor '{self.name}' must have exactly ONE Method with Active=true, found {len(active)}"
            )
        return active[0]


def load_ssot(site_root: Path) -> Dict[str, Any]:
    path = site_root / "etc" / "config.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing config.json at {path}")
    with open(path, "r", encoding="utf-8") as f:
        ssot = json.load(f)
    if "site" not in ssot:
        raise ValueError("SSOT config.json must include top-level 'site'")
    return ssot


def list_sensors(ssot: Dict[str, Any]) -> List[str]:
    sensors = []
    for k, v in ssot.items():
        if k in ("site", "debug"):
            continue
        if isinstance(v, dict) and "Features" in v:
            sensors.append(k)
    return sensors


def get_sensor(ssot: Dict[str, Any], sensor_name: str) -> SensorSpec:
    if sensor_name not in ssot:
        raise KeyError(f"Unknown sensor '{sensor_name}' in SSOT")
    return SensorSpec(name=sensor_name, cfg=ssot[sensor_name])
