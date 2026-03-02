from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Tuple

META_KEYS = {"site", "debug", "timezone"}


def load_ssot(site_root: Path) -> Dict[str, Any]:
    """
    site_root = .../LIO
    SSOT file = .../LIO/etc/config.json
    """
    ssot_path = site_root / "etc" / "config.json"
    with open(ssot_path, "r") as f:
        return json.load(f)


def ssot_as_data(ssot: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keep only sensor blocks, keyed by sensor_key (e.g. Mill1PinionConditionScore).
    """
    return {k: v for k, v in ssot.items() if k not in META_KEYS and isinstance(v, dict)}


def build_name_maps(data: Dict[str, Any]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Returns:
      desc_to_key: Model_Description -> sensor_key
      key_to_desc: sensor_key -> Model_Description
    """
    desc_to_key: Dict[str, str] = {}
    key_to_desc: Dict[str, str] = {}
    for sensor_key, cfg in data.items():
        desc = str(cfg.get("Model_Description", sensor_key))
        desc_to_key[desc] = sensor_key
        key_to_desc[sensor_key] = desc
    return desc_to_key, key_to_desc