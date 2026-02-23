# APM/LIO/src/models/base.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any
import pandas as pd


@dataclass
class ModelResult:
    score: pd.Series
    feature_contrib: pd.DataFrame


class BaseModel:
    """
    Base for runtime models.
    NOTE: sensor_name is required so models can identify where they belong in SSOT.
    """

    name: str = "BaseModel"

    def __init__(self, *, sensor_cfg: Dict[str, Any], method_cfg: Dict[str, Any], sensor_name: str):
        self.sensor_cfg = sensor_cfg
        self.method_cfg = method_cfg
        self.sensor_name = sensor_name

    def score(self, *, df_wide: pd.DataFrame, feature_displaynames_to_tags: Dict[str, str]) -> ModelResult:
        raise NotImplementedError