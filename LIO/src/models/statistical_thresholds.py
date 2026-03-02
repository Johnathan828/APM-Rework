# APM/LIO/src/models/statistical_thresholds.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd


@dataclass
class ModelResult:
    score: pd.Series
    feature_contrib: pd.DataFrame


class StatisticalThresholdsModel:
    """
    StatisticalThresholds runtime (FixedThresholds parity):
      - Loads trained high/low thresholds from:
          <site_root>/etc/trained_params/<SensorName>/StatisticalThresholds/trained_thresholds.json
      - Flags each feature displayname as 1.0 when its raw tag breaches the trained thresholds, else 0.0.
      - Score = 1 - mean(flags across features), clipped to [0, 1].

    Expected artifact JSON (per your screenshot):
    {
      "ok": true,
      "sensor": "...",
      "method": "StatisticalThresholds",
      "trained_at_utc": "...",
      "thresholds": {
        "high": { "<raw_tag>": <float>, ... },
        "low":  { "<raw_tag>": <float>, ... }
      }
    }
    """

    def __init__(
        self,
        *,
        sensor_cfg: Dict[str, Any],
        method_cfg: Dict[str, Any],
        sensor_name: str,
        site_root: Path,
    ):
        self.sensor_cfg = sensor_cfg
        self.method_cfg = method_cfg
        self.sensor_name = sensor_name
        self.site_root = Path(site_root)

        self.artifact_path = (
            self.site_root
            / "etc"
            / "trained_params"
            / sensor_name
            / "StatisticalThresholds"
            / "trained_thresholds.json"
        ).resolve()

        if not self.artifact_path.exists():
            raise FileNotFoundError(f"{sensor_name}: missing StatisticalThresholds artifact: {self.artifact_path}")

        with open(self.artifact_path, "r", encoding="utf-8") as f:
            art = json.load(f)

        thresh = (art.get("thresholds", {}) or {})
        self.high_map: Dict[str, float] = (thresh.get("high", {}) or {})
        self.low_map: Dict[str, float] = (thresh.get("low", {}) or {})

        if not self.high_map and not self.low_map:
            raise ValueError(f"{sensor_name}: artifact has no thresholds.high/low: {self.artifact_path}")

    def score(self, *, df_wide: pd.DataFrame, feature_displaynames_to_tags: Dict[str, str]) -> ModelResult:
        if df_wide is None or df_wide.empty:
            idx = pd.DatetimeIndex([])
            return ModelResult(score=pd.Series(dtype=float, index=idx), feature_contrib=pd.DataFrame(index=idx))

        df_wide = df_wide.copy()
        df_wide.index = pd.to_datetime(df_wide.index)

        flags: Dict[str, pd.Series] = {}

        # For each feature displayname, map to raw tag, then compare against trained thresholds for that tag.
        for disp, tag in feature_displaynames_to_tags.items():
            if tag not in df_wide.columns:
                # Tag missing => non-contributing (keeps score from collapsing)
                flags[disp] = pd.Series(0.0, index=df_wide.index)
                continue

            s = pd.to_numeric(df_wide[tag], errors="coerce")

            hi = self.high_map.get(tag, None)
            lo = self.low_map.get(tag, None)

            outside = pd.Series(False, index=df_wide.index)
            if hi is not None:
                outside = outside | (s > float(hi))
            if lo is not None:
                outside = outside | (s < float(lo))

            outside = outside.fillna(False)
            flags[disp] = outside.astype(float)

        feat_df = pd.DataFrame(flags, index=df_wide.index).fillna(0.0).sort_index()

        if feat_df.empty:
            score = pd.Series(1.0, index=df_wide.index).sort_index()
            return ModelResult(score=score, feature_contrib=feat_df)

        violation_ratio = feat_df.mean(axis=1).fillna(0.0)
        score = (1.0 - violation_ratio).clip(0.0, 1.0)

        return ModelResult(score=score, feature_contrib=feat_df)