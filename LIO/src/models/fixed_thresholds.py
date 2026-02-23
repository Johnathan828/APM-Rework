# APM/LIO/src/models/fixed_thresholds.py
from __future__ import annotations

from typing import Dict
import pandas as pd

from .base import BaseModel, ModelResult


class FixedThresholdsModel(BaseModel):
    name = "FixedThresholds"

    def score(self, *, df_wide: pd.DataFrame, feature_displaynames_to_tags: Dict[str, str]) -> ModelResult:
        """
        Runtime FixedThresholds (config-only):
          - For each feature tag, check high/low threshold violations.
          - violation_ratio per timestamp = mean over features of (outside bounds)
          - score = 1 - violation_ratio
          - feature_contrib = per-feature violation indicator (0/1)
        """
        high_map = (self.method_cfg.get("high") or {})
        low_map = (self.method_cfg.get("low") or {})

        contrib_cols = {}
        viol_flags = []

        for disp, tag in feature_displaynames_to_tags.items():
            if tag not in df_wide.columns:
                contrib_cols[disp] = pd.Series(0.0, index=df_wide.index)
                continue

            s = pd.to_numeric(df_wide[tag], errors="coerce")

            hi = high_map.get(tag, None)
            lo = low_map.get(tag, None)

            outside = pd.Series(False, index=df_wide.index)
            if hi is not None:
                outside = outside | (s > float(hi))
            if lo is not None:
                outside = outside | (s < float(lo))

            outside = outside.fillna(False)
            viol_flags.append(outside.astype(float))
            contrib_cols[disp] = outside.astype(float)

        if viol_flags:
            viol_df = pd.concat(viol_flags, axis=1)
            violation_ratio = viol_df.mean(axis=1)
        else:
            violation_ratio = pd.Series(0.0, index=df_wide.index)

        score = (1.0 - violation_ratio).clip(0.0, 1.0)
        feature_contrib = pd.DataFrame(contrib_cols, index=df_wide.index).fillna(0.0)

        return ModelResult(score=score, feature_contrib=feature_contrib)