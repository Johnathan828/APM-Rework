# APM/LIO/src/models/isolation_forest.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd

from .base import BaseModel, ModelResult


@dataclass
class IFArtifacts:
    model: Any
    scaler: Optional[Any]
    feature_tags: List[str]
    meta: Dict[str, Any]


class IsolationForestModel(BaseModel):
    """
    IsolationForest runtime scorer.

    CRITICAL RUNTIME RULE:
      - DO NOT RESAMPLE in runtime.
      - resample_rule is TRAINING ONLY.
      - Runtime must score on the native df_wide index (e.g., 15s), so the gate/filter tag aligns.
    """

    name = "IsolationForest"

    def __init__(self, *, sensor_cfg: Dict[str, Any], method_cfg: Dict[str, Any], sensor_name: str, site_root: Path):
        super().__init__(sensor_cfg=sensor_cfg, method_cfg=method_cfg, sensor_name=sensor_name)
        self.site_root = site_root
        self.art: Optional[IFArtifacts] = None

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    def _artifact_dir(self) -> Path:
        # Your convention: LIO/etc/trained_params/<SensorName>/<Method>/
        return self.site_root / "etc" / "trained_params" / self.sensor_name / self.name

    def _joblib_path(self) -> Path:
        return self._artifact_dir() / "trained_isolation_forest.joblib"

    def _json_path(self) -> Path:
        return self._artifact_dir() / "trained_isolation_forest.json"

    # ------------------------------------------------------------------
    # Load trained artifacts
    # ------------------------------------------------------------------
    def try_load(self) -> bool:
        joblib_path = self._joblib_path()
        json_path = self._json_path()

        if (not joblib_path.exists()) or (not json_path.exists()):
            return False

        # joblib load
        try:
            import joblib
        except Exception as e:
            raise ImportError("joblib is required to load IsolationForest artifacts") from e

        obj = joblib.load(joblib_path)

        # json meta
        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        # Be tolerant to different saved structures
        model = obj.get("model", obj.get("iforest", None))
        scaler = obj.get("scaler", obj.get("standard_scaler", None))

        feature_tags = (
            meta.get("features")
            or meta.get("feature_tags")
            or obj.get("feature_tags")
            or obj.get("features")
            or []
        )
        feature_tags = list(feature_tags) if feature_tags else []

        if model is None:
            raise ValueError(f"{self.sensor_name}: IsolationForest artifact missing 'model' key")

        if not feature_tags:
            # last resort: if the scaler has feature_names_in_
            try:
                feature_tags = list(getattr(scaler, "feature_names_in_", [])) if scaler is not None else []
            except Exception:
                feature_tags = []

        if not feature_tags:
            raise ValueError(
                f"{self.sensor_name}: IsolationForest artifact missing feature list. "
                f"Expected meta['features'] or meta['feature_tags']."
            )

        self.art = IFArtifacts(model=model, scaler=scaler, feature_tags=feature_tags, meta=meta)
        return True

    # ------------------------------------------------------------------
    # Runtime scoring
    # ------------------------------------------------------------------
    def score(
        self,
        *,
        df_wide: pd.DataFrame,
        feature_displaynames_to_tags: Dict[str, str],
    ) -> ModelResult:
        """
        Returns:
          score in [0, 1] where 1 = normal, 0 = anomalous (parity with alert_engine: score <= alarm_thresh triggers)
          feature_contrib: placeholder or per-tag trigger flags (here: zeros; trigger column handled elsewhere)
        """
        if self.art is None:
            ok = self.try_load()
            if not ok:
                raise FileNotFoundError(
                    f"{self.sensor_name}: IsolationForest artifacts not found at {self._artifact_dir()}"
                )

        art = self.art
        assert art is not None

        # IMPORTANT: preserve df_wide index (do NOT resample)
        idx = df_wide.index

        # Build X in the EXACT trained feature order
        X_df = pd.DataFrame(index=idx)
        for tag in art.feature_tags:
            if tag in df_wide.columns:
                X_df[tag] = pd.to_numeric(df_wide[tag], errors="coerce")
            else:
                # If a trained feature is missing in runtime pull, fill with NaN -> then ffill/bfill/0
                X_df[tag] = np.nan

        # Basic fill strategy (no resample)
        X_df = X_df.ffill().bfill().fillna(0.0)

        X = X_df.values

        # Apply scaler if present
        if art.scaler is not None:
            try:
                X = art.scaler.transform(X)
            except Exception:
                # If scaler fails for any reason, fall back without scaling
                X = X_df.values

        # decision_function: higher = more normal
        normality = pd.Series(art.model.decision_function(X), index=idx)

        # Normalize to [0,1] as "health score"
        # Prefer training-normalization if it exists, otherwise per-window normalize
        meta = art.meta or {}
        norm = meta.get("normality_norm", {}) if isinstance(meta.get("normality_norm", {}), dict) else {}

        mn = norm.get("min", None)
        mx = norm.get("max", None)

        if mn is None or mx is None:
            mn = float(normality.min())
            mx = float(normality.max())
        else:
            mn = float(mn)
            mx = float(mx)

        if (mx - mn) < 1e-12:
            score = pd.Series(1.0, index=idx)
        else:
            score = ((normality - mn) / (mx - mn)).clip(0.0, 1.0)

        # Feature contributions: IF doesn’t provide native per-feature contributions.
        # Keep zeros so the email table still renders.
        contrib_cols: Dict[str, pd.Series] = {}
        for disp in feature_displaynames_to_tags.keys():
            contrib_cols[disp] = pd.Series(0.0, index=idx)
        feature_contrib = pd.DataFrame(contrib_cols, index=idx).fillna(0.0)

        return ModelResult(score=score, feature_contrib=feature_contrib)