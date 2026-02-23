# APM/LIO/src/models/isolation_forest.py
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Any, Optional, List

import pandas as pd

from .base import BaseModel, ModelResult
from .paths import model_dir_for


class IsolationForestModel(BaseModel):
    """
    Classic IF on features (wide frame).
    Score output is normalized to [0,1] where LOWER = more anomalous (for parity with alert_thresh logic).
    """

    name = "IsolationForest"

    def __init__(self, sensor_cfg: Dict[str, Any], method_cfg: Dict[str, Any]):
        super().__init__(sensor_cfg=sensor_cfg, method_cfg=method_cfg)
        self.model = None
        self.feature_order: List[str] = []

    def try_load(self, *, site_root: Path) -> bool:
        d = model_dir_for(site_root=site_root, sensor_name=self.sensor_name, method_name=self.name)
        path = d / "iforest.pkl"
        if not path.exists():
            return False
        with open(path, "rb") as f:
            obj = pickle.load(f)
        self.model = obj["model"]
        self.feature_order = obj.get("feature_order", []) or []
        return True

    def save(self, *, site_root: Path) -> None:
        if self.model is None:
            raise ValueError("No trained model to save")
        d = model_dir_for(site_root=site_root, sensor_name=self.sensor_name, method_name=self.name)
        d.mkdir(parents=True, exist_ok=True)
        path = d / "iforest.pkl"
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "feature_order": self.feature_order}, f)

    def train(self, *, df_wide: pd.DataFrame, feature_displaynames_to_tags: Dict[str, str]) -> None:
        try:
            from sklearn.ensemble import IsolationForest
        except Exception as e:
            raise ImportError("scikit-learn required for IsolationForest. Install sklearn in your env.") from e

        # Build X from available tags only
        tags = [tag for _, tag in feature_displaynames_to_tags.items() if tag in df_wide.columns]
        if not tags:
            raise ValueError("No feature tags found in df_wide for IsolationForest training")

        X = df_wide[tags].apply(pd.to_numeric, errors="coerce").fillna(method="ffill").fillna(method="bfill").fillna(0.0)

        n_estimators = int(self.method_cfg.get("n_estimators", 100))
        contamination = float(self.method_cfg.get("contamination", 0.01))
        random_state = int(self.method_cfg.get("random_state", 42))

        m = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
        )
        m.fit(X.values)

        self.model = m
        self.feature_order = tags

    def score(self, *, df_wide: pd.DataFrame, feature_displaynames_to_tags: Dict[str, str]) -> ModelResult:
        if self.model is None:
            raise ValueError("IsolationForestModel not trained/loaded. Run train first.")

        tags = self.feature_order
        X = df_wide.reindex(columns=tags).apply(pd.to_numeric, errors="coerce").fillna(method="ffill").fillna(method="bfill").fillna(0.0)

        # sklearn decision_function: higher = more normal
        normality = pd.Series(self.model.decision_function(X.values), index=df_wide.index)

        # normalize to [0,1] as "score": 1 normal, 0 anomalous (lower triggers alerts)
        mn = float(normality.min())
        mx = float(normality.max())
        if mx - mn < 1e-9:
            score = pd.Series(1.0, index=df_wide.index)
        else:
            score = ((normality - mn) / (mx - mn)).clip(0.0, 1.0)

        # contributions: for IF we don’t have per-feature contributions without extra work;
        # keep a placeholder zeros for parity with email layout.
        contrib_cols = {}
        for disp in feature_displaynames_to_tags.keys():
            contrib_cols[disp] = pd.Series(0.0, index=df_wide.index)
        feature_contrib = pd.DataFrame(contrib_cols, index=df_wide.index).fillna(0.0)

        return ModelResult(score=score, feature_contrib=feature_contrib)