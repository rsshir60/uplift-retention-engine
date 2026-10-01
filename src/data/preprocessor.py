"""
src/data/preprocessor.py

Feature engineering and preprocessing for the uplift pipeline.

Design choices
──────────────
• StandardScaler fitted only on training rows — never on val/test.
• Interaction terms (product pairs of top features) add non-linearity
  that tree models exploit for CATE estimation.
• Pipeline is serialised via joblib so the API loads identical transforms
  at inference time without re-fitting.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = [f"f{i}" for i in range(12)]
ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "models"


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add hand-picked interaction terms motivated by churn domain knowledge:
      • recency × frequency  (f0 × f1)
      • monetary × tenure    (f2 × f3)
      • support_vol squared  (f4²)
    """
    out = df.copy()
    out["f0_x_f1"] = out["f0"] * out["f1"]
    out["f2_x_f3"] = out["f2"] * out["f3"]
    out["f4_sq"] = out["f4"] ** 2
    out["f5_x_f6"] = out["f5"] * out["f6"]
    return out


def get_feature_names(with_interactions: bool = True) -> list[str]:
    base = FEATURE_COLS.copy()
    if with_interactions:
        base += ["f0_x_f1", "f2_x_f3", "f4_sq", "f5_x_f6"]
    return base


class UpliftPreprocessor:
    """
    Stateful transformer that:
      1. Engineers interaction features.
      2. Scales all features with StandardScaler.

    fit_transform() → use on training data.
    transform()     → use on val / test / live data.
    """

    def __init__(self, with_interactions: bool = True):
        self.with_interactions = with_interactions
        self.feature_names_: list[str] = []
        self._pipeline: Pipeline | None = None

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        df_feat = (
            add_interaction_features(df)
            if self.with_interactions
            else df.copy()
        )
        self.feature_names_ = get_feature_names(self.with_interactions)
        X = df_feat[self.feature_names_].values.astype(np.float32)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self._pipeline = Pipeline([("scaler", scaler)])

        logger.info(
            f"Preprocessor fitted on {len(df):,} rows, "
            f"{len(self.feature_names_)} features."
        )
        return X_scaled

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self._pipeline is None:
            raise RuntimeError("Call fit_transform() before transform().")
        df_feat = (
            add_interaction_features(df)
            if self.with_interactions
            else df.copy()
        )
        X = df_feat[self.feature_names_].values.astype(np.float32)
        return self._pipeline.transform(X)

    def save(self, path: Path | None = None) -> Path:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        path = path or ARTIFACT_DIR / "preprocessor.joblib"
        joblib.dump(self, path)
        logger.info(f"Preprocessor saved → {path}")
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> "UpliftPreprocessor":
        path = path or ARTIFACT_DIR / "preprocessor.joblib"
        obj = joblib.load(path)
        logger.info(f"Preprocessor loaded ← {path}")
        return obj
