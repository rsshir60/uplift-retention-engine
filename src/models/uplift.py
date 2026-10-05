"""
src/models/uplift.py

Implements three metalearner families for Individual Treatment Effect (ITE)
estimation, all backed by XGBoost base learners.

┌─────────────┬──────────────────────────────────────────────────────────────┐
│ Learner     │ When to use                                                  │
├─────────────┼──────────────────────────────────────────────────────────────┤
│ T-Learner   │ Simple baseline; two separate models μ₀, μ₁                  │
│ X-Learner   │ Better in imbalanced treatment splits; cross-fitted CATE     │
│ DR-Learner  │ Doubly-robust; most statistically efficient estimator        │
└─────────────┴──────────────────────────────────────────────────────────────┘

All learners expose a common interface:
  .fit(X_train, T_train, Y_train)
  .predict_ite(X)          → np.ndarray of shape (n,)
  .save() / .load()

EconML is used for X-Learner and DR-Learner to keep the implementation lean
and production-validated.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import joblib
import numpy as np
from econml.dr import DRLearner
from econml.metalearners import TLearner, XLearner
from loguru import logger
from xgboost import XGBClassifier, XGBRegressor

ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "models"

_XGB_CLF_PARAMS = dict(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    eval_metric="logloss",
    use_label_encoder=False,
    random_state=42,
    n_jobs=-1,
    tree_method="hist",
)

_XGB_REG_PARAMS = dict(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    eval_metric="rmse",
    random_state=42,
    n_jobs=-1,
    tree_method="hist",
)


# ─────────────────────────────────────────────────────────────────────────────
# T-Learner  (baseline)
# ─────────────────────────────────────────────────────────────────────────────

class TLearnerModel:
    """
    Two-model approach:
        μ₀(x) = E[Y | X=x, T=0]
        μ₁(x) = E[Y | X=x, T=1]
        τ̂(x)  = μ₁(x) − μ₀(x)
    """

    def __init__(self):
        self._model = TLearner(
            models=XGBClassifier(**_XGB_CLF_PARAMS)
        )
        self.name = "t_learner"

    def fit(
        self,
        X: np.ndarray,
        T: np.ndarray,
        Y: np.ndarray,
    ) -> "TLearnerModel":
        logger.info("Fitting T-Learner …")
        self._model.fit(Y, T, X=X)
        logger.info("T-Learner fit complete.")
        return self

    def predict_ite(self, X: np.ndarray) -> np.ndarray:
        return self._model.effect(X).ravel()

    def save(self, path: Path | None = None) -> Path:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        path = path or ARTIFACT_DIR / f"{self.name}.joblib"
        joblib.dump(self, path)
        logger.info(f"{self.name} saved → {path}")
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> "TLearnerModel":
        path = path or ARTIFACT_DIR / "t_learner.joblib"
        obj = joblib.load(path)
        logger.info(f"TLearnerModel loaded ← {path}")
        return obj


# ─────────────────────────────────────────────────────────────────────────────
# X-Learner  (primary model)
# ─────────────────────────────────────────────────────────────────────────────

class XLearnerModel:
    """
    Cross-fitted imputation of counterfactual outcomes, then a CATE model
    on the imputed pseudo-outcomes.  Performs significantly better when
    treatment and control group sizes are unbalanced.

    Uses a propensity model (XGBClassifier) for inverse-probability weighting.
    """

    def __init__(self):
        self._model = XLearner(
            models=XGBClassifier(**_XGB_CLF_PARAMS),
            propensity_model=XGBClassifier(**{**_XGB_CLF_PARAMS, "n_estimators": 200}),
            cate_models=XGBRegressor(**_XGB_REG_PARAMS),
        )
        self.name = "x_learner"

    def fit(
        self,
        X: np.ndarray,
        T: np.ndarray,
        Y: np.ndarray,
    ) -> "XLearnerModel":
        logger.info("Fitting X-Learner …")
        self._model.fit(Y, T, X=X)
        logger.info("X-Learner fit complete.")
        return self

    def predict_ite(self, X: np.ndarray) -> np.ndarray:
        return self._model.effect(X).ravel()

    def save(self, path: Path | None = None) -> Path:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        path = path or ARTIFACT_DIR / f"{self.name}.joblib"
        joblib.dump(self, path)
        logger.info(f"{self.name} saved → {path}")
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> "XLearnerModel":
        path = path or ARTIFACT_DIR / "x_learner.joblib"
        obj = joblib.load(path)
        logger.info(f"XLearnerModel loaded ← {path}")
        return obj


# ─────────────────────────────────────────────────────────────────────────────
# DR-Learner  (doubly-robust, most efficient)
# ─────────────────────────────────────────────────────────────────────────────

class DRLearnerModel:
    """
    Doubly-Robust learner: consistent if *either* the outcome model or the
    propensity model is correctly specified.  Best asymptotic efficiency.

    Implemented via EconML's DRLearner with cross-fitting (cv=3).
    """

    def __init__(self):
        self._model = DRLearner(
            model_propensity=XGBClassifier(**{**_XGB_CLF_PARAMS, "n_estimators": 200}),
            model_regression=XGBClassifier(**_XGB_CLF_PARAMS),
            model_final=XGBRegressor(**_XGB_REG_PARAMS),
            cv=3,
            random_state=42,
        )
        self.name = "dr_learner"

    def fit(
        self,
        X: np.ndarray,
        T: np.ndarray,
        Y: np.ndarray,
    ) -> "DRLearnerModel":
        logger.info("Fitting DR-Learner (cross-fitted, cv=3) …")
        self._model.fit(Y, T, X=X)
        logger.info("DR-Learner fit complete.")
        return self

    def predict_ite(self, X: np.ndarray) -> np.ndarray:
        return self._model.effect(X).ravel()

    def save(self, path: Path | None = None) -> Path:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        path = path or ARTIFACT_DIR / f"{self.name}.joblib"
        joblib.dump(self, path)
        logger.info(f"{self.name} saved → {path}")
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> "DRLearnerModel":
        path = path or ARTIFACT_DIR / "dr_learner.joblib"
        obj = joblib.load(path)
        logger.info(f"DRLearnerModel loaded ← {path}")
        return obj


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

LearnerType = Literal["t_learner", "x_learner", "dr_learner"]

_REGISTRY: dict[str, type] = {
    "t_learner": TLearnerModel,
    "x_learner": XLearnerModel,
    "dr_learner": DRLearnerModel,
}


def build_learner(name: LearnerType = "x_learner"):
    """Instantiate a learner by name."""
    if name not in _REGISTRY:
        raise ValueError(f"Unknown learner '{name}'. Choose from {list(_REGISTRY)}")
    return _REGISTRY[name]()


def load_learner(name: LearnerType = "x_learner"):
    """Load a serialised learner by name."""
    path = ARTIFACT_DIR / f"{name}.joblib"
    return joblib.load(path)
