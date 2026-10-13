"""
src/api/predictor.py

Model-loading singleton and inference logic for the FastAPI layer.

Design
──────
• Models are loaded once at startup via lifespan and cached in module-level
  dicts — zero-copy on every request.
• Preprocessing (StandardScaler + interaction features) is also loaded once.
• Segment labelling uses ITE thresholds calibrated from domain knowledge:
    persuadable   : ITE > 0.05
    sleeping_dog  : ITE < -0.02  (treatment harms this customer)
    sure_thing    : ITE > 0.00 but low absolute score (model sees low variance)
    lost_cause    : ITE ≤ 0.00 and control outcome very low
    neutral       : everything else
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from src.data.preprocessor import UpliftPreprocessor
from src.models.uplift import load_learner
from src.explain.shap_explainer import UpliftExplainer

ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "models"

# ─────────────────────────────────────────────────────────────────────────────
# Module-level caches (populated during FastAPI lifespan startup)
# ─────────────────────────────────────────────────────────────────────────────

_preprocessor: UpliftPreprocessor | None = None
_learners: dict[str, Any] = {}
_explainers: dict[str, UpliftExplainer] = {}


def _segment(ite: float) -> str:
    if ite >= 0.05:
        return "persuadable"
    elif ite < -0.02:
        return "sleeping_dog"
    elif 0.0 < ite < 0.05:
        return "sure_thing"
    elif -0.02 <= ite <= 0.0:
        return "neutral"
    else:
        return "lost_cause"


# ─────────────────────────────────────────────────────────────────────────────
# Startup / shutdown
# ─────────────────────────────────────────────────────────────────────────────

def load_all_artifacts(learner_names: list[str] | None = None) -> None:
    """
    Called during FastAPI lifespan startup.
    Loads preprocessor + every available serialised learner.
    """
    global _preprocessor

    # Preprocessor
    prep_path = ARTIFACT_DIR / "preprocessor.joblib"
    if prep_path.exists():
        _preprocessor = UpliftPreprocessor.load(prep_path)
    else:
        raise FileNotFoundError(
            f"Preprocessor not found at {prep_path}. Run scripts/train.py first."
        )

    # Learners
    available = learner_names or ["t_learner", "x_learner", "dr_learner"]
    for name in available:
        path = ARTIFACT_DIR / f"{name}.joblib"
        if path.exists():
            _learners[name] = load_learner(name)
            _explainers[name] = UpliftExplainer(
                _learners[name],
                feature_names=_preprocessor.feature_names_,
            )
            logger.info(f"Loaded learner + explainer: {name}")
        else:
            logger.warning(f"Learner '{name}' not found at {path} — skipping.")

    if not _learners:
        raise RuntimeError("No trained learners found. Run scripts/train.py first.")


def loaded_model_names() -> list[str]:
    return list(_learners.keys())


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def _features_to_df(customers: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(customers)


def predict_batch(
    customers: list[dict],
    model_name: str = "x_learner",
    threshold: float = 0.05,
    explain: bool = False,
) -> list[dict]:
    """
    Core inference function.

    Parameters
    ----------
    customers : list of feature dicts (keys f0 … f11)
    model_name: one of loaded learner names
    threshold : ITE ≥ threshold → recommend offer
    explain   : compute per-instance SHAP top-5 contributions

    Returns
    -------
    list of result dicts, one per customer
    """
    if _preprocessor is None or model_name not in _learners:
        raise RuntimeError(
            f"Model '{model_name}' not loaded. Available: {list(_learners)}"
        )

    df = _features_to_df(customers)
    X = _preprocessor.transform(df)

    learner = _learners[model_name]
    ites = learner.predict_ite(X)  # shape (n,)

    shap_values = None
    if explain and model_name in _explainers:
        explainer = _explainers[model_name]
        shap_values = explainer.compute(X)

    results = []
    for i, ite in enumerate(ites):
        rec: dict[str, Any] = {
            "customer_index": i,
            "predicted_ite": round(float(ite), 6),
            "recommend_offer": bool(ite >= threshold),
            "segment": _segment(float(ite)),
        }
        if shap_values is not None:
            contribs = _explainers[model_name].instance_explanation(
                shap_values, idx=i, top_n=5
            )
            rec["shap_contributions"] = contribs
        else:
            rec["shap_contributions"] = None
        results.append(rec)

    return results
