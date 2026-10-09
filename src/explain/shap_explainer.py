"""
src/explain/shap_explainer.py

SHAP-based interpretability for the uplift models.

Design
──────
• For T-Learner / X-Learner we explain the *treated-arm* outcome model (μ₁)
  because that captures the response surface most directly linked to CATE.
• For DR-Learner we explain the CATE model (model_final).
• TreeExplainer is used (O(n · TL) — fast for XGBoost).
• Outputs:
    - Global feature importance bar plot
    - Beeswarm summary plot
    - Waterfall plot for a single customer (top persuadable)
    - Raw shap_values array for downstream use
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import shap
from loguru import logger

PLOT_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "plots"


def _extract_xgb_model(learner: Any) -> Any:
    """
    Pull the XGBoost model that directly predicts on feature space from
    various EconML wrapper types.
    """
    name = type(learner).__name__

    if name == "TLearnerModel":
        # EconML TLearner stores two clones: models_[0]=control, models_[1]=treated
        return learner._model.models_[1]

    elif name == "XLearnerModel":
        # EconML XLearner: cate_models_[0] is the CATE regressor on treated arm
        return learner._model.cate_models_[0]

    elif name == "DRLearnerModel":
        # EconML DRLearner: model_final_ is the final CATE model
        return learner._model.model_final_

    else:
        raise TypeError(f"Unsupported learner type: {name}")


class UpliftExplainer:
    """
    Wraps SHAP TreeExplainer for uplift models.

    Usage
    -----
    explainer = UpliftExplainer(learner, feature_names)
    shap_vals  = explainer.compute(X_test)
    explainer.plot_summary(shap_vals, X_test)
    explainer.plot_waterfall(shap_vals, idx=0)
    """

    def __init__(
        self,
        learner: Any,
        feature_names: list[str],
    ):
        self.feature_names = feature_names
        self.model_name = type(learner).__name__
        xgb_model = _extract_xgb_model(learner)
        self._explainer = shap.TreeExplainer(xgb_model)
        logger.info(f"SHAP TreeExplainer initialised for {self.model_name}")

    def compute(self, X: np.ndarray) -> np.ndarray:
        """Return SHAP values array of shape (n, n_features)."""
        logger.info(f"Computing SHAP values for {len(X):,} samples …")
        sv = self._explainer.shap_values(X)
        # For binary classifiers SHAP returns list [class0, class1]; take class-1
        if isinstance(sv, list):
            sv = sv[1]
        logger.info("SHAP computation done.")
        return sv

    def plot_summary(
        self,
        shap_values: np.ndarray,
        X: np.ndarray,
        max_display: int = 16,
        save: bool = True,
    ) -> Path:
        """Beeswarm summary — shows direction and magnitude of each feature."""
        PLOT_DIR.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 7))
        shap.summary_plot(
            shap_values,
            X,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False,
            plot_type="dot",
        )
        plt.title(f"SHAP Summary — {self.model_name}", fontsize=13, pad=12)
        plt.tight_layout()
        path = PLOT_DIR / f"shap_summary_{self.model_name.lower()}.png"
        if save:
            plt.savefig(path, dpi=150, bbox_inches="tight")
            logger.info(f"SHAP summary plot saved → {path}")
        plt.close()
        return path

    def plot_bar_importance(
        self,
        shap_values: np.ndarray,
        X: np.ndarray,
        max_display: int = 16,
        save: bool = True,
    ) -> Path:
        """Mean |SHAP| bar chart — global feature importance."""
        PLOT_DIR.mkdir(parents=True, exist_ok=True)
        mean_abs = np.abs(shap_values).mean(axis=0)
        order = np.argsort(mean_abs)[::-1][:max_display]
        names = [self.feature_names[i] for i in order]
        vals = mean_abs[order]

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(names[::-1], vals[::-1], color="#2196F3", alpha=0.85)
        ax.set_xlabel("Mean |SHAP value|", fontsize=12)
        ax.set_title(f"Global Feature Importance — {self.model_name}", fontsize=13)
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()

        path = PLOT_DIR / f"shap_bar_{self.model_name.lower()}.png"
        if save:
            fig.savefig(path, dpi=150)
            logger.info(f"SHAP bar plot saved → {path}")
        plt.close(fig)
        return path

    def plot_waterfall(
        self,
        shap_values: np.ndarray,
        X: np.ndarray,
        idx: int = 0,
        save: bool = True,
    ) -> Path:
        """Waterfall plot for a single customer — shows per-feature contribution."""
        PLOT_DIR.mkdir(parents=True, exist_ok=True)
        expl = shap.Explanation(
            values=shap_values[idx],
            base_values=self._explainer.expected_value
            if not isinstance(self._explainer.expected_value, list)
            else self._explainer.expected_value[1],
            data=X[idx],
            feature_names=self.feature_names,
        )
        shap.waterfall_plot(expl, show=False)
        plt.title(f"SHAP Waterfall — Customer #{idx}", fontsize=12)
        plt.tight_layout()
        path = PLOT_DIR / f"shap_waterfall_{self.model_name.lower()}_idx{idx}.png"
        if save:
            plt.savefig(path, dpi=150, bbox_inches="tight")
            logger.info(f"SHAP waterfall saved → {path}")
        plt.close()
        return path

    def top_features(
        self,
        shap_values: np.ndarray,
        top_n: int = 5,
    ) -> list[dict[str, float]]:
        """Return top-n features by mean |SHAP| as a list of dicts (for API)."""
        mean_abs = np.abs(shap_values).mean(axis=0)
        order = np.argsort(mean_abs)[::-1][:top_n]
        return [
            {"feature": self.feature_names[i], "mean_abs_shap": round(float(mean_abs[i]), 6)}
            for i in order
        ]

    def instance_explanation(
        self,
        shap_values: np.ndarray,
        idx: int = 0,
        top_n: int = 5,
    ) -> list[dict[str, float]]:
        """
        Return top-n SHAP contributions for a single customer
        (sign preserved — positive = pushes toward higher ITE).
        """
        sv = shap_values[idx]
        order = np.argsort(np.abs(sv))[::-1][:top_n]
        return [
            {
                "feature": self.feature_names[i],
                "shap_value": round(float(sv[i]), 6),
            }
            for i in order
        ]
