"""
scripts/train.py

End-to-end training pipeline:
  1. Load dataset (synthetic or Criteo)
  2. Preprocess + engineer features
  3. Fit T-Learner, X-Learner, DR-Learner
  4. Evaluate on test set (Qini, AUUC, decile chart, business ROI)
  5. Log all metrics + artifacts to MLflow

Usage
─────
  python scripts/train.py --source synthetic --learners x_learner dr_learner
  python scripts/train.py --source criteo --sample-frac 0.05
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import time
from typing import Optional

import mlflow
import mlflow.xgboost
import numpy as np
import typer
from loguru import logger

from src.data.loader import (
    FEATURE_COLS,
    OUTCOME_COL,
    TREATMENT_COL,
    load_dataset,
    split_dataset,
)
from src.data.preprocessor import UpliftPreprocessor
from src.models.evaluator import (
    auuc,
    net_revenue_saved,
    plot_qini_curve,
    plot_uplift_decile,
    qini_coefficient,
    uplift_by_decile,
)
from src.models.uplift import build_learner, LearnerType
from src.explain.shap_explainer import UpliftExplainer

app = typer.Typer(add_completion=False)

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "uplift-retention-engine"


def _fit_and_eval(
    name: LearnerType,
    X_train: np.ndarray,
    T_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    T_test: np.ndarray,
    Y_test: np.ndarray,
    preprocessor: UpliftPreprocessor,
    threshold: float,
) -> dict:
    """Train one learner, compute all metrics, save artefacts. Returns metrics dict."""
    learner = build_learner(name)

    t0 = time.time()
    learner.fit(X_train, T_train, Y_train)
    train_time = time.time() - t0
    logger.info(f"{name} trained in {train_time:.1f}s")

    ites_test = learner.predict_ite(X_test)

    qini = qini_coefficient(Y_test, T_test, ites_test)
    auuc_score = auuc(Y_test, T_test, ites_test)
    roi_metrics = net_revenue_saved(
        ites_test, T_test, Y_test, targeting_threshold=threshold
    )
    decile_df = uplift_by_decile(Y_test, T_test, ites_test)

    metrics = {
        "qini_coefficient": qini,
        "auuc": auuc_score,
        "train_time_s": round(train_time, 2),
        **{f"biz_{k}": v for k, v in roi_metrics.items()},
    }
    logger.info(f"{name} | Qini={qini:.4f}  AUUC={auuc_score:.4f}")

    # Save model + preprocessor
    learner.save()
    preprocessor.save()

    # Plots
    qini_path = plot_qini_curve(
        Y_test, T_test, {name: ites_test}, save=True
    )
    decile_path = plot_uplift_decile(decile_df, model_name=name, save=True)

    # SHAP (on 2,000 sample for speed)
    shap_idx = np.random.choice(len(X_test), min(2000, len(X_test)), replace=False)
    explainer = UpliftExplainer(learner, preprocessor.feature_names_)
    shap_vals = explainer.compute(X_test[shap_idx])
    summary_path = explainer.plot_summary(shap_vals, X_test[shap_idx])
    bar_path = explainer.plot_bar_importance(shap_vals, X_test[shap_idx])

    return {
        "metrics": metrics,
        "plots": [qini_path, decile_path, summary_path, bar_path],
        "ites_test": ites_test,
    }


@app.command()
def train(
    source: str = typer.Option("synthetic", help="'synthetic' or 'criteo'"),
    sample_frac: float = typer.Option(0.10, help="Criteo sample fraction"),
    n_samples: int = typer.Option(150_000, help="Synthetic dataset size"),
    learners: Optional[list[str]] = typer.Option(
        None,
        "--learners",
        help="Learners to train (space-separated). Default: x_learner dr_learner",
    ),
    threshold: float = typer.Option(0.05, help="ITE recommendation threshold"),
    random_state: int = typer.Option(42),
):
    if learners is None:
        learners = ["x_learner", "dr_learner"]

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # ── Data ──────────────────────────────────────────────────────────────────
    logger.info(f"Loading dataset: source={source}")
    df = load_dataset(source=source, sample_frac=sample_frac, n_samples=n_samples, random_state=random_state)
    train_df, val_df, test_df = split_dataset(df, random_state=random_state)

    # ── Preprocessing ─────────────────────────────────────────────────────────
    prep = UpliftPreprocessor(with_interactions=True)
    X_train = prep.fit_transform(train_df)
    X_val   = prep.transform(val_df)
    X_test  = prep.transform(test_df)

    T_train = train_df[TREATMENT_COL].values.astype(np.int8)
    Y_train = train_df[OUTCOME_COL].values.astype(np.int8)
    T_val   = val_df[TREATMENT_COL].values.astype(np.int8)
    Y_val   = val_df[OUTCOME_COL].values.astype(np.int8)
    T_test  = test_df[TREATMENT_COL].values.astype(np.int8)
    Y_test  = test_df[OUTCOME_COL].values.astype(np.int8)

    # ── Train each learner ────────────────────────────────────────────────────
    all_ites: dict[str, np.ndarray] = {}

    for name in learners:
        with mlflow.start_run(run_name=name):
            mlflow.log_params(
                {
                    "learner": name,
                    "data_source": source,
                    "n_train": len(X_train),
                    "n_test": len(X_test),
                    "threshold": threshold,
                    "with_interactions": True,
                }
            )
            result = _fit_and_eval(
                name=name,
                X_train=X_train, T_train=T_train, Y_train=Y_train,
                X_test=X_test,  T_test=T_test,   Y_test=Y_test,
                preprocessor=prep,
                threshold=threshold,
            )
            mlflow.log_metrics(result["metrics"])
            for plot_path in result["plots"]:
                mlflow.log_artifact(str(plot_path))
            all_ites[name] = result["ites_test"]

    # ── Combined Qini chart (all models in one plot) ───────────────────────────
    if len(all_ites) > 1:
        combined_path = plot_qini_curve(Y_test, T_test, all_ites, save=True)
        logger.info(f"Combined Qini plot → {combined_path}")

    logger.info("Training complete. Run `mlflow ui` to explore results.")


if __name__ == "__main__":
    app()
