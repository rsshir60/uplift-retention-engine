"""
scripts/evaluate.py

Load a previously trained learner and produce a full evaluation report:
  - Qini coefficient, AUUC
  - Uplift-by-decile table (CSV + chart)
  - Net revenue simulation at several targeting thresholds
  - SHAP global importance chart

Usage
─────
  python scripts/evaluate.py --model x_learner --source synthetic
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import typer
from loguru import logger

from src.data.loader import OUTCOME_COL, TREATMENT_COL, load_dataset, split_dataset
from src.data.preprocessor import UpliftPreprocessor
from src.explain.shap_explainer import UpliftExplainer
from src.models.evaluator import (
    auuc,
    net_revenue_saved,
    plot_qini_curve,
    plot_uplift_decile,
    qini_coefficient,
    uplift_by_decile,
)
from src.models.uplift import load_learner

ARTIFACT_DIR = ROOT / "artifacts"
REPORT_DIR = ARTIFACT_DIR / "reports"

app = typer.Typer(add_completion=False)


@app.command()
def evaluate(
    model: str = typer.Option("x_learner", help="Learner name to evaluate"),
    source: str = typer.Option("synthetic", help="'synthetic' or 'criteo'"),
    n_samples: int = typer.Option(150_000),
    sample_frac: float = typer.Option(0.10),
    random_state: int = typer.Option(42),
    thresholds: str = typer.Option(
        "0.02,0.05,0.10,0.15",
        help="Comma-separated ITE thresholds for business simulation",
    ),
):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Reproduce test split ──────────────────────────────────────────────────
    logger.info("Loading dataset for evaluation …")
    df = load_dataset(source=source, n_samples=n_samples, sample_frac=sample_frac, random_state=random_state)
    _, _, test_df = split_dataset(df, random_state=random_state)

    prep = UpliftPreprocessor.load()
    X_test = prep.transform(test_df)
    T_test = test_df[TREATMENT_COL].values.astype(np.int8)
    Y_test = test_df[OUTCOME_COL].values.astype(np.int8)

    # ── Load model ────────────────────────────────────────────────────────────
    logger.info(f"Loading learner: {model}")
    learner = load_learner(model)
    ites = learner.predict_ite(X_test)

    # ── Core uplift metrics ───────────────────────────────────────────────────
    q = qini_coefficient(Y_test, T_test, ites)
    a = auuc(Y_test, T_test, ites)
    logger.info(f"Qini coefficient : {q:.6f}")
    logger.info(f"AUUC (normalised): {a:.6f}")

    # ── Decile table ──────────────────────────────────────────────────────────
    decile_df = uplift_by_decile(Y_test, T_test, ites)
    csv_path = REPORT_DIR / f"decile_{model}.csv"
    decile_df.to_csv(csv_path, index=False)
    logger.info(f"Decile table saved → {csv_path}")
    plot_uplift_decile(decile_df, model_name=model)

    # ── Qini curve ────────────────────────────────────────────────────────────
    plot_qini_curve(Y_test, T_test, {model: ites})

    # ── Business simulation at multiple thresholds ────────────────────────────
    print("\n── Business Simulation ──────────────────────────────")
    thr_list = [float(t) for t in thresholds.split(",")]
    for thr in thr_list:
        res = net_revenue_saved(
            ites, T_test, Y_test,
            offer_cost=10.0,
            revenue_per_retention=120.0,
            targeting_threshold=thr,
        )
        print(
            f"  threshold={thr:.2f} | "
            f"targeting={res['targeting_rate_pct']:.1f}% | "
            f"profit=\${res['profit_usd']:,.0f} | "
            f"ROI={res['roi_pct']:.1f}% | "
            f"vs_naive=\${res['profit_vs_naive_usd']:,.0f}"
        )

    # ── SHAP global importance ────────────────────────────────────────────────
    shap_idx = np.random.choice(len(X_test), min(2000, len(X_test)), replace=False)
    explainer = UpliftExplainer(learner, prep.feature_names_)
    shap_vals = explainer.compute(X_test[shap_idx])
    explainer.plot_bar_importance(shap_vals, X_test[shap_idx])
    explainer.plot_summary(shap_vals, X_test[shap_idx])

    top5 = explainer.top_features(shap_vals, top_n=5)
    print("\n── Top-5 Features by Mean |SHAP| ────────────────────")
    for rank, feat in enumerate(top5, 1):
        print(f"  {rank}. {feat['feature']:<12} {feat['mean_abs_shap']:.6f}")


if __name__ == "__main__":
    app()
