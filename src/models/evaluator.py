"""
src/models/evaluator.py

Uplift-specific evaluation metrics that go beyond standard ML scores.

Metrics implemented
───────────────────
  qini_coefficient   : Area between model Qini curve and random baseline.
  auuc               : Area Under the Uplift Curve (normalised).
  uplift_by_decile   : Mean CATE per ITE-score decile (calibration check).
  net_revenue_saved  : Business simulation — incremental retention value.
  plot_qini_curve    : Saves Qini + random baseline figure.
  plot_uplift_decile : Bar chart of uplift by decile.

All plot functions return the figure path so callers can log it to MLflow.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

PLOT_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "plots"


# ─────────────────────────────────────────────────────────────────────────────
# Core curve builders
# ─────────────────────────────────────────────────────────────────────────────

def _uplift_curve(
    y_true: np.ndarray,
    treatment: np.ndarray,
    uplift_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sort population by descending predicted ITE and compute the cumulative
    actual uplift at each population percentile.

    Returns (fraction_targeted, cumulative_uplift).
    """
    n = len(y_true)
    order = np.argsort(-uplift_scores)
    y_sorted = y_true[order]
    t_sorted = treatment[order]

    cumulative_uplift = []
    for k in range(1, n + 1):
        ys = y_sorted[:k]
        ts = t_sorted[:k]
        treated_mask = ts == 1
        control_mask = ts == 0
        n_t = treated_mask.sum()
        n_c = control_mask.sum()
        if n_t == 0 or n_c == 0:
            cumulative_uplift.append(0.0)
            continue
        lift = (ys[treated_mask].sum() / n_t) - (ys[control_mask].sum() / n_c)
        cumulative_uplift.append(lift * k)  # absolute incremental conversions

    fractions = np.arange(1, n + 1) / n
    return fractions, np.array(cumulative_uplift)


def _random_baseline(
    y_true: np.ndarray,
    treatment: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Qini curve for a random model (straight line from 0 to total uplift)."""
    n = len(y_true)
    yt, yc = y_true[treatment == 1], y_true[treatment == 0]
    nt, nc = len(yt), len(yc)
    total_uplift = (yt.sum() / nt - yc.sum() / nc) * n
    fractions = np.array([0.0, 1.0])
    baseline = np.array([0.0, total_uplift])
    return fractions, baseline


# ─────────────────────────────────────────────────────────────────────────────
# Public metric functions
# ─────────────────────────────────────────────────────────────────────────────

def qini_coefficient(
    y_true: np.ndarray,
    treatment: np.ndarray,
    uplift_scores: np.ndarray,
) -> float:
    """
    Qini coefficient = (AUC of model curve − AUC of random baseline) / n.
    Higher is better; 0 = random, 1 = perfect ordering.
    """
    fracs, curve = _uplift_curve(y_true, treatment, uplift_scores)
    _, baseline = _random_baseline(y_true, treatment)
    baseline_interp = np.interp(fracs, [0.0, 1.0], baseline)
    qini = np.trapz(curve - baseline_interp, fracs)
    # Normalise by the area of perfect model (approximated by max cumulative uplift)
    max_area = np.trapz(
        np.sort(curve)[::-1] - baseline_interp, fracs
    )
    if max_area == 0:
        return 0.0
    return float(qini / max_area)


def auuc(
    y_true: np.ndarray,
    treatment: np.ndarray,
    uplift_scores: np.ndarray,
    normalise: bool = True,
) -> float:
    """Area Under the Uplift Curve, optionally normalised to [0, 1]."""
    fracs, curve = _uplift_curve(y_true, treatment, uplift_scores)
    area = np.trapz(curve, fracs)
    if normalise:
        random_area = np.trapz(
            np.interp(fracs, [0.0, 1.0], [0.0, curve[-1]]), fracs
        )
        return float(area - random_area)
    return float(area)


def uplift_by_decile(
    y_true: np.ndarray,
    treatment: np.ndarray,
    uplift_scores: np.ndarray,
) -> pd.DataFrame:
    """
    Bin population into 10 equal-sized buckets by predicted ITE (desc order).
    For each decile report:
      - mean predicted ITE
      - observed uplift (treatment conversion − control conversion rate)
      - n_treated, n_control
    """
    df = pd.DataFrame(
        {"y": y_true, "t": treatment, "score": uplift_scores}
    )
    df["decile"] = pd.qcut(
        df["score"].rank(method="first"),
        q=10,
        labels=[f"D{i}" for i in range(10, 0, -1)],
    )
    rows = []
    for decile, grp in df.groupby("decile", observed=True):
        nt = (grp["t"] == 1).sum()
        nc = (grp["t"] == 0).sum()
        conv_t = grp.loc[grp["t"] == 1, "y"].mean() if nt > 0 else np.nan
        conv_c = grp.loc[grp["t"] == 0, "y"].mean() if nc > 0 else np.nan
        obs_uplift = (conv_t - conv_c) if (nt > 0 and nc > 0) else np.nan
        rows.append(
            {
                "decile": decile,
                "mean_predicted_ite": grp["score"].mean(),
                "observed_uplift": obs_uplift,
                "n_treated": nt,
                "n_control": nc,
                "conversion_treated": conv_t,
                "conversion_control": conv_c,
            }
        )
    result = pd.DataFrame(rows)
    logger.info("\n" + result.to_string(index=False))
    return result


def net_revenue_saved(
    uplift_scores: np.ndarray,
    treatment: np.ndarray,
    y_true: np.ndarray,
    offer_cost: float = 10.0,
    revenue_per_retention: float = 120.0,
    targeting_threshold: float = 0.05,
) -> dict[str, float]:
    """
    Business simulation:
      - Target only users with predicted ITE > threshold.
      - Revenue = retentions_incremental × revenue_per_retention.
      - Cost    = n_targeted × offer_cost.
      - ROI     = (Revenue − Cost) / Cost × 100.

    Compared against the naive strategy of targeting everyone.
    """
    mask_targeted = uplift_scores >= targeting_threshold
    n_targeted = mask_targeted.sum()

    # Estimate incremental retentions in targeted group
    grp = pd.DataFrame(
        {"y": y_true, "t": treatment, "score": uplift_scores}
    )[mask_targeted]
    nt = (grp["t"] == 1).sum()
    nc = (grp["t"] == 0).sum()
    if nt == 0 or nc == 0:
        obs_uplift = 0.0
    else:
        obs_uplift = (
            grp.loc[grp["t"] == 1, "y"].mean()
            - grp.loc[grp["t"] == 0, "y"].mean()
        )
    incremental_retentions = obs_uplift * n_targeted

    revenue = incremental_retentions * revenue_per_retention
    cost = n_targeted * offer_cost
    profit = revenue - cost
    roi = (profit / cost * 100) if cost > 0 else 0.0

    # Naive baseline: target everyone
    n_all = len(uplift_scores)
    nt_all = (treatment == 1).sum()
    nc_all = (treatment == 0).sum()
    if nt_all > 0 and nc_all > 0:
        uplift_all = (
            y_true[treatment == 1].mean() - y_true[treatment == 0].mean()
        )
    else:
        uplift_all = 0.0
    naive_revenue = uplift_all * n_all * revenue_per_retention
    naive_cost = n_all * offer_cost
    naive_profit = naive_revenue - naive_cost

    results = {
        "n_targeted": int(n_targeted),
        "targeting_rate_pct": round(n_targeted / len(uplift_scores) * 100, 2),
        "incremental_retentions": round(incremental_retentions, 1),
        "revenue_usd": round(revenue, 2),
        "cost_usd": round(cost, 2),
        "profit_usd": round(profit, 2),
        "roi_pct": round(roi, 2),
        "naive_profit_usd": round(naive_profit, 2),
        "profit_vs_naive_usd": round(profit - naive_profit, 2),
    }
    logger.info("Business simulation:\n" + "\n".join(f"  {k}: {v}" for k, v in results.items()))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────────────────────────────────────

def plot_qini_curve(
    y_true: np.ndarray,
    treatment: np.ndarray,
    models: dict[str, np.ndarray],
    save: bool = True,
) -> Path:
    """
    Overlay Qini curves for multiple models.

    Parameters
    ----------
    models : {"model_name": uplift_scores_array, …}
    """
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6))

    _, baseline = _random_baseline(y_true, treatment)
    ax.plot([0, 1], baseline, "--", color="grey", label="Random baseline", lw=1.5)

    colors = ["#2196F3", "#E91E63", "#4CAF50", "#FF9800"]
    for (name, scores), color in zip(models.items(), colors):
        fracs, curve = _uplift_curve(y_true, treatment, scores)
        qini = qini_coefficient(y_true, treatment, scores)
        ax.plot(fracs, curve, label=f"{name}  (Qini={qini:.4f})", color=color, lw=2)

    ax.set_xlabel("Fraction of population targeted", fontsize=12)
    ax.set_ylabel("Cumulative incremental conversions", fontsize=12)
    ax.set_title("Qini Curve — Uplift Model Comparison", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    path = PLOT_DIR / "qini_curve.png"
    if save:
        fig.savefig(path, dpi=150)
        logger.info(f"Qini curve saved → {path}")
    plt.close(fig)
    return path


def plot_uplift_decile(
    decile_df: pd.DataFrame,
    model_name: str = "x_learner",
    save: bool = True,
) -> Path:
    """Bar chart of predicted vs observed uplift per decile."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(decile_df))
    w = 0.35
    ax.bar(x - w / 2, decile_df["mean_predicted_ite"], w, label="Predicted ITE", color="#2196F3", alpha=0.85)
    ax.bar(x + w / 2, decile_df["observed_uplift"], w, label="Observed uplift", color="#E91E63", alpha=0.85)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(decile_df["decile"].tolist(), rotation=45)
    ax.set_xlabel("Decile (D10 = highest predicted ITE)", fontsize=12)
    ax.set_ylabel("Uplift", fontsize=12)
    ax.set_title(f"Predicted vs Observed Uplift by Decile — {model_name}", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    path = PLOT_DIR / f"uplift_decile_{model_name}.png"
    if save:
        fig.savefig(path, dpi=150)
        logger.info(f"Decile plot saved → {path}")
    plt.close(fig)
    return path
