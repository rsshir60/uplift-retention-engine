"""
src/data/loader.py

Handles two data sources:
  1. Criteo Uplift dataset (public benchmark, ~14M rows) — downloaded on demand.
  2. Synthetic A/B simulation — generated locally when Criteo is unavailable.

Criteo schema
─────────────
  f0 … f11  : anonymised numeric features
  treatment  : 0 (control) / 1 (treated with retention offer)
  visit      : 1 if user visited within 2 weeks (intermediate outcome)
  conversion : 1 if user converted / retained (primary outcome)
"""

from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from loguru import logger
from sklearn.model_selection import train_test_split

CRITEO_URL = (
    "https://go.criteo.net/criteo-research-uplift-v2.1.csv.gz"
)
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

FEATURE_COLS = [f"f{i}" for i in range(12)]
TREATMENT_COL = "treatment"
OUTCOME_COL = "conversion"


# ─────────────────────────────────────────────────────────────────────────────
# Criteo download
# ─────────────────────────────────────────────────────────────────────────────

def download_criteo(force: bool = False) -> Path:
    """Download the Criteo Uplift v2.1 CSV (gzipped) if not already cached."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = RAW_DIR / "criteo_uplift.csv.gz"
    if dest.exists() and not force:
        logger.info(f"Criteo dataset already cached at {dest}")
        return dest
    logger.info("Downloading Criteo Uplift dataset (~300 MB) …")
    resp = requests.get(CRITEO_URL, stream=True, timeout=120)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                logger.info(f"  {pct:.1f}%")
    logger.info("Download complete.")
    return dest


def load_criteo(sample_frac: float = 0.10, random_state: int = 42) -> pd.DataFrame:
    """
    Load Criteo dataset, sample it to ``sample_frac`` for tractable training,
    and return a clean DataFrame with FEATURE_COLS + treatment + conversion.
    """
    path = download_criteo()
    logger.info(f"Reading Criteo CSV from {path} …")
    df = pd.read_csv(path, compression="gzip")
    df = df.rename(columns=str.lower)
    df = df[FEATURE_COLS + [TREATMENT_COL, "visit", OUTCOME_COL]].dropna()
    df = df.sample(frac=sample_frac, random_state=random_state).reset_index(drop=True)
    logger.info(
        f"Loaded {len(df):,} rows | treatment rate: "
        f"{df[TREATMENT_COL].mean():.3f} | conversion rate: "
        f"{df[OUTCOME_COL].mean():.4f}"
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic A/B data generator
# ─────────────────────────────────────────────────────────────────────────────

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_synthetic(
    n_samples: int = 150_000,
    n_features: int = 12,
    treatment_prob: float = 0.50,
    base_conversion_rate: float = 0.08,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Simulate an A/B retention experiment with heterogeneous treatment effects.

    Persuadables (CATE > 0): ~30 % of population.
    Sure things  (always convert): ~15 %.
    Lost causes  (never convert):  ~20 %.
    Sleeping dogs (harmed by offer): ~5 %.

    The remaining ~30 % are neutral — treatment has no effect.
    """
    rng = np.random.default_rng(random_state)

    X = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    feature_names = [f"f{i}" for i in range(n_features)]

    # Propensity (pure random A/B — ignorability satisfied)
    treatment = rng.binomial(1, treatment_prob, n_samples).astype(np.int8)

    # Latent customer segment
    segment_logit = (
        0.4 * X[:, 0]
        - 0.3 * X[:, 1]
        + 0.2 * X[:, 2] * X[:, 3]
        + 0.5 * X[:, 4]
    )
    segment_prob = _sigmoid(segment_logit)

    # Baseline conversion probability (control world)
    mu0 = _sigmoid(
        -2.5
        + 0.6 * X[:, 0]
        + 0.3 * X[:, 1]
        - 0.4 * X[:, 5]
        + 0.2 * X[:, 6]
    )

    # Heterogeneous CATE: positive for persuadables, negative for sleeping dogs
    cate = (
        0.15 * segment_prob          # persuadables benefit most
        - 0.05 * (1 - segment_prob)  # sleeping dogs are slightly harmed
        + 0.03 * X[:, 7]
        - 0.02 * X[:, 8]
    )

    mu1 = np.clip(mu0 + cate, 1e-6, 1 - 1e-6)

    # Observed outcome
    p_outcome = np.where(treatment == 1, mu1, mu0)
    conversion = rng.binomial(1, p_outcome).astype(np.int8)

    df = pd.DataFrame(X, columns=feature_names)
    df[TREATMENT_COL] = treatment
    df[OUTCOME_COL] = conversion
    df["true_cate"] = cate.astype(np.float32)  # kept for evaluation only

    logger.info(
        f"Synthetic data: {n_samples:,} rows | "
        f"treatment={df[TREATMENT_COL].mean():.3f} | "
        f"conversion={df[OUTCOME_COL].mean():.4f} | "
        f"mean CATE={cate.mean():.4f}"
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Unified loader
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(
    source: str = "synthetic",
    sample_frac: float = 0.10,
    n_samples: int = 150_000,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Parameters
    ----------
    source : 'synthetic' | 'criteo'
    sample_frac : fraction of Criteo rows to keep (ignored for synthetic)
    n_samples : synthetic dataset size (ignored for criteo)
    """
    if source == "criteo":
        return load_criteo(sample_frac=sample_frac, random_state=random_state)
    elif source == "synthetic":
        return generate_synthetic(n_samples=n_samples, random_state=random_state)
    else:
        raise ValueError(f"Unknown source '{source}'. Choose 'synthetic' or 'criteo'.")


def split_dataset(
    df: pd.DataFrame,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified split on treatment × conversion to preserve both rates."""
    strat = df[TREATMENT_COL].astype(str) + "_" + df[OUTCOME_COL].astype(str)
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=strat, random_state=random_state
    )
    strat_tv = train_val[TREATMENT_COL].astype(str) + "_" + train_val[OUTCOME_COL].astype(str)
    val_rel = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=val_rel, stratify=strat_tv, random_state=random_state
    )
    logger.info(
        f"Split: train={len(train):,}  val={len(val):,}  test={len(test):,}"
    )
    return train, val, test
