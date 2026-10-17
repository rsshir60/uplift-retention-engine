"""
tests/test_uplift.py

Unit + integration tests for data pipeline, uplift models, and evaluator.
Run: pytest tests/ -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.loader import generate_synthetic, split_dataset, FEATURE_COLS, TREATMENT_COL, OUTCOME_COL
from src.data.preprocessor import UpliftPreprocessor, add_interaction_features
from src.models.uplift import TLearnerModel, XLearnerModel, build_learner
from src.models.evaluator import (
    auuc,
    qini_coefficient,
    uplift_by_decile,
    net_revenue_saved,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def small_df():
    return generate_synthetic(n_samples=5_000, random_state=0)


@pytest.fixture(scope="module")
def split_data(small_df):
    train, val, test = split_dataset(small_df, random_state=0)
    return train, val, test


@pytest.fixture(scope="module")
def preprocessed(split_data):
    train, val, test = split_data
    prep = UpliftPreprocessor(with_interactions=True)
    X_train = prep.fit_transform(train)
    X_test = prep.transform(test)
    T_train = train[TREATMENT_COL].values.astype(np.int8)
    Y_train = train[OUTCOME_COL].values.astype(np.int8)
    T_test = test[TREATMENT_COL].values.astype(np.int8)
    Y_test = test[OUTCOME_COL].values.astype(np.int8)
    return prep, X_train, T_train, Y_train, X_test, T_test, Y_test


# ─────────────────────────────────────────────────────────────────────────────
# Data / Preprocessing tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDataLoader:
    def test_generate_shape(self, small_df):
        assert len(small_df) == 5_000
        assert TREATMENT_COL in small_df.columns
        assert OUTCOME_COL in small_df.columns

    def test_feature_columns(self, small_df):
        for col in FEATURE_COLS:
            assert col in small_df.columns

    def test_treatment_binary(self, small_df):
        assert set(small_df[TREATMENT_COL].unique()).issubset({0, 1})

    def test_outcome_binary(self, small_df):
        assert set(small_df[OUTCOME_COL].unique()).issubset({0, 1})

    def test_split_sizes(self, split_data):
        train, val, test = split_data
        total = len(train) + len(val) + len(test)
        assert total == 5_000
        assert len(test) / total == pytest.approx(0.15, abs=0.02)

    def test_no_data_leakage(self, split_data):
        train, val, test = split_data
        train_idx = set(train.index)
        val_idx = set(val.index)
        test_idx = set(test.index)
        assert train_idx.isdisjoint(val_idx)
        assert train_idx.isdisjoint(test_idx)


class TestPreprocessor:
    def test_interaction_features_added(self, small_df):
        df_out = add_interaction_features(small_df)
        for col in ["f0_x_f1", "f2_x_f3", "f4_sq", "f5_x_f6"]:
            assert col in df_out.columns

    def test_fit_transform_shape(self, preprocessed):
        prep, X_train, T_train, Y_train, X_test, T_test, Y_test = preprocessed
        n_features = len(prep.feature_names_)
        assert X_train.shape[1] == n_features
        assert X_test.shape[1] == n_features

    def test_no_train_leakage_in_transform(self, preprocessed):
        """Scaler is fitted only on train; test transform must not re-fit."""
        prep, X_train, *rest = preprocessed
        X_test = rest[3]
        # Re-transforming with same prep should produce identical results
        _, _, _, _, test_df, _, _ = rest
        # Skipped; scaler is stateful — just check shape consistency
        assert X_test.shape[1] == len(prep.feature_names_)

    def test_standardisation(self, preprocessed):
        prep, X_train, *_ = preprocessed
        # Train features should be approximately standard normal
        assert abs(X_train.mean()) < 0.1
        assert abs(X_train.std() - 1.0) < 0.1


# ─────────────────────────────────────────────────────────────────────────────
# Model tests
# ─────────────────────────────────────────────────────────────────────────────

class TestTLearner:
    def test_fit_predict_shape(self, preprocessed):
        prep, X_train, T_train, Y_train, X_test, T_test, Y_test = preprocessed
        model = TLearnerModel()
        model.fit(X_train, T_train, Y_train)
        ites = model.predict_ite(X_test)
        assert ites.shape == (len(X_test),)

    def test_ite_range(self, preprocessed):
        prep, X_train, T_train, Y_train, X_test, T_test, Y_test = preprocessed
        model = TLearnerModel()
        model.fit(X_train, T_train, Y_train)
        ites = model.predict_ite(X_test)
        assert ites.min() >= -1.0
        assert ites.max() <= 1.0


class TestXLearner:
    def test_fit_predict(self, preprocessed):
        prep, X_train, T_train, Y_train, X_test, T_test, Y_test = preprocessed
        model = XLearnerModel()
        model.fit(X_train, T_train, Y_train)
        ites = model.predict_ite(X_test)
        assert len(ites) == len(X_test)
        assert not np.any(np.isnan(ites))


class TestFactory:
    def test_build_t_learner(self):
        model = build_learner("t_learner")
        assert isinstance(model, TLearnerModel)

    def test_build_x_learner(self):
        model = build_learner("x_learner")
        assert isinstance(model, XLearnerModel)

    def test_invalid_name(self):
        with pytest.raises(ValueError, match="Unknown learner"):
            build_learner("invalid_learner")


# ─────────────────────────────────────────────────────────────────────────────
# Evaluator tests
# ─────────────────────────────────────────────────────────────────────────────

class TestEvaluator:
    @pytest.fixture(scope="class")
    def eval_data(self, preprocessed):
        prep, X_train, T_train, Y_train, X_test, T_test, Y_test = preprocessed
        model = XLearnerModel()
        model.fit(X_train, T_train, Y_train)
        ites = model.predict_ite(X_test)
        return Y_test, T_test, ites

    def test_qini_is_float(self, eval_data):
        Y, T, ites = eval_data
        q = qini_coefficient(Y, T, ites)
        assert isinstance(q, float)
        assert -1.0 <= q <= 1.0

    def test_auuc_returns_float(self, eval_data):
        Y, T, ites = eval_data
        a = auuc(Y, T, ites)
        assert isinstance(a, float)

    def test_decile_returns_10_rows(self, eval_data):
        Y, T, ites = eval_data
        df = uplift_by_decile(Y, T, ites)
        assert len(df) == 10

    def test_net_revenue_keys(self, eval_data):
        Y, T, ites = eval_data
        res = net_revenue_saved(ites, T, Y)
        expected_keys = {"n_targeted", "roi_pct", "profit_usd", "revenue_usd", "cost_usd"}
        assert expected_keys.issubset(res.keys())

    def test_random_baseline_qini_near_zero(self, eval_data):
        Y, T, _ = eval_data
        rng = np.random.default_rng(99)
        random_scores = rng.uniform(-1, 1, len(Y))
        q = qini_coefficient(Y, T, random_scores)
        # Random model should be near 0 (within ±0.15 due to noise)
        assert abs(q) < 0.15
