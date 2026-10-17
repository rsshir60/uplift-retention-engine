"""
tests/test_api.py

Integration tests for the FastAPI serving layer.
Uses httpx.AsyncClient with ASGITransport — no real server needed.

Run: pytest tests/test_api.py -v --asyncio-mode=auto
"""

from __future__ import annotations

import numpy as np
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

# Patch predictor before importing app
from unittest.mock import patch, MagicMock

MOCK_LEARNER_NAMES = ["x_learner"]


def _mock_predict_batch(customers, model_name, threshold, explain):
    """Returns deterministic fake ITE scores for testing."""
    results = []
    for i, _ in enumerate(customers):
        ite = 0.08 if i % 3 == 0 else (-0.03 if i % 3 == 1 else 0.01)
        segment_map = {True: "persuadable", False: "neutral"}
        recommend = ite >= threshold
        results.append(
            {
                "customer_index": i,
                "predicted_ite": ite,
                "recommend_offer": recommend,
                "segment": "persuadable" if ite >= 0.05 else "sleeping_dog" if ite < -0.02 else "neutral",
                "shap_contributions": (
                    [{"feature": "f0", "shap_value": 0.04}] if explain else None
                ),
            }
        )
    return results


@pytest.fixture(scope="module")
def app_client():
    """Create test client with mocked model loading."""
    with (
        patch("src.api.predictor.load_all_artifacts", return_value=None),
        patch("src.api.predictor.loaded_model_names", return_value=MOCK_LEARNER_NAMES),
        patch("src.api.predictor.predict_batch", side_effect=_mock_predict_batch),
    ):
        from src.api.main import app
        transport = ASGITransport(app=app)
        client = AsyncClient(transport=transport, base_url="http://test")
        return client, app


SAMPLE_CUSTOMER = {
    "f0": 1.2, "f1": -0.5, "f2": 0.8, "f3": 2.1,
    "f4": 0.0, "f5": -1.1, "f6": 0.3, "f7": 0.9,
    "f8": -0.2, "f9": 1.5, "f10": 0.1, "f11": -0.7,
}


@pytest.mark.asyncio
async def test_health(app_client):
    client, _ = app_client
    resp = await client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "loaded_models" in body


@pytest.mark.asyncio
async def test_recommend_batch_basic(app_client):
    client, _ = app_client
    payload = {
        "customers": [SAMPLE_CUSTOMER] * 3,
        "model": "x_learner",
        "threshold": 0.05,
        "explain": False,
    }
    resp = await client.post("/recommend", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["n_customers"] == 3
    assert len(body["recommendations"]) == 3


@pytest.mark.asyncio
async def test_recommend_batch_with_explain(app_client):
    client, _ = app_client
    payload = {
        "customers": [SAMPLE_CUSTOMER],
        "model": "x_learner",
        "threshold": 0.05,
        "explain": True,
    }
    resp = await client.post("/recommend", json=payload)
    assert resp.status_code == 200
    rec = resp.json()["recommendations"][0]
    # SHAP contributions should be present
    assert rec["shap_contributions"] is not None
    assert isinstance(rec["shap_contributions"], list)


@pytest.mark.asyncio
async def test_recommend_single(app_client):
    client, _ = app_client
    resp = await client.post(
        "/recommend/single?model=x_learner&threshold=0.05&explain=false",
        json=SAMPLE_CUSTOMER,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "predicted_ite" in body
    assert "recommend_offer" in body
    assert "segment" in body


@pytest.mark.asyncio
async def test_recommend_invalid_model(app_client):
    client, _ = app_client
    payload = {
        "customers": [SAMPLE_CUSTOMER],
        "model": "nonexistent_model",
        "threshold": 0.05,
        "explain": False,
    }
    resp = await client.post("/recommend", json=payload)
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_recommend_threshold_effect(app_client):
    """High threshold → fewer recommendations."""
    client, _ = app_client
    payload_low = {
        "customers": [SAMPLE_CUSTOMER] * 6,
        "model": "x_learner",
        "threshold": -1.0,
        "explain": False,
    }
    payload_high = {
        "customers": [SAMPLE_CUSTOMER] * 6,
        "model": "x_learner",
        "threshold": 0.99,
        "explain": False,
    }
    r_low  = await client.post("/recommend", json=payload_low)
    r_high = await client.post("/recommend", json=payload_high)
    assert r_low.json()["n_recommended"] >= r_high.json()["n_recommended"]


@pytest.mark.asyncio
async def test_batch_size_limit(app_client):
    """1001 customers → 422 validation error."""
    client, _ = app_client
    payload = {
        "customers": [SAMPLE_CUSTOMER] * 1001,
        "model": "x_learner",
        "threshold": 0.05,
        "explain": False,
    }
    resp = await client.post("/recommend", json=payload)
    assert resp.status_code == 422
