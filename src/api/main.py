"""
src/api/main.py

FastAPI application — exposes three endpoints:

  GET  /health           → liveness + loaded models list
  POST /recommend        → batch ITE scoring + offer recommendation
  POST /recommend/single → single-customer shortcut with SHAP explanation

Run locally:
    uvicorn src.api.main:app --reload --port 8000

Docker:
    docker compose up
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.api.predictor import (
    load_all_artifacts,
    loaded_model_names,
    predict_batch,
)
from src.api.schemas import (
    BatchRecommendRequest,
    BatchRecommendResponse,
    CustomerFeatures,
    CustomerRecommendation,
    HealthResponse,
    SHAPContribution,
)

APP_VERSION = "1.0.0"


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading uplift models …")
    load_all_artifacts()
    logger.info(f"Models ready: {loaded_model_names()}")
    yield
    logger.info("Shutting down.")


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Uplift Retention Engine",
    description=(
        "Causal ML API that scores customers by their Individual Treatment Effect "
        "(ITE) and recommends personalised retention offers — targeting only "
        "persuadable users to maximise net incremental revenue."
    ),
    version=APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    """Liveness check and model inventory."""
    return HealthResponse(
        status="ok",
        loaded_models=loaded_model_names(),
        version=APP_VERSION,
    )


@app.post("/recommend", response_model=BatchRecommendResponse, tags=["inference"])
async def recommend_batch(request: BatchRecommendRequest) -> BatchRecommendResponse:
    """
    Score a batch of up to 1,000 customers and return:
    - Predicted ITE (Individual Treatment Effect)
    - Whether to send a retention offer (ITE ≥ threshold)
    - Customer segment (persuadable / sure_thing / sleeping_dog / neutral)
    - Optional SHAP feature contributions per customer
    """
    if request.model not in loaded_model_names():
        raise HTTPException(
            status_code=422,
            detail=(
                f"Model '{request.model}' not loaded. "
                f"Available: {loaded_model_names()}"
            ),
        )

    customers_dicts = [c.model_dump() for c in request.customers]

    try:
        raw_results = predict_batch(
            customers=customers_dicts,
            model_name=request.model,
            threshold=request.threshold,
            explain=request.explain,
        )
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(exc))

    recommendations = []
    for r in raw_results:
        contribs = None
        if r["shap_contributions"] is not None:
            contribs = [
                SHAPContribution(**c) for c in r["shap_contributions"]
            ]
        recommendations.append(
            CustomerRecommendation(
                customer_index=r["customer_index"],
                predicted_ite=r["predicted_ite"],
                recommend_offer=r["recommend_offer"],
                segment=r["segment"],
                shap_contributions=contribs,
            )
        )

    n_recommended = sum(1 for r in recommendations if r.recommend_offer)

    return BatchRecommendResponse(
        model_used=request.model,
        threshold=request.threshold,
        n_customers=len(recommendations),
        n_recommended=n_recommended,
        recommendations=recommendations,
    )


@app.post(
    "/recommend/single",
    response_model=CustomerRecommendation,
    tags=["inference"],
)
async def recommend_single(
    customer: CustomerFeatures,
    model: str = Query(default="x_learner"),
    threshold: float = Query(default=0.05, ge=-1.0, le=1.0),
    explain: bool = Query(default=True),
) -> CustomerRecommendation:
    """
    Score a single customer. Returns SHAP explanation by default — useful for
    integration with customer-service dashboards or A/B experiment overrides.
    """
    if model not in loaded_model_names():
        raise HTTPException(
            status_code=422,
            detail=f"Model '{model}' not loaded. Available: {loaded_model_names()}",
        )

    raw_results = predict_batch(
        customers=[customer.model_dump()],
        model_name=model,
        threshold=threshold,
        explain=explain,
    )
    r = raw_results[0]
    contribs = None
    if r["shap_contributions"] is not None:
        contribs = [SHAPContribution(**c) for c in r["shap_contributions"]]

    return CustomerRecommendation(
        customer_index=0,
        predicted_ite=r["predicted_ite"],
        recommend_offer=r["recommend_offer"],
        segment=r["segment"],
        shap_contributions=contribs,
    )
