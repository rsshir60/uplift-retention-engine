"""
src/api/schemas.py

Pydantic v2 request/response models for the FastAPI serving layer.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator


# ─────────────────────────────────────────────────────────────────────────────
# Request
# ─────────────────────────────────────────────────────────────────────────────

class CustomerFeatures(BaseModel):
    """Raw feature vector for a single customer.  Maps to f0 … f11."""

    f0: float = Field(..., description="Anonymised feature 0 (e.g. recency score)")
    f1: float = Field(..., description="Anonymised feature 1 (e.g. frequency)")
    f2: float = Field(..., description="Anonymised feature 2 (e.g. monetary value)")
    f3: float = Field(..., description="Anonymised feature 3 (e.g. tenure days)")
    f4: float = Field(..., description="Anonymised feature 4 (e.g. support tickets)")
    f5: float = Field(..., description="Anonymised feature 5")
    f6: float = Field(..., description="Anonymised feature 6")
    f7: float = Field(..., description="Anonymised feature 7")
    f8: float = Field(..., description="Anonymised feature 8")
    f9: float = Field(..., description="Anonymised feature 9")
    f10: float = Field(..., description="Anonymised feature 10")
    f11: float = Field(..., description="Anonymised feature 11")

    model_config = {"json_schema_extra": {
        "example": {
            "f0": 1.2, "f1": -0.5, "f2": 0.8, "f3": 2.1,
            "f4": 0.0, "f5": -1.1, "f6": 0.3, "f7": 0.9,
            "f8": -0.2, "f9": 1.5, "f10": 0.1, "f11": -0.7,
        }
    }}


class BatchRecommendRequest(BaseModel):
    """Batch prediction — up to 1,000 customers per call."""

    customers: list[CustomerFeatures] = Field(
        ..., min_length=1, max_length=1000
    )
    model: str = Field(
        default="x_learner",
        description="Learner to use: 't_learner' | 'x_learner' | 'dr_learner'",
    )
    threshold: float = Field(
        default=0.05,
        ge=-1.0,
        le=1.0,
        description="ITE threshold above which the offer is recommended",
    )
    explain: bool = Field(
        default=False,
        description="If True, return per-customer SHAP feature contributions",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Response
# ─────────────────────────────────────────────────────────────────────────────

class SHAPContribution(BaseModel):
    feature: str
    shap_value: float


class CustomerRecommendation(BaseModel):
    customer_index: int
    predicted_ite: float = Field(..., description="Individual Treatment Effect score")
    recommend_offer: bool = Field(
        ..., description="True when predicted_ite >= threshold"
    )
    segment: str = Field(
        ...,
        description=(
            "persuadable | sure_thing | lost_cause | sleeping_dog | neutral"
        ),
    )
    shap_contributions: list[SHAPContribution] | None = None


class BatchRecommendResponse(BaseModel):
    model_used: str
    threshold: float
    n_customers: int
    n_recommended: int
    recommendations: list[CustomerRecommendation]


class HealthResponse(BaseModel):
    status: str
    loaded_models: list[str]
    version: str
