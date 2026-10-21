# Uplift Retention Engine

> **Causal ML pipeline that identifies *persuadable* customers and recommends personalised retention offers — maximising net incremental revenue instead of wasting budget on customers who would stay or leave regardless.**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688.svg)](https://fastapi.tiangolo.com/)
[![EconML](https://img.shields.io/badge/EconML-0.15-orange.svg)](https://econml.azurewebsites.net/)
[![MLflow](https://img.shields.io/badge/MLflow-2.13-0194E2.svg)](https://mlflow.org/)

---

## The Problem with Standard Churn Prediction

A binary churn classifier tells you *who might leave*. It does not tell you *whether an intervention will change their behaviour*. Sending a \$10 discount to a customer who would have stayed anyway wastes budget. Sending it to a "lost cause" who will leave regardless is equally wasteful.

**Uplift modelling** estimates the **Individual Treatment Effect (ITE)** — the causal difference in outcome probability caused by sending the offer:

```
ITE(x) = P(retain | offer, x) − P(retain | no offer, x)
```

The four customer segments:

| Segment | ITE | Action |
|---|---|---|
| **Persuadable** | high positive | ✅ Send offer |
| **Sure thing** | near zero, high base rate | ⏩ Skip — saves regardless |
| **Lost cause** | near zero, low base rate | ⏩ Skip — leaves regardless |
| **Sleeping dog** | negative | ⛔ Skip — offer makes things worse |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Data Layer                                                     │
│  ┌──────────────────┐    ┌──────────────────────────────────┐  │
│  │ Criteo Uplift v2 │    │ Synthetic A/B Simulator           │  │
│  │ (13.9M rows, pub)│    │ (heterogeneous CATE, 4 segments) │  │
│  └────────┬─────────┘    └──────────────┬───────────────────┘  │
│           └──────────────┬──────────────┘                       │
│                  UpliftPreprocessor                             │
│          (interaction features + StandardScaler)                │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│  Modelling Layer  (EconML + XGBoost)                            │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │  T-Learner  │  │  X-Learner  │  │  DR-Learner (doubly-    │ │
│  │  (baseline) │  │  (primary)  │  │   robust, cross-fitted) │ │
│  └──────┬──────┘  └──────┬──────┘  └────────────┬────────────┘ │
│         └────────────────┼─────────────────────  │              │
│                    ITE scores                     │              │
└──────────────────────────┬────────────────────────┘             
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│  Evaluation Layer                                               │
│  • Qini coefficient + Qini curve                                │
│  • AUUC (Area Under Uplift Curve)                               │
│  • Uplift by decile (predicted vs observed)                     │
│  • Business ROI simulation (\$/retained user, ROI %)            │
│  • SHAP feature importance + waterfall plots                    │
│  • MLflow experiment tracking                                   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│  Serving Layer  (FastAPI)                                       │
│                                                                 │
│  POST /recommend         → batch ITE scoring (up to 1,000)     │
│  POST /recommend/single  → single customer + SHAP explanation  │
│  GET  /health            → liveness + loaded model list        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quickstart

### 1. Install

```bash
git clone https://github.com/yourname/uplift-retention-engine
cd uplift-retention-engine
pip install -r requirements.txt
```

### 2. Train

```bash
# Synthetic data (fast, no download needed)
python scripts/train.py --source synthetic --learners x_learner dr_learner

# Criteo public benchmark (~300 MB download, 10% sample)
python scripts/train.py --source criteo --sample-frac 0.10
```

Training writes models to `artifacts/models/`, plots to `artifacts/plots/`, and logs all metrics to MLflow.

### 3. Evaluate

```bash
python scripts/evaluate.py --model x_learner
```

Sample output:
```
── Business Simulation ──────────────────────────────
  threshold=0.02 | targeting=48.3% | profit=$184,200 | ROI=212% | vs_naive=$+62,400
  threshold=0.05 | targeting=29.1% | profit=$201,800 | ROI=348% | vs_naive=$+80,000
  threshold=0.10 | targeting=14.7% | profit=$178,600 | ROI=512% | vs_naive=$+56,800

── Top-5 Features by Mean |SHAP| ────────────────────
  1. f0           0.084213
  2. f0_x_f1      0.061042
  3. f4_sq        0.053119
  4. f2           0.041887
  5. f5           0.038204
```

### 4. View experiment tracking

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# → open http://localhost:5000
```

### 5. Serve the API

```bash
# Local dev
uvicorn src.api.main:app --reload --port 8000

# Docker (API + MLflow UI)
docker compose up --build
```

### 6. Call the API

```bash
# Single customer with SHAP explanation
curl -X POST http://localhost:8000/recommend/single \
  -H "Content-Type: application/json" \
  -d '{
    "f0": 1.2, "f1": -0.5, "f2": 0.8, "f3": 2.1,
    "f4": 0.0, "f5": -1.1, "f6": 0.3, "f7": 0.9,
    "f8": -0.2, "f9": 1.5, "f10": 0.1, "f11": -0.7
  }'
```

Response:
```json
{
  "customer_index": 0,
  "predicted_ite": 0.082341,
  "recommend_offer": true,
  "segment": "persuadable",
  "shap_contributions": [
    {"feature": "f0", "shap_value": 0.041200},
    {"feature": "f0_x_f1", "shap_value": 0.028900},
    {"feature": "f4_sq", "shap_value": -0.012300},
    {"feature": "f2", "shap_value": 0.009800},
    {"feature": "f5", "shap_value": -0.007400}
  ]
}
```

---

## Model Comparison

| Learner | Estimator Type | Qini (test) | AUUC | Notes |
|---------|----------------|-------------|------|-------|
| T-Learner | Two separate μ₀, μ₁ models | ~0.18 | ~0.04 | Baseline; biased under overlap violations |
| **X-Learner** | Cross-fitted pseudo-outcomes | **~0.24** | **~0.06** | **Primary model**; handles imbalanced splits well |
| DR-Learner | Doubly-robust, cv=3 | ~0.23 | ~0.06 | Consistent if either outcome or propensity is correct |

*Scores on 150k synthetic dataset. Criteo scores will differ.*

---

## Evaluation Metrics

### Why not AUC / F1?

Standard classification metrics measure *predictive accuracy*, not *causal effect ordering*. A model that perfectly predicts churn can have Qini = 0 if it cannot distinguish persuadables from sure-things.

### Qini Coefficient

Measures how well the model orders customers by their treatment effect. Defined as the area between the model's Qini curve and the random baseline, normalised:

```
Qini = ∫[model curve] - ∫[random baseline]
```

### Uplift by Decile

Splits the test population into 10 buckets ranked by predicted ITE. A well-calibrated model shows:
- Top deciles: high predicted ITE AND high observed uplift
- Bottom deciles: low/negative predicted ITE AND low/negative observed uplift

### Business ROI Simulation

Translates ITE scores into dollar impact:

```
revenue  = incremental_retentions × $120/user
cost     = n_targeted × $10/offer
profit   = revenue − cost
ROI      = profit / cost × 100%
```

Compared against the naive strategy of sending offers to the entire at-risk population.

---

## Project Structure

```
uplift-retention-engine/
├── src/
│   ├── data/
│   │   ├── loader.py          # Criteo download + synthetic A/B generator
│   │   └── preprocessor.py   # Interaction features + StandardScaler pipeline
│   ├── models/
│   │   ├── uplift.py          # T-Learner, X-Learner, DR-Learner (EconML + XGBoost)
│   │   └── evaluator.py      # Qini, AUUC, decile analysis, ROI simulation
│   ├── explain/
│   │   └── shap_explainer.py  # SHAP TreeExplainer, summary/bar/waterfall plots
│   └── api/
│       ├── main.py            # FastAPI app (lifespan, routes)
│       ├── predictor.py       # Model cache + inference logic
│       └── schemas.py         # Pydantic v2 request/response models
├── scripts/
│   ├── train.py               # End-to-end training + MLflow logging
│   └── evaluate.py            # Standalone evaluation report
├── tests/
│   ├── test_uplift.py         # Unit + integration tests (data, models, evaluator)
│   └── test_api.py            # Async FastAPI integration tests (httpx)
├── artifacts/
│   ├── models/                # Serialised learners + preprocessor (joblib)
│   └── plots/                 # Qini curves, decile charts, SHAP plots
├── Dockerfile
├── docker-compose.yml         # API + MLflow UI services
├── Makefile
└── requirements.txt
```

---

## Running Tests

```bash
pytest tests/ -v --asyncio-mode=auto
```

```
tests/test_uplift.py::TestDataLoader::test_generate_shape        PASSED
tests/test_uplift.py::TestDataLoader::test_treatment_binary      PASSED
tests/test_uplift.py::TestDataLoader::test_no_data_leakage       PASSED
tests/test_uplift.py::TestPreprocessor::test_standardisation     PASSED
tests/test_uplift.py::TestTLearner::test_fit_predict_shape       PASSED
tests/test_uplift.py::TestXLearner::test_fit_predict             PASSED
tests/test_uplift.py::TestEvaluator::test_qini_is_float          PASSED
tests/test_uplift.py::TestEvaluator::test_decile_returns_10_rows PASSED
tests/test_api.py::test_health                                   PASSED
tests/test_api.py::test_recommend_batch_basic                    PASSED
tests/test_api.py::test_recommend_single                         PASSED
tests/test_api.py::test_batch_size_limit                         PASSED
```

---

## Dataset

**Option A — Synthetic (default)**  
Generated locally with controlled heterogeneous treatment effects across four customer segments. Requires no download. Use for rapid iteration.

**Option B — Criteo Uplift v2.1 (public benchmark)**  
13.9M rows, 12 anonymised features, binary treatment/conversion labels from a real online advertising A/B test.  
Download: `python scripts/train.py --source criteo`  
Paper: [Diemert et al., 2018](https://dl.acm.org/doi/10.1145/3219819.3219943)

---

## Tech Stack

| Component | Library |
|-----------|---------|
| Causal ML | `econml` 0.15 |
| Base learner | `xgboost` 2.0 |
| Interpretability | `shap` 0.45 |
| API serving | `fastapi` + `uvicorn` |
| Experiment tracking | `mlflow` 2.13 |
| Data | `pandas`, `numpy` |
| Testing | `pytest`, `httpx` |
| Containerisation | Docker + Compose |
