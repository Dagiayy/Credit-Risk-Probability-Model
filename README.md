# Credit Risk Probability Model

A credit-risk scoring service built for **Bati Bank's** Buy-Now-Pay-Later
(BNPL) pilot with an eCommerce partner. It scores a customer's likelihood of
default from their transaction behaviour — enabling responsible lending
decisions for customers who have no traditional credit history.

There is no historical "did this customer default" label in the source data
(the pilot hasn't lent to anyone yet), so the model is trained against a
**proxy target**: an RFM (Recency/Frequency/Monetary) behavioural clustering
that flags disengaged customers as higher-risk. See
[Proxy target rationale](#proxy-target-rationale) below.

---

## Table of contents

- [Engineering Audit & Upgrades](#engineering-audit--upgrades)
- [Architecture](#architecture)
- [Why customer-grain, not transaction-grain](#why-customer-grain-not-transaction-grain)
- [Proxy target rationale](#proxy-target-rationale)
- [Project structure](#project-structure)
- [Getting started](#getting-started)
- [Running the pipeline](#running-the-pipeline)
- [Running the API](#running-the-api)
- [Explainability](#explainability)
- [Testing](#testing)
- [Docker](#docker)
- [CI/CD](#cicd)
- [Configuration](#configuration)
- [Data quality & validation](#data-quality--validation)
- [Limitations & next steps](#limitations--next-steps)

---

## Engineering Audit & Upgrades

This repository has undergone a comprehensive senior-level engineering audit and upgrade pass to bring the credit risk pipeline from prototype to production-grade quality. For full details on issues identified (P0/P1/P2) and specific technical remediations, see [`CHANGES.md`](CHANGES.md).

### Key Upgrade Highlights
- **Target Leakage Fix:** Eliminated `FraudResult` leakage from request schemas (`CustomerTransactionsRequest`). FastAPI serving runs feature pipelines on raw transaction inputs.
- **Customer-Grain Aggregation:** Standardized feature engineering to aggregate transaction history per `CustomerId` before splitting, ensuring leak-free train/test evaluation.
- **Explainability:** Added SHAP-based (`TreeExplainer` / `KernelExplainer`) local factor attributions via `?explain=true` parameter on prediction endpoints to meet Basel-II regulatory transparency standards.
- **Model Promotion & Registry:** Integrated MLflow Model Registry alias resolution (`champion`) with automated ROC-AUC gain promotion gates to prevent serving performance regressions.
- **Data Validation & Quality:** Integrated `pandera` data contracts (`src/data_schema.py`) and consolidated data quality reporting (`duplicate_report`, `data_quality_report`).
- **Production DevOps & API:** Replaced inline execution with FastAPI `lifespan` handler, readiness/liveness health probes (`/health`, `/ready`), non-root Docker runtime, and GitHub Actions CI workflow.

---

## Architecture

```
Raw transactions CSV
        |
        v
  [schema validation]          data_schema.py  (pandera contract, fail-fast)
        |
        v
  [customer aggregation]       feature_eng_process.build_customer_features
        |                      transaction rows -> ONE ROW PER CUSTOMER
        v
  [RFM + proxy labeling]       proxy_labeling.py  (KMeans on Recency/Frequency/Monetary)
        |
        v
  [train/test split]           customer-grain split (train.py) -- see below
        |
        v
  [preprocessing pipeline]     impute -> one-hot encode -> scale (fit on TRAIN only)
        |
        v
  [model training]             LogisticRegression + RandomForest, MLflow-tracked
        |
        v
  [MLflow Model Registry]      versioned model + `champion` alias
        |
        v
  [FastAPI /api/v1/predict]    raw transactions in -> risk_probability + SHAP factors out
```

The feature pipeline fitted at training time is persisted (`joblib`) and
logged as an MLflow artifact alongside the model it was trained with. The
API and the batch `predict.py` CLI both load *that exact pipeline* — not a
hand-reconstructed one — so serving-time features can never silently drift
from what the model was trained on.

## Why customer-grain, not transaction-grain

The original version of this project aggregated features per customer (e.g.
`total_transaction_amount`) and then **merged those aggregates onto every
transaction row**, training on ~95k transaction rows with a customer-level
label repeated across each customer's transactions. That has two serious
problems for a model whose target is inherently defined per customer:

1. **Leakage** — a customer's aggregate features are computed from *all* of
   that customer's transactions, including ones that would later land in
   the test split. A row's own held-out siblings leak into its features.
2. **Pseudo-replication** — a customer with 500 transactions is worth 500x
   the weight of a customer with 1 transaction, and a random train/test
   split done at the row level lets the same customer appear on both sides.

This project now aggregates to **one row per `CustomerId`** before
splitting, imputing, encoding, or scaling. The model's unit of prediction
(a customer) matches its unit of training, and the customer-grain train/test
split is leak-free by construction.

## Proxy target rationale

Basel-II-style alternative-data credit scoring commonly proxies default risk
with engagement clustering when no repayment history exists yet:
`proxy_labeling.py` clusters customers on standardized Recency/Frequency/Monetary
via KMeans (k configurable, default 3) and labels the least-engaged cluster
(lowest combined frequency + monetary engagement) as `is_high_risk = 1`. The
module logs the silhouette score and per-cluster RFM profile on every run so
the choice is auditable rather than a black box — check the logs (or
`reports/training_report.json`) before trusting a given run's labels.

This proxy should be replaced with real repayment outcomes as soon as the
BNPL pilot generates them; the codebase isolates it to `proxy_labeling.py`
specifically so that swap is a one-module change.

## Project structure

```
credit-risk-model/
├── .github/workflows/ci.yml     # Lint, test, coverage, Docker build check
├── data/                        # Not committed — see data/README.md
├── notebooks/1.0-eda.ipynb      # Exploratory analysis
├── reports/                     # Generated: plots, training_report.json, risk_scores.csv
├── src/
│   ├── config.py                 # Env-driven settings (single source of truth for paths/URIs/knobs)
│   ├── logging_utils.py          # Structured logging setup
│   ├── data_schema.py            # pandera schema — raw data contract
│   ├── data_processing.py        # EDA utilities, data-quality report, outlier flags
│   ├── feature_eng_process.py    # Customer-grain aggregation + preprocessing pipeline + IV diagnostic
│   ├── proxy_labeling.py         # RFM + KMeans proxy target, with audit logging
│   ├── model_registry.py         # Resolves the `champion` model + its paired feature pipeline
│   ├── explainability.py         # SHAP per-prediction top factors
│   ├── train.py                  # End-to-end training + conditional promotion
│   ├── predict.py                # Batch scoring CLI
│   └── api/
│       ├── main.py               # FastAPI app: /health, /ready, /api/v1/predict
│       └── pydantic_models.py    # Request/response schemas (raw transaction fields only)
├── tests/                        # pytest suite (synthetic-data fixtures, no real dataset required)
├── Dockerfile                    # API image (non-root, healthcheck)
├── docker-compose.yml            # mlflow + credit-risk-api services, + `trainer` profile
├── requirements.txt / requirements-dev.txt
├── .env.example
└── CHANGES.md                    # Detailed audit + upgrade log
```

## Getting started

```bash
python -m venv .venv
source .venv/Scripts/activate      # Windows Git Bash; use .venv\Scripts\activate.bat on cmd
pip install -r requirements-dev.txt
cp .env.example .env               # optional — defaults work out of the box
```

Place the raw Xente CSV at `data/raw/data.csv` (see `data/README.md`).

## Running the pipeline

Start an MLflow tracking server (or point `MLFLOW_TRACKING_URI` at one you
already have running):

```bash
docker-compose up mlflow -d
# MLflow UI: http://localhost:5000
```

Train end-to-end (validates data, builds customer features, computes proxy
labels, fits preprocessing on train only, trains + compares models, and
promotes the winner to `champion` if it beats the current one):

```bash
python -m src.train
```

Batch-score a CSV of raw transactions:

```bash
python -m src.predict --input data/raw/data.csv --output reports/risk_scores.csv
```

## Running the API

```bash
docker-compose up --build
# API docs: http://localhost:8000/docs
# Health:   http://localhost:8000/health   (liveness — always 200 once the process is up)
# Ready:    http://localhost:8000/ready    (readiness — 503 until a champion model loads)
```

Or locally without Docker (with an MLflow server running and a champion
model already trained):

```bash
uvicorn src.api.main:app --reload
```

Example request — score one customer from their raw transaction history
(this is the important change from the original API: it takes raw
transaction fields, not ~30 pre-engineered features, and it never accepts
`FraudResult`, which would leak the answer):

```bash
curl -X POST "http://localhost:8000/api/v1/predict?explain=true" \
  -H "Content-Type: application/json" \
  -d '{
    "transactions": [
      {
        "TransactionId": "T1", "BatchId": "B1", "AccountId": "A1",
        "SubscriptionId": "S1", "CustomerId": "C1", "CurrencyCode": "UGX",
        "CountryCode": "256", "ProviderId": "ProviderId_1", "ProductId": "ProductId_1",
        "ProductCategory": "airtime", "ChannelId": "ChannelId_1",
        "Amount": 1000, "Value": 1000,
        "TransactionStartTime": "2019-03-01T12:00:00", "PricingStrategy": 2
      }
    ]
  }'
```

Response:

```json
{
  "customer_id": "C1",
  "risk_probability": 0.31,
  "is_high_risk": false,
  "risk_band": "medium",
  "model_version": "3",
  "top_factors": [
    {"feature": "numeric__recency_days", "shap_value": 0.12, "direction": "increases_risk"}
  ]
}
```

## Explainability

`?explain=true` runs SHAP (`TreeExplainer` for tree models, falling back to
a model-agnostic explainer) against the served model and returns the
top-5 features that pushed the score up or down. This exists because credit
scoring is regulated: Basel II expects lenders to be able to justify a
decision, not just produce a number.

## Testing

```bash
pytest tests/ --cov=src --cov-report=term-missing
```

The suite runs entirely on synthetic fixtures (`tests/conftest.py`) — no
real dataset or running MLflow server required. It covers: schema
validation, customer-grain aggregation correctness (including the leakage
fix — see `test_build_customer_features_transaction_count_matches_source`),
the preprocessing pipeline's train/transform column stability, proxy
labeling, config overrides, and the API (health/ready/predict, including a
regression test asserting `FraudResult` can never be part of the request
schema).

## Docker

```bash
docker-compose up --build              # mlflow + credit-risk-api
docker-compose --profile train run --rm trainer   # one-off training job
```

The API image only ships `src/` (see `.dockerignore`) — it loads its model
from the MLflow registry, so it never needs the raw dataset or notebooks.
The `trainer` service bind-mounts the repo because training needs
`data/` and writes `reports/`.

## CI/CD

`.github/workflows/ci.yml` runs on every push and PR to `main`: flake8,
pytest with coverage, and a Docker build-verification job.

## Configuration

Every path, URI, and tunable lives in `src/config.py` (pydantic-settings),
overridable via environment variables or `.env` — see `.env.example`. Key
knobs: `MLFLOW_TRACKING_URI`, `MODEL_ALIAS` (defaults to `champion`),
`N_RISK_CLUSTERS`, `TEST_SIZE`, `PROMOTION_MIN_ROC_AUC_GAIN`.

Promotion to `champion` is not automatic-by-default: `train.py` only moves
the alias if the new model's ROC-AUC is at least as good as the current
champion's (configurable via `PROMOTION_MIN_ROC_AUC_GAIN`), so a bad
retraining run can't silently regress production.

## Data quality & validation

`src/data_schema.py` defines a pandera contract for the raw dataset (column
presence, types, `FraudResult ∈ {0,1}`, non-negative `Value`, unique
`TransactionId`). `load_data()` validates against it by default and raises
`DataValidationError` with a readable failure summary rather than letting a
malformed CSV silently corrupt downstream features.

## Limitations & next steps

- **Proxy labels, not real outcomes.** Replace `proxy_labeling.py`'s
  clustering with actual repayment/default data as soon as the BNPL pilot
  produces it, and re-validate that WOE/IV-style feature relationships still
  hold against real defaults.
- **Cold-start customers.** A customer with only one transaction gets
  `std_transaction_amount = 0` and no meaningful recency signal — the model
  will be systematically less confident here; consider a minimum-history
  gate or a separate cold-start heuristic before using this for lending
  decisions on brand-new customers.
- **No drift monitoring yet.** `model_registry.get_champion_metrics()` and
  `reports/training_report.json` give you a manual before/after comparison
  each retrain; there's no automated data/prediction drift detection in
  production. A reasonable next step is periodic population-stability-index
  checks on `risk_probability` against the training distribution.
- **Single-region behavioural assumption.** RFM windows and outlier
  thresholds are computed globally; if the customer base spans very
  different transaction cadences (e.g. business vs. consumer accounts),
  segment-specific thresholds would likely improve the proxy label quality.
