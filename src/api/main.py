"""FastAPI credit-risk scoring service.

Fixes vs. the original API:
- No longer accepts `FraudResult` (or any pre-engineered feature) from
  callers — takes raw transaction records and runs the real feature
  pipeline, so serving-time features can never drift from training-time
  features and the model can never be handed its own leaked target.
- Model is resolved via the MLflow registry `champion` alias, not a
  hard-coded version number, and loading happens once at startup behind a
  health check instead of unguarded module-level code.
- Structured logging, /health and /ready probes, versioned routes, and
  explicit error responses instead of unhandled 500s.
"""
from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from src.api.pydantic_models import (
    CustomerTransactionsRequest,
    HealthStatus,
    RiskPrediction,
)
from src.config import settings
from src.explainability import RiskExplainer
from src.feature_eng_process import build_customer_features
from src.logging_utils import configure_logging, get_logger
from src.model_registry import LoadedModel, load_champion_model

configure_logging(settings.log_level, settings.log_json)
logger = get_logger(__name__)


def _risk_band(probability: float) -> str:
    if probability < 0.33:
        return "low"
    if probability < 0.66:
        return "medium"
    return "high"


async def _load_model_in_background(app: FastAPI) -> None:
    try:
        # mlflow's HTTP client retries with backoff on connection failure,
        # which can take well over a minute. Running this in a thread (off
        # the event loop) instead of awaiting it directly in `lifespan`
        # matters: FastAPI/uvicorn won't accept *any* connections — including
        # `/health` — until `lifespan` reaches its `yield`. Blocking there
        # would mean an unreachable MLflow server makes the whole API
        # unreachable too, defeating the point of a liveness probe.
        loaded_model = await asyncio.to_thread(load_champion_model)
        app.state.loaded_model = loaded_model
        app.state.explainer = RiskExplainer(loaded_model.model, loaded_model.feature_names)
        logger.info("Champion model version=%s ready", loaded_model.version)
    except Exception as exc:
        logger.error("Failed to load champion model: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.loaded_model = None
    app.state.explainer = None
    load_task = asyncio.create_task(_load_model_in_background(app))
    yield
    load_task.cancel()


app = FastAPI(title=settings.api_title, version=settings.api_version, lifespan=lifespan)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info("%s %s -> %s (%.1fms)", request.method, request.url.path, response.status_code, duration_ms)
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.get("/health", response_model=HealthStatus, tags=["ops"])
def health() -> HealthStatus:
    """Liveness probe — always 200 if the process is up."""
    loaded: LoadedModel | None = app.state.loaded_model
    return HealthStatus(
        status="ok", model_loaded=loaded is not None, model_version=loaded.version if loaded else None
    )


@app.get("/ready", response_model=HealthStatus, tags=["ops"])
def ready() -> HealthStatus:
    """Readiness probe — 503 until a champion model is loaded."""
    loaded: LoadedModel | None = app.state.loaded_model
    if loaded is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return HealthStatus(status="ready", model_loaded=True, model_version=loaded.version)


@app.post("/api/v1/predict", response_model=RiskPrediction, tags=["scoring"])
def predict_risk(request: CustomerTransactionsRequest, explain: bool = False) -> RiskPrediction:
    loaded: LoadedModel | None = app.state.loaded_model
    if loaded is None:
        raise HTTPException(status_code=503, detail="Model not loaded; try again shortly")

    raw_df = pd.DataFrame([t.model_dump() for t in request.transactions])
    customer_id = raw_df["CustomerId"].iloc[0]

    try:
        customer_df = build_customer_features(raw_df)
        X = customer_df.drop(columns=["CustomerId"])
        X_proc = loaded.feature_pipeline.transform(X)
        X_proc = X_proc.reindex(columns=loaded.feature_names, fill_value=0)
    except Exception as exc:
        logger.warning("Feature engineering failed for customer %s: %s", customer_id, exc)
        raise HTTPException(status_code=422, detail=f"Could not featurize input: {exc}") from exc

    probability = float(loaded.model.predict_proba(X_proc)[0][1])

    top_factors = None
    if explain and app.state.explainer is not None:
        try:
            top_factors = app.state.explainer.top_factors(X_proc)
        except Exception as exc:
            logger.warning("SHAP explanation failed for customer %s: %s", customer_id, exc)

    return RiskPrediction(
        customer_id=str(customer_id),
        risk_probability=probability,
        is_high_risk=probability >= 0.5,
        risk_band=_risk_band(probability),
        model_version=loaded.version,
        top_factors=top_factors,
    )
