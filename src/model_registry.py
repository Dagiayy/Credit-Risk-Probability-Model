"""Single place that knows how to load "the model that should serve traffic".

The original code hard-coded `MODEL_VERSION = 2` in both `predict.py` and
`api/main.py` — promoting a new model meant editing and redeploying source
code. This module resolves the serving model via an MLflow Registry
**alias** (`champion`, see `config.settings.model_alias`) instead, which
`train.py` moves after a new model proves itself. It also loads the fitted
preprocessing pipeline that was logged alongside the model, so callers never
have to reconstruct feature engineering by hand.
"""
from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from src.config import settings
from src.logging_utils import get_logger

logger = get_logger(__name__)

PIPELINE_ARTIFACT_SUBPATH = "feature_pipeline/feature_pipeline.joblib"


class ModelNotFoundError(RuntimeError):
    """Raised when neither the alias nor the fallback version can be resolved."""


@dataclass
class LoadedModel:
    model: Any
    feature_pipeline: Any
    feature_names: list[str]
    version: str
    run_id: str


def _resolve_model_uri(client: MlflowClient) -> tuple[str, str]:
    """Return (model_uri, version) preferring the registry alias."""
    name = settings.model_name
    try:
        mv = client.get_model_version_by_alias(name, settings.model_alias)
        return f"models:/{name}/{mv.version}", mv.version
    except Exception as exc:  # alias not set yet, or registry unreachable in that way
        logger.warning("Could not resolve alias '%s' for model '%s': %s", settings.model_alias, name, exc)

    if settings.fallback_model_version is not None:
        version = str(settings.fallback_model_version)
        return f"models:/{name}/{version}", version

    raise ModelNotFoundError(
        f"No model version found for alias '{settings.model_alias}' on '{name}', "
        "and no MODEL_FALLBACK_VERSION configured."
    )


def load_champion_model() -> LoadedModel:
    """Load the current champion model + its paired feature pipeline from MLflow."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = MlflowClient()

    model_uri, version = _resolve_model_uri(client)
    logger.info("Loading model from %s", model_uri)
    model = mlflow.sklearn.load_model(model_uri)

    mv = client.get_model_version(settings.model_name, version)
    run_id = mv.run_id

    with tempfile.TemporaryDirectory() as tmp_dir:
        local_path = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path=PIPELINE_ARTIFACT_SUBPATH, dst_path=tmp_dir
        )
        feature_pipeline = joblib.load(local_path)

    feature_names_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="feature_pipeline/feature_names.txt", dst_path=tempfile.mkdtemp()
    )
    feature_names = Path(feature_names_path).read_text().splitlines()

    logger.info(
        "Loaded champion model version=%s run_id=%s (%s features)", version, run_id, len(feature_names)
    )
    return LoadedModel(
        model=model,
        feature_pipeline=feature_pipeline,
        feature_names=feature_names,
        version=version,
        run_id=run_id,
    )


def promote_to_champion(run_id: str, version: str) -> None:
    """Point the `champion` alias at this model version."""
    client = MlflowClient()
    client.set_registered_model_alias(settings.model_name, settings.model_alias, version)
    client.set_model_version_tag(settings.model_name, version, "promoted_by", "automated-training-pipeline")
    logger.info("Promoted model version %s to alias '%s'", version, settings.model_alias)


def get_champion_metrics() -> dict[str, float] | None:
    """Fetch the recorded metrics of the current champion, if one exists."""
    client = MlflowClient()
    try:
        mv = client.get_model_version_by_alias(settings.model_name, settings.model_alias)
    except Exception:
        return None
    run = client.get_run(mv.run_id)
    return dict(run.data.metrics)
