"""Train, evaluate, and (conditionally) promote a credit-risk model.

Pipeline: raw transactions -> customer-level features -> proxy labels ->
customer-level train/test split -> fit preprocessing on TRAIN ONLY -> train
candidate models -> log everything to MLflow -> promote the best candidate
to the `champion` alias only if it beats the current champion.

Run with `python -m src.train`.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.config import settings
from src.data_processing import load_data
from src.feature_eng_process import build_customer_features, build_preprocessing_pipeline
from src.logging_utils import get_logger
from src.model_registry import get_champion_metrics, promote_to_champion
from src.proxy_labeling import calculate_rfm, label_high_risk_customers

logger = get_logger(__name__)


def build_labeled_customer_dataset(raw_df: pd.DataFrame) -> pd.DataFrame:
    """raw transactions -> one row per customer, with engineered features + proxy label."""
    customer_features = build_customer_features(raw_df)
    rfm = calculate_rfm(raw_df)
    risk_labels = label_high_risk_customers(rfm)
    labeled = customer_features.merge(
        risk_labels[["CustomerId", "is_high_risk"]], on="CustomerId", how="inner"
    )
    return labeled


def evaluate_model(model, X_test, y_test) -> dict[str, float]:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


def train_and_log_model(
    name: str,
    model,
    params: dict[str, Any],
    X_train,
    y_train,
    X_test,
    y_test,
    feature_pipeline,
    feature_names: list[str],
) -> tuple[Any, dict[str, float], str]:
    with mlflow.start_run(run_name=name) as run:
        model.set_params(**params)
        model.fit(X_train, y_train)

        metrics = evaluate_model(model, X_test, y_test)

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.set_tag("model_family", name)

        signature = infer_signature(X_test, model.predict(X_test))
        mlflow.sklearn.log_model(
            model,
            artifact_path=f"{name.lower()}_model",
            input_example=X_test.iloc[:1],
            signature=signature,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline_path = Path(tmp_dir) / "feature_pipeline.joblib"
            joblib.dump(feature_pipeline, pipeline_path)
            mlflow.log_artifact(str(pipeline_path), artifact_path="feature_pipeline")

            names_path = Path(tmp_dir) / "feature_names.txt"
            names_path.write_text("\n".join(feature_names))
            mlflow.log_artifact(str(names_path), artifact_path="feature_pipeline")

        logger.info("Run %s (%s): %s", run.info.run_id, name, metrics)
        return model, metrics, run.info.run_id


def main() -> None:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    logger.info("Loading and validating raw data...")
    raw_df = load_data()

    logger.info("Building customer-level feature table and proxy labels...")
    labeled_df = build_labeled_customer_dataset(raw_df)
    os.makedirs(os.path.dirname(settings.labeled_data_path), exist_ok=True)
    labeled_df.to_csv(settings.labeled_data_path, index=False)

    y = labeled_df["is_high_risk"]
    X = labeled_df.drop(columns=["is_high_risk", "Cluster"], errors="ignore")

    logger.info("Splitting at customer grain (one row per customer -> no leakage across split)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=settings.test_size, random_state=settings.random_state, stratify=y
    )

    logger.info("Fitting preprocessing pipeline on TRAIN only...")
    feature_pipeline = build_preprocessing_pipeline()
    X_train_proc = feature_pipeline.fit_transform(X_train.drop(columns=["CustomerId"]))
    X_test_proc = feature_pipeline.transform(X_test.drop(columns=["CustomerId"]))
    feature_names = list(X_train_proc.columns)

    os.makedirs(os.path.dirname(settings.feature_pipeline_path), exist_ok=True)
    joblib.dump(feature_pipeline, settings.feature_pipeline_path)
    Path(settings.feature_names_path).write_text("\n".join(feature_names))

    logger.info("Baseline (majority-class DummyClassifier) for sanity comparison...")
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train_proc, y_train)
    baseline_metrics = {
        "accuracy": accuracy_score(y_test, baseline.predict(X_test_proc)),
        "roc_auc": 0.5,
    }
    logger.info("Baseline metrics: %s", baseline_metrics)

    candidates = {
        "LogisticRegression": (
            LogisticRegression(),
            {"C": 1.0, "solver": "lbfgs", "max_iter": 1000, "class_weight": "balanced"},
        ),
        "RandomForest": (
            RandomForestClassifier(),
            {
                "n_estimators": 300,
                "max_depth": 8,
                "min_samples_leaf": 5,
                "class_weight": "balanced",
                "random_state": settings.random_state,
            },
        ),
    }

    results: dict[str, dict] = {}
    for name, (model, params) in candidates.items():
        trained_model, metrics, run_id = train_and_log_model(
            name, model, params, X_train_proc, y_train, X_test_proc, y_test,
            feature_pipeline, feature_names,
        )
        results[name] = {"model": trained_model, "metrics": metrics, "run_id": run_id}

    best_name = max(results, key=lambda n: results[n]["metrics"]["roc_auc"])
    best = results[best_name]
    logger.info("Best candidate: %s (roc_auc=%.4f)", best_name, best["metrics"]["roc_auc"])

    model_uri = f"runs:/{best['run_id']}/{best_name.lower()}_model"
    registered = mlflow.register_model(model_uri=model_uri, name=settings.model_name)
    version = registered.version

    client = MlflowClient()
    client.set_model_version_tag(settings.model_name, version, "model_family", best_name)
    for metric_name, value in best["metrics"].items():
        client.set_model_version_tag(settings.model_name, version, f"metric_{metric_name}", f"{value:.4f}")

    champion_metrics = get_champion_metrics()
    required_roc_auc = (
        (champion_metrics or {}).get("roc_auc", 0.0) + settings.promotion_min_roc_auc_gain
    )
    should_promote = champion_metrics is None or best["metrics"]["roc_auc"] >= required_roc_auc

    if should_promote:
        promote_to_champion(best["run_id"], version)
        logger.info("Promoted %s v%s to champion.", best_name, version)
    else:
        logger.info(
            "New model (roc_auc=%.4f) did not beat current champion (roc_auc=%.4f); not promoted.",
            best["metrics"]["roc_auc"],
            champion_metrics.get("roc_auc", 0.0),
        )

    report = {
        "baseline": baseline_metrics,
        **{name: r["metrics"] for name, r in results.items()},
        "selected_model": best_name,
        "selected_version": version,
        "promoted": should_promote,
    }
    os.makedirs(settings.reports_dir, exist_ok=True)
    report_path = Path(settings.reports_dir) / "training_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    logger.info("Training report written to %s", report_path)


if __name__ == "__main__":
    main()
