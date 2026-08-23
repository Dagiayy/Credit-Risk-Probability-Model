"""Batch inference CLI.

Takes a raw transactions CSV (same schema as training input), builds the
identical customer-level features + preprocessing used at training time via
the champion model's paired feature pipeline, and writes out per-customer
risk scores. This is what you'd run for an overnight batch scoring job; the
API in `src/api/main.py` wraps the same logic for a single customer's
transactions on demand.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import settings
from src.data_processing import load_data
from src.feature_eng_process import build_customer_features
from src.logging_utils import get_logger
from src.model_registry import load_champion_model

logger = get_logger(__name__)


def score_customers(raw_df: pd.DataFrame) -> pd.DataFrame:
    loaded = load_champion_model()

    customer_df = build_customer_features(raw_df)
    X = customer_df.drop(columns=["CustomerId"])
    X_proc = loaded.feature_pipeline.transform(X)
    X_proc = X_proc.reindex(columns=loaded.feature_names, fill_value=0)

    probabilities = loaded.model.predict_proba(X_proc)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    return pd.DataFrame(
        {
            "CustomerId": customer_df["CustomerId"],
            "risk_probability": probabilities,
            "is_high_risk": predictions,
            "model_version": loaded.version,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-score customers for credit risk.")
    parser.add_argument("--input", default=str(settings.raw_data_path), help="Raw transactions CSV")
    parser.add_argument(
        "--output", default=str(Path(settings.reports_dir) / "risk_scores.csv"), help="Output CSV path"
    )
    args = parser.parse_args()

    raw_df = load_data(args.input)
    scores = score_customers(raw_df)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.output, index=False)
    logger.info("Wrote %s risk scores to %s", len(scores), args.output)


if __name__ == "__main__":
    main()
