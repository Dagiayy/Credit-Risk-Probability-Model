"""Customer-level feature engineering.

Design note — the most important fix in this module vs. the original
implementation:

The original pipeline engineered features at *transaction* grain: it merged
each customer's aggregate stats (total spend, avg spend, ...) onto every one
of that customer's transaction rows, then trained a model on those rows
directly. That has two serious, non-obvious problems for a model whose
target ("is this *customer* high risk?") is inherently defined at customer
grain:

1. **Leakage** — a customer's aggregate features (e.g. `total_transaction_amount`)
   are computed from *all* of that customer's transactions, including ones
   that would later land in the test split. Every training row for that
   customer already "knows" about its own held-out siblings.
2. **Pseudo-replication** — every transaction of a customer carries the same
   label, so a customer with 500 transactions is (silently) worth 500x the
   weight of a customer with 1 transaction in training and in every metric.
   Train/test splits done at the row level also let the same customer appear
   on both sides of the split.

The fix: aggregate to **one row per `CustomerId`** first, and only then
impute/encode/scale. This makes the model's unit of prediction match its
unit of training, and makes a customer-level train/test split leak-free by
construction (see `train.py`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import settings
from src.data_processing import add_outlier_flags
from src.logging_utils import get_logger

logger = get_logger(__name__)


def _mode_or_nan(series: pd.Series):
    m = series.mode()
    return m.iloc[0] if not m.empty else None


def build_customer_features(
    df: pd.DataFrame,
    numeric_cols: list[str] | None = None,
    categorical_cols: list[str] | None = None,
    group_col: str = "CustomerId",
) -> pd.DataFrame:
    """Collapse a transaction-level dataframe into one row per customer.

    Returns a dataframe with `group_col` plus behavioural/RFM-style features.
    This is the grain the model is actually trained and served at.
    """
    numeric_cols = numeric_cols or settings.numeric_cols
    categorical_cols = categorical_cols or settings.categorical_cols

    df = df.copy()
    df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"], errors="coerce")
    df = add_outlier_flags(df, numeric_cols=numeric_cols)

    snapshot_date = df["TransactionStartTime"].max() + pd.Timedelta(days=1)

    numeric_agg = (
        df.groupby(group_col)["Amount"]
        .agg(
            total_transaction_amount="sum",
            avg_transaction_amount="mean",
            std_transaction_amount="std",
            transaction_count="count",
            min_transaction_amount="min",
            max_transaction_amount="max",
        )
        .reset_index()
    )
    numeric_agg["std_transaction_amount"] = numeric_agg["std_transaction_amount"].fillna(0)

    recency_tenure = (
        df.groupby(group_col)["TransactionStartTime"]
        .agg(first_transaction="min", last_transaction="max")
        .reset_index()
    )
    recency_tenure["recency_days"] = (
        snapshot_date - recency_tenure["last_transaction"]
    ).dt.days
    recency_tenure["tenure_days"] = (
        recency_tenure["last_transaction"] - recency_tenure["first_transaction"]
    ).dt.days
    recency_tenure = recency_tenure.drop(columns=["first_transaction", "last_transaction"])

    outlier_rate_cols = [f"{c}_outlier_flag" for c in numeric_cols if f"{c}_outlier_flag" in df.columns]
    outlier_rates = (
        df.groupby(group_col)[outlier_rate_cols].mean().reset_index() if outlier_rate_cols else None
    )

    categorical_modes = (
        df.groupby(group_col)[categorical_cols]
        .agg(lambda s: _mode_or_nan(s))
        .reset_index()
    )

    behavioural_hour_mode = (
        df.groupby(group_col)["TransactionStartTime"]
        .agg(lambda s: int(s.dt.hour.mode().iloc[0]) if s.dt.hour.mode().size else None)
        .reset_index()
        .rename(columns={"TransactionStartTime": "most_common_transaction_hour"})
    )

    customer_df = numeric_agg.merge(recency_tenure, on=group_col, how="left")
    if outlier_rates is not None:
        customer_df = customer_df.merge(outlier_rates, on=group_col, how="left")
    customer_df = customer_df.merge(categorical_modes, on=group_col, how="left")
    customer_df = customer_df.merge(behavioural_hour_mode, on=group_col, how="left")

    logger.info(
        "Built customer-level feature table: %s customers from %s transactions",
        customer_df.shape[0],
        df.shape[0],
    )
    return customer_df


def build_preprocessing_pipeline(
    numeric_cols: list[str] | None = None,
    categorical_cols: list[str] | None = None,
) -> Pipeline:
    """Impute + encode + scale a customer-level feature table (fit on TRAIN only)."""
    numeric_cols = numeric_cols or [
        "total_transaction_amount",
        "avg_transaction_amount",
        "std_transaction_amount",
        "transaction_count",
        "min_transaction_amount",
        "max_transaction_amount",
        "recency_days",
        "tenure_days",
        "most_common_transaction_hour",
        "Amount_outlier_flag",
        "Value_outlier_flag",
        "PricingStrategy_outlier_flag",
    ]
    categorical_cols = categorical_cols or settings.categorical_cols

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_cols),
            ("categorical", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )
    pipeline = Pipeline(steps=[("preprocessor", preprocessor)])
    # Keep column names through the pipeline (sklearn >= 1.2) so downstream
    # consumers (SHAP, the API response, saved feature-name manifests) don't
    # have to reverse-engineer positional array indices.
    pipeline.set_output(transform="pandas")
    return pipeline


def get_output_feature_names(pipeline: Pipeline) -> list[str]:
    """Recover human-readable column names for the ColumnTransformer output."""
    preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]
    return list(preprocessor.get_feature_names_out())


# --------- WOE/IV diagnostic (feature-selection report, not used for encoding) ------------

def calculate_woe_iv(df: pd.DataFrame, categorical_cols: list[str], target_col: str) -> dict[str, float]:
    """Information Value per categorical column — a standard credit-scoring
    diagnostic for feature *selection*, kept separate from encoding (which
    uses one-hot encoding inside the sklearn pipeline above for simplicity
    and to avoid an extra heavyweight dependency for a modest feature set).
    """
    iv_dict: dict[str, float] = {}
    eps = 1e-4
    n_good = max(1, len(df[df[target_col] == 0]))
    n_bad = max(1, len(df[df[target_col] == 1]))

    for col in categorical_cols:
        iv_total = 0.0
        for val in df[col].dropna().unique():
            subset = df[df[col] == val]
            good = len(subset[subset[target_col] == 0])
            bad = len(subset[subset[target_col] == 1])
            dist_good = good / n_good
            dist_bad = bad / n_bad
            woe = float(np.log((dist_good + eps) / (dist_bad + eps)))
            iv_total += (dist_good - dist_bad) * woe
        iv_dict[col] = iv_total
        logger.info("Feature: %s, IV: %.4f", col, iv_total)

    return iv_dict


# --------- Example usage -----------

if __name__ == "__main__":
    from src.config import settings as _settings

    df = pd.read_csv(_settings.cleaned_data_path)
    customer_df = build_customer_features(df)
    customer_df.to_csv(_settings.customer_features_path, index=False)
    logger.info("Saved customer-level feature table to %s", _settings.customer_features_path)
