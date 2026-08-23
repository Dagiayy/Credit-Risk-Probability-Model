"""EDA utilities and raw-data ingestion for the Xente transactions dataset.

`load_data` is the single point where a CSV on disk becomes a validated
dataframe: it enforces the schema in `data_schema.py` so downstream stages
(feature engineering, labeling, training) can assume clean, typed input
instead of re-deriving assumptions about the data ad hoc.
"""
from __future__ import annotations

import os
from typing import Dict

import matplotlib

matplotlib.use("Agg")  # headless-safe: never try to open a GUI window
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from src.config import settings  # noqa: E402
from src.data_schema import validate_raw_transactions  # noqa: E402
from src.logging_utils import get_logger  # noqa: E402

logger = get_logger(__name__)


def load_data(path: str | None = None, validate: bool = True) -> pd.DataFrame:
    """Load the raw transactions CSV, optionally validating it against the schema."""
    path = path or str(settings.raw_data_path)
    df = pd.read_csv(path)
    logger.info("Loaded raw data: %s rows, %s columns from %s", df.shape[0], df.shape[1], path)
    if validate:
        df = validate_raw_transactions(df)
        logger.info("Raw data passed schema validation")
    return df


def overview(df: pd.DataFrame) -> None:
    logger.info("Data overview — shape=%s", df.shape)
    logger.info("Dtypes:\n%s", df.dtypes)
    logger.info("Head:\n%s", df.head())


def summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    return df.describe(include="all")


def missing_values(df: pd.DataFrame) -> pd.Series:
    missing = df.isnull().sum()
    return missing[missing > 0]


def duplicate_report(df: pd.DataFrame, key_col: str = "TransactionId") -> Dict[str, int]:
    """Report exact row duplicates and duplicate primary keys."""
    report = {
        "exact_duplicate_rows": int(df.duplicated().sum()),
    }
    if key_col in df.columns:
        report[f"duplicate_{key_col}"] = int(df[key_col].duplicated().sum())
    return report


def detect_outliers(df: pd.DataFrame) -> Dict[str, int]:
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    outlier_counts: Dict[str, int] = {}
    for col in numeric_cols:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        outliers = df[(df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)]
        outlier_counts[col] = len(outliers)
    return outlier_counts


def add_outlier_flags(df: pd.DataFrame, numeric_cols: list[str] | None = None) -> pd.DataFrame:
    """Add boolean outlier-flag columns (IQR method) and drop constant columns."""
    df = df.copy()
    if "CountryCode" in df.columns and df["CountryCode"].nunique(dropna=False) <= 1:
        df = df.drop(columns=["CountryCode"])

    numeric_cols = numeric_cols or settings.numeric_cols
    for col in numeric_cols:
        if col not in df.columns:
            continue
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        df[f"{col}_outlier_flag"] = ((df[col] < lower) | (df[col] > upper)).astype(int)

    return df


def data_quality_report(df: pd.DataFrame) -> Dict[str, object]:
    """A single consolidated report used both in the notebook and by CI smoke checks."""
    return {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "n_customers": int(df["CustomerId"].nunique()) if "CustomerId" in df else None,
        "missing_values": missing_values(df).to_dict(),
        "duplicates": duplicate_report(df),
        "outliers_iqr": detect_outliers(df),
    }


def plot_numerical_distributions(df: pd.DataFrame, output_folder: str | None = None) -> None:
    output_folder = output_folder or str(settings.plots_dir)
    os.makedirs(output_folder, exist_ok=True)
    for col in df.select_dtypes(include=["int64", "float64"]).columns:
        plt.figure(figsize=(6, 4))
        df[col].hist(bins=50)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f"{col}_hist.png"))
        plt.close()


def plot_categorical_distributions(df: pd.DataFrame, output_folder: str | None = None) -> None:
    output_folder = output_folder or str(settings.plots_dir)
    os.makedirs(output_folder, exist_ok=True)
    for col in df.select_dtypes(include=["object", "category"]).columns:
        plt.figure(figsize=(6, 4))
        df[col].value_counts().nlargest(10).plot(kind="bar")
        plt.title(f"Top 10 Categories in {col}")
        plt.ylabel("Frequency")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f"{col}_bar.png"))
        plt.close()


def correlation_matrix(df: pd.DataFrame, output_file: str | None = None) -> pd.DataFrame:
    output_file = output_file or os.path.join(str(settings.plots_dir), "correlation_matrix.png")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    numeric_df = df.select_dtypes(include=["int64", "float64"])
    corr = numeric_df.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    return corr


def plot_boxplots(df: pd.DataFrame, output_folder: str | None = None) -> None:
    output_folder = output_folder or os.path.join(str(settings.plots_dir), "boxplots")
    os.makedirs(output_folder, exist_ok=True)
    for col in df.select_dtypes(include=["int64", "float64"]).columns:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=df[col])
        plt.title(f"Boxplot of {col}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f"{col}_boxplot.png"))
        plt.close()


def save_processed_data(df: pd.DataFrame, path: str | None = None) -> None:
    path = path or str(settings.cleaned_data_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved cleaned data to: %s (%s rows)", path, len(df))
