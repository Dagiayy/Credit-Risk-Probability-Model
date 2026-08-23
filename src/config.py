"""Centralized, environment-driven configuration.

Every path, URI, and tunable constant that used to be hard-coded across
data_processing.py / feature_eng_process.py / train.py / predict.py / api/main.py
lives here instead. Values can be overridden via environment variables or a
`.env` file (see `.env.example`), which is what makes the same code work
unchanged across a laptop, CI, and a container.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Also load .env into the real process environment (not just this Settings
# model) so libraries that read os.environ directly — notably mlflow's HTTP
# client tuning vars like MLFLOW_HTTP_REQUEST_MAX_RETRIES — pick it up too.
load_dotenv(PROJECT_ROOT / ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Data paths -------------------------------------------------
    raw_data_path: Path = Field(default=PROJECT_ROOT / "data" / "raw" / "data.csv")
    cleaned_data_path: Path = Field(default=PROJECT_ROOT / "data" / "processed" / "data_cleaned.csv")
    customer_features_path: Path = Field(
        default=PROJECT_ROOT / "data" / "processed" / "customer_features.csv"
    )
    labeled_data_path: Path = Field(
        default=PROJECT_ROOT / "data" / "processed" / "customer_features_labeled.csv"
    )
    feature_names_path: Path = Field(default=PROJECT_ROOT / "data" / "processed" / "feature_names.txt")
    feature_pipeline_path: Path = Field(
        default=PROJECT_ROOT / "data" / "processed" / "feature_pipeline.joblib"
    )
    plots_dir: Path = Field(default=PROJECT_ROOT / "reports" / "plots")
    reports_dir: Path = Field(default=PROJECT_ROOT / "reports")

    # --- MLflow -------------------------------------------------------
    mlflow_tracking_uri: str = Field(default="http://localhost:5000")
    mlflow_experiment_name: str = Field(default="Credit Risk Modeling")
    model_name: str = Field(default="CreditRiskModel")
    # MLflow Model Registry alias used to mark the model that should actually
    # serve traffic. Aliases (unlike raw version numbers) let us promote a
    # new champion without touching any code or redeploying the API image.
    model_alias: str = Field(default="champion")
    # Fallback used only when no alias has been set yet (e.g. first ever run).
    fallback_model_version: int | None = Field(default=None)

    # --- Modeling -------------------------------------------------------
    random_state: int = Field(default=42)
    test_size: float = Field(default=0.2)
    n_risk_clusters: int = Field(default=3)
    high_risk_min_precision: float = Field(default=0.0)
    # A newly trained model is only promoted to `champion` if its ROC-AUC on
    # the held-out test set is at least this much better than the current
    # champion's recorded ROC-AUC (0.0 = "at least as good").
    promotion_min_roc_auc_gain: float = Field(default=0.0)

    # --- Categorical / numeric columns used by the feature pipeline -----
    categorical_cols: List[str] = Field(
        default=["ProductCategory", "ChannelId", "ProviderId"]
    )
    numeric_cols: List[str] = Field(default=["Amount", "Value", "PricingStrategy"])
    id_cols: List[str] = Field(
        default=[
            "TransactionId",
            "BatchId",
            "AccountId",
            "SubscriptionId",
            "CustomerId",
            "CurrencyCode",
        ]
    )

    # --- API -------------------------------------------------------------
    api_title: str = Field(default="Credit Risk Scoring API")
    api_version: str = Field(default="1.0.0")
    log_level: str = Field(default="INFO")
    log_json: bool = Field(default=False)


settings = Settings()
