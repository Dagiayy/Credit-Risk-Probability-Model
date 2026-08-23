"""Proxy target construction: RFM + KMeans.

There is no ground-truth "defaulted on a loan" label in this dataset (Bati
Bank has no lending history with these customers yet — that's the whole
point of a BNPL pilot). Basel-II-style alternative-data scoring commonly
proxies default risk with engagement/behavioural clustering: customers who
transact rarely, recently disengaged, and spend little are treated as the
higher-risk segment for a first model, to be replaced with real repayment
outcomes once the pilot generates them.

This module keeps that approach but makes the clustering decision auditable
instead of a black box: it reports silhouette score and per-cluster RFM
profiles so a reviewer can sanity-check *why* a cluster was labeled high-risk.
"""
from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.config import settings
from src.logging_utils import get_logger

logger = get_logger(__name__)


def calculate_rfm(df: pd.DataFrame, snapshot_date_col: str = "TransactionStartTime") -> pd.DataFrame:
    df = df.copy()
    df[snapshot_date_col] = pd.to_datetime(df[snapshot_date_col])
    snapshot_date = df[snapshot_date_col].max() + pd.Timedelta(days=1)

    rfm = df.groupby("CustomerId").agg(
        Recency=(snapshot_date_col, lambda x: (snapshot_date - x.max()).days),
        Frequency=("CustomerId", "count"),
        Monetary=("Amount", "sum"),
    )
    return rfm.reset_index()


def label_high_risk_customers(
    rfm_df: pd.DataFrame, n_clusters: int | None = None, random_state: int | None = None
) -> pd.DataFrame:
    """Cluster customers on scaled RFM and label the least-engaged cluster high-risk.

    Returns a dataframe with CustomerId, is_high_risk, and diagnostic columns
    (cluster id, silhouette score) so the choice can be audited.
    """
    n_clusters = n_clusters or settings.n_risk_clusters
    random_state = random_state if random_state is not None else settings.random_state

    rfm_df = rfm_df.copy()
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_df[["Recency", "Frequency", "Monetary"]])

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    rfm_df["Cluster"] = kmeans.fit_predict(rfm_scaled)

    sil_score = (
        silhouette_score(rfm_scaled, rfm_df["Cluster"]) if rfm_df["Cluster"].nunique() > 1 else float("nan")
    )
    logger.info("KMeans clustering (k=%s) silhouette score: %.4f", n_clusters, sil_score)

    cluster_profile = rfm_df.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean()
    logger.info("Per-cluster RFM profile:\n%s", cluster_profile)

    # Least-engaged cluster = high recency (long since last purchase), low
    # frequency, low monetary value -> lowest (Frequency + Monetary), scaled.
    engagement_score = (
        rfm_df.groupby("Cluster")[["Frequency", "Monetary"]]
        .mean()
        .apply(lambda col: (col - col.mean()) / col.std(ddof=0))
        .sum(axis=1)
    )
    high_risk_cluster = engagement_score.idxmin()
    logger.info("Cluster %s selected as high-risk (lowest engagement score)", high_risk_cluster)

    rfm_df["is_high_risk"] = (rfm_df["Cluster"] == high_risk_cluster).astype(int)
    high_risk_rate = rfm_df["is_high_risk"].mean()
    logger.info("Proxy high-risk rate: %.2f%% of customers", high_risk_rate * 100)

    return rfm_df[["CustomerId", "is_high_risk", "Cluster"]]


if __name__ == "__main__":
    from src.config import settings as _settings

    df = pd.read_csv(_settings.customer_features_path)
    raw_df = pd.read_csv(_settings.raw_data_path)

    rfm = calculate_rfm(raw_df)
    risk_labels = label_high_risk_customers(rfm)

    df_labeled = df.merge(risk_labels[["CustomerId", "is_high_risk"]], on="CustomerId", how="left")
    df_labeled.to_csv(_settings.labeled_data_path, index=False)
    logger.info("Saved labeled customer dataset to %s", _settings.labeled_data_path)
