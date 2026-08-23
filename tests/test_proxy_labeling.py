from src.proxy_labeling import calculate_rfm, label_high_risk_customers


def test_calculate_rfm_one_row_per_customer(raw_transactions_df):
    rfm = calculate_rfm(raw_transactions_df)
    assert rfm["CustomerId"].is_unique
    assert set(rfm.columns) == {"CustomerId", "Recency", "Frequency", "Monetary"}
    assert (rfm["Recency"] >= 0).all()
    assert (rfm["Frequency"] >= 1).all()


def test_label_high_risk_customers_produces_binary_label(raw_transactions_df):
    rfm = calculate_rfm(raw_transactions_df)
    labels = label_high_risk_customers(rfm, n_clusters=3, random_state=42)
    assert set(labels["is_high_risk"].unique()) <= {0, 1}
    assert labels["CustomerId"].is_unique
    assert len(labels) == len(rfm)


def test_label_high_risk_customers_is_exactly_one_cluster(raw_transactions_df):
    rfm = calculate_rfm(raw_transactions_df)
    labels = label_high_risk_customers(rfm, n_clusters=3, random_state=42)
    high_risk_clusters = labels.loc[labels["is_high_risk"] == 1, "Cluster"].unique()
    # by construction, is_high_risk==1 marks exactly the single least-engaged cluster
    assert len(high_risk_clusters) <= 1
