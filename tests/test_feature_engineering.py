import pandas as pd

from src.feature_eng_process import (
    build_customer_features,
    build_preprocessing_pipeline,
    calculate_woe_iv,
    get_output_feature_names,
)


def test_build_customer_features_one_row_per_customer(raw_transactions_df):
    out = build_customer_features(raw_transactions_df)
    assert out["CustomerId"].is_unique
    assert set(out["CustomerId"]) == set(raw_transactions_df["CustomerId"])


def test_build_customer_features_transaction_count_matches_source(raw_transactions_df):
    out = build_customer_features(raw_transactions_df)
    expected_counts = raw_transactions_df.groupby("CustomerId").size()
    merged = out.set_index("CustomerId")["transaction_count"]
    assert (merged == expected_counts).all()


def test_build_customer_features_std_is_zero_not_nan_for_single_transaction():
    df = pd.DataFrame(
        {
            "TransactionId": ["t1"],
            "CustomerId": ["c1"],
            "Amount": [100.0],
            "Value": [100.0],
            "TransactionStartTime": ["2020-01-01T00:00:00"],
            "ProductCategory": ["airtime"],
            "ChannelId": ["ChannelId_1"],
            "ProviderId": ["ProviderId_1"],
            "PricingStrategy": [1],
            "CountryCode": ["256"],
        }
    )
    out = build_customer_features(df)
    assert out.loc[0, "std_transaction_amount"] == 0


def test_preprocessing_pipeline_fit_transform_is_numeric(customer_features_df):
    pipeline = build_preprocessing_pipeline()
    X = customer_features_df.drop(columns=["CustomerId"])
    transformed = pipeline.fit_transform(X)
    assert transformed.shape[0] == X.shape[0]
    assert not transformed.isnull().any().any()


def test_preprocessing_pipeline_transform_after_fit_has_stable_columns(customer_features_df):
    pipeline = build_preprocessing_pipeline()
    X = customer_features_df.drop(columns=["CustomerId"])
    split = len(X) // 2
    train_out = pipeline.fit_transform(X.iloc[:split])
    test_out = pipeline.transform(X.iloc[split:])
    assert list(train_out.columns) == list(test_out.columns)


def test_get_output_feature_names_matches_transform_columns(customer_features_df):
    pipeline = build_preprocessing_pipeline()
    X = customer_features_df.drop(columns=["CustomerId"])
    transformed = pipeline.fit_transform(X)
    assert get_output_feature_names(pipeline) == list(transformed.columns)


def test_calculate_woe_iv_returns_finite_scores(raw_transactions_df):
    iv = calculate_woe_iv(raw_transactions_df, ["ProductCategory"], target_col="FraudResult")
    assert "ProductCategory" in iv
    assert iv["ProductCategory"] == iv["ProductCategory"]  # not NaN
