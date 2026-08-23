import pandas as pd

from src.data_processing import (
    add_outlier_flags,
    data_quality_report,
    detect_outliers,
    duplicate_report,
    missing_values,
)


def test_add_outlier_flags_drops_constant_country_code(raw_transactions_df):
    df = add_outlier_flags(raw_transactions_df)
    assert "CountryCode" not in df.columns  # constant column, single UGX country code
    assert "Amount_outlier_flag" in df.columns
    assert set(df["Amount_outlier_flag"].unique()) <= {0, 1}


def test_add_outlier_flags_keeps_varying_columns(raw_transactions_df):
    df = raw_transactions_df.copy()
    df.loc[df.index[0], "CountryCode"] = "999"  # introduce variance
    out = add_outlier_flags(df)
    assert "CountryCode" in out.columns


def test_missing_values_reports_only_nonzero_columns():
    df = pd.DataFrame({"a": [1, None, 3], "b": [1, 2, 3]})
    report = missing_values(df)
    assert list(report.index) == ["a"]
    assert report["a"] == 1


def test_duplicate_report(raw_transactions_df):
    df = pd.concat([raw_transactions_df, raw_transactions_df.iloc[[0]]], ignore_index=True)
    report = duplicate_report(df)
    assert report["exact_duplicate_rows"] == 1
    assert report["duplicate_TransactionId"] == 1


def test_detect_outliers_returns_counts_per_numeric_column(raw_transactions_df):
    counts = detect_outliers(raw_transactions_df)
    assert "Amount" in counts
    assert counts["Amount"] >= 0


def test_data_quality_report_shape(raw_transactions_df):
    report = data_quality_report(raw_transactions_df)
    assert report["n_rows"] == len(raw_transactions_df)
    assert report["n_customers"] == raw_transactions_df["CustomerId"].nunique()
