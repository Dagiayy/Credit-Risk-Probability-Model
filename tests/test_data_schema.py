import pytest

from src.data_schema import DataValidationError, validate_raw_transactions


def test_validate_raw_transactions_passes_for_valid_data(raw_transactions_df):
    validated = validate_raw_transactions(raw_transactions_df)
    assert len(validated) == len(raw_transactions_df)


def test_validate_raw_transactions_rejects_missing_column(raw_transactions_df):
    bad = raw_transactions_df.drop(columns=["Amount"])
    with pytest.raises(DataValidationError):
        validate_raw_transactions(bad)


def test_validate_raw_transactions_rejects_invalid_fraud_result(raw_transactions_df):
    bad = raw_transactions_df.copy()
    bad.loc[0, "FraudResult"] = 5  # only 0/1 allowed
    with pytest.raises(DataValidationError):
        validate_raw_transactions(bad)


def test_validate_raw_transactions_rejects_duplicate_transaction_id(raw_transactions_df):
    bad = raw_transactions_df.copy()
    bad.loc[1, "TransactionId"] = bad.loc[0, "TransactionId"]
    with pytest.raises(DataValidationError):
        validate_raw_transactions(bad)
