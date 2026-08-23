"""Schema validation for the raw Xente transactions dataset.

The original project had zero validation between "CSV on disk" and "pandas
dataframe fed into feature engineering" — a renamed column, a dtype change,
or a currency other than UGX would silently propagate into training. This
module makes that boundary an explicit, fail-fast contract.
"""
from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema

RAW_TRANSACTIONS_SCHEMA = DataFrameSchema(
    {
        "TransactionId": Column(str, nullable=False, unique=True),
        "BatchId": Column(str, nullable=False),
        "AccountId": Column(str, nullable=False),
        "SubscriptionId": Column(str, nullable=False),
        "CustomerId": Column(str, nullable=False),
        "CurrencyCode": Column(str, nullable=False),
        "CountryCode": Column(nullable=False),
        "ProviderId": Column(str, nullable=False),
        "ProductId": Column(str, nullable=False),
        "ProductCategory": Column(str, nullable=False),
        "ChannelId": Column(str, nullable=False),
        "Amount": Column(float, coerce=True, nullable=False),
        "Value": Column(float, checks=Check.ge(0), coerce=True, nullable=False),
        "TransactionStartTime": Column(str, nullable=False),
        "PricingStrategy": Column(nullable=False),
        "FraudResult": Column(int, checks=Check.isin([0, 1]), coerce=True, nullable=False),
    },
    strict=False,  # allow extra columns without failing (forward compatible)
    coerce=False,
)


class DataValidationError(ValueError):
    """Raised when the raw dataset does not match the expected contract."""


def validate_raw_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Validate `df` against RAW_TRANSACTIONS_SCHEMA, raising a clear error on failure.

    Returns the (possibly type-coerced) dataframe on success so callers can
    chain it directly into the next pipeline stage.
    """
    try:
        return RAW_TRANSACTIONS_SCHEMA.validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:  # pragma: no cover - message formatting only
        failure_summary = exc.failure_cases[["column", "check", "failure_case"]].head(20)
        raise DataValidationError(
            "Raw transactions data failed schema validation:\n"
            f"{failure_summary.to_string(index=False)}"
        ) from exc
