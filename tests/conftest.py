import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def raw_transactions_df() -> pd.DataFrame:
    """Synthetic transactions mimicking the Xente schema, spanning several customers."""
    rng = np.random.default_rng(42)
    n_customers = 20
    rows = []
    tx_id = 0
    start = pd.Timestamp("2019-01-01")

    for c in range(n_customers):
        customer_id = f"CustomerId_{c}"
        n_tx = rng.integers(1, 15)
        for _ in range(n_tx):
            tx_id += 1
            amount = float(rng.normal(1000, 500))
            rows.append(
                {
                    "TransactionId": f"TransactionId_{tx_id}",
                    "BatchId": f"BatchId_{tx_id % 5}",
                    "AccountId": f"AccountId_{c}",
                    "SubscriptionId": f"SubscriptionId_{c}",
                    "CustomerId": customer_id,
                    "CurrencyCode": "UGX",
                    "CountryCode": "256",
                    "ProviderId": f"ProviderId_{rng.integers(1, 6)}",
                    "ProductId": f"ProductId_{rng.integers(1, 20)}",
                    "ProductCategory": rng.choice(["airtime", "data_bundles", "utility_bill", "tv"]),
                    "ChannelId": f"ChannelId_{rng.integers(1, 4)}",
                    "Amount": amount,
                    "Value": abs(amount),
                    "TransactionStartTime": (
                        start + pd.Timedelta(days=int(rng.integers(0, 365)), hours=int(rng.integers(0, 24)))
                    ).isoformat(),
                    "PricingStrategy": int(rng.integers(0, 4)),
                    "FraudResult": 0,
                }
            )

    return pd.DataFrame(rows)


@pytest.fixture
def customer_features_df(raw_transactions_df):
    from src.feature_eng_process import build_customer_features

    return build_customer_features(raw_transactions_df)
