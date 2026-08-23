"""API request/response schemas.

Deliberately modeled on *raw* transaction fields — the same shape as a row
in the source dataset — rather than the ~30 pre-engineered model features
the original API demanded from callers. A loan-application-time caller has
raw transaction history, not `total_transaction_amount` or one-hot columns;
making them compute those by hand duplicated (and could silently diverge
from) the training feature pipeline. The API now runs that pipeline itself.

`FraudResult` is intentionally not part of this schema: it is the historical
fraud label from the source dataset. Accepting it as a prediction input
would (a) leak the answer for historical rows and (b) be unavailable for a
real new applicant, so it must never be a serving-time input.
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, model_validator


class TransactionRecord(BaseModel):
    TransactionId: str
    BatchId: str
    AccountId: str
    SubscriptionId: str
    CustomerId: str
    CurrencyCode: str
    CountryCode: str
    ProviderId: str
    ProductId: str
    ProductCategory: str
    ChannelId: str
    Amount: float
    Value: float = Field(ge=0)
    TransactionStartTime: str
    PricingStrategy: int


class CustomerTransactionsRequest(BaseModel):
    """A batch of a single customer's transaction history to score."""

    transactions: List[TransactionRecord] = Field(min_length=1)

    @model_validator(mode="after")
    def _all_same_customer(self) -> "CustomerTransactionsRequest":
        customer_ids = {t.CustomerId for t in self.transactions}
        if len(customer_ids) != 1:
            raise ValueError(
                f"All transactions in one request must belong to the same CustomerId, got {customer_ids}"
            )
        return self


class RiskFactor(BaseModel):
    feature: str
    shap_value: float
    direction: str


class RiskPrediction(BaseModel):
    customer_id: str
    risk_probability: float = Field(ge=0, le=1)
    is_high_risk: bool
    risk_band: str
    model_version: str
    top_factors: Optional[List[RiskFactor]] = None


class HealthStatus(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[str] = None
