from pydantic import BaseModel
from typing import List


class CustomerData(BaseModel):
    FraudResult: int
    Amount_outlier_flag: int
    Value_outlier_flag: int
    PricingStrategy_outlier_flag: int
    total_transaction_amount: float
    avg_transaction_amount: float
    transaction_count: int
    std_transaction_amount: float
    transaction_hour: int
    transaction_day: int
    transaction_month: int
    transaction_year: int
    ProductCategory_airtime: int
    ProductCategory_data_bundles: int
    ProductCategory_financial_services: int
    ProductCategory_movies: int
    ProductCategory_other: int
    ProductCategory_ticket: int
    ProductCategory_transport: int
    ProductCategory_tv: int
    ProductCategory_utility_bill: int
    ChannelId_ChannelId_1: int
    ChannelId_ChannelId_2: int
    ChannelId_ChannelId_3: int
    ChannelId_ChannelId_5: int
    ProviderId_ProviderId_1: int
    ProviderId_ProviderId_2: int
    ProviderId_ProviderId_3: int
    ProviderId_ProviderId_4: int
    ProviderId_ProviderId_5: int
    ProviderId_ProviderId_6: int
    Amount: float
    Value: float
    PricingStrategy: int


class RiskPrediction(BaseModel):
    risk_probability: float
