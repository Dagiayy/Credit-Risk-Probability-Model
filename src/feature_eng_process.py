import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# --------- Custom Transformers ------------

class AggregateFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, group_col='CustomerId'):
        self.group_col = group_col
        self.agg_features_ = None

    def fit(self, X, y=None):
        agg_df = X.groupby(self.group_col).agg(
            total_transaction_amount=('Amount', 'sum'),
            avg_transaction_amount=('Amount', 'mean'),
            transaction_count=('Amount', 'count'),
            std_transaction_amount=('Amount', 'std')
        ).reset_index()
        agg_df['std_transaction_amount'] = agg_df['std_transaction_amount'].fillna(0)
        self.agg_features_ = agg_df
        return self

    def transform(self, X):
        X = X.merge(self.agg_features_, on=self.group_col, how='left')
        return X

class DatetimeFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, datetime_col='TransactionStartTime'):
        self.datetime_col = datetime_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[self.datetime_col] = pd.to_datetime(X[self.datetime_col], errors='coerce')
        X['transaction_hour'] = X[self.datetime_col].dt.hour
        X['transaction_day'] = X[self.datetime_col].dt.day
        X['transaction_month'] = X[self.datetime_col].dt.month
        X['transaction_year'] = X[self.datetime_col].dt.year
        return X

class CategoricalImputer(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_cols):
        self.categorical_cols = categorical_cols
        self.fill_values_ = {}

    def fit(self, X, y=None):
        for col in self.categorical_cols:
            self.fill_values_[col] = X[col].mode()[0]
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.categorical_cols:
            X[col] = X[col].fillna(self.fill_values_[col])
        return X

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_cols):
        self.categorical_cols = categorical_cols
        self.ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    def fit(self, X, y=None):
        self.ohe.fit(X[self.categorical_cols])
        return self

    def transform(self, X):
        ohe_array = self.ohe.transform(X[self.categorical_cols])
        ohe_df = pd.DataFrame(ohe_array, columns=self.ohe.get_feature_names_out(self.categorical_cols))
        ohe_df.index = X.index
        X = X.drop(columns=self.categorical_cols)
        X = pd.concat([X, ohe_df], axis=1)
        return X


class NumericImputerScaler(BaseEstimator, TransformerMixin):
    def __init__(self, numeric_cols):
        self.numeric_cols = numeric_cols
        self.imputer = SimpleImputer(strategy='median')
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        X_num = X[self.numeric_cols]
        X_imputed = self.imputer.fit_transform(X_num)
        self.scaler.fit(X_imputed)
        return self

    def transform(self, X):
        X_num = X[self.numeric_cols]
        X_imputed = self.imputer.transform(X_num)
        X_scaled = self.scaler.transform(X_imputed)
        X_scaled_df = pd.DataFrame(X_scaled, columns=self.numeric_cols, index=X.index)
        X = X.drop(columns=self.numeric_cols)
        X = pd.concat([X, X_scaled_df], axis=1)
        return X

# --------- WOE/IV Calculation ------------

def calculate_woe_iv(df, categorical_cols, target_col='FraudResult'):
    iv_dict = {}
    eps = 0.0001  # To prevent division by zero

    for col in categorical_cols:
        lst = []
        for val in df[col].unique():
            total = len(df[df[col] == val])
            good = len(df[(df[col] == val) & (df[target_col] == 0)])
            bad = len(df[(df[col] == val) & (df[target_col] == 1)])
            dist_good = good / max(1, len(df[df[target_col] == 0]))
            dist_bad = bad / max(1, len(df[df[target_col] == 1]))

            woe = np.log((dist_good + eps) / (dist_bad + eps))
            iv = (dist_good - dist_bad) * woe
            lst.append(iv)

        iv_total = sum(lst)
        iv_dict[col] = iv_total
        print(f"Feature: {col}, IV: {iv_total:.4f}")

    return iv_dict

# --------- Pipeline Builder -----------

def build_feature_engineering_pipeline():
    categorical_cols = ['ProductCategory', 'ChannelId', 'ProviderId']
    numeric_cols = ['Amount', 'Value', 'PricingStrategy']

    pipeline = Pipeline([
        ('aggregate_features', AggregateFeatures(group_col='CustomerId')),
        ('datetime_features', DatetimeFeatures(datetime_col='TransactionStartTime')),
        ('cat_imputer', CategoricalImputer(categorical_cols=categorical_cols)),
        ('cat_encoder', CategoricalEncoder(categorical_cols=categorical_cols)),
        ('num_imputer_scaler', NumericImputerScaler(numeric_cols=numeric_cols)),
    ])

    return pipeline

# --------- Example usage -----------

if __name__ == "__main__":
    df = pd.read_csv('data/processed/data_cleaned.csv')

    pipeline = build_feature_engineering_pipeline()
    df_transformed = pipeline.fit_transform(df)

    # Calculate IV before encoding
    iv_scores = calculate_woe_iv(df, categorical_cols=['ProductCategory', 'ChannelId', 'ProviderId'], target_col='FraudResult')

    df_transformed.to_csv('data/processed/feature_engineered_data.csv', index=False)
    print("✅ Saved feature engineered data to '../data/processed/feature_engineered_data.csv'")
