import mlflow
import pandas as pd

def load_model(model_name="CreditRiskModel", version=2):
    model_uri = f"models:/{model_name}/{version}"
    model = mlflow.sklearn.load_model(model_uri)
    return model

def predict(df_features):
    model = load_model()
    preds = model.predict(df_features)
    return preds

if __name__ == "__main__":
    df = pd.read_csv('data/processed/feature_engineered_labeled.csv')

    drop_cols = ['TransactionId', 'TransactionStartTime', 'CustomerId', 'BatchId',
                 'AccountId', 'SubscriptionId', 'CurrencyCode', 'ProductId']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

    if 'is_high_risk' in df.columns:
        df = df.drop(columns=['is_high_risk'])

    X = df
    predictions = predict(X)
    print(predictions)
