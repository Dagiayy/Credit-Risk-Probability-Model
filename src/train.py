import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
import mlflow  # type: ignore
import mlflow.sklearn  # type: ignore
from mlflow.models.signature import infer_signature  # type: ignore
from mlflow.tracking import MlflowClient  # type: ignore
import os

# Constants
RANDOM_STATE = 42
EXPERIMENT_NAME = "Credit Risk Modeling"
DATA_PATH = "data/processed/feature_engineered_labeled.csv"
FEATURE_OUTPUT_PATH = "data/processed/feature_names.txt"

def load_data(path):
    return pd.read_csv(path)

def preprocess(df):
    drop_cols = ['TransactionId', 'TransactionStartTime', 'CustomerId', 'BatchId']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

    y = df['is_high_risk']
    X = df.drop(columns=['is_high_risk'])

    numeric_cols = X.select_dtypes(include=['number']).columns
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    non_numeric_cols = X.select_dtypes(exclude=['number']).columns
    if len(non_numeric_cols) > 0:
        print(f"⚠️ Dropping non-numeric columns: {list(non_numeric_cols)}")
        X = X.drop(columns=non_numeric_cols)

    return X, y

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba)
    }

def train_and_log_model(name, model, params, X_train, y_train, X_test, y_test):
    with mlflow.start_run(run_name=name) as run:
        run_id = run.info.run_id

        model.set_params(**params)
        model.fit(X_train, y_train)

        metrics = evaluate_model(model, X_test, y_test)

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        input_example = X_test.iloc[:1]
        signature = infer_signature(X_test, model.predict(X_test))

        mlflow.sklearn.log_model(
            model,
            artifact_path=name.lower() + "_model",
            input_example=input_example,
            signature=signature
        )

        print(f"\n✅ Results for {name}:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

        os.makedirs(os.path.dirname(FEATURE_OUTPUT_PATH), exist_ok=True)
        with open(FEATURE_OUTPUT_PATH, "w") as f:
            f.write("\n".join(X_train.columns.tolist()))
        print(f"📝 Feature names written to {FEATURE_OUTPUT_PATH}")

        return model, run_id

def main():
    # ✅ Dynamically detect MLflow URI (default to localhost)
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(EXPERIMENT_NAME)

    print("📥 Loading and preprocessing data...")
    df = load_data(DATA_PATH)
    X, y = preprocess(df)

    print("🔀 Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    print("🤖 Training models and logging to MLflow...")

    # Logistic Regression
    lr_model = LogisticRegression()
    lr_params = {'C': 1.0, 'solver': 'lbfgs', 'max_iter': 500}
    train_and_log_model("LogisticRegression", lr_model, lr_params, X_train, y_train, X_test, y_test)

    # Random Forest
    rf_model = RandomForestClassifier()
    rf_params = {'n_estimators': 100, 'max_depth': 5, 'random_state': RANDOM_STATE}
    rf_model, run_id = train_and_log_model("RandomForest", rf_model, rf_params, X_train, y_train, X_test, y_test)

    # Register model
    model_uri = f"runs:/{run_id}/randomforest_model"
    print(f"📦 Registering the RandomForest model from URI: {model_uri}")
    result = mlflow.register_model(model_uri=model_uri, name="CreditRiskModel")
    version = result.version

    # ✅ Add custom tags instead of deprecated stage
    client = MlflowClient()
    client.set_model_version_tag("CreditRiskModel", version, "stage", "production")
    client.set_model_version_tag("CreditRiskModel", version, "promoted_by", "dagmawi")

    print(f"🚀 Model version {version} tagged as 'production'")

if __name__ == "__main__":
    main()
