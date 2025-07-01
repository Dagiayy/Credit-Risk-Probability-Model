import os
import mlflow
import mlflow.sklearn
from fastapi import FastAPI
from src.api.pydantic_models import CustomerData, RiskPrediction
import pandas as pd

app = FastAPI(title="Credit Risk Scoring API")

# Set MLflow tracking URI (defaults to http://mlflow:5000 if env variable not set)
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))

# Load model from MLflow registry
MODEL_NAME = "CreditRiskModel"
MODEL_VERSION = 2
model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
model = mlflow.sklearn.load_model(model_uri)

@app.post("/predict", response_model=RiskPrediction)
def predict_risk(data: CustomerData):
    df = pd.DataFrame([data.dict()])
    probability = model.predict_proba(df)[0][1]  # Probability of high risk
    return RiskPrediction(risk_probability=probability)
