import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

import src.api.main as api_main
from src.feature_eng_process import build_customer_features, build_preprocessing_pipeline
from src.model_registry import LoadedModel

app = api_main.app


@pytest.fixture(autouse=True)
def no_network_model_loading(monkeypatch):
    """Prevent the app's startup lifespan from making real MLflow network calls.

    Without this, every `TestClient(app)` context triggers `load_champion_model()`,
    which tries to reach `MLFLOW_TRACKING_URI` and retries with backoff on failure —
    turning a synthetic-data unit test suite into a multi-minute run dependent on
    network behaviour. Tests set `app.state.loaded_model` explicitly instead.
    """

    def _fail(*args, **kwargs):
        raise RuntimeError("no MLflow server in tests")

    monkeypatch.setattr(api_main, "load_champion_model", _fail)


def _stub_loaded_model(raw_transactions_df):
    customer_df = build_customer_features(raw_transactions_df)
    X = customer_df.drop(columns=["CustomerId"])
    pipeline = build_preprocessing_pipeline()
    X_proc = pipeline.fit_transform(X)

    y = [i % 2 for i in range(len(X_proc))]
    model = LogisticRegression(max_iter=200)
    model.fit(X_proc, y)

    return LoadedModel(
        model=model,
        feature_pipeline=pipeline,
        feature_names=list(X_proc.columns),
        version="test-1",
        run_id="test-run",
    )


def test_health_always_ok():
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


def test_ready_503_when_model_not_loaded():
    with TestClient(app) as client:
        client.app.state.loaded_model = None
        resp = client.get("/ready")
        assert resp.status_code == 503


def test_predict_returns_risk_probability(raw_transactions_df):
    with TestClient(app) as client:
        client.app.state.loaded_model = _stub_loaded_model(raw_transactions_df)
        client.app.state.explainer = None

        target_customer = raw_transactions_df["CustomerId"].iloc[0]
        subset = raw_transactions_df[raw_transactions_df["CustomerId"] == target_customer]
        payload = {"transactions": subset.to_dict(orient="records")}

        resp = client.post("/api/v1/predict", json=payload)

        assert resp.status_code == 200
        body = resp.json()
        assert 0.0 <= body["risk_probability"] <= 1.0
        assert body["customer_id"] == target_customer
        assert body["risk_band"] in {"low", "medium", "high"}
        assert body["top_factors"] is None  # explain defaults to False


def test_predict_503_when_model_not_loaded(raw_transactions_df):
    with TestClient(app) as client:
        client.app.state.loaded_model = None
        target_customer = raw_transactions_df["CustomerId"].iloc[0]
        subset = raw_transactions_df[raw_transactions_df["CustomerId"] == target_customer]
        resp = client.post("/api/v1/predict", json={"transactions": subset.to_dict(orient="records")})
        assert resp.status_code == 503


def test_predict_rejects_mixed_customers(raw_transactions_df):
    rows = raw_transactions_df.groupby("CustomerId").head(1).head(2).to_dict(orient="records")
    assert len({r["CustomerId"] for r in rows}) == 2  # sanity check on fixture

    with TestClient(app) as client:
        resp = client.post("/api/v1/predict", json={"transactions": rows})
        assert resp.status_code == 422


def test_transaction_schema_does_not_accept_fraud_result():
    """FraudResult must not be part of the accepted schema (target-leakage guard)."""
    from src.api.pydantic_models import TransactionRecord

    assert "FraudResult" not in TransactionRecord.model_fields
