"""Integration tests for the inference API."""
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def sample_transaction():
    return {"Time": 42000.0, "Amount": 149.62, **{f"V{i}": 0.0 for i in range(1, 29)}}


# ── Health & metadata ─────────────────────────────────
def test_health_returns_healthy_with_model_loaded(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "healthy"
    assert body["model_loaded"] is True
    assert body["roc_auc"] > 0.5


def test_model_info_exposes_lineage(client):
    r = client.get("/model/info")
    assert r.status_code == 200
    body = r.json()
    assert body["algorithm"] == "XGBoost"
    assert body["n_features"] == 30
    assert len(body["feature_names"]) == 30
    assert "roc_auc" in body["metrics"]
    assert body["training_rows"] > 0


def test_root_endpoint(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


# ── Single prediction ─────────────────────────────────
def test_predict_returns_valid_probability(client, sample_transaction):
    r = client.post("/predict", json=sample_transaction)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert isinstance(body["is_fraud"], bool)
    assert body["risk_band"] in ("LOW", "MEDIUM", "HIGH")
    assert body["model_version"]


def test_predict_flag_matches_threshold(client, sample_transaction):
    body = client.post("/predict", json=sample_transaction).json()
    assert body["is_fraud"] == (body["fraud_probability"] >= body["threshold"])


def test_predict_meets_latency_budget(client, sample_transaction):
    """Inference must stay within the 15ms serving target."""
    latencies = [
        client.post("/predict", json=sample_transaction).json()["inference_latency_ms"]
        for _ in range(20)
    ]
    assert max(latencies) < 15.0, f"p100 latency {max(latencies):.2f}ms exceeds 15ms budget"


def test_predict_is_deterministic(client, sample_transaction):
    a = client.post("/predict", json=sample_transaction).json()["fraud_probability"]
    b = client.post("/predict", json=sample_transaction).json()["fraud_probability"]
    assert a == b


def test_high_risk_features_raise_probability(client, sample_transaction):
    """Shifting the discriminative features should increase the fraud score."""
    baseline = client.post("/predict", json=sample_transaction).json()["fraud_probability"]

    suspicious = {**sample_transaction}
    for feat in ("V3", "V4", "V10", "V11", "V12", "V14", "V16", "V17", "V18"):
        suspicious[feat] = 3.0
    elevated = client.post("/predict", json=suspicious).json()["fraud_probability"]

    assert elevated > baseline


# ── Validation ────────────────────────────────────────
def test_negative_amount_rejected(client, sample_transaction):
    bad = {**sample_transaction, "Amount": -10.0}
    assert client.post("/predict", json=bad).status_code == 422


def test_missing_required_field_rejected(client, sample_transaction):
    bad = {k: v for k, v in sample_transaction.items() if k != "Amount"}
    assert client.post("/predict", json=bad).status_code == 422


# ── Batch prediction ──────────────────────────────────
def test_batch_returns_one_prediction_per_input(client, sample_transaction):
    r = client.post("/predict/batch", json={"transactions": [sample_transaction] * 25})
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 25
    assert len(body["predictions"]) == 25
    assert body["throughput_per_sec"] > 0


def test_empty_batch_rejected(client):
    assert client.post("/predict/batch", json={"transactions": []}).status_code == 422


def test_batch_amortises_per_row_latency(client, sample_transaction):
    """Batched inference should beat the single-prediction path per row."""
    single = client.post("/predict", json=sample_transaction).json()["inference_latency_ms"]
    batch = client.post(
        "/predict/batch", json={"transactions": [sample_transaction] * 100}
    ).json()["predictions"][0]["inference_latency_ms"]
    assert batch < single


# ── Observability ─────────────────────────────────────
def test_metrics_endpoint_exposes_custom_metrics(client, sample_transaction):
    client.post("/predict", json=sample_transaction)
    body = client.get("/metrics").text
    assert "fraud_predictions_total" in body
    assert "fraud_inference_latency_seconds" in body
    assert "fraud_model_roc_auc" in body


def test_drift_endpoint_reports_window_state(client):
    r = client.get("/drift")
    assert r.status_code == 200
    body = r.json()
    assert "ready" in body
    assert body["window_size"] > 0
