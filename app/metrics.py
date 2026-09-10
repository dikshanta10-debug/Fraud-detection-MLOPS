"""
Custom Prometheus metrics for the fraud detection service.

These sit alongside the default HTTP metrics that
prometheus-fastapi-instrumentator registers automatically.
"""
from prometheus_client import Counter, Histogram, Gauge

# ── Prediction outcomes ───────────────────────────────
PREDICTIONS_TOTAL = Counter(
    "fraud_predictions_total",
    "Total number of transactions scored, labelled by outcome",
    labelnames=["outcome"],  # "fraud" | "legitimate"
)

# ── Inference latency ─────────────────────────────────
# Buckets chosen around the sub-15ms target so p50/p95/p99 land
# in meaningful ranges rather than all collapsing into one bucket.
INFERENCE_LATENCY = Histogram(
    "fraud_inference_latency_seconds",
    "Model inference latency in seconds (excludes HTTP overhead)",
    buckets=(0.001, 0.0025, 0.005, 0.0075, 0.010, 0.015, 0.025, 0.050, 0.100, 0.250),
)

# ── Score distribution ────────────────────────────────
FRAUD_PROBABILITY = Histogram(
    "fraud_probability_distribution",
    "Distribution of predicted fraud probabilities",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

# ── Feature drift ─────────────────────────────────────
FEATURE_DRIFT_PSI = Gauge(
    "fraud_feature_drift_psi",
    "Population Stability Index per feature vs the training baseline",
    labelnames=["feature"],
)

DRIFTED_FEATURES = Gauge(
    "fraud_drifted_feature_count",
    "Number of features whose PSI exceeds the alert threshold",
)

# ── Model metadata ────────────────────────────────────
MODEL_INFO = Gauge(
    "fraud_model_info",
    "Champion model metadata exposed as labels (value is always 1)",
    labelnames=["version", "algorithm"],
)

MODEL_ROC_AUC = Gauge(
    "fraud_model_roc_auc",
    "ROC-AUC of the currently loaded champion model on the held-out test set",
)


def record_prediction(probability: float, is_fraud: bool, latency_seconds: float) -> None:
    """Record a single prediction across all relevant metrics."""
    PREDICTIONS_TOTAL.labels(outcome="fraud" if is_fraud else "legitimate").inc()
    INFERENCE_LATENCY.observe(latency_seconds)
    FRAUD_PROBABILITY.observe(probability)


def set_model_metrics(version: str, algorithm: str, roc_auc: float) -> None:
    """Publish champion model metadata once at startup."""
    MODEL_INFO.labels(version=version, algorithm=algorithm).set(1)
    MODEL_ROC_AUC.set(roc_auc)
