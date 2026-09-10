"""Request and response models for the inference API."""
from typing import Literal
from pydantic import BaseModel, Field, ConfigDict


class Transaction(BaseModel):
    """
    A single transaction to be scored.

    Feature names mirror the Kaggle Credit Card Fraud dataset:
    V1..V28 are PCA-anonymised features, plus Time and Amount.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "Time": 42000.0,
                "Amount": 149.62,
                **{f"V{i}": 0.0 for i in range(1, 29)},
            }
        }
    )

    Time: float = Field(..., description="Seconds elapsed since the first transaction in the dataset")
    Amount: float = Field(..., ge=0, description="Transaction amount")

    V1: float = 0.0
    V2: float = 0.0
    V3: float = 0.0
    V4: float = 0.0
    V5: float = 0.0
    V6: float = 0.0
    V7: float = 0.0
    V8: float = 0.0
    V9: float = 0.0
    V10: float = 0.0
    V11: float = 0.0
    V12: float = 0.0
    V13: float = 0.0
    V14: float = 0.0
    V15: float = 0.0
    V16: float = 0.0
    V17: float = 0.0
    V18: float = 0.0
    V19: float = 0.0
    V20: float = 0.0
    V21: float = 0.0
    V22: float = 0.0
    V23: float = 0.0
    V24: float = 0.0
    V25: float = 0.0
    V26: float = 0.0
    V27: float = 0.0
    V28: float = 0.0


class PredictionResponse(BaseModel):
    """Scoring result for a single transaction."""

    fraud_probability: float = Field(..., description="Model output probability in [0, 1]")
    is_fraud: bool = Field(..., description="True when probability exceeds the decision threshold")
    threshold: float = Field(..., description="Decision threshold applied")
    risk_band: Literal["LOW", "MEDIUM", "HIGH"] = Field(..., description="Bucketed risk level")
    inference_latency_ms: float = Field(..., description="Model inference time in milliseconds")
    model_version: str = Field(..., description="Version of the champion model that produced this score")


class BatchPredictionRequest(BaseModel):
    transactions: list[Transaction] = Field(..., min_length=1, max_length=1000)


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    count: int
    flagged_count: int
    total_latency_ms: float
    throughput_per_sec: float


class DriftFeature(BaseModel):
    feature: str
    psi: float
    status: Literal["STABLE", "WARN", "ALERT"]


class DriftResponse(BaseModel):
    samples_observed: int
    window_size: int
    ready: bool = Field(..., description="False until the observation window has filled")
    features: list[DriftFeature]
    drifted_feature_count: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None = None
    roc_auc: float | None = None


class ModelInfoResponse(BaseModel):
    model_version: str
    trained_at: str
    algorithm: str
    n_features: int
    feature_names: list[str]
    hyperparameters: dict
    metrics: dict
    training_rows: int
    fraud_rate_percent: float
