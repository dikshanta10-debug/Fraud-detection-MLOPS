"""Inference endpoints."""
from __future__ import annotations

import time

import numpy as np
from fastapi import APIRouter, HTTPException

from app.config import settings
from app.model_loader import champion, risk_band
from app.metrics import record_prediction
from app.schemas import (
    Transaction,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    DriftResponse,
)
from app.state import drift_monitor

router = APIRouter(tags=["inference"])


def _to_vector(txn: Transaction) -> list[float]:
    """Convert a request payload into the model's expected column order."""
    payload = txn.model_dump()
    return [float(payload[name]) for name in champion.feature_names]


@router.post("/predict", response_model=PredictionResponse)
async def predict(transaction: Transaction) -> PredictionResponse:
    """
    Score a single transaction.

    Latency reported in the response covers model inference only, excluding
    HTTP parsing and serialisation, so it is directly comparable to the
    Prometheus histogram.
    """
    if not champion.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    vector = _to_vector(transaction)
    features = np.array([vector], dtype=np.float32)

    start = time.perf_counter()
    probability = float(champion.predict_proba(features)[0])
    latency_s = time.perf_counter() - start

    threshold = champion.threshold
    is_fraud = probability >= threshold

    record_prediction(probability, is_fraud, latency_s)
    drift_monitor.observe(vector)

    return PredictionResponse(
        fraud_probability=round(probability, 6),
        is_fraud=is_fraud,
        threshold=threshold,
        risk_band=risk_band(probability),
        inference_latency_ms=round(latency_s * 1000, 3),
        model_version=champion.version,
    )


@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """
    Score up to 1000 transactions in one call.

    Batching amortises the DMatrix construction cost across rows, so
    per-transaction latency here is substantially lower than the
    single-prediction path.
    """
    if not champion.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    vectors = [_to_vector(t) for t in request.transactions]
    features = np.array(vectors, dtype=np.float32)

    start = time.perf_counter()
    probabilities = champion.predict_proba(features)
    total_latency_s = time.perf_counter() - start

    threshold = champion.threshold
    per_row_latency_s = total_latency_s / len(vectors)

    predictions = []
    flagged = 0
    for vector, prob in zip(vectors, probabilities):
        prob = float(prob)
        is_fraud = prob >= threshold
        flagged += int(is_fraud)

        record_prediction(prob, is_fraud, per_row_latency_s)
        drift_monitor.observe(vector)

        predictions.append(
            PredictionResponse(
                fraud_probability=round(prob, 6),
                is_fraud=is_fraud,
                threshold=threshold,
                risk_band=risk_band(prob),
                inference_latency_ms=round(per_row_latency_s * 1000, 4),
                model_version=champion.version,
            )
        )

    return BatchPredictionResponse(
        predictions=predictions,
        count=len(predictions),
        flagged_count=flagged,
        total_latency_ms=round(total_latency_s * 1000, 3),
        throughput_per_sec=round(len(vectors) / max(total_latency_s, 1e-9), 1),
    )


@router.get("/drift", response_model=DriftResponse)
async def get_drift() -> DriftResponse:
    """
    Current Population Stability Index per feature against the training baseline.

    Returns ready=false until DRIFT_WINDOW_SIZE predictions have been observed.
    """
    result = drift_monitor.compute()
    return DriftResponse(
        samples_observed=result.get("samples_observed", 0),
        window_size=settings.DRIFT_WINDOW_SIZE,
        ready=result.get("ready", False),
        features=result.get("features", []),
        drifted_feature_count=result.get("drifted_feature_count", 0),
    )
