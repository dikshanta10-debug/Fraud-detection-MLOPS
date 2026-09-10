"""Liveness, readiness and model introspection endpoints."""
from fastapi import APIRouter, HTTPException

from app.model_loader import champion
from app.schemas import HealthResponse, ModelInfoResponse

router = APIRouter(tags=["operations"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Readiness probe — reports whether the champion model is serving."""
    return HealthResponse(
        status="healthy" if champion.is_loaded else "degraded",
        model_loaded=champion.is_loaded,
        model_version=champion.version if champion.is_loaded else None,
        roc_auc=champion.roc_auc if champion.is_loaded else None,
    )


@router.get("/model/info", response_model=ModelInfoResponse)
async def model_info() -> ModelInfoResponse:
    """
    Full lineage for the model currently in production: hyperparameters,
    held-out metrics, training set size and the MLflow run that produced it.
    """
    if not champion.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    md = champion.metadata
    return ModelInfoResponse(
        model_version=md.get("model_version", "unknown"),
        trained_at=md.get("trained_at", "unknown"),
        algorithm=md.get("algorithm", "XGBoost"),
        n_features=md.get("n_features", len(champion.feature_names)),
        feature_names=champion.feature_names,
        hyperparameters=md.get("hyperparameters", {}),
        metrics=md.get("metrics", {}),
        training_rows=md.get("training_rows", 0),
        fraud_rate_percent=md.get("fraud_rate_percent", 0.0),
    )
