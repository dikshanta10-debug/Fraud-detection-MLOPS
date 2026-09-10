"""
Fraud Detection Inference API.

Serves the champion XGBoost model behind a REST interface, exports
Prometheus telemetry, and monitors feature drift against the training
distribution.

Run locally:
    uvicorn app.main:app --reload
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from app.config import settings
from app.metrics import set_model_metrics
from app.model_loader import champion
from app.routers import health, predict

logging.basicConfig(
    level=settings.LOG_LEVEL,
    format="%(asctime)s  %(levelname)-7s %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the champion model once at startup rather than per request.

    A cold model load costs tens of milliseconds; doing it per request would
    dominate the latency budget and make the sub-15ms target unreachable.
    """
    logger.info("Starting %s v%s", settings.APP_NAME, settings.APP_VERSION)
    try:
        champion.load(settings.model_path, settings.metadata_path)
        set_model_metrics(
            version=champion.version,
            algorithm=champion.algorithm,
            roc_auc=champion.roc_auc,
        )
    except FileNotFoundError as exc:
        # Start in degraded mode so /health can report the problem rather
        # than the container crash-looping with no diagnostics.
        logger.error("Model load failed: %s", exc)

    yield

    logger.info("Shutting down")


app = FastAPI(
    title="Fraud Detection API",
    description=(
        "Real-time fraud scoring backed by an XGBoost classifier, with MLflow "
        "model lineage, Prometheus telemetry and PSI-based feature drift monitoring."
    ),
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

# Registers default HTTP metrics and exposes /metrics for Prometheus scraping.
Instrumentator(
    should_group_status_codes=False,
    excluded_handlers=["/metrics", "/health"],
).instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)

app.include_router(health.router)
app.include_router(predict.router)


@app.get("/", include_in_schema=False)
async def root():
    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "model_loaded": champion.is_loaded,
        "docs": "/docs",
        "metrics": "/metrics",
    }
