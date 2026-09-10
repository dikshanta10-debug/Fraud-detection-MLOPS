# ── Stage 1: build dependencies ───────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

# libgomp is required by XGBoost at runtime; build tools only in this stage.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: runtime ──────────────────────────────────
FROM python:3.12-slim

WORKDIR /app

# XGBoost needs libgomp1 present in the final image.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

# Run as a non-root user — never ship a container that serves traffic as root.
RUN useradd --create-home --uid 1000 appuser

COPY --chown=appuser:appuser app/ ./app/
COPY --chown=appuser:appuser training/ ./training/
COPY --chown=appuser:appuser models/ ./models/

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# Single worker: the model lives in process memory, so additional workers
# multiply RSS. Scale horizontally with more replicas instead.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
