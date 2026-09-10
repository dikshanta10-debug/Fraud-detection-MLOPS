# Real-Time Fraud Detection Pipeline with MLOps Tracking

An end-to-end machine learning system that scores payment transactions for fraud in real time. Covers the full model lifecycle: training with experiment tracking, champion selection, containerised serving, production telemetry, and feature drift monitoring.

Built to demonstrate that a model is only half of an ML system — the other half is versioning it, serving it under a latency budget, and knowing when it has gone stale.

---

## Architecture

```
                         ┌──────────────────────────────┐
   creditcard.csv  ─────►│  training/train.py           │
   (or synthetic)        │  • stratified 60/20/20 split │
                         │  • 4 hyperparameter variants │
                         │  • scale_pos_weight for      │
                         │    class imbalance           │
                         │  • threshold tuned on val    │
                         └──────────┬───────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
          ┌──────────────────┐          ┌─────────────────────┐
          │  MLflow          │          │  models/            │
          │  • params        │          │  • champion.json    │
          │  • metrics       │          │  • metadata.json    │
          │  • artifacts     │          │  • baseline_stats   │
          │  • run lineage   │          └──────────┬──────────┘
          └──────────────────┘                     │
                                                   ▼
                                    ┌──────────────────────────┐
    POST /predict  ────────────────►│  FastAPI inference API   │
    POST /predict/batch             │  • model loaded at       │
    GET  /drift                     │    startup, not per call │
    GET  /model/info                │  • PSI drift monitor     │
    GET  /metrics   ◄───────────────│  • Prometheus telemetry  │
    GET  /health                    └──────────────────────────┘
                                                   │
                                    ┌──────────────┴───────────┐
                                    ▼                          ▼
                            ┌──────────────┐          ┌──────────────┐
                            │  Prometheus  │─────────►│   Grafana    │
                            └──────────────┘          └──────────────┘
```

---

## Measured results

Trained on 100,000 transactions with a 0.40% fraud rate, stratified 60/20/20 split.

### Model performance (held-out test set)

| Metric | Value |
|---|---|
| ROC-AUC | **0.948** |
| PR-AUC | **0.876** |
| Precision | **0.909** |
| Recall | **0.750** |
| F1 | **0.822** |
| Flag rate | 0.33% of traffic |

Accuracy is deliberately not reported. At a 0.40% fraud rate, a model predicting "legitimate" for every transaction scores 99.6% accuracy while catching zero fraud. **PR-AUC is the selection metric** — it stays discriminative under heavy class imbalance where ROC-AUC saturates.

### Serving performance

| Scenario | Mean | p95 | p99 |
|---|---|---|---|
| Model inference | **0.42 ms** | 0.51 ms | 0.66 ms |
| End-to-end HTTP | **1.48 ms** | 1.67 ms | 1.83 ms |

Batch throughput: **~12,000 rows/sec** at batch size 100 (0.0067 ms per row).

All well inside the 15 ms serving budget.

### Test suite

33 tests, **94% coverage**. CI blocks any merge that drops below 90%.

---

## Quick start

### 1. Install

```bash
git clone <your-repo-url>
cd fraud-detection-mlops

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Get data (optional)

Download [creditcard.csv](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) into `data/`.

If the file is absent, the pipeline generates a synthetic dataset with the same schema and class imbalance so everything runs immediately. Synthetic mode is recorded in the run metadata, so results are never mistaken for the real benchmark.

### 3. Train

```bash
python -m training.train
```

Trains four hyperparameter variants, logs each to MLflow, and promotes the best by validation PR-AUC to `models/champion.json`.

```
==============================================================================
VARIANT         VAL PR-AUC  TEST ROC-AUC   PRECISION    RECALL        F1
==============================================================================
aggressive          0.8761        0.9484      0.9091    0.7500    0.8219 *
deeper              0.8760        0.9363      0.8806    0.7375    0.8027
regularised         0.8732        0.9452      0.8889    0.7000    0.7832
baseline            0.8715        0.9445      0.8939    0.7375    0.8082
==============================================================================
* champion — promoted to models/champion.json
```

### 4. Serve

```bash
uvicorn app.main:app --reload
```

Interactive docs at http://localhost:8000/docs

### 5. Or run the full stack

```bash
docker compose up --build
```

| Service | URL |
|---|---|
| Inference API | http://localhost:8000/docs |
| MLflow UI | http://localhost:5000 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |

### 6. Benchmark

```bash
python -m scripts.benchmark -n 2000
```

### 7. Test

```bash
pytest tests/ --cov --cov-report=term-missing
```

---

## API

| Endpoint | Method | Description |
|---|---|---|
| `/predict` | POST | Score a single transaction |
| `/predict/batch` | POST | Score up to 1,000 transactions |
| `/drift` | GET | Current PSI per feature vs training baseline |
| `/model/info` | GET | Champion model lineage, hyperparameters, metrics |
| `/health` | GET | Readiness probe |
| `/metrics` | GET | Prometheus scrape endpoint |

### Example

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"Time": 42000.0, "Amount": 149.62, "V1": 0.0, "V2": 0.0, ... "V28": 0.0}'
```

```json
{
  "fraud_probability": 0.000363,
  "is_fraud": false,
  "threshold": 0.62,
  "risk_band": "LOW",
  "inference_latency_ms": 0.412,
  "model_version": "v1.0.0-aggressive"
}
```

---

## Engineering decisions

**Class imbalance via `scale_pos_weight`, not SMOTE.**
XGBoost's `scale_pos_weight` reweights the gradient contribution of positive examples directly. SMOTE synthesises interpolated minority samples, which for PCA-transformed features produces points that do not correspond to any real transaction. Reweighting is also cheaper and leaves the validation distribution untouched.

**PR-AUC for champion selection.**
With 0.4% positives, ROC-AUC compresses into a narrow band near 1.0 and stops distinguishing between models. PR-AUC responds to precision/recall trade-offs in the minority class, which is what actually matters.

**Decision threshold tuned on validation, never on test.**
The default 0.5 threshold is rarely optimal under imbalance. Tuning on validation and reporting on test keeps the held-out estimate honest.

**Model loaded once at startup.**
A cold XGBoost load costs tens of milliseconds. Loading per request would dominate the latency budget. The lifespan handler loads once; if the artefact is missing, the service starts degraded so `/health` can report the failure instead of crash-looping.

**Native XGBoost JSON, not pickle.**
Portable across library versions and avoids unpickling arbitrary bytes inside a serving process.

**PSI for drift, computed on a rolling window.**
Population Stability Index is the standard drift metric in credit risk. Thresholds follow industry convention: < 0.10 stable, 0.10–0.25 monitor, > 0.25 retrain.

---

## Project layout

```
fraud-detection-mlops/
├── app/
│   ├── main.py              FastAPI app, lifespan model loading
│   ├── config.py            Settings (env-overridable)
│   ├── schemas.py           Pydantic request/response models
│   ├── model_loader.py      Champion model wrapper
│   ├── metrics.py           Prometheus counters, histograms, gauges
│   ├── drift.py             PSI calculation + rolling window monitor
│   ├── state.py             Shared singletons
│   └── routers/
│       ├── predict.py       /predict, /predict/batch, /drift
│       └── health.py        /health, /model/info
├── training/
│   ├── data_loader.py       Loading, splitting, baseline stats
│   ├── train.py             MLflow-tracked training + champion selection
│   └── evaluate.py          Metrics + threshold tuning
├── tests/                   33 tests, 94% coverage
├── scripts/benchmark.py     Latency + throughput measurement
├── monitoring/              Prometheus + Grafana config
├── .github/workflows/ci.yml Tests, coverage gate, Docker smoke test
├── Dockerfile               Multi-stage, non-root
└── docker-compose.yml       API + MLflow + Prometheus + Grafana
```

---

## Prometheus metrics

| Metric | Type | Description |
|---|---|---|
| `fraud_predictions_total{outcome}` | Counter | Predictions by outcome |
| `fraud_inference_latency_seconds` | Histogram | Inference latency distribution |
| `fraud_probability_distribution` | Histogram | Score distribution |
| `fraud_feature_drift_psi{feature}` | Gauge | PSI per feature |
| `fraud_drifted_feature_count` | Gauge | Features exceeding the alert threshold |
| `fraud_model_roc_auc` | Gauge | Champion ROC-AUC |

Useful queries:

```promql
# p99 inference latency
histogram_quantile(0.99, rate(fraud_inference_latency_seconds_bucket[5m]))

# fraud flag rate
rate(fraud_predictions_total{outcome="fraud"}[5m])
  / rate(fraud_predictions_total[5m])

# features currently drifting
fraud_feature_drift_psi > 0.25
```

---

## What would change for production

- Model registry with staged promotion (staging → production) instead of a file on disk
- Async feature store lookups for real-time aggregates (velocity, device history)
- Shadow deployment to compare a challenger against the champion on live traffic
- Alertmanager rules firing on sustained PSI breach or latency regression
- Scheduled retraining triggered by drift rather than a fixed cadence
