"""
Train fraud detection models, track every run in MLflow, and promote the
best performer to champion.

Run:
    python -m training.train                 # default 4-variant sweep
    python -m training.train --quick         # single fast run
    python -m training.train --n-rows 200000 # larger synthetic dataset

Every run logs hyperparameters, dataset split sizes, metrics and the model
artefact to MLflow. The champion is selected by validation PR-AUC — the
right selection metric for imbalanced classification, since ROC-AUC can
look deceptively strong when the negative class dominates.
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import numpy as np
import xgboost as xgb

from app.config import settings, ROOT_DIR
from training.data_loader import (
    load_dataset,
    split_dataset,
    compute_baseline_stats,
    generate_synthetic_data,
)
from training.evaluate import compute_metrics, tune_threshold, format_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s")
logger = logging.getLogger(__name__)

DATA_PATH = ROOT_DIR / "data" / "creditcard.csv"


# ── Hyperparameter variants ───────────────────────────
# Deliberately spans a range of depth/regularisation so the MLflow
# comparison view shows a meaningful trade-off, not four near-identical runs.
VARIANTS: list[dict] = [
    {
        "name": "baseline",
        "max_depth": 4,
        "eta": 0.10,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "n_estimators": 200,
    },
    {
        "name": "deeper",
        "max_depth": 7,
        "eta": 0.10,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "n_estimators": 300,
    },
    {
        "name": "regularised",
        "max_depth": 5,
        "eta": 0.05,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "min_child_weight": 5,
        "reg_lambda": 3.0,
        "reg_alpha": 0.5,
        "n_estimators": 400,
    },
    {
        "name": "aggressive",
        "max_depth": 9,
        "eta": 0.15,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "min_child_weight": 1,
        "n_estimators": 250,
    },
]


def train_variant(variant: dict, data: dict, scale_pos_weight: float) -> tuple[xgb.Booster, dict, dict]:
    """
    Train one hyperparameter configuration and evaluate it.

    scale_pos_weight is the class-imbalance correction: it multiplies the
    gradient contribution of positive (fraud) examples so the model cannot
    minimise loss by ignoring them entirely.
    """
    params = {k: v for k, v in variant.items() if k not in ("name", "n_estimators")}
    params.update({
        "objective": "binary:logistic",
        "eval_metric": ["aucpr", "auc"],
        "scale_pos_weight": scale_pos_weight,
        "tree_method": "hist",
        "seed": 42,
    })

    dtrain = xgb.DMatrix(data["X_train"], label=data["y_train"], feature_names=data["feature_names"])
    dval = xgb.DMatrix(data["X_val"], label=data["y_val"], feature_names=data["feature_names"])
    dtest = xgb.DMatrix(data["X_test"], label=data["y_test"], feature_names=data["feature_names"])

    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=variant["n_estimators"],
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=30,
        verbose_eval=False,
    )

    val_proba = booster.predict(dval)
    test_proba = booster.predict(dtest)

    # Tune the decision threshold on validation only — never on test.
    best_threshold, _ = tune_threshold(data["y_val"].to_numpy(), val_proba, metric="f1")

    val_metrics = compute_metrics(data["y_val"].to_numpy(), val_proba, threshold=best_threshold)
    test_metrics = compute_metrics(data["y_test"].to_numpy(), test_proba, threshold=best_threshold)

    return booster, val_metrics, test_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train fraud detection models with MLflow tracking")
    parser.add_argument("--quick", action="store_true", help="Train only the baseline variant")
    parser.add_argument("--n-rows", type=int, default=100_000, help="Rows for synthetic data")
    parser.add_argument("--no-synthetic", action="store_true", help="Fail if real dataset is absent")
    args = parser.parse_args()

    # ── Load data ─────────────────────────────────────
    if DATA_PATH.exists():
        df, is_synthetic = load_dataset(DATA_PATH, allow_synthetic=not args.no_synthetic)
    else:
        if args.no_synthetic:
            raise SystemExit(f"{DATA_PATH} not found and --no-synthetic was set.")
        logger.warning("Real dataset absent — generating %s synthetic rows.", f"{args.n_rows:,}")
        df = generate_synthetic_data(n_rows=args.n_rows)
        is_synthetic = True

    fraud_rate = df["Class"].mean()
    logger.info(
        "Dataset: %s rows | %s fraud (%.4f%%) | source=%s",
        f"{len(df):,}",
        f"{int(df['Class'].sum()):,}",
        fraud_rate * 100,
        "synthetic" if is_synthetic else "kaggle",
    )

    data = split_dataset(df)
    logger.info(
        "Split: train=%s val=%s test=%s",
        f"{len(data['X_train']):,}", f"{len(data['X_val']):,}", f"{len(data['X_test']):,}",
    )

    # Class-imbalance correction ratio.
    n_neg = int((data["y_train"] == 0).sum())
    n_pos = int((data["y_train"] == 1).sum())
    scale_pos_weight = n_neg / max(n_pos, 1)
    logger.info("scale_pos_weight = %.1f  (neg=%s pos=%s)", scale_pos_weight, f"{n_neg:,}", n_pos)

    # ── MLflow setup ──────────────────────────────────
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    Path(settings.MLFLOW_ARTIFACT_ROOT).mkdir(parents=True, exist_ok=True)

    # Create the experiment with an explicit artifact root on first run.
    if mlflow.get_experiment_by_name(settings.MLFLOW_EXPERIMENT_NAME) is None:
        mlflow.create_experiment(
            settings.MLFLOW_EXPERIMENT_NAME,
            artifact_location=settings.MLFLOW_ARTIFACT_ROOT,
        )
    mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)

    variants = VARIANTS[:1] if args.quick else VARIANTS
    results = []

    for variant in variants:
        logger.info("── Training variant: %s ──", variant["name"])

        with mlflow.start_run(run_name=variant["name"]) as run:
            booster, val_metrics, test_metrics = train_variant(variant, data, scale_pos_weight)

            # Log everything needed to reproduce this run.
            mlflow.log_params({
                **{k: v for k, v in variant.items() if k != "name"},
                "variant": variant["name"],
                "scale_pos_weight": round(scale_pos_weight, 2),
                "objective": "binary:logistic",
                "best_iteration": booster.best_iteration,
            })
            mlflow.log_params({
                "train_rows": len(data["X_train"]),
                "val_rows": len(data["X_val"]),
                "test_rows": len(data["X_test"]),
                "n_features": len(data["feature_names"]),
                "fraud_rate_pct": round(fraud_rate * 100, 4),
                "data_source": "synthetic" if is_synthetic else "kaggle",
            })
            mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
            mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

            # Persist the booster as an MLflow artefact for lineage.
            model_file = ROOT_DIR / "models" / f"{variant['name']}.json"
            model_file.parent.mkdir(parents=True, exist_ok=True)
            booster.save_model(str(model_file))
            mlflow.log_artifact(str(model_file), artifact_path="model")

            logger.info(format_report(test_metrics, f"{variant['name']} — test set"))

            results.append({
                "name": variant["name"],
                "booster": booster,
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
                "hyperparameters": {k: v for k, v in variant.items() if k != "name"},
                "run_id": run.info.run_id,
            })

    # ── Champion selection ────────────────────────────
    # PR-AUC on validation is the selection criterion. ROC-AUC saturates
    # near 1.0 under heavy imbalance and stops discriminating between models.
    champion = max(results, key=lambda r: r["val_metrics"]["pr_auc"])
    logger.info(
        "Champion: %s (val PR-AUC %.4f, test ROC-AUC %.4f)",
        champion["name"],
        champion["val_metrics"]["pr_auc"],
        champion["test_metrics"]["roc_auc"],
    )

    models_dir = ROOT_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    champion["booster"].save_model(str(models_dir / settings.MODEL_FILE))

    metadata = {
        "model_version": f"v1.0.0-{champion['name']}",
        "algorithm": "XGBoost",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "mlflow_run_id": champion["run_id"],
        "feature_names": data["feature_names"],
        "n_features": len(data["feature_names"]),
        "hyperparameters": champion["hyperparameters"],
        "decision_threshold": champion["test_metrics"]["threshold"],
        "metrics": champion["test_metrics"],
        "validation_metrics": champion["val_metrics"],
        "training_rows": len(data["X_train"]),
        "fraud_rate_percent": round(fraud_rate * 100, 4),
        "data_source": "synthetic" if is_synthetic else "kaggle",
        "variants_compared": [r["name"] for r in results],
    }
    with open(models_dir / settings.METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)

    # Drift baseline captured from the training distribution.
    baseline = compute_baseline_stats(data["X_train"])
    with open(models_dir / settings.BASELINE_STATS_FILE, "w") as f:
        json.dump(baseline, f)

    logger.info("Champion written to %s", models_dir / settings.MODEL_FILE)
    logger.info("Metadata written to %s", models_dir / settings.METADATA_FILE)
    logger.info("Drift baseline written to %s", models_dir / settings.BASELINE_STATS_FILE)

    # ── Comparison summary ────────────────────────────
    print("\n" + "=" * 78)
    print(f"{'VARIANT':<14}{'VAL PR-AUC':>12}{'TEST ROC-AUC':>14}{'PRECISION':>12}{'RECALL':>10}{'F1':>10}")
    print("=" * 78)
    for r in sorted(results, key=lambda x: -x["val_metrics"]["pr_auc"]):
        m, v = r["test_metrics"], r["val_metrics"]
        star = " *" if r["name"] == champion["name"] else "  "
        print(
            f"{r['name']:<14}{v['pr_auc']:>12.4f}{m['roc_auc']:>14.4f}"
            f"{m['precision']:>12.4f}{m['recall']:>10.4f}{m['f1']:>10.4f}{star}"
        )
    print("=" * 78)
    print("* champion — promoted to models/champion.json\n")
    print(f"Inspect all runs:  mlflow ui --backend-store-uri {settings.MLFLOW_TRACKING_URI}\n")


if __name__ == "__main__":
    main()
