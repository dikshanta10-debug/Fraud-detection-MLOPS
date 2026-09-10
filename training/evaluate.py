"""
Evaluation metrics for a severely imbalanced binary classifier.

Accuracy is deliberately excluded from the headline metrics: with a 0.17%
fraud rate, a model that predicts "legitimate" for every transaction scores
99.83% accuracy while catching zero fraud. ROC-AUC and PR-AUC are the
metrics that actually reflect performance here.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5) -> dict:
    """Full metric suite at a given decision threshold."""
    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        # Threshold-independent — the primary model quality signals.
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        # Threshold-dependent.
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        # Raw confusion matrix cells — what a fraud analyst actually cares about.
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        # Operational rate: proportion of all traffic sent for manual review.
        "flag_rate": float(y_pred.mean()),
        "threshold": float(threshold),
    }


def tune_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric: str = "f1",
    n_steps: int = 99,
) -> tuple[float, float]:
    """
    Sweep decision thresholds and return the one maximising the chosen metric.

    The default 0.5 threshold is rarely optimal for imbalanced problems —
    tuning on the validation set typically buys a meaningful recall gain at
    an acceptable false-positive cost.

    Returns:
        (best_threshold, best_score)
    """
    best_threshold, best_score = 0.5, -1.0

    for t in np.linspace(0.01, 0.99, n_steps):
        scores = compute_metrics(y_true, y_proba, threshold=float(t))
        if scores[metric] > best_score:
            best_score, best_threshold = scores[metric], float(t)

    return round(best_threshold, 4), round(best_score, 4)


def format_report(metrics: dict, title: str = "Evaluation") -> str:
    """Human-readable metric summary for console output."""
    return (
        f"\n{title}\n"
        f"{'-' * len(title)}\n"
        f"  ROC-AUC          {metrics['roc_auc']:.4f}\n"
        f"  PR-AUC           {metrics['pr_auc']:.4f}\n"
        f"  Precision        {metrics['precision']:.4f}\n"
        f"  Recall           {metrics['recall']:.4f}\n"
        f"  F1               {metrics['f1']:.4f}\n"
        f"  Threshold        {metrics['threshold']:.4f}\n"
        f"  Flag rate        {metrics['flag_rate'] * 100:.3f}% of traffic\n"
        f"  TP {metrics['true_positives']}  FP {metrics['false_positives']}  "
        f"FN {metrics['false_negatives']}  TN {metrics['true_negatives']}\n"
    )
