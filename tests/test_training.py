"""Unit tests for data handling, evaluation metrics and drift detection."""
import numpy as np
import pytest

from training.data_loader import (
    generate_synthetic_data,
    split_dataset,
    compute_baseline_stats,
    FEATURE_COLUMNS,
)
from training.evaluate import compute_metrics, tune_threshold
from app.drift import calculate_psi


# ── Data generation ───────────────────────────────────
def test_synthetic_data_has_expected_schema():
    df = generate_synthetic_data(n_rows=5_000)
    assert len(df) == 5_000
    assert set(FEATURE_COLUMNS + ["Class"]) == set(df.columns)


def test_synthetic_data_preserves_class_imbalance():
    df = generate_synthetic_data(n_rows=20_000, fraud_rate=0.004)
    rate = df["Class"].mean()
    assert 0.002 < rate < 0.007, f"fraud rate {rate:.4f} outside expected band"


def test_synthetic_data_is_reproducible():
    a = generate_synthetic_data(n_rows=2_000, seed=7)
    b = generate_synthetic_data(n_rows=2_000, seed=7)
    assert a.equals(b)


# ── Splitting ─────────────────────────────────────────
def test_split_preserves_class_ratio_across_folds():
    """Stratification matters: an unstratified split can leave zero positives in test."""
    df = generate_synthetic_data(n_rows=20_000)
    data = split_dataset(df)

    overall = df["Class"].mean()
    for fold in ("y_train", "y_val", "y_test"):
        assert data[fold].mean() == pytest.approx(overall, abs=0.003)


def test_split_has_no_row_leakage():
    df = generate_synthetic_data(n_rows=10_000)
    data = split_dataset(df)
    total = len(data["X_train"]) + len(data["X_val"]) + len(data["X_test"])
    assert total == len(df)

    train_idx = set(data["X_train"].index)
    test_idx = set(data["X_test"].index)
    assert train_idx.isdisjoint(test_idx)


# ── Metrics ───────────────────────────────────────────
def test_perfect_predictions_score_one():
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.01, 0.02, 0.98, 0.99])
    m = compute_metrics(y_true, y_proba, threshold=0.5)
    assert m["roc_auc"] == 1.0
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0


def test_confusion_matrix_cells_sum_to_sample_count():
    y_true = np.array([0, 0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.9, 0.8, 0.3])
    m = compute_metrics(y_true, y_proba, threshold=0.5)
    total = m["true_positives"] + m["false_positives"] + m["true_negatives"] + m["false_negatives"]
    assert total == len(y_true)


def test_accuracy_is_not_reported_as_headline_metric():
    """Guards against reintroducing accuracy, which is misleading under imbalance."""
    y_true = np.array([0] * 99 + [1])
    y_proba = np.array([0.01] * 100)
    m = compute_metrics(y_true, y_proba)
    assert "accuracy" not in m
    assert m["recall"] == 0.0  # caught nothing despite 99% accuracy


def test_threshold_tuning_beats_default():
    rng = np.random.default_rng(0)
    y_true = np.array([0] * 950 + [1] * 50)
    y_proba = np.concatenate([rng.beta(2, 8, 950), rng.beta(6, 4, 50)])

    best_t, best_f1 = tune_threshold(y_true, y_proba, metric="f1")
    default_f1 = compute_metrics(y_true, y_proba, threshold=0.5)["f1"]

    assert 0.0 < best_t < 1.0
    assert best_f1 >= default_f1


# ── Drift / PSI ───────────────────────────────────────
def test_psi_near_zero_for_identical_distributions():
    rng = np.random.default_rng(42)
    baseline_values = rng.standard_normal(10_000)
    edges = np.quantile(baseline_values, np.linspace(0, 1, 11))
    edges[0], edges[-1] = -np.inf, np.inf
    counts, _ = np.histogram(baseline_values, bins=edges)
    percents = (counts / counts.sum()).tolist()

    live = rng.standard_normal(2_000)
    psi = calculate_psi(percents, live, edges.tolist())
    assert psi < 0.10, f"PSI {psi} should indicate no drift"


def test_psi_flags_shifted_distribution():
    rng = np.random.default_rng(42)
    baseline_values = rng.standard_normal(10_000)
    edges = np.quantile(baseline_values, np.linspace(0, 1, 11))
    edges[0], edges[-1] = -np.inf, np.inf
    counts, _ = np.histogram(baseline_values, bins=edges)
    percents = (counts / counts.sum()).tolist()

    shifted = rng.standard_normal(2_000) + 3.0
    psi = calculate_psi(percents, shifted, edges.tolist())
    assert psi > 0.25, f"PSI {psi} should exceed the alert threshold"


def test_psi_handles_empty_input():
    assert calculate_psi([0.5, 0.5], np.array([]), [-np.inf, 0.0, np.inf]) == 0.0


# ── Baseline stats ────────────────────────────────────
def test_baseline_stats_cover_every_feature():
    df = generate_synthetic_data(n_rows=5_000)
    data = split_dataset(df)
    stats = compute_baseline_stats(data["X_train"])

    assert stats["feature_names"] == FEATURE_COLUMNS
    for feature in FEATURE_COLUMNS:
        assert feature in stats["features"]
        assert sum(stats["features"][feature]["bin_percents"]) == pytest.approx(1.0, abs=1e-6)
