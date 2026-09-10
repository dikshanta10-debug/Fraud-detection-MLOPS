"""Tests for the rolling-window DriftMonitor."""
import json

import numpy as np
import pytest

from app.drift import DriftMonitor
from training.data_loader import generate_synthetic_data, split_dataset, compute_baseline_stats


@pytest.fixture(scope="module")
def baseline_file(tmp_path_factory):
    """Write a real drift baseline derived from synthetic training data."""
    df = generate_synthetic_data(n_rows=8_000)
    data = split_dataset(df)
    stats = compute_baseline_stats(data["X_train"])

    path = tmp_path_factory.mktemp("models") / "baseline_stats.json"
    path.write_text(json.dumps(stats))
    return path


def test_monitor_disabled_when_baseline_missing(tmp_path):
    monitor = DriftMonitor(baseline_path=tmp_path / "nope.json", window_size=10)
    assert monitor.enabled is False

    monitor.observe([0.0] * 30)  # must not raise
    assert monitor.samples_observed == 0


def test_monitor_not_ready_before_window_fills(baseline_file):
    monitor = DriftMonitor(baseline_path=baseline_file, window_size=50)
    for _ in range(10):
        monitor.observe([0.0] * 30)

    result = monitor.compute()
    assert result["ready"] is False
    assert result["samples_observed"] == 10


def test_monitor_reports_stable_for_in_distribution_traffic(baseline_file):
    monitor = DriftMonitor(baseline_path=baseline_file, window_size=200)
    rng = np.random.default_rng(42)

    # Mimic the training distribution: Time uniform, Amount lognormal, V ~ N(0,1).
    for _ in range(200):
        vector = [
            float(rng.uniform(0, 172_800)),
            float(rng.lognormal(3.2, 1.1)),
            *rng.standard_normal(28).tolist(),
        ]
        monitor.observe(vector)

    result = monitor.compute()
    assert result["ready"] is True
    assert result["drifted_feature_count"] <= 3, "in-distribution traffic should not alert broadly"


def test_monitor_detects_shifted_traffic(baseline_file):
    monitor = DriftMonitor(baseline_path=baseline_file, window_size=200)
    rng = np.random.default_rng(42)

    # Every V feature shifted by 4 sigma — an unmistakable population change.
    for _ in range(200):
        vector = [
            float(rng.uniform(0, 172_800)),
            float(rng.lognormal(3.2, 1.1)),
            *(rng.standard_normal(28) + 4.0).tolist(),
        ]
        monitor.observe(vector)

    result = monitor.compute()
    assert result["ready"] is True
    assert result["drifted_feature_count"] >= 20, "large shift should trip most features"


def test_window_is_bounded(baseline_file):
    """The buffer must not grow without limit under sustained traffic."""
    monitor = DriftMonitor(baseline_path=baseline_file, window_size=100)
    for _ in range(500):
        monitor.observe([0.0] * 30)

    assert len(monitor._buffer) == 100
    assert monitor.samples_observed == 500
