"""
Feature drift monitoring using the Population Stability Index (PSI).

PSI compares the distribution of live inference traffic against the
distribution seen during training. It is the standard drift metric in
credit risk and fraud modelling.

    PSI = sum over bins of  (actual% - expected%) * ln(actual% / expected%)

Interpretation (industry convention):
    PSI < 0.10   -> no significant population change
    0.10 - 0.25  -> moderate shift, worth monitoring
    PSI > 0.25   -> significant shift, model likely needs retraining
"""
from __future__ import annotations

import json
import threading
from collections import deque
from pathlib import Path

import numpy as np

from app.config import settings
from app.metrics import FEATURE_DRIFT_PSI, DRIFTED_FEATURES

# Small constant preventing log(0) / division-by-zero when a bin is empty.
_EPSILON = 1e-6


def calculate_psi(
    baseline_percents: list[float],
    live_values: np.ndarray,
    bin_edges: list[float],
) -> float:
    """
    Compute PSI for one feature.

    Args:
        baseline_percents: proportion of training rows falling in each bin.
        live_values: observed values from live inference traffic.
        bin_edges: bin boundaries computed at training time.
    """
    if live_values.size == 0:
        return 0.0

    live_counts, _ = np.histogram(live_values, bins=bin_edges)
    live_percents = live_counts / max(live_counts.sum(), 1)

    expected = np.asarray(baseline_percents, dtype=float) + _EPSILON
    actual = live_percents.astype(float) + _EPSILON

    psi = float(np.sum((actual - expected) * np.log(actual / expected)))
    return round(psi, 4)


class DriftMonitor:
    """
    Maintains a rolling window of recent inference inputs and computes
    PSI per feature against the training baseline.

    Thread-safe: FastAPI serves requests from a thread pool, so the
    observation buffer is guarded by a lock.
    """

    def __init__(self, baseline_path: Path, window_size: int) -> None:
        self.window_size = window_size
        self._lock = threading.Lock()
        self._buffer: deque[list[float]] = deque(maxlen=window_size)
        self._samples_seen = 0

        self.baseline: dict = {}
        self.feature_names: list[str] = []
        self.enabled = False

        if baseline_path.exists():
            with open(baseline_path) as f:
                self.baseline = json.load(f)
            self.feature_names = self.baseline.get("feature_names", [])
            self.enabled = True

    def observe(self, feature_vector: list[float]) -> None:
        """Record one inference input into the rolling window."""
        if not self.enabled:
            return
        with self._lock:
            self._buffer.append(feature_vector)
            self._samples_seen += 1

        # Recompute PSI once the window is full, then every window_size samples.
        if self._samples_seen % self.window_size == 0:
            self.compute()

    def compute(self) -> dict:
        """Compute PSI for every feature and publish to Prometheus."""
        with self._lock:
            if len(self._buffer) < self.window_size:
                return {
                    "ready": False,
                    "samples_observed": self._samples_seen,
                    "features": [],
                }
            matrix = np.array(self._buffer, dtype=float)

        results = []
        drifted = 0
        stats = self.baseline.get("features", {})

        for idx, name in enumerate(self.feature_names):
            fstat = stats.get(name)
            if not fstat:
                continue

            psi = calculate_psi(
                baseline_percents=fstat["bin_percents"],
                live_values=matrix[:, idx],
                bin_edges=fstat["bin_edges"],
            )

            if psi > settings.PSI_ALERT_THRESHOLD:
                status = "ALERT"
                drifted += 1
            elif psi > settings.PSI_WARN_THRESHOLD:
                status = "WARN"
            else:
                status = "STABLE"

            FEATURE_DRIFT_PSI.labels(feature=name).set(psi)
            results.append({"feature": name, "psi": psi, "status": status})

        DRIFTED_FEATURES.set(drifted)

        return {
            "ready": True,
            "samples_observed": self._samples_seen,
            "features": results,
            "drifted_feature_count": drifted,
        }

    @property
    def samples_observed(self) -> int:
        return self._samples_seen
