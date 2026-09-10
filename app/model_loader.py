"""
Loads the champion model into memory once at application startup.

The model is loaded from disk in XGBoost's native JSON format rather than
pickled, which keeps it portable across library versions and avoids the
security concerns of unpickling arbitrary files in a serving process.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import xgboost as xgb

from app.config import settings

logger = logging.getLogger(__name__)


class ChampionModel:
    """Wraps the loaded booster plus the metadata needed to serve it."""

    def __init__(self) -> None:
        self.booster: xgb.Booster | None = None
        self.metadata: dict = {}
        self.feature_names: list[str] = []
        self.is_loaded = False

    def load(self, model_path: Path, metadata_path: Path) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"No champion model at {model_path}. "
                "Run `python -m training.train` before starting the API."
            )

        self.booster = xgb.Booster()
        self.booster.load_model(str(model_path))

        if metadata_path.exists():
            with open(metadata_path) as f:
                self.metadata = json.load(f)
            self.feature_names = self.metadata.get("feature_names", [])
        else:
            logger.warning("Metadata file missing — serving without version info.")
            self.metadata = {}

        self.is_loaded = True
        logger.info(
            "Loaded champion model version=%s roc_auc=%.4f",
            self.version,
            self.roc_auc,
        )

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """
        Score one or more rows.

        Args:
            features: shape (n_samples, n_features), column order must match
                      self.feature_names.
        Returns:
            Array of fraud probabilities, shape (n_samples,).
        """
        if not self.is_loaded or self.booster is None:
            raise RuntimeError("Model not loaded")

        dmatrix = xgb.DMatrix(features, feature_names=self.feature_names)
        return self.booster.predict(dmatrix)

    # ── Convenience accessors ─────────────────────────
    @property
    def version(self) -> str:
        return self.metadata.get("model_version", "unknown")

    @property
    def algorithm(self) -> str:
        return self.metadata.get("algorithm", "xgboost")

    @property
    def roc_auc(self) -> float:
        return float(self.metadata.get("metrics", {}).get("roc_auc", 0.0))

    @property
    def threshold(self) -> float:
        """Decision threshold tuned during training; falls back to config."""
        return float(self.metadata.get("decision_threshold", settings.FRAUD_THRESHOLD))


# Module-level singleton, populated by the FastAPI lifespan handler.
champion = ChampionModel()


def risk_band(probability: float) -> str:
    """Bucket a probability into a human-readable risk level."""
    if probability >= 0.80:
        return "HIGH"
    if probability >= 0.40:
        return "MEDIUM"
    return "LOW"
