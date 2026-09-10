"""Centralised configuration. All values overridable via environment variables."""
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = two levels up from this file (app/config.py -> app/ -> root/)
ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Service ───────────────────────────────────────
    APP_NAME: str = "fraud-detection-api"
    APP_VERSION: str = "1.0.0"
    LOG_LEVEL: str = "INFO"

    # ── Model artefacts ───────────────────────────────
    MODEL_DIR: Path = ROOT_DIR / "models"
    MODEL_FILE: str = "champion.json"
    METADATA_FILE: str = "metadata.json"
    BASELINE_STATS_FILE: str = "baseline_stats.json"

    # ── Decision threshold ────────────────────────────
    # Probability above which a transaction is flagged as fraud.
    # Tuned on the validation set to maximise F1 — see training/train.py.
    FRAUD_THRESHOLD: float = 0.50

    # ── MLflow ────────────────────────────────────────
    # SQLite backend: the file store was deprecated in MLflow 3.x, and a DB
    # backend is required for the model registry and run comparison UI.
    MLFLOW_TRACKING_URI: str = f"sqlite:///{ROOT_DIR / 'mlflow.db'}"
    MLFLOW_ARTIFACT_ROOT: str = str(ROOT_DIR / "mlartifacts")
    MLFLOW_EXPERIMENT_NAME: str = "fraud-detection"

    # ── Drift monitoring ──────────────────────────────
    # Population Stability Index thresholds (industry standard):
    #   PSI < 0.10  -> no significant shift
    #   0.10-0.25   -> moderate shift, monitor
    #   PSI > 0.25  -> significant shift, retrain
    DRIFT_WINDOW_SIZE: int = 500
    PSI_WARN_THRESHOLD: float = 0.10
    PSI_ALERT_THRESHOLD: float = 0.25

    @property
    def model_path(self) -> Path:
        return self.MODEL_DIR / self.MODEL_FILE

    @property
    def metadata_path(self) -> Path:
        return self.MODEL_DIR / self.METADATA_FILE

    @property
    def baseline_stats_path(self) -> Path:
        return self.MODEL_DIR / self.BASELINE_STATS_FILE


settings = Settings()
