"""
Shared singletons.

Kept in their own module so routers can import them without creating a
circular dependency with app.main.
"""
from app.config import settings
from app.drift import DriftMonitor

drift_monitor = DriftMonitor(
    baseline_path=settings.baseline_stats_path,
    window_size=settings.DRIFT_WINDOW_SIZE,
)
