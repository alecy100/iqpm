from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "iqPM"
    database_url: str = "postgresql+psycopg2://iqpm:iqpm@postgres:5432/iqpm"

    # Seeding + simulation
    seed_csv_path: Path = Path("/data/synthetic_timeseries.csv")
    ai4i_csv_path: Path = Path("/data/ai4i2020.csv")
    stream_autostart: bool = False
    stream_interval_seconds: float = 1.0  # wall-clock seconds between simulated steps
    stream_step_seconds: int = 30  # simulated seconds per step (matches generate_synthetic.py)

    # Labels. A label_horizon value is only knowable once the horizon has elapsed,
    # so labels are hidden from training until timestamp + label_reveal_lag_hours.
    label_horizon_hours: float = 24.0
    label_reveal_lag_hours: float = 24.0

    # Training + governance
    rolling_window_days: int = 5
    min_supervised_rows: int = 1000
    min_f1_for_promotion: float = 0.7
    automl_time_budget_seconds: int = 30
    anomaly_contamination: float = 0.02
    failure_probability_threshold: float = 0.5
    mlflow_tracking_uri: str = "http://mlflow:5000"

    # Drift monitor
    drift_monitor_enabled: bool = True
    drift_check_interval_seconds: int = 300
    drift_window_hours: float = 6.0
    # Calibrated against generate_synthetic.py's lifecycle walk: a 6h window normally spans part of a
    # healthy/degrading/failed cycle, which alone measures mean PSI ~3-8 against the training window.
    # The threshold is set above that baseline so it fires on an unusual shift, not the expected cycling.
    drift_threshold: float = 12.0

    # LLM
    groq_api_key: str | None = None
    llm_model: str = "openai/gpt-oss-120b"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
