from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "iqPM"
    database_url: str = "postgresql+psycopg2://iqpm:iqpm@postgres:5432/iqpm"
    csv_path: Path = Path("/data/synthetic_timeseries.csv")
    stream_interval_seconds: float = 1.0
    label_reveal_lag_hours: float = 6.0
    rolling_window_days: int = 5
    min_supervised_rows: int = 1000
    min_f1_for_promotion: float = 0.7
    model_dir: Path = Path("/models")
    mlflow_tracking_uri: str = "http://mlflow:5000"
    groq_api_key: str | None = None

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache
def get_settings() -> Settings:
    return Settings()
