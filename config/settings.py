"""Typed settings loaded from config.yaml + .env overrides."""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic_settings import BaseSettings


ROOT_DIR = Path(__file__).parent.parent


def _load_yaml() -> dict:
    config_path = ROOT_DIR / "config" / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


class Settings(BaseSettings):
    """Runtime settings. Values can be overridden via environment variables."""

    # Reddit credentials - set in .env or environment
    reddit_client_id: str = ""
    reddit_client_secret: str = ""
    reddit_user_agent: str = "f1_prediction_bot/1.0"

    # Derived project paths (resolved relative to project root)
    root_dir: Path = ROOT_DIR
    data_raw_dir: Path = ROOT_DIR / "data" / "raw"
    data_processed_dir: Path = ROOT_DIR / "data" / "processed"
    data_cache_dir: Path = ROOT_DIR / "data" / "cache"
    data_predictions_dir: Path = ROOT_DIR / "data" / "predictions"
    models_dir: Path = ROOT_DIR / "models"

    # UI configuration (env-overridable)
    ui_density_default: str = "dense"
    ui_enable_live_widgets: bool = False
    ui_task_poll_seconds: int = 5
    ui_max_visible_blocks: int = 6

    class Config:
        env_file = ROOT_DIR / ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

    def ensure_dirs(self) -> None:
        """Create all data directories if they don't exist."""
        for d in [
            self.data_raw_dir,
            self.data_processed_dir,
            self.data_cache_dir,
            self.data_predictions_dir,
            self.models_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)


# Singleton
settings = Settings()
yaml_cfg = _load_yaml()



