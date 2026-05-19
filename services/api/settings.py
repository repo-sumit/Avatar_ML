from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Single source of truth for Windows-side configuration.

    All values come from .env (loaded automatically). The most important one is
    COLAB_INFERENCE_URL — it is the single pointer to the remote inference
    worker. Swapping Colab for a paid GPU is an env-var change, nothing more.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    colab_inference_url: str = ""
    storage_dir: Path = Path("./storage")
    database_url: str = "sqlite:///./data/avatar_ml.sqlite"
    stun_servers: str = "stun:stun.l.google.com:19302"

    colab_http_timeout: int = 120
    colab_ws_recv_timeout: int = 30
    colab_healthcheck_interval: int = 10


settings = Settings()
