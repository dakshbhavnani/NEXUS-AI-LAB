
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# Project root:
# config.py -> core -> nexus -> src -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """
    Central configuration manager for NEXUS AI LAB.

    Values can be loaded from environment variables
    or the project's .env file.
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    PROJECT_ROOT: Path = PROJECT_ROOT

    # Application information
    APP_NAME: str = "NEXUS AI LAB"
    APP_VERSION: str = "0.1.0"
    APP_ENV: str = "development"
    DEBUG: bool = True

    # API configuration
    API_HOST: str = "127.0.0.1"
    API_PORT: int = Field(default=8000, ge=1, le=65535)

    # Logging configuration
    LOG_LEVEL: str = "INFO"

    # Storage directories
    DATA_DIR: Path = PROJECT_ROOT / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    EXTERNAL_DATA_DIR: Path = DATA_DIR / "external"

    ARTIFACTS_DIR: Path = PROJECT_ROOT / "artifacts"
    MODELS_DIR: Path = ARTIFACTS_DIR / "models"
    REPORTS_DIR: Path = ARTIFACTS_DIR / "reports"

    LOGS_DIR: Path = PROJECT_ROOT / "logs"


@lru_cache
def get_settings() -> Settings:
    """
    Return a cached settings instance.

    This avoids repeatedly loading configuration
    whenever another module requests application settings.
    """
    return Settings()
