
from dataclasses import dataclass
import logging

from src.nexus.core.config import Settings, get_settings
from src.nexus.core.logging_config import get_logger


@dataclass(frozen=True)
class ApplicationContext:
    """Shared application resources for NEXUS components."""

    settings: Settings
    logger: logging.Logger


def initialize_application() -> ApplicationContext:
    """
    Initialize the NEXUS application foundation.

    Loads settings, prepares required directories,
    initializes logging, and returns shared resources.
    """

    settings = get_settings()

    # Prepare required application directories.
    required_directories = [
        settings.RAW_DATA_DIR,
        settings.PROCESSED_DATA_DIR,
        settings.EXTERNAL_DATA_DIR,
        settings.MODELS_DIR,
        settings.REPORTS_DIR,
        settings.LOGS_DIR,
    ]

    for directory in required_directories:
        directory.mkdir(parents=True, exist_ok=True)

    # Initialize the central application logger.
    logger = get_logger("application")

    logger.info(
        "%s v%s initialized successfully | Environment: %s",
        settings.APP_NAME,
        settings.APP_VERSION,
        settings.APP_ENV,
    )

    return ApplicationContext(
        settings=settings,
        logger=logger,
    )
