
import logging
from logging.handlers import RotatingFileHandler

from src.nexus.core.config import get_settings


def setup_logging() -> logging.Logger:
    """
    Configure centralized logging for NEXUS AI LAB.

    Logs are written to:
    1. The terminal (console)
    2. logs/nexus.log (rotating log file)
    """

    settings = get_settings()

    # Create the log directory if it does not exist.
    settings.LOGS_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("nexus")

    # Validate the configured logging level.
    level_name = settings.LOG_LEVEL.upper()
    level = getattr(logging, level_name, None)

    if not isinstance(level, int):
        raise ValueError(
            f"Invalid LOG_LEVEL: {settings.LOG_LEVEL}"
        )

    logger.setLevel(level)
    logger.propagate = False

    # Prevent duplicate handlers if setup is called repeatedly.
    if not getattr(logger, "_nexus_configured", False):

        formatter = logging.Formatter(
            fmt=(
                "%(asctime)s | %(levelname)s | "
                "%(name)s | %(message)s"
            ),
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Console logging.
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        # Rotating file logging.
        file_handler = RotatingFileHandler(
            filename=settings.LOGS_DIR / "nexus.log",
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

        logger._nexus_configured = True

    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """
    Get the main NEXUS logger or a named child logger.

    Example:
        get_logger("agents.dataset_discovery")
    """

    setup_logging()

    if name:
        return logging.getLogger(f"nexus.{name}")

    return logging.getLogger("nexus")
