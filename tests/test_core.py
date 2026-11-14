
import logging

from src.nexus.core.config import get_settings
from src.nexus.core.application import initialize_application


def test_application_settings():
    settings = get_settings()

    assert settings.APP_NAME == "NEXUS AI LAB"
    assert settings.APP_VERSION == "0.1.0"
    assert settings.PROJECT_ROOT.exists()


def test_required_directories_exist():
    settings = get_settings()

    directories = [
        settings.RAW_DATA_DIR,
        settings.PROCESSED_DATA_DIR,
        settings.EXTERNAL_DATA_DIR,
        settings.MODELS_DIR,
        settings.REPORTS_DIR,
        settings.LOGS_DIR,
    ]

    for directory in directories:
        assert directory.exists()
        assert directory.is_dir()


def test_application_initialization():
    app = initialize_application()

    assert app.settings.APP_NAME == "NEXUS AI LAB"
    assert app.logger.name == "nexus.application"



def test_repeated_initialization_no_duplicate_handlers():
    from logging.handlers import RotatingFileHandler

    def get_application_handlers():
        parent_logger = logging.getLogger("nexus")

        return [
            handler
            for handler in parent_logger.handlers
            if type(handler) in (logging.StreamHandler, RotatingFileHandler)
        ]

    first = initialize_application()
    handlers_before = get_application_handlers()

    second = initialize_application()
    handlers_after = get_application_handlers()

    assert first.logger is second.logger
    assert len(handlers_after) == 2
    assert handlers_before == handlers_after
