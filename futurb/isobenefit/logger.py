from __future__ import annotations

import logging
import logging.config
import os
import sys

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.path.pardir)
sys.path.insert(0, BASE_DIR)
LOGS_DIR = os.path.join(BASE_DIR, "logs")
LOG_FILENAME = "isobenefit-cities.log"

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {"simple": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"}},
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "simple",
            "stream": "ext://sys.stdout",
        },
        "general_file": {
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "formatter": "simple",
            "filename": os.path.join(LOGS_DIR, LOG_FILENAME),
        },
    },
    "loggers": {
        "isobenefit-cities": {
            "handlers": ["console", "general_file"],
            "level": "DEBUG",
        }
    },
}


def configure_logging(console_only: bool = False) -> None:
    """ """
    if console_only:
        logging.config.dictConfig(LOGGING)
        return

    if not os.path.exists(LOGS_DIR):
        os.makedirs(LOGS_DIR)

    if not os.path.exists(os.path.join(LOGS_DIR, LOG_FILENAME)):
        open(os.path.join(LOGS_DIR, LOG_FILENAME), "a").close()

    logging.config.dictConfig(LOGGING)


def get_logger() -> logging.Logger:
    return logging.getLogger(LOG_FILENAME)
