"""Logging helpers for the detection layer.

The scripts talk to the user with print(), but the detection layer also has
to report the backend it picked and any runtime fallback, so it uses the
logging module and writes to stdout. The launcher captures stdout, so the
messages still show up in its log panel.
"""

import logging
import sys

LOGGER_NAME = "face_detection"


def get_logger() -> logging.Logger:
    """Return the shared logger without touching its handlers."""
    return logging.getLogger(LOGGER_NAME)


def configure_logging() -> logging.Logger:
    """Attach a stdout handler once and let the logger emit INFO and above."""
    logger = get_logger()
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger
