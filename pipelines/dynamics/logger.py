"""Step loggers for the dynamics pipeline.

Each step logs to the console (ChemLink's shared format, above any live
progress bars) and to the run's ``simulation.log`` (full detail, aligned,
nothing truncated). Use ``extra={"file_only": True}`` for diagnostics that
belong in the log file but would only be noise on screen.
"""
import logging
import warnings
from contextlib import contextmanager
from typing import Optional

from utils.logger import (
    ConsoleHandler,
    ConsoleOnlyFilter as _ConsoleOnlyFilter,
    FileFormatter,
    attach_file,
)

# Kept for anything that still formats records itself; the file handler uses
# FileFormatter, which produces the same fields in aligned columns.
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(lineno)d | %(message)s"

__all__ = ["get_step_logger", "warnings_to_log", "_ConsoleOnlyFilter", "FileFormatter", "LOG_FORMAT"]


@contextmanager
def warnings_to_log(logger):
    """Persist Python warnings from a preparation step without printing them."""
    with warnings.catch_warnings():
        warnings.simplefilter("always")

        def log_warning(message, category, filename, lineno, file=None, line=None):
            logger.warning(
                "%s:%s: %s: %s", filename, lineno, category.__name__, message,
                extra={"file_only": True},
            )

        warnings.showwarning = log_warning
        yield


def get_step_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not any(getattr(handler, "_step_stream_handler", False) for handler in logger.handlers):
        stream_handler = ConsoleHandler()
        stream_handler._step_stream_handler = True
        logger.addHandler(stream_handler)

    if log_file:
        attach_file(logger, log_file)

    logger.propagate = False
    return logger
