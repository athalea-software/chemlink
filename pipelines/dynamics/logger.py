import logging
import sys
from pathlib import Path
from typing import Optional


LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(lineno)d | %(message)s"


class _ConsoleOnlyFilter(logging.Filter):
    """Hide records explicitly reserved for persistent diagnostic logs."""

    def filter(self, record: logging.LogRecord) -> bool:
        return not bool(getattr(record, "file_only", False))


def get_step_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(LOG_FORMAT)

    if not any(getattr(handler, "_step_stream_handler", False) for handler in logger.handlers):
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        stream_handler.addFilter(_ConsoleOnlyFilter())
        stream_handler._step_stream_handler = True
        logger.addHandler(stream_handler)

    if log_file:
        file_path = Path(log_file).expanduser().resolve()
        file_path.parent.mkdir(parents=True, exist_ok=True)
        if not any(getattr(handler, "_step_log_path", None) == str(file_path) for handler in logger.handlers):
            file_handler = logging.FileHandler(file_path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            file_handler._step_log_path = str(file_path)
            logger.addHandler(file_handler)

    logger.propagate = False
    return logger
