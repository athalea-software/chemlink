"""Logging configuration for ChemLink project."""

import logging
import os
import sys


class _ConsoleFormatter(logging.Formatter):
    """Render bracketed levels and highlight informational messages."""

    # True-colour blue avoids terminal-theme remapping. Only the level name
    # receives this colour.
    _INFO_COLOR = "\033[38;2;0;128;255m"
    _RESET = "\033[0m"

    def __init__(self, use_color: bool) -> None:
        super().__init__()
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        line = f"[{record.levelname}] {record.getMessage()}"
        if self.use_color and record.levelno == logging.INFO:
            return f"[{self._INFO_COLOR}{record.levelname}{self._RESET}] {record.getMessage()}"
        return line


def setup_logger(name: str, level: int = logging.WARNING) -> logging.Logger:
    """Configure and return a logger with consistent formatting.
    
    Args:
        name: Name of the logger (typically __name__)
        level: Logging level (default: WARNING)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers if logger already configured
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    # Console handler with simple format
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    
    use_color = bool(getattr(sys.stdout, "isatty", lambda: False)())
    use_color = use_color and not os.environ.get("NO_COLOR")
    formatter = _ConsoleFormatter(use_color=use_color)
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get existing or create new logger.
    
    Args:
        name: Name of the logger
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)
