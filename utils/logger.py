"""Logging for ChemLink — one console format, one file format, everywhere.

Console (via the shared Rich console, so lines land cleanly above any live
progress bars)::

    11:42:03  • INFO   topology       Generated topology with AMBER99SB-ILDN
    11:42:04  ▲ WARN   ions           Neutralization added 3 NA ions
    11:42:05  ✗ ERROR  energy-min     Energy Minimization failed:
                                      │ Fatal error: …
                                      │ … 38 more line(s) — full output in simulation.log

File (plain text, aligned, nothing truncated, one session banner per run)::

    ══ ChemLink 2.0.0 · session 2026-10-14 11:42:03 · host node01 · pid 4121 ══
    2026-10-14 11:42:03.120  INFO      dynamics.topology:212        Generated topology …
    2026-10-14 11:42:05.884  ERROR     dynamics.energy_min:101      Energy Minimization failed:
                                                                    │ Fatal error: …

Records carrying ``extra={"file_only": True}`` never reach the console.
Console verbosity is global (``set_console_level``) and driven by the CLI's
``--verbose`` / ``--quiet`` flags or ``CHEMLINK_LOG_LEVEL``.
"""
from __future__ import annotations

import logging
import os
import re
import socket
from datetime import datetime

try:
    from .theme import get_console, get_err_console, register_aliases, shared
    from .style import format_log_line
except ImportError:  # pragma: no cover
    from theme import get_console, get_err_console, register_aliases, shared  # type: ignore
    from style import format_log_line  # type: ignore

register_aliases(__name__)

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
VERSION = "2.0.0"


# ── Global console threshold ──────────────────────────────────────────────────

def _initial_level() -> int:
    name = os.environ.get("CHEMLINK_LOG_LEVEL", "").strip().upper()
    return getattr(logging, name, logging.INFO) if name else logging.INFO


class _LevelBox:
    def __init__(self) -> None:
        self.level = _initial_level()


def _level_box() -> _LevelBox:
    return shared("console_log_level", _LevelBox)


def set_console_level(level: int) -> None:
    """Minimum level any ChemLink console handler prints (files keep everything)."""
    _level_box().level = level


def console_level() -> int:
    return _level_box().level


# ── Naming ────────────────────────────────────────────────────────────────────

_NAME_PREFIXES = ("chemlink.", "pipelines.", "utils.", "adapters.", "hpc.")


def short_name(name: str) -> str:
    """``chemlink.pipelines.dynamics.steps.energy_min`` → ``dynamics.energy_min``."""
    for prefix in _NAME_PREFIXES:
        if name.startswith(prefix):
            name = name[len(prefix):]
    for prefix in _NAME_PREFIXES:
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name.replace(".steps.", ".").replace("docking.peptide_docking.", "peptide.").replace("peptide_docking.", "peptide.")


def component_name(name: str) -> str:
    """The last segment, hyphenated, for the console's component column."""
    last = short_name(name).rsplit(".", 1)[-1]
    return last.replace("_", "-")


# ── Filters & handlers ────────────────────────────────────────────────────────

class ConsoleOnlyFilter(logging.Filter):
    """Drop records explicitly reserved for persistent diagnostic logs."""

    def filter(self, record: logging.LogRecord) -> bool:
        return not bool(getattr(record, "file_only", False))


class ConsoleHandler(logging.Handler):
    """Pretty, colour-aware console output through the shared Rich console.

    The stream is resolved at emit time (never captured at construction),
    so redirected stdout and live progress regions both work.
    """

    _chemlink_console = True

    def __init__(self, level: int = logging.NOTSET, *, log_file_hint: str = "", stderr: bool = False) -> None:
        super().__init__(level)
        self.log_file_hint = log_file_hint
        self.stderr = stderr
        self.addFilter(ConsoleOnlyFilter())

    def emit(self, record: logging.LogRecord) -> None:
        if record.levelno < console_level():
            return
        try:
            message = record.getMessage()
            if record.exc_info and record.levelno >= logging.ERROR:
                exc_type, exc, _tb = record.exc_info
                if exc is not None:
                    message += f"\n{exc_type.__name__ if exc_type else 'Error'}: {exc}"
            message = _ANSI_RE.sub("", message)
            hint = f"full output in {os.path.basename(self.log_file_hint)}" if self.log_file_hint else ""
            console = get_err_console() if self.stderr else get_console()
            line = format_log_line(
                record.levelname,
                message,
                component=getattr(record, "component", "") or component_name(record.name),
                when=datetime.fromtimestamp(record.created),
                show_time=console.width >= 70,
                more_hint=hint,
            )
            console.print(line, soft_wrap=True)
        except Exception:  # pragma: no cover — logging must never raise
            self.handleError(record)


class FileFormatter(logging.Formatter):
    """Aligned plain-text records; continuation lines hang under the message."""

    def __init__(self) -> None:
        super().__init__()

    def format(self, record: logging.LogRecord) -> str:
        stamp = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")
        stamp += f".{int(record.msecs):03d}"
        where = f"{short_name(record.name)}:{record.lineno}"
        head = f"{stamp}  {record.levelname:<8}  {where:<30} "
        message = _ANSI_RE.sub("", record.getMessage())
        if record.exc_info:
            message += "\n" + self.formatException(record.exc_info)
        if record.stack_info:
            message += "\n" + self.formatStack(record.stack_info)
        lines = message.splitlines() or [""]
        pad = " " * len(head)
        out = head + lines[0]
        for line in lines[1:]:
            out += f"\n{pad}│ {line}"
        return out


def _session_banner() -> str:
    try:
        host = socket.gethostname()
    except Exception:  # pragma: no cover
        host = "?"
    text = (
        f" ChemLink {VERSION} · session {datetime.now():%Y-%m-%d %H:%M:%S}"
        f" · host {host} · pid {os.getpid()} "
    )
    return "\n" + "══" + text + "═" * max(4, 96 - len(text)) + "\n"


class SessionFileHandler(logging.FileHandler):
    """A ``FileHandler`` that opens each run with a session banner so appended
    logs (resumes, re-analyses) stay easy to navigate."""

    def __init__(self, path: str, *, encoding: str = "utf-8") -> None:
        super().__init__(path, encoding=encoding)
        self.setFormatter(FileFormatter())
        banners = shared("log_session_banners", set)
        if path not in banners:
            banners.add(path)
            try:
                self.stream.write(_session_banner())
                self.stream.flush()
            except Exception:  # pragma: no cover
                pass


# ── Back-compat formatter name ────────────────────────────────────────────────

class _ConsoleFormatter(logging.Formatter):
    """Kept for imports from older code: a plain ``LEVEL  message`` format."""

    def __init__(self, use_color: bool = False) -> None:
        super().__init__()
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        return f"{record.levelname:<7} {record.getMessage()}"


# ── Public API ────────────────────────────────────────────────────────────────

def setup_logger(name: str, level: int = logging.WARNING) -> logging.Logger:
    """A module logger that prints through the ChemLink console format.

    ``level`` is the logger's own threshold (INFO for chatty pipeline steps,
    WARNING for helpers); the global console level filters on top of it.
    """
    logger = logging.getLogger(name)
    if any(getattr(h, "_chemlink_console", False) for h in logger.handlers):
        return logger
    logger.setLevel(level)
    logger.addHandler(ConsoleHandler())
    logger.propagate = False
    return logger


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def attach_file(logger: logging.Logger, log_file: str) -> logging.Handler:
    """Add (once) a session file handler for ``log_file`` to ``logger``."""
    path = os.path.abspath(os.path.expanduser(log_file))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for handler in logger.handlers:
        if getattr(handler, "_step_log_path", None) == path:
            return handler
    handler = SessionFileHandler(path)
    handler._step_log_path = path  # type: ignore[attr-defined]
    logger.addHandler(handler)
    for h in logger.handlers:
        if isinstance(h, ConsoleHandler) and not h.log_file_hint:
            h.log_file_hint = path
    return handler


def capture_warnings_to_console() -> None:
    """Route ``warnings.warn`` through the log format — on stderr, like Python's
    own warnings, so machine-readable stdout (``--json``) stays clean."""
    logging.captureWarnings(True)
    py_warnings = logging.getLogger("py.warnings")
    if not any(getattr(h, "_chemlink_console", False) for h in py_warnings.handlers):
        py_warnings.addHandler(ConsoleHandler(logging.WARNING, stderr=True))
        py_warnings.propagate = False
