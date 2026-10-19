"""Subprocess retry + timeout utility."""
from __future__ import annotations

import logging
import subprocess
import time
from typing import Callable, List, Optional

_log = logging.getLogger(__name__)


class SubprocessError(RuntimeError):
    """Raised when a subprocess exhausts all retries."""

    def __init__(self, cmd: List[str], returncode: int, stderr: str) -> None:
        self.cmd        = cmd
        self.returncode = returncode
        self.stderr     = stderr
        super().__init__(
            f"Command failed (rc={returncode}): {' '.join(cmd)}\n{stderr}"
        )


def run_subprocess(
    cmd: List[str],
    *,
    timeout: Optional[int] = None,
    retries: int = 1,
    cwd: Optional[str] = None,
    env: Optional[dict] = None,
    input_data: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    file_only: bool = False,
    on_retry: Optional[Callable[[int, Exception], Optional[List[str]]]] = None,
) -> subprocess.CompletedProcess:
    """
    Run a subprocess with optional timeout and retry support.

    Parameters
    ----------
    cmd        : Command + arguments list.
    timeout    : Per-attempt wall-clock limit in seconds (None = unlimited).
    retries    : Total number of attempts (1 = no retry on failure).
    cwd        : Working directory for the subprocess.
    env        : Environment override; None inherits the current environment.
    input_data : Text written to stdin (replaces subprocess.Popen for interactive tools).
    logger     : Logger for warnings; falls back to the module-level logger.
    file_only  : Mark diagnostics for file-only handlers and retain full stderr.
    on_retry   : Called as on_retry(attempt, last_exception) before each retry.
                 Return a new cmd list to use on the next attempt, or None to
                 reuse the current command unchanged.
    """
    log = logger or _log
    current_cmd = list(cmd)
    last_exc: Exception = RuntimeError("No attempts made")

    for attempt in range(1, retries + 1):
        try:
            return subprocess.run(
                current_cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd,
                env=env,
                input=input_data,
            )
        except subprocess.TimeoutExpired as exc:
            last_exc = exc
            log.warning(
                "Attempt %d/%d — timed out after %ds: %s",
                attempt, retries, timeout, " ".join(current_cmd),
                extra={"file_only": file_only},
            )
        except subprocess.CalledProcessError as exc:
            last_exc = exc
            stderr_snippet = (exc.stderr or "").strip()
            if not file_only:
                stderr_snippet = stderr_snippet[:400]
            log.warning(
                "Attempt %d/%d — rc=%d: %s\n%s",
                attempt, retries, exc.returncode,
                " ".join(current_cmd), stderr_snippet,
                extra={"file_only": file_only},
            )
        except Exception as exc:
            last_exc = exc
            log.warning("Attempt %d/%d — unexpected error: %s", attempt, retries, exc,
                        extra={"file_only": file_only})

        if attempt < retries:
            wait = 2 ** (attempt - 1)   # 1 s, 2 s, 4 s …
            log.info("Retrying in %ds …", wait, extra={"file_only": file_only})
            time.sleep(wait)
            if on_retry is not None:
                new_cmd = on_retry(attempt, last_exc)
                if new_cmd is not None:
                    current_cmd = new_cmd

    if isinstance(last_exc, subprocess.CalledProcessError):
        raise SubprocessError(
            list(last_exc.cmd) if last_exc.cmd else current_cmd,
            last_exc.returncode,
            last_exc.stderr or "",
        )
    if isinstance(last_exc, subprocess.TimeoutExpired):
        raise SubprocessError(current_cmd, -1, f"Timed out after {timeout}s")
    raise last_exc
