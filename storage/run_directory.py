"""Safe helpers for user-named ChemLink run directories."""

import re
from pathlib import Path
from typing import Optional


_RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def validate_run_name(name: str) -> str:
    """Validate a portable directory name, without accepting path syntax."""
    if not isinstance(name, str) or not name or name != name.strip():
        raise ValueError("Run name must be a non-empty name without surrounding spaces")
    if name in {".", ".."} or not _RUN_NAME.fullmatch(name):
        raise ValueError(
            "Invalid run name. Use only ASCII letters, numbers, '.', '_' or '-', "
            "starting with a letter or number; paths are not allowed."
        )
    return name


def create_named_run_directory(base: str, run_name: str) -> str:
    """Create ``base/run_name`` and fail rather than reuse existing results."""
    safe_name = validate_run_name(run_name)
    base_path = Path(base).expanduser().resolve()
    base_path.mkdir(parents=True, exist_ok=True)
    run_path = base_path / safe_name
    try:
        run_path.mkdir()
    except FileExistsError as exc:
        raise FileExistsError(
            f"Run directory already exists: {run_path}. Choose another --run-name."
        ) from exc
    return str(run_path)


def indexed_run_name(run_name: Optional[str], index: int, total: int) -> Optional[str]:
    """Return an exact custom name for one run or a stable suffix for a batch."""
    if run_name is None:
        return None
    safe_name = validate_run_name(run_name)
    return safe_name if total == 1 else f"{safe_name}_{index + 1:03d}"
