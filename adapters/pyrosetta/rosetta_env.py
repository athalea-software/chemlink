"""Resolver for the PyRosetta Python environment.

PyRosetta has no single subprocess "binary" to wrap the way the other
adapters do -- it's a library, imported from a dedicated conda env's
Python interpreter, and every actual trial runs in its own subprocess
(see steps/_flexpepdock_trial_worker.py) since PyRosetta is not
thread-safe and not meant to be re-initialized in-process. This class only
resolves that interpreter and checks it can actually `import pyrosetta`
(requires an academic license file to be installed, which is a per-cluster
concern this class deliberately does not manage).
"""

import os
import subprocess
from typing import Optional


class RosettaEnvironment:
    """Resolves and validates the PyRosetta-capable Python interpreter."""

    DEFAULT_PYTHON = os.environ.get(
        "PYROSETTA_PYTHON", "/nfs/chemlink/miniconda/envs/pyrosetta/bin/python3"
    )

    def __init__(self, python_executable: Optional[str] = None):
        self.python_executable = python_executable or self.DEFAULT_PYTHON

    def is_available(self, timeout_seconds: int = 30) -> bool:
        """Return True if `import pyrosetta` succeeds in the configured interpreter.

        Used to degrade gracefully (skip the Rosetta stage, not crash the
        pipeline) when PyRosetta or its license isn't installed on a given
        machine -- see FlexPepDockRefinement.
        """
        if not os.path.isfile(self.python_executable):
            return False
        try:
            result = subprocess.run(
                [self.python_executable, "-c", "import pyrosetta"],
                capture_output=True, text=True, timeout=timeout_seconds,
            )
        except (subprocess.TimeoutExpired, OSError):
            return False
        return result.returncode == 0

    def unavailable_reason(self, timeout_seconds: int = 30) -> str:
        """Human-readable reason PyRosetta isn't usable (best-effort, for logs)."""
        if not os.path.isfile(self.python_executable):
            return f"interpreter not found: {self.python_executable}"
        try:
            result = subprocess.run(
                [self.python_executable, "-c", "import pyrosetta"],
                capture_output=True, text=True, timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return "`import pyrosetta` timed out"
        except OSError as exc:
            return f"could not launch interpreter: {exc}"
        if result.returncode == 0:
            return ""
        return (result.stderr or result.stdout or "unknown import error").strip()[-500:]
