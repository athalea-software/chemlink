"""Adapter for HADDOCK3 command-line execution.

HADDOCK3 ships two entry points needed here: `haddock3-restraints` (builds
an ambiguous restraints table, `ambig.tbl`, from two active/passive residue
lists) and `haddock3` itself (runs a workflow described by a `run.toml`
through CNS: topoaa -> rigidbody -> seletop -> flexref -> emref). Both come
bundled with their own precompiled CNS binary, so no separate CNS license
or install is needed -- see this project's dedicated `haddock3` conda env.
"""

import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Haddock3RunResult:
    """Structured output of a `haddock3 run.toml` run."""

    run_dir: str
    log_file: str


class Haddock3Adapter:
    """Wrapper around the `haddock3` / `haddock3-restraints` binaries."""

    DEFAULT_EXECUTABLE = os.environ.get(
        "HADDOCK3_BIN", "/nfs/chemlink/miniconda/envs/haddock3/bin/haddock3"
    )
    DEFAULT_RESTRAINTS_EXECUTABLE = os.environ.get(
        "HADDOCK3_RESTRAINTS_BIN", "/nfs/chemlink/miniconda/envs/haddock3/bin/haddock3-restraints"
    )

    def __init__(self, executable: Optional[str] = None, restraints_executable: Optional[str] = None):
        self.executable = self._resolve_executable(executable or self.DEFAULT_EXECUTABLE)
        self.restraints_executable = self._resolve_executable(
            restraints_executable or self.DEFAULT_RESTRAINTS_EXECUTABLE
        )

    @staticmethod
    def _resolve_executable(executable: str) -> str:
        if os.path.isabs(executable) and os.path.isfile(executable):
            return executable
        resolved = shutil.which(executable)
        return resolved or executable

    def validate(self) -> None:
        """Ensure both haddock3 entry points exist and are executable."""
        for executable, name in (
            (self.executable, "haddock3"),
            (self.restraints_executable, "haddock3-restraints"),
        ):
            if not os.path.isfile(executable):
                raise FileNotFoundError(f"{name} not found: {executable}")
            if not os.access(executable, os.X_OK):
                raise RuntimeError(f"{name} is not executable: {executable}")

    def build_ambig_tbl(
        self,
        receptor_actpass: str,
        ligand_actpass: str,
        out_tbl: str,
        cwd: str,
        receptor_segid: str = "R",
        ligand_segid: str = "A",
        timeout_seconds: int = 60,
    ) -> str:
        """Run `haddock3-restraints active_passive_to_ambig` and write ambig.tbl."""
        cmd = [
            self.restraints_executable, "active_passive_to_ambig",
            receptor_actpass, ligand_actpass,
            "--segid-one", receptor_segid, "--segid-two", ligand_segid,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_seconds, cwd=cwd)
        if result.returncode != 0 or not result.stdout.strip():
            tail = (result.stderr or result.stdout or "").strip()[-500:]
            raise RuntimeError(f"haddock3-restraints active_passive_to_ambig failed: {tail or 'no output'}")
        with open(out_tbl, "w", encoding="utf-8") as handle:
            handle.write(result.stdout)
        return out_tbl

    def run_workflow(self, run_toml: str, cwd: str, timeout_seconds: int = 7200) -> Haddock3RunResult:
        """Run `haddock3 run.toml` and return the run directory it produced.

        The run directory name comes from `run.toml`'s own `run_dir = "..."`
        key (relative to `cwd`) -- this method doesn't parse the toml itself,
        the caller passes the same name it wrote there.
        """
        cmd = [self.executable, run_toml]
        log_file = os.path.join(cwd, "haddock3.log")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_seconds, cwd=cwd)
        with open(log_file, "w", encoding="utf-8") as handle:
            handle.write(result.stdout)
            handle.write(result.stderr)

        if result.returncode != 0:
            tail = (result.stderr or result.stdout or "").strip()[-800:]
            raise RuntimeError(f"haddock3 failed for {run_toml}: {tail or 'no output'}")

        return Haddock3RunResult(run_dir=cwd, log_file=log_file)
