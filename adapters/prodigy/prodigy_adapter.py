"""Adapter for PRODIGY (PROtein binDIng enerGY prediction) command-line execution.

PRODIGY estimates binding affinity (dG, kcal/mol) from a complex PDB's
interface contacts alone -- no docking/refinement, so it's cheap (~seconds
per complex) and useful as an independent cross-check against LightDock's
own DFIRE score. Installed with `pip install prodigy-prot` in the `bio`
environment and run as a separate program.
"""

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ProdigyResult:
    """Structured output of a `prodigy` run."""

    dg_kcal_mol: float
    log_file: str


class ProdigyAdapter:
    """Wrapper around the `prodigy` binary."""

    DEFAULT_EXECUTABLE = os.environ.get("PRODIGY_BIN", "prodigy")

    _DG_RE = re.compile(r"Predicted binding affinity \(kcal\.mol-1\)\s*:\s*(-?\d+\.?\d*)")

    def __init__(self, executable: Optional[str] = None):
        self.executable = self._resolve_executable(executable or self.DEFAULT_EXECUTABLE)

    @staticmethod
    def _resolve_executable(executable: str) -> str:
        if os.path.isabs(executable) and os.path.isfile(executable):
            return executable
        resolved = shutil.which(executable)
        return resolved or executable

    def validate(self) -> None:
        """Ensure the prodigy executable exists and is runnable."""
        if not os.path.isfile(self.executable):
            raise FileNotFoundError(f"prodigy not found: {self.executable}")
        if not os.access(self.executable, os.X_OK):
            raise RuntimeError(f"prodigy is not executable: {self.executable}")

    def predict(
        self,
        complex_pdb: str,
        receptor_chain: str,
        ligand_chain: str,
        cwd: str,
        timeout_seconds: int = 120,
    ) -> ProdigyResult:
        """Run prodigy on one complex PDB and parse its predicted dG.

        `receptor_chain`/`ligand_chain` are passed as prodigy's
        `--selection A B`-style chain groups (one chain each here, since
        every complex this project produces is a single-chain receptor
        against a single-chain peptide).
        """
        cmd = [
            self.executable, complex_pdb,
            "--selection", receptor_chain, ligand_chain,
        ]
        log_file = os.path.join(cwd, f"prodigy_{os.path.basename(complex_pdb)}.log")
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_seconds, cwd=cwd,
        )
        combined = (result.stdout or "") + (result.stderr or "")
        with open(log_file, "w", encoding="utf-8") as handle:
            handle.write(combined)

        match = self._DG_RE.search(combined)
        if not match:
            tail = combined.strip()[-500:]
            raise RuntimeError(f"prodigy did not report a binding affinity for {complex_pdb}: {tail or 'no output'}")
        return ProdigyResult(dg_kcal_mol=float(match.group(1)), log_file=log_file)
