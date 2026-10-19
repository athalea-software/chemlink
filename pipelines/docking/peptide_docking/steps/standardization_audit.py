"""Standardization-chemistry audit step for the robustness pipeline.

For every gate candidate, rebuilds its PDB with
utils.peptide_processor.PeptideProcessor(standardize=True) and records
which positions (if any) got reverted to ALA/GLY -- this tells a "real
chemistry" candidate apart from one whose docked pose stands in for
non-canonical residues, and is also the signal used to flag likely-
unsupported stapled/macrocyclic peptides before their results are trusted.

Each candidate runs in its own subprocess with a hard timeout: RDKit's
ETKDG embedding hangs unkillable by SIGTERM on a small fraction of
real-world peptides (see PeptidePreparation, same root cause) -- only
killing the OS process (subprocess.run(timeout=...)) reliably recovers.
"""

import csv
import json
import logging
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from typing import Dict, List, Optional

from .....storage.file_manager import create_folder, list_files_in_directory
from .....utils.logger import setup_logger
from .....utils.progress import step_bar_iter

logger = setup_logger(__name__, level=logging.INFO)

# Parent directory of the `chemlink` package -- see peptide_preparation.py's
# identical computation for why this is needed as cwd/PYTHONPATH.
_PKG_PARENT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", ".."))


class StandardizationAudit:
    """Audit non-canonical/modified chemistry for every gate candidate."""

    def __init__(
        self,
        peptide_input_path: str,
        candidates: List[str],
        output_path: str,
        timeout_seconds: int = 180,
    ):
        self.peptide_input_path = peptide_input_path
        self.candidates = candidates
        self.output_path = output_path
        self.timeout_seconds = timeout_seconds
        self.csv_path = os.path.join(self.output_path, "standardization_audit.csv")

    def _find_sdf(self, cid: str) -> Optional[str]:
        candidate = os.path.join(self.peptide_input_path, f"{cid}.sdf")
        if os.path.isfile(candidate):
            return candidate
        matches = list_files_in_directory(self.peptide_input_path, [f"{cid}.sdf"])
        return matches[0] if matches else None

    def _existing_rows(self) -> Dict[str, dict]:
        """Rows already audited from a prior (interrupted) run -- see robust_pipeline --resume."""
        if not os.path.isfile(self.csv_path):
            return {}
        with open(self.csv_path, newline="") as fh:
            return {row["peptide"]: row for row in csv.DictReader(fh)}

    def _audit_one_isolated(self, cid: str) -> dict:
        sdf_path = self._find_sdf(cid)
        if sdf_path is None:
            return {"peptide": cid, "success": False, "timeout": False,
                    "error": "source SDF not found"}

        env = dict(os.environ)
        env["PYTHONPATH"] = _PKG_PARENT + os.pathsep + env.get("PYTHONPATH", "")
        for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[_var] = "1"
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "chemlink.pipelines.docking.peptide_docking.steps.standardization_audit",
                 os.path.abspath(sdf_path)],
                capture_output=True, text=True, timeout=self.timeout_seconds,
                cwd=_PKG_PARENT, env=env,
            )
        except subprocess.TimeoutExpired:
            return {"peptide": cid, "success": False, "timeout": True,
                    "error": f"exceeded {self.timeout_seconds}s timeout"}

        for line in reversed(proc.stdout.splitlines()):
            line = line.strip()
            if line.startswith("{"):
                try:
                    result = json.loads(line)
                    result["timeout"] = False
                    return result
                except json.JSONDecodeError:
                    continue
        return {"peptide": cid, "success": False, "timeout": False,
                "error": f"no result line on stdout (exit {proc.returncode}): "
                         f"{proc.stderr[-500:] if proc.stderr else '(no stderr)'}"}

    def run(self, n_workers: Optional[int] = None) -> Dict[str, int]:
        """Audit every candidate not already present in standardization_audit.csv."""
        create_folder(self.output_path)
        existing = self._existing_rows()
        pending = [cid for cid in self.candidates if cid not in existing]

        rows: Dict[str, dict] = dict(existing)
        if pending:
            if n_workers is None:
                n_workers = cpu_count()
            n_workers = max(1, min(n_workers, len(pending)))
            logger.info(f"Standardization audit: {len(pending)} candidate(s), {n_workers} worker(s)")

            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                results = list(step_bar_iter(
                    pool.map(self._audit_one_isolated, pending), "Standardization Audit",
                    total=len(pending), unit="peptide", colour="magenta",
                ))

            # A transient crash (subprocess died before printing its JSON
            # result line, e.g. "No module named chemlink..." under heavy
            # NFS contention from this many concurrent subprocesses) is the
            # same story as a timeout under a different symptom -- a genuine
            # chemistry failure prints a clean JSON result line before
            # exiting, so only a hard crash before that point leaves stdout
            # without one (see _audit_one_isolated's "no result line"
            # fallback). Retrying just those can't hide a real per-candidate
            # failure behind a retry.
            def _needs_retry(r: dict) -> bool:
                if r.get("timeout"):
                    return True
                return (not r.get("success")
                        and "no result line on stdout" in (r.get("error") or ""))

            retry_indices = [i for i, r in enumerate(results) if _needs_retry(r)]
            if retry_indices and n_workers > 1:
                original_timeout = self.timeout_seconds
                self.timeout_seconds = original_timeout * 2
                try:
                    for i in step_bar_iter(retry_indices, "Retrying timeouts/crashes", unit="peptide", colour="yellow"):
                        results[i] = self._audit_one_isolated(pending[i])
                finally:
                    self.timeout_seconds = original_timeout

            for result in results:
                rows[result["peptide"]] = result

        fieldnames = ["peptide", "success", "n_residues", "n_standardized",
                      "standardized_residues", "unsupported_reason", "error"]
        with open(self.csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for cid in self.candidates:
                row = rows.get(cid, {"peptide": cid, "success": False, "error": "not audited"})
                writer.writerow({k: row.get(k, "") for k in fieldnames})

        n_ok = sum(1 for cid in self.candidates if rows.get(cid, {}).get("success"))
        n_modified = sum(
            1 for cid in self.candidates
            if rows.get(cid, {}).get("success") and int(rows[cid].get("n_standardized") or 0) > 0
        )
        return {"audited": len(self.candidates), "successful": n_ok,
                "modified_chemistry": n_modified, "csv": self.csv_path}


# ── Subprocess entrypoint (one candidate) ───────────────────────────────────

def _audit_worker(sdf_path: str) -> None:
    from .....utils.peptide_processor import PeptideProcessor

    peptide_name = os.path.splitext(os.path.basename(sdf_path))[0]
    try:
        processor = PeptideProcessor(standardize=True)
        info = processor.build_pdb(sdf_path, os.devnull)
    except Exception as exc:  # noqa: BLE001 - reported to the parent process, not raised here
        print(json.dumps({"peptide": peptide_name, "success": False, "error": str(exc)}))
        return

    std = processor.standardized_residues
    print(json.dumps({
        "peptide": peptide_name,
        "success": True,
        "n_residues": info.get("residues", 0),
        "n_standardized": len(std),
        "standardized_residues": json.dumps(std),
        "unsupported_reason": "",
        "error": "",
    }))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python -m chemlink.pipelines.docking.peptide_docking.steps.standardization_audit <sdf>",
              file=sys.stderr)
        sys.exit(2)
    _audit_worker(sys.argv[1])
