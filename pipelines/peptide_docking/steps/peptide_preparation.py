"""Peptide preparation pipeline step for peptide-protein docking."""

import json
import logging
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from multiprocessing import cpu_count
from typing import Optional, List, Dict, Tuple
from ....utils.progress import step_bar_iter

from ....storage.file_manager import create_folder, list_files_in_directory
from ....utils.peptide_processor import deduplicate_peptides
from ....utils.logger import setup_logger

logger = setup_logger(__name__, level=logging.INFO)

# Parent directory of the `chemlink` package (i.e. the dir containing
# chemlink/), needed as cwd/PYTHONPATH so subprocesses can `python -m
# chemlink.utils.peptide_processor` regardless of the caller's own cwd.
# This file lives at chemlink/pipelines/peptide_docking/steps/<this>.py --
# four levels up from its own directory reaches that parent.
_PKG_PARENT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))


class PeptidePreparation:
    """Batch peptide preparation for peptide-protein docking.

    Rebuilds a clean, standard-residue PDB for each peptide SDF via
    utils.peptide_processor -- see that module for why this doesn't reuse
    the small-molecule ligand path (Open Babel's residue perception is
    unreliable on peptides).
    """

    def __init__(
        self,
        input_path: str,
        output_path: str,
        standardize: bool = False,
        dedup: bool = False,
        timeout_seconds: int = 180,
    ):
        """Initialize peptide preparation pipeline.

        Args:
            input_path: Directory containing input peptide SDF files
            output_path: Base directory for output files
            standardize: If True, non-canonical/modified peptides are
                standardized (mutated to ALA / linearized) so they are dockable
                instead of excluded, and the full original molecule is also
                saved to ``original_peptides_pdb/`` so only the true molecule
                proceeds to the dynamics (MD) stage. See
                utils.peptide_processor.PeptideProcessor(standardize=...).
            dedup: If True, group input SDFs by real structure (canonical
                SMILES, isotopes stripped) before preparing anything, and
                only prepare one representative per group. Isotope-labeled
                variants of the same compound score near-identically in
                docking (see the memantine batch this session), so this can
                save real compute at scale without changing the ranking.
            timeout_seconds: Hard per-peptide wall-clock limit. Each peptide
                is prepared in its own subprocess (`python -m
                chemlink.utils.peptide_processor`), which is what makes the
                limit enforceable -- a hung RDKit embed on a pathological
                topology (observed in practice: two molecules stuck >10-20
                min each, no larger than others that finished in seconds)
                can only be killed by terminating its OS process; an
                in-process timeout can't interrupt a C-extension call that
                won't yield control back.
        """
        self.input_path = input_path
        self.output_path = output_path
        self.standardize = standardize
        self.dedup = dedup
        self.timeout_seconds = timeout_seconds

    def _collect_peptide_files(self) -> List[str]:
        """Collect peptide source files (SDF), deduplicating first if enabled."""
        all_files = list_files_in_directory(self.input_path, ["*.sdf"])
        if not self.dedup:
            return all_files

        groups = deduplicate_peptides(self.input_path)
        representatives = {min(cids) for cids in groups.values()}
        n_dropped = len(all_files) - len(representatives)
        if n_dropped:
            logger.info(
                f"Dedup: {len(all_files)} SDF -> {len(representatives)} distinct structure(s) "
                f"({n_dropped} isotope/near-duplicate variant(s) skipped)"
            )
            dedup_log = os.path.join(self.output_path, "dedup_groups.json")
            create_folder(self.output_path)
            with open(dedup_log, "w") as fh:
                json.dump(groups, fh, indent=2)
        return [f for f in all_files
                if os.path.splitext(os.path.basename(f))[0] in representatives]

    def _get_shard_config(self) -> Optional[Tuple[int, int]]:
        """Return (shard_index, shard_count) from environment if configured.

        Mirrors LigandPreparation's SLURM array sharding support.
        """
        shard_index_env = os.getenv("CHEMLINK_SHARD_INDEX")
        shard_count_env = os.getenv("CHEMLINK_SHARD_COUNT")

        if shard_index_env is not None or shard_count_env is not None:
            if shard_index_env is None or shard_count_env is None:
                raise RuntimeError(
                    "Both CHEMLINK_SHARD_INDEX and CHEMLINK_SHARD_COUNT must be set"
                )
            try:
                shard_index = int(shard_index_env)
                shard_count = int(shard_count_env)
            except ValueError as exc:
                raise RuntimeError(
                    "CHEMLINK_SHARD_INDEX and CHEMLINK_SHARD_COUNT must be integers"
                ) from exc
        elif os.getenv("SLURM_ARRAY_TASK_ID") is not None:
            try:
                slurm_task_id = int(os.getenv("SLURM_ARRAY_TASK_ID", "0"))
            except ValueError as exc:
                raise RuntimeError("SLURM_ARRAY_TASK_ID must be an integer") from exc

            slurm_task_count = os.getenv("SLURM_ARRAY_TASK_COUNT")
            slurm_task_min = os.getenv("SLURM_ARRAY_TASK_MIN")
            slurm_task_max = os.getenv("SLURM_ARRAY_TASK_MAX")

            if slurm_task_count is not None:
                shard_count = int(slurm_task_count)
                if slurm_task_min is not None:
                    shard_index = slurm_task_id - int(slurm_task_min)
                else:
                    shard_index = slurm_task_id
            elif slurm_task_min is not None and slurm_task_max is not None:
                task_min = int(slurm_task_min)
                task_max = int(slurm_task_max)
                shard_count = (task_max - task_min) + 1
                shard_index = slurm_task_id - task_min
            else:
                return None
        else:
            return None

        if shard_count <= 1:
            return None
        if shard_index < 0 or shard_index >= shard_count:
            raise RuntimeError(
                f"Invalid shard index {shard_index} for shard count {shard_count}"
            )
        return shard_index, shard_count

    @staticmethod
    def _select_shard_files(files: List[str], shard_index: int, shard_count: int) -> List[str]:
        """Round-robin shard assignment for balanced distribution."""
        return files[shard_index::shard_count]

    def _write_error_report(self, failed: list, timestamp: str) -> str:
        log_path = os.path.join(self.output_path, f"peptide_preparation_errors_{timestamp}.txt")
        with open(log_path, "w") as f:
            f.write("Peptide Preparation Error Report\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*80}\n\n")
            f.write(f"Summary: {len(failed)} peptide(s) failed to prepare\n\n")
            f.write(f"{'='*80}\n\n")
            for i, error in enumerate(failed, 1):
                f.write(f"Error #{i}\n{'-'*80}\n")
                f.write(f"Peptide: {error['peptide']}\n")
                f.write(f"File: {error['file']}\n")
                f.write(f"Error: {error['error']}\n")
                f.write(f"\nTraceback:\n{error['traceback']}\n")
                f.write(f"{'='*80}\n\n")
        return log_path

    @staticmethod
    def _excluded_category(reason: str) -> str:
        """Classify an exclusion warning by why DFIRE can't score the peptide.

        Both come from utils.peptide_processor.process_peptide(); the
        text itself is the only signal available here since the (name,
        success, error_details, warnings) contract doesn't carry a
        separate category field.
        """
        if "modified terminus" in reason:
            return "modified_termini"
        if "non-canonical" in reason:
            return "non_canonical"
        return "other"

    def _write_excluded_report(self, excluded: List[Tuple[str, str, str]], timestamp: str) -> str:
        """List peptides excluded from docking (modified termini, non-canonical residues, ...)."""
        log_path = os.path.join(self.output_path, f"peptide_preparation_excluded_{timestamp}.txt")
        with open(log_path, "w") as f:
            f.write("Peptide Preparation - Excluded From Docking\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*80}\n\n")
            f.write(f"Summary: {len(excluded)} peptide(s) built/diagnosed successfully but excluded\n\n")
            f.write(f"{'='*80}\n\n")
            for peptide_name, category, reason in excluded:
                f.write(f"Peptide: {peptide_name}\n")
                f.write(f"Category: {category}\n")
                f.write(f"Reason: {reason}\n\n")
        return log_path

    def _prepare_one_isolated(self, sdf_path: str) -> dict:
        """Prepare one peptide in its own subprocess, hard-killed at timeout_seconds.

        Why a subprocess and not a plain function call: a hung RDKit embed
        on a pathological topology can't be interrupted from inside the
        same process (a signal handler can't preempt a C-extension call
        that never yields back to the interpreter) -- only killing the OS
        process works. subprocess.run(timeout=...) does exactly that.
        """
        env = dict(os.environ)
        env["PYTHONPATH"] = _PKG_PARENT + os.pathsep + env.get("PYTHONPATH", "")
        # Cap each subprocess to one core. RDKit's embedding calls out to
        # BLAS/OpenMP, which by default grabs as many threads as it sees
        # cores -- with N workers running concurrently that oversubscribes
        # real CPU regardless of how low n_workers is set, and the
        # contention alone can push a borderline-slow peptide over the
        # timeout (observed: 1 timeout/10 running one at a time, 3/10 at
        # n_workers=4 on an 8-core box, same peptides, same timeout).
        for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[_var] = "1"
        peptide_name = os.path.splitext(os.path.basename(sdf_path))[0]
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "chemlink.utils.peptide_processor",
                 os.path.abspath(sdf_path), os.path.abspath(self.output_path),
                 "1" if self.standardize else "0"],
                capture_output=True, text=True, timeout=self.timeout_seconds,
                cwd=_PKG_PARENT, env=env,
            )
        except subprocess.TimeoutExpired:
            return {"peptide": peptide_name, "success": False, "timeout": True,
                    "warnings": [], "error": f"exceeded {self.timeout_seconds}s timeout",
                    "qc_status": "SKIPPED", "qc_issues": []}

        # The entrypoint prints its JSON result line before exiting either way
        # (success or a clean process_peptide failure); only a crash before
        # that point (import error, etc.) leaves stdout without it.
        for line in reversed(proc.stdout.splitlines()):
            line = line.strip()
            if line.startswith("{"):
                try:
                    result = json.loads(line)
                    result["timeout"] = False
                    return result
                except json.JSONDecodeError:
                    continue
        return {"peptide": peptide_name, "success": False, "timeout": False,
                "warnings": [], "qc_status": "SKIPPED", "qc_issues": [],
                "error": f"no result line on stdout (exit {proc.returncode}): "
                         f"{proc.stderr[-500:] if proc.stderr else '(no stderr)'}"}

    def prepare(self, n_workers: Optional[int] = None) -> Dict[str, int]:
        """Prepare all peptides found in input directory.

        Each peptide runs in its own subprocess with a
        `self.timeout_seconds` hard limit and a post-build QC pass
        (utils.peptide_processor.validate_prepared_pdb) -- see
        `_prepare_one_isolated` and this class's __init__ docstring for why.

        Args:
            n_workers: Number of parallel workers (None = all CPUs, 1 = sequential)

        Returns:
            {'successful', 'failed', 'timeout', 'excluded',
             'excluded_modified_termini', 'excluded_non_canonical',
             'qc_warn', 'qc_fail'}
        """
        create_folder(f"{self.output_path}/prepared_peptides_pdb")
        if self.standardize:
            create_folder(f"{self.output_path}/original_peptides_pdb")

        files = self._collect_peptide_files()
        if not files:
            logger.warning(f"No peptide files found in {self.input_path}")
            return {"successful": 0, "failed": 0}

        shard_cfg = self._get_shard_config()
        total_files = len(files)
        if shard_cfg:
            shard_index, shard_count = shard_cfg
            files = self._select_shard_files(files, shard_index, shard_count)
            logger.info(
                f"SLURM sharding enabled: shard {shard_index + 1}/{shard_count} "
                f"processing {len(files)}/{total_files} peptide(s)"
            )
            if not files:
                logger.warning("No peptides assigned to this shard.")
                return {"successful": 0, "failed": 0}

        if n_workers is None:
            n_workers = cpu_count()
        n_workers = max(1, min(n_workers, len(files)))

        successful: List[str] = []
        failed: List[dict] = []
        excluded: List[Tuple[str, str, str]] = []
        timed_out: List[str] = []
        qc_warn: List[Tuple[str, List[str]]] = []
        qc_fail: List[Tuple[str, List[str]]] = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        logger.info(
            f"Preparing {len(files)} peptide(s) using {n_workers} worker(s), "
            f"{self.timeout_seconds}s timeout each..."
        )

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            results = list(step_bar_iter(
                pool.map(self._prepare_one_isolated, files), "Peptide Preparation",
                total=len(files), unit="peptide", colour="magenta",
            ))

        # Retry timeouts AND transient subprocess crashes alone (n_workers=1,
        # 2x the budget). Running N peptides concurrently oversubscribes real
        # CPU even with one thread per subprocess -- observed in practice:
        # 1/10 timed out run one at a time, 3/10 timed out at n_workers=4 on
        # the SAME peptides and timeout, purely from contention. A peptide
        # that was merely slowed down by a crowded run gets a fair,
        # uncontended shot here; one that's genuinely pathological (the
        # 145949006-style cases seen this session) still times out and is
        # reported as such.
        #
        # A transient crash (subprocess died before printing its JSON result
        # line) is the same story under a different symptom: on a real
        # 721-peptide/24-worker run on this project's NFS mount, 412/721
        # failed with the identical "No module named chemlink..." even
        # though the module unquestionably exists -- 309 succeeded running
        # that exact same command seconds apart, and the semaglutide
        # reference itself was one of the 412 casualties. A genuine
        # chemistry failure (e.g. NonCanonicalResidueError) is caught
        # inside the entrypoint and prints a clean JSON result line before
        # exiting; only a hard crash before that point leaves stdout without
        # one (see _prepare_one_isolated's "no result line" fallback), so
        # retrying just those can't hide a real per-peptide failure behind a
        # retry -- it only re-gives an uncontended shot to a process that
        # never got to run this project's code at all.
        if n_workers > 1:
            def _needs_retry(r: dict) -> bool:
                if r.get("timeout"):
                    return True
                return (not r.get("success")
                        and "no result line on stdout" in (r.get("error") or ""))

            retry_indices = [i for i, r in enumerate(results) if _needs_retry(r)]
            if retry_indices:
                logger.info(f"Retrying {len(retry_indices)} timeout/transient-crash case(s) alone "
                           f"(uncontended, {self.timeout_seconds * 2}s budget)...")
                original_timeout = self.timeout_seconds
                self.timeout_seconds = original_timeout * 2
                try:
                    for i in step_bar_iter(retry_indices, "Retrying timeouts/crashes", unit="peptide", colour="yellow"):
                        results[i] = self._prepare_one_isolated(files[i])
                finally:
                    self.timeout_seconds = original_timeout

        manifest_path = os.path.join(self.output_path, f"peptide_preparation_manifest_{timestamp}.jsonl")
        create_folder(self.output_path)
        with open(manifest_path, "w") as manifest_fh:
            for result in results:
                manifest_fh.write(json.dumps(result) + "\n")
                peptide_name = result.get("peptide", "unknown_peptide")

                if result.get("timeout"):
                    timed_out.append(peptide_name)
                    continue

                if result.get("success"):
                    successful.append(peptide_name)
                    warnings = result.get("warnings") or []
                    if warnings:
                        reason = "; ".join(warnings)
                        excluded.append((peptide_name, self._excluded_category(reason), reason))
                    qc_status = result.get("qc_status")
                    if qc_status == "WARN":
                        qc_warn.append((peptide_name, result.get("qc_issues") or []))
                    elif qc_status == "FAIL":
                        qc_fail.append((peptide_name, result.get("qc_issues") or []))
                else:
                    failed.append({
                        "peptide": peptide_name, "file": self.input_path,
                        "error": result.get("error", "unknown error"),
                        "traceback": "",
                    })

        excluded_modified_termini = sum(1 for _, cat, _ in excluded if cat == "modified_termini")
        excluded_non_canonical = sum(1 for _, cat, _ in excluded if cat == "non_canonical")

        logger.info(
            f"Successfully prepared: {len(successful)}/{len(files)} peptides "
            f"({len(excluded)} excluded from docking: "
            f"{excluded_modified_termini} modified termini, "
            f"{excluded_non_canonical} non-canonical; "
            f"{len(timed_out)} timed out; "
            f"QC: {len(qc_warn)} warn, {len(qc_fail)} fail -- see manifest)"
        )
        logger.info(f"Manifest: {manifest_path}")

        if timed_out:
            logger.warning(f"Timed out (>{self.timeout_seconds}s): {len(timed_out)} - {timed_out}")

        if qc_fail:
            logger.error(f"QC FAIL (untrustworthy, check manifest before docking): "
                         f"{[name for name, _ in qc_fail]}")

        if excluded:
            excluded_log = self._write_excluded_report(excluded, timestamp)
            logger.warning(f"Excluded: {len(excluded)} - See: {excluded_log}")

        if failed:
            error_log = self._write_error_report(failed, timestamp)
            logger.error(f"Failed: {len(failed)} - See: {error_log}")

        return {
            "successful": len(successful),
            "failed": len(failed),
            "timeout": len(timed_out),
            "excluded": len(excluded),
            "excluded_modified_termini": excluded_modified_termini,
            "excluded_non_canonical": excluded_non_canonical,
            "qc_warn": len(qc_warn),
            "qc_fail": len(qc_fail),
        }
