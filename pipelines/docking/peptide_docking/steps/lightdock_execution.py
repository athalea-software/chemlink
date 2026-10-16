"""LightDock execution step for peptide-protein docking."""

import os
import shutil
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from multiprocessing import cpu_count
from typing import Any, Dict, List, Optional

from .....adapters.lightdock.lightdock_adapter import LightDockAdapter
from .....storage.file_manager import create_folder, list_files_in_directory, find_compound_name
from .....utils.logger import setup_logger
from .....utils.progress import job_bar, CTRL_C_HINT, console as _progress_console
from .....utils.style import log_line

logger = setup_logger(__name__)


@dataclass(frozen=True)
class LightDockJob:
    """Single peptide docking task (one receptor, fixed across all jobs)."""

    peptide_pdb: str
    peptide_name: str


class LightDockExecution:
    """Run LightDock peptide-protein docking for one receptor against many peptides.

    Unlike DockingExecution (AutoDock-GPU, N receptors x M ligands), this
    targets the common peptide-docking case of ONE receptor against MANY
    peptides, and is built around the two costly, receptor-only LightDock
    steps measured empirically not worth paying per-peptide:

      - ANM for the receptor. LightDock recomputes it from scratch on
        every call with no caching, so it's simply skipped (anm_rec=0,
        rigid receptor); receptor flexibility is handled properly
        downstream by the MD stage instead.
      - The swarm starting-position calculation (SASA + occlusion
        filter): ~485s for a ~380-residue receptor, separate from and
        larger than the ANM cost. LightDock reuses it for free if an
        `init/` folder already exists in the job's working directory
        (its own documented behavior), and that folder only depends on
        the receptor + restraints + swarm-to-surface distance -- not on
        which peptide is docked. So it's computed ONCE, against the
        first peptide purely as a vehicle to get through the CLI, using
        a *fixed* distance (instead of the default per-ligand-diameter
        one) so the result is valid for every other peptide too. It's
        then copied into each peptide's isolated job directory before
        that peptide's own setup() call.

    anm_lig defaults to 4, not LightDock's usual 10: ANM needs at least
    ~(n_modes + 6) degrees of freedom, and LightDock's coarse-grained ANM
    gives 3 per residue -- a 5-residue peptide only has 15, so anm_lig=10
    raises a ValueError from ProDy's eigensolver ("Requested eigenvalue
    indices are not valid"). Worse, anm_lig can't just be raised for
    individual larger peptides: every peptide in a batch shares the same
    init/ folder, and LightDock validates its starting-position files
    against a fixed `7 + anm_rec + anm_lig` line width, so anm_lig must
    be one constant safe for the smallest peptide in the whole batch.
    """

    def __init__(
        self,
        receptor_pdb: str,
        peptides_dir: str,
        output_path: str,
        restraints_file: Optional[str] = None,
        gso_steps: int = 10,
        fixed_distance: float = 15.0,
        glowworms: int = 20,
        anm_lig: int = 4,
        cores_per_job: Optional[int] = None,
        swarms: Optional[int] = None,
        setup_executable: Optional[str] = None,
        run_executable: Optional[str] = None,
    ):
        self.receptor_pdb = os.path.abspath(receptor_pdb)
        self.peptides_dir = peptides_dir
        # Absolute so setup.json / job dirs stay valid when passed to
        # lightdock3.py, which runs with cwd=job_dir (a relative path would be
        # re-resolved against that cwd and doubled -> "file does not exist").
        self.output_path = os.path.abspath(output_path)
        self.restraints_file = os.path.abspath(restraints_file) if restraints_file else None
        self.gso_steps = gso_steps
        self.fixed_distance = fixed_distance
        self.glowworms = glowworms
        self.anm_lig = anm_lig
        self.cores_per_job = cores_per_job
        self.swarms = swarms  # override the SASA-density swarm count (mostly for testing)
        self.adapter = LightDockAdapter(setup_executable, run_executable)

        self.results_dir = os.path.join(self.output_path, "lightdock_results")
        self.shared_dir = os.path.join(self.results_dir, "_shared_init")
        self.log_file = os.path.join(self.output_path, "lightdock_docking.log")

    def _log(self, level: str, message: str, echo: bool = True) -> None:
        """Write a timestamped message to the step log and optionally stdout."""
        create_folder(self.output_path)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] [{level}] {message}"
        if echo:
            _progress_console.print(log_line(level, message, phase="lightdock"), soft_wrap=True)
        with open(self.log_file, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def _build_jobs(self) -> List[LightDockJob]:
        """Build the peptide job list from prepared peptide PDBs."""
        peptide_files = list_files_in_directory(self.peptides_dir, ["*.pdb"])
        if not peptide_files:
            raise RuntimeError(f"No prepared peptide files found in: {self.peptides_dir}")
        return [
            LightDockJob(peptide_pdb=p, peptide_name=find_compound_name(p))
            for p in peptide_files
        ]

    def _prepare_shared_init(self, vehicle_peptide_pdb: str) -> None:
        """Run setup() once with a fixed distance to produce a reusable init/ folder."""
        shared_init = os.path.join(self.shared_dir, "init")
        if os.path.isdir(shared_init):
            self._log("INFO", "Shared init/ already present, skipping precompute")
            return

        create_folder(self.shared_dir)
        shutil.copy(self.receptor_pdb, os.path.join(self.shared_dir, "receptor.pdb"))
        shutil.copy(vehicle_peptide_pdb, os.path.join(self.shared_dir, "ligand.pdb"))

        self._log("INFO", f"Precomputing shared swarm positions (fixed_distance={self.fixed_distance} A)...")
        self.adapter.setup(
            receptor_pdb="receptor.pdb",
            ligand_pdb="ligand.pdb",
            cwd=self.shared_dir,
            restraints_file=self.restraints_file,
            glowworms=self.glowworms,
            anm_rec=0,
            anm_lig=self.anm_lig,
            fixed_distance=self.fixed_distance,
            swarms=self.swarms,
        )
        if not os.path.isdir(shared_init):
            raise RuntimeError(f"Shared init/ was not generated at {shared_init}")
        self._log("INFO", "Shared swarm positions ready")

    def _run_single(self, job: LightDockJob) -> Dict[str, Any]:
        """Run setup + simulation for a single peptide, reusing the shared init/."""
        job_dir = os.path.join(self.results_dir, job.peptide_name)
        create_folder(job_dir)

        shutil.copy(self.receptor_pdb, os.path.join(job_dir, "receptor.pdb"))
        shutil.copy(job.peptide_pdb, os.path.join(job_dir, "ligand.pdb"))
        job_init = os.path.join(job_dir, "init")
        if not os.path.isdir(job_init):
            shutil.copytree(os.path.join(self.shared_dir, "init"), job_init)

        setup_result = self.adapter.setup(
            receptor_pdb="receptor.pdb",
            ligand_pdb="ligand.pdb",
            cwd=job_dir,
            restraints_file=self.restraints_file,
            glowworms=self.glowworms,
            anm_rec=0,
            anm_lig=self.anm_lig,
            fixed_distance=self.fixed_distance,
            swarms=self.swarms,
        )
        run_result = self.adapter.run(
            setup_file=setup_result.setup_file,
            steps=self.gso_steps,
            cwd=job_dir,
            cores=self.cores_per_job,
        )
        return {
            "peptide": job.peptide_name,
            "job_dir": job_dir,
            "swarms": setup_result.swarms,
            "completed_swarms": len(run_result.swarm_dirs),
        }

    def run(self, n_workers: Optional[int] = None) -> Dict[str, int]:
        """Run LightDock docking for every prepared peptide against the receptor.

        Args:
            n_workers: Number of peptides to dock in parallel (each using
                `cores_per_job` cores internally). None = sequential.

        Returns:
            {'successful': int, 'failed': int, 'total': int}
        """
        create_folder(self.results_dir)
        with open(self.log_file, "w", encoding="utf-8") as handle:
            handle.write("=== LIGHTDOCK PEPTIDE-PROTEIN DOCKING ===\n")

        jobs = self._build_jobs()
        self._prepare_shared_init(jobs[0].peptide_pdb)

        if n_workers is None:
            n_workers = 1
        n_workers = max(1, min(n_workers, len(jobs)))

        # Auto-split cores across parallel jobs unless the caller pinned
        # --cores-per-job, same policy as pipelines.docking.steps.vina_execution:
        # total threads = n_workers * cores_per_job, kept <= available cores so
        # concurrent LightDock runs never oversubscribe the machine.
        cores_mode = "explicit"
        if self.cores_per_job is None:
            self.cores_per_job = max(1, cpu_count() // n_workers)
            cores_mode = "auto"

        self._log(
            "INFO",
            f"Docking {len(jobs)} peptide(s) using {n_workers} worker(s), "
            f"{self.cores_per_job} core(s) each ({cores_mode}; {cpu_count()} core(s) available)",
        )

        successful = 0
        failed = 0
        results: List[Dict[str, Any]] = []

        _progress_console.print(CTRL_C_HINT, end="")
        with job_bar("  LightDock", len(jobs)) as progress:
            def _record_success(job: LightDockJob, result: Dict[str, Any]) -> None:
                nonlocal successful
                results.append(result)
                successful += 1
                status_text = f"OK {job.peptide_name} ({result['completed_swarms']}/{result['swarms']} swarms)"
                progress.set_postfix_str(status_text)
                self._log("INFO", status_text, echo=False)

            def _record_failure(job: LightDockJob, exc: Exception) -> None:
                nonlocal failed
                failed += 1
                full_error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
                self._log("ERROR", f"FAILED {job.peptide_name}:\n{full_error}", echo=False)
                exc_str = str(exc)
                if len(exc_str) > 100:
                    exc_str = exc_str[:97] + "..."
                progress.set_postfix_str(f"FAILED {job.peptide_name}: {exc_str}")

            if n_workers == 1:
                for job in jobs:
                    try:
                        result = self._run_single(job)
                        _record_success(job, result)
                    except Exception as exc:
                        _record_failure(job, exc)
                    finally:
                        progress.update(1)
            else:
                with ThreadPoolExecutor(max_workers=n_workers) as executor:
                    future_map = {executor.submit(self._run_single, job): job for job in jobs}
                    for future in as_completed(future_map):
                        job = future_map[future]
                        try:
                            result = future.result()
                            _record_success(job, result)
                        except Exception as exc:
                            _record_failure(job, exc)
                        finally:
                            progress.update(1)

        self._log("INFO", f"LightDock docking finished: {successful} successful, {failed} failed")
        return {"successful": successful, "failed": failed, "total": len(jobs)}
