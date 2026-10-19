"""High-level orchestration for the peptide-protein docking workflow (LightDock)."""

import csv
import os
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from ....storage.file_manager import list_files_in_directory
from .steps import LightDockExecution, PeptidePreparation
from ....utils.style import status_line
from ....utils.theme import get_console


@dataclass
class PeptideDockingResult:
    """Structured result for the peptide docking pipeline."""

    peptide_preparation: Dict[str, int]
    lightdock_execution: Optional[Dict[str, Any]] = None


class PeptideDockingPipeline:
    """Compose peptide preparation and LightDock execution.

    Separate from DockingPipeline (AutoDock-GPU, small-molecule ligands):
    this targets peptide-protein docking of many peptides against a
    single receptor, using LightDock instead. See
    pipelines/docking/peptide_docking/steps/lightdock_execution.py for why the
    receptor-side setup cost is shared across peptides instead of paid
    per-peptide.
    """

    def __init__(
        self,
        peptide_input_path: str,
        receptor_pdb: str,
        output_path: str,
        restraints_file: Optional[str] = None,
        gso_steps: int = 10,
        fixed_distance: float = 15.0,
        glowworms: int = 20,
        anm_lig: int = 4,
        cores_per_job: Optional[int] = None,
        swarms: Optional[int] = None,
        standardize: bool = False,
        dedup: bool = False,
        prep_timeout_seconds: int = 180,
    ):
        self.peptide_input_path = peptide_input_path
        self.receptor_pdb = receptor_pdb
        self.output_path = output_path
        self.restraints_file = restraints_file
        self.gso_steps = gso_steps
        self.fixed_distance = fixed_distance
        self.glowworms = glowworms
        self.anm_lig = anm_lig
        self.cores_per_job = cores_per_job
        self.swarms = swarms
        # Standardize modified peptides so they can be docked (instead of
        # excluding them) and keep the complete original molecule for the MD stage.
        self.standardize = standardize
        # See PeptidePreparation.__init__ for the reasoning behind each: dedup
        # groups isotopic/near-identical variants before docking (it does not
        # change the ranking, it saves compute); prep_timeout_seconds kills
        # hung preparation subprocesses instead of blocking the batch.
        self.dedup = dedup
        self.prep_timeout_seconds = prep_timeout_seconds

    @property
    def prepared_peptides_path(self) -> str:
        """Return the output directory used by peptide preparation.

        If peptide_input_path already contains prepared .pdb files, treat
        it as already-prepared (useful when starting from a later step).
        """
        if list_files_in_directory(self.peptide_input_path, ["*.pdb"]):
            return self.peptide_input_path
        return f"{self.output_path}/prepared_peptides_pdb"

    def run_peptide_preparation(self, n_workers: Optional[int] = None) -> Dict[str, int]:
        """Prepare peptide inputs, rebuilding clean PDBs from source SDFs."""
        return PeptidePreparation(
            input_path=self.peptide_input_path,
            output_path=self.output_path,
            standardize=self.standardize,
            dedup=self.dedup,
            timeout_seconds=self.prep_timeout_seconds,
        ).prepare(n_workers=n_workers)

    def run_lightdock_execution(self, n_workers: Optional[int] = None) -> Dict[str, int]:
        """Run LightDock docking for every prepared peptide against the receptor."""
        return LightDockExecution(
            receptor_pdb=self.receptor_pdb,
            peptides_dir=self.prepared_peptides_path,
            output_path=self.output_path,
            restraints_file=self.restraints_file,
            gso_steps=self.gso_steps,
            fixed_distance=self.fixed_distance,
            glowworms=self.glowworms,
            anm_lig=self.anm_lig,
            cores_per_job=self.cores_per_job,
            swarms=self.swarms,
        ).run(n_workers=n_workers)

    def _restraint_residues(self) -> Set[Tuple[str, int]]:
        """Parse restraints_file into {(chain, resnum), ...} for the receptor side.

        Restraint lines look like "R R.LYS.197" (role, chain.resname.resnum);
        only the receptor's chain/resnum are needed here.
        """
        residues: Set[Tuple[str, int]] = set()
        if not self.restraints_file or not os.path.isfile(self.restraints_file):
            return residues
        with open(self.restraints_file) as fh:
            for line in fh:
                parts = line.split()
                if len(parts) < 2:
                    continue
                chain_resname_resnum = parts[1].split(".")
                if len(chain_resname_resnum) != 3:
                    continue
                chain, _resname, resnum = chain_resname_resnum
                try:
                    residues.add((chain, int(resnum)))
                except ValueError:
                    continue
        return residues

    @staticmethod
    def _read_atoms(pdb_path: str, chain_filter=None) -> List[Tuple[str, int, Tuple[float, float, float]]]:
        atoms = []
        with open(pdb_path, errors="ignore") as fh:
            for line in fh:
                if not line.startswith("ATOM"):
                    continue
                chain = line[21]
                if chain_filter and chain != chain_filter:
                    continue
                try:
                    resnum = int(line[22:26])
                    xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
                except ValueError:
                    continue
                atoms.append((chain, resnum, xyz))
        return atoms

    def _site_distance(self, job_dir: str, swarm: str, glowworm: str,
                        restraint_residues: Set[Tuple[str, int]]) -> Optional[float]:
        """Min distance (A) between the best pose's ligand and the restraint residues.

        A low value (roughly <10 A) means the docked peptide is actually
        touching the site it was told to search near; a high one means the
        search wandered elsewhere despite the restraints -- see this
        session's notes on how to observe where a peptide "stuck" without a
        crystal reference for every candidate.
        """
        swarm_dir = os.path.join(job_dir, f"swarm_{swarm}")
        pose_pdb = os.path.join(swarm_dir, f"lightdock_{glowworm}.pdb")
        if not os.path.isfile(pose_pdb):
            subprocess.run(
                ["lgd_generate_conformations.py", "../receptor.pdb", "../ligand.pdb",
                 f"gso_{self.gso_steps}.out", str(self.glowworms), "--setup", "../setup.json"],
                cwd=swarm_dir, capture_output=True, text=True,
            )
        if not os.path.isfile(pose_pdb):
            return None

        ligand_atoms = self._read_atoms(pose_pdb, chain_filter="A")
        receptor_atoms = [a for a in self._read_atoms(pose_pdb, chain_filter="R")
                          if (a[0], a[1]) in restraint_residues]
        if not ligand_atoms or not receptor_atoms:
            return None

        lig_xyz = np.array([a[2] for a in ligand_atoms])
        rec_xyz = np.array([a[2] for a in receptor_atoms])
        dists = np.sqrt(((lig_xyz[:, None, :] - rec_xyz[None, :, :]) ** 2).sum(axis=2))
        return float(dists.min())

    def run_ranking(self, reference_cids: Optional[Set[str]] = None) -> str:
        """Rank every docked peptide by its best LightDock (DFIRE) score.

        Writes ranking.csv to output_path with one row per peptide, sorted
        best-first, including a `site_distance_A` column: the minimum
        distance between the best pose's peptide atoms and the restraint
        residues. Low (~<10 A) means it's actually touching the requested
        site; high means the search wandered elsewhere despite the
        restraints -- see this session's notes on verifying where a peptide
        "stuck" without a crystal reference for every candidate.

        This is a triage ranking, not a validated affinity prediction --
        see the session's 7KI0 redocking test: LightDock's best-scored pose
        still ran ~40-50 A RMSD from the true structure even with the
        correct restraints. Useful for comparing candidates against each
        other, not for trusting any single pose or absolute score.

        Args:
            reference_cids: Known compounds (e.g. an approved drug already
                in the batch, like semaglutide/56843331 in this project) to
                flag with is_reference=1 and report the rank position of --
                "4 candidates scored above the reference" is the actionable
                read, not any individual score in isolation.
        """
        reference_cids = reference_cids or set()
        restraint_residues = self._restraint_residues()
        results_dir = os.path.join(self.output_path, "lightdock_results")
        rows = []
        if os.path.isdir(results_dir):
            for peptide_name in sorted(os.listdir(results_dir)):
                job_dir = os.path.join(results_dir, peptide_name)
                if peptide_name == "_shared_init" or not os.path.isdir(job_dir):
                    continue
                swarm_dirs = [d for d in os.listdir(job_dir) if d.startswith("swarm_")]
                if not swarm_dirs:
                    continue
                subprocess.run(
                    ["lgd_rank.py", str(len(swarm_dirs)), str(self.gso_steps)],
                    cwd=job_dir, capture_output=True, text=True,
                )
                rank_file = os.path.join(job_dir, "rank_by_scoring.list")
                if not os.path.isfile(rank_file):
                    continue
                with open(rank_file) as fh:
                    next(fh, None)
                    best = fh.readline().split()
                if not best:
                    continue

                swarm, glowworm = best[0], best[1]
                site_distance = None
                if restraint_residues:
                    site_distance = self._site_distance(job_dir, swarm, glowworm, restraint_residues)

                rows.append({
                    "peptide": peptide_name, "best_score": best[-1],
                    "best_swarm": swarm, "best_glowworm": glowworm,
                    "swarms_completed": len(swarm_dirs),
                    "site_distance_A": f"{site_distance:.2f}" if site_distance is not None else "",
                    "is_reference": 1 if peptide_name in reference_cids else 0,
                })

        rows.sort(key=lambda r: float(r["best_score"]), reverse=True)
        ranking_path = os.path.join(self.output_path, "ranking.csv")
        with open(ranking_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=[
                "peptide", "best_score", "best_swarm", "best_glowworm",
                "swarms_completed", "site_distance_A", "is_reference"])
            writer.writeheader()
            writer.writerows(rows)

        for rank, row in enumerate(rows, 1):
            if row["is_reference"]:
                get_console().print(status_line(
                    "info",
                    f"Reference [bold]{row['peptide']}[/] ranks {rank}/{len(rows)} "
                    f"(score {row['best_score']}) — {rank - 1} candidate(s) above it, "
                    f"{len(rows) - rank} below it",
                ), soft_wrap=True)

        return ranking_path

    def run_full_pipeline(
        self,
        peptide_workers: Optional[int] = None,
        docking_workers: Optional[int] = None,
        reference_cids: Optional[Set[str]] = None,
    ) -> PeptideDockingResult:
        """Run peptide preparation followed by LightDock execution and ranking."""
        peptide_stats = self.run_peptide_preparation(n_workers=peptide_workers)
        if peptide_stats.get("successful", 0) <= 0:
            raise RuntimeError(
                "Peptide preparation finished without successful outputs, so docking cannot continue."
            )

        docking_stats = self.run_lightdock_execution(n_workers=docking_workers)
        ranking_path = self.run_ranking(reference_cids=reference_cids)
        docking_stats["ranking_file"] = ranking_path
        return PeptideDockingResult(
            peptide_preparation=peptide_stats,
            lightdock_execution=docking_stats,
        )
