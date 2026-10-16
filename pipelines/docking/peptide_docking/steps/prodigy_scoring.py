"""PRODIGY cross-check step for the robustness pipeline.

Runs PRODIGY's contact-based binding-affinity estimator on each gate
candidate's best LightDock pose -- a cheap (~seconds/peptide), independent
signal that doesn't depend on DFIRE, useful as a sanity cross-check
alongside it rather than a replacement for it.
"""

import csv
import logging
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from typing import Dict, List, Optional

from .....adapters.prodigy.prodigy_adapter import ProdigyAdapter
from .....storage.file_manager import create_folder
from .....utils.logger import setup_logger
from .....utils.progress import step_bar_iter

logger = setup_logger(__name__, level=logging.INFO)


class ProdigyScoring:
    """Score every gate candidate's best pose with PRODIGY."""

    def __init__(
        self,
        lightdock_results_dir: str,
        gate_rows: List[dict],
        output_path: str,
        gso_steps: int = 10,
        glowworms: int = 20,
        executable: Optional[str] = None,
        receptor_chain: str = "R",
        ligand_chain: str = "A",
    ):
        self.lightdock_results_dir = lightdock_results_dir
        self.gate_rows = gate_rows
        self.output_path = output_path
        self.gso_steps = gso_steps
        self.glowworms = glowworms
        self.receptor_chain = receptor_chain
        self.ligand_chain = ligand_chain
        self.adapter = ProdigyAdapter(executable)
        self.csv_path = os.path.join(self.output_path, "prodigy_scores.csv")

    def _pose_pdb(self, cid: str, swarm: str, glowworm: str) -> Optional[str]:
        """Locate (or generate, same as PeptideDockingPipeline._site_distance) the
        best-pose complex PDB for one candidate."""
        job_dir = os.path.join(self.lightdock_results_dir, cid)
        swarm_dir = os.path.join(job_dir, f"swarm_{swarm}")
        pose_pdb = os.path.join(swarm_dir, f"lightdock_{glowworm}.pdb")
        if not os.path.isfile(pose_pdb):
            subprocess.run(
                ["lgd_generate_conformations.py", "../receptor.pdb", "../ligand.pdb",
                 f"gso_{self.gso_steps}.out", str(self.glowworms), "--setup", "../setup.json"],
                cwd=swarm_dir, capture_output=True, text=True,
            )
        return pose_pdb if os.path.isfile(pose_pdb) else None

    def _existing_rows(self) -> Dict[str, dict]:
        if not os.path.isfile(self.csv_path):
            return {}
        with open(self.csv_path, newline="") as fh:
            return {row["peptide"]: row for row in csv.DictReader(fh)}

    def _score_one(self, row: dict) -> dict:
        cid = row["peptide"]
        pose_pdb = self._pose_pdb(cid, row["best_swarm"], row["best_glowworm"])
        if pose_pdb is None:
            return {"peptide": cid, "dg_kcal_mol": "", "error": "best pose PDB not found/generated"}
        try:
            result = self.adapter.predict(
                complex_pdb=pose_pdb, receptor_chain=self.receptor_chain,
                ligand_chain=self.ligand_chain, cwd=os.path.dirname(pose_pdb),
            )
            return {"peptide": cid, "dg_kcal_mol": result.dg_kcal_mol, "error": ""}
        except Exception as exc:  # noqa: BLE001 - one failure must not stop the batch
            logger.warning(f"{cid}: prodigy failed - {exc}")
            return {"peptide": cid, "dg_kcal_mol": "", "error": str(exc)}

    def run(self, n_workers: Optional[int] = None) -> Dict[str, int]:
        """Score every gate candidate not already present in prodigy_scores.csv."""
        create_folder(self.output_path)
        existing = self._existing_rows()
        pending = [row for row in self.gate_rows if row["peptide"] not in existing]

        rows: Dict[str, dict] = dict(existing)
        if pending:
            if n_workers is None:
                n_workers = min(cpu_count(), len(pending))
            n_workers = max(1, n_workers)
            logger.info(f"PRODIGY scoring: {len(pending)} candidate(s), {n_workers} worker(s)")
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                results = list(step_bar_iter(
                    pool.map(self._score_one, pending), "PRODIGY Scoring",
                    total=len(pending), unit="peptide", colour="cyan",
                ))
            for result in results:
                rows[result["peptide"]] = result

        with open(self.csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["peptide", "dg_kcal_mol", "error"])
            writer.writeheader()
            for row in self.gate_rows:
                cid = row["peptide"]
                writer.writerow(rows.get(cid, {"peptide": cid, "dg_kcal_mol": "", "error": "not scored"}))

        n_ok = sum(1 for row in self.gate_rows if rows.get(row["peptide"], {}).get("dg_kcal_mol") not in (None, ""))
        return {"scored": len(self.gate_rows), "successful": n_ok, "csv": self.csv_path}
