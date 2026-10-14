"""HADDOCK3 refinement step for the robustness pipeline.

Runs each gate candidate through a full HADDOCK3 workflow (topoaa ->
rigidbody -> seletop -> flexref -> emref) using the same restraint
residues LightDock was given, and reports two signals per candidate:
top_score (the rank-1 model's HADDOCK score) and convergence_pct (the
fraction of the OTHER sampled models that still land near the top one,
dockq >= 0.10) -- convergence is the more meaningful robustness signal,
since a single low top_score can come from one lucky sample rather than a
real, repeatable binding mode.
"""

import csv
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from typing import Dict, List, Optional, Set, Tuple

from ....adapters.haddock3.haddock3_adapter import Haddock3Adapter
from ....storage.file_manager import create_folder
from ....utils.logger import setup_logger
from ....utils.progress import step_bar_iter
from ._pdb_utils import chain_letters, filter_chain

logger = setup_logger(__name__, level=logging.INFO)

_MAX_LIGAND_CHAINS = 2  # see chain_letters() docstring


class _UnsupportedCandidate(Exception):
    """Raised for a candidate this extraction method can't handle correctly."""


_RUN_TOML_TEMPLATE = """\
run_dir = "run"
ncores = {ncores}

molecules = [
    "receptor.pdb",
    "ligand.pdb"
]

[topoaa]

[rigidbody]
ambig_fname = "ambig.tbl"
sampling = {sampling}

[seletop]
select = {seletop_select}

[flexref]
ambig_fname = "ambig.tbl"

[emref]
ambig_fname = "ambig.tbl"
"""


class Haddock3Refinement:
    """Refine every gate candidate through a HADDOCK3 workflow."""

    def __init__(
        self,
        lightdock_results_dir: str,
        gate_candidates: List[str],
        receptor_pdb: str,
        restraint_residues: Set[Tuple[str, int]],
        output_path: str,
        sampling: int = 500,
        seletop_select: int = 40,
        ncores: Optional[int] = None,
        executable: Optional[str] = None,
        restraints_executable: Optional[str] = None,
        timeout_seconds: int = 7200,
    ):
        self.lightdock_results_dir = lightdock_results_dir
        self.gate_candidates = gate_candidates
        self.receptor_pdb = os.path.abspath(receptor_pdb)
        self.restraint_residues = restraint_residues
        self.output_path = output_path
        self.sampling = sampling
        self.seletop_select = seletop_select
        # None = auto-split at run() time (see there): HADDOCK3's own `ncores`
        # general parameter controls how many CNS worker PROCESSES run per
        # candidate (default 4, see haddock/modules/defaults.yaml) -- CNS
        # itself is single-threaded (confirmed empirically: each CNS process
        # pins at ~100% CPU / 1 thread regardless of OMP_NUM_THREADS, since it
        # doesn't use OpenMP at all), so total cores used by this stage is
        # n_workers (concurrent candidates) * ncores (CNS processes/candidate),
        # not n_workers alone.
        self.ncores = ncores
        self.timeout_seconds = timeout_seconds
        self.adapter = Haddock3Adapter(executable, restraints_executable)
        self.jobs_dir = os.path.join(self.output_path, "haddock3_jobs")
        self.csv_path = os.path.join(self.output_path, "haddock3_results.csv")

    @staticmethod
    def _peptide_resnums(pdb_path: str) -> List[int]:
        resnums = []
        seen = set()
        with open(pdb_path, errors="ignore") as fh:
            for line in fh:
                if not line.startswith("ATOM"):
                    continue
                try:
                    resnum = int(line[22:26])
                except ValueError:
                    continue
                if resnum not in seen:
                    seen.add(resnum)
                    resnums.append(resnum)
        return resnums

    def _prepare_job_dir(self, cid: str) -> Optional[str]:
        src_ligand = os.path.join(self.lightdock_results_dir, cid, "ligand.pdb")
        if not os.path.isfile(src_ligand):
            return None
        n_chains = len(chain_letters(src_ligand))
        if n_chains > _MAX_LIGAND_CHAINS:
            raise _UnsupportedCandidate(
                f"unsupported:multi_chain_peptide ({n_chains} PDB chains in ligand.pdb -- "
                f"the phantom-chain filter is only validated for the single-extra-chain case, "
                f"see _pdb_utils.chain_letters)"
            )
        job_dir = os.path.join(self.jobs_dir, cid)
        create_folder(job_dir)

        ligand_pdb = os.path.join(job_dir, "ligand.pdb")
        filter_chain(src_ligand, ligand_pdb, keep_chain="A")

        receptor_pdb = os.path.join(job_dir, "receptor.pdb")
        if not os.path.isfile(receptor_pdb):
            filter_chain(self.receptor_pdb, receptor_pdb, keep_chain="R")

        receptor_resnums = sorted(resnum for _chain, resnum in self.restraint_residues)
        ligand_resnums = self._peptide_resnums(ligand_pdb)
        if not receptor_resnums or not ligand_resnums:
            return None

        receptor_actpass = os.path.join(job_dir, "receptor_actpass.txt")
        with open(receptor_actpass, "w") as fh:
            fh.write(" ".join(str(r) for r in receptor_resnums) + "\n\n")
        ligand_actpass = os.path.join(job_dir, "ligand_actpass.txt")
        with open(ligand_actpass, "w") as fh:
            fh.write(" ".join(str(r) for r in ligand_resnums) + "\n\n")

        ambig_tbl = os.path.join(job_dir, "ambig.tbl")
        self.adapter.build_ambig_tbl(
            receptor_actpass="receptor_actpass.txt", ligand_actpass="ligand_actpass.txt",
            out_tbl=ambig_tbl, cwd=job_dir,
        )

        run_toml = os.path.join(job_dir, "run.toml")
        with open(run_toml, "w") as fh:
            fh.write(_RUN_TOML_TEMPLATE.format(
                sampling=self.sampling, seletop_select=self.seletop_select, ncores=self.ncores,
            ))
        return job_dir

    @staticmethod
    def _parse_capri_ss(tsv_path: str) -> Tuple[Optional[float], Optional[float]]:
        with open(tsv_path, newline="") as fh:
            rows = list(csv.DictReader(fh, delimiter="\t"))
        if not rows:
            return None, None
        top_score = None
        dockqs_rest = []
        for row in rows:
            try:
                rank = int(float(row["caprieval_rank"]))
                score = float(row["score"])
                dockq = float(row["dockq"])
            except (KeyError, ValueError):
                continue
            if rank == 1:
                top_score = score
            else:
                dockqs_rest.append(dockq)
        convergence_pct = None
        if dockqs_rest:
            convergence_pct = 100.0 * sum(1 for d in dockqs_rest if d >= 0.10) / len(dockqs_rest)
        return top_score, convergence_pct

    def _refine_one(self, cid: str) -> dict:
        analysis_tsv = os.path.join(self.jobs_dir, cid, "run", "analysis", "4_emref_analysis", "capri_ss.tsv")
        if os.path.isfile(analysis_tsv):
            top_score, convergence_pct = self._parse_capri_ss(analysis_tsv)
            if top_score is not None:
                return {"peptide": cid, "top_score": top_score,
                        "convergence_pct": convergence_pct if convergence_pct is not None else "", "error": ""}

        try:
            job_dir = self._prepare_job_dir(cid)
        except _UnsupportedCandidate as exc:
            return {"peptide": cid, "top_score": "", "convergence_pct": "", "error": str(exc)}
        if job_dir is None:
            return {"peptide": cid, "top_score": "", "convergence_pct": "",
                     "error": "missing ligand.pdb or no restraint/peptide residues to refine against"}

        try:
            self.adapter.run_workflow("run.toml", cwd=job_dir, timeout_seconds=self.timeout_seconds)
        except Exception as exc:  # noqa: BLE001 - one candidate's failure must not stop the batch
            logger.warning(f"{cid}: HADDOCK3 failed - {exc}")
            return {"peptide": cid, "top_score": "", "convergence_pct": "", "error": str(exc)}

        if not os.path.isfile(analysis_tsv):
            return {"peptide": cid, "top_score": "", "convergence_pct": "",
                    "error": f"HADDOCK3 finished without {analysis_tsv}"}

        top_score, convergence_pct = self._parse_capri_ss(analysis_tsv)
        if top_score is None:
            return {"peptide": cid, "top_score": "", "convergence_pct": "", "error": "capri_ss.tsv had no rank-1 model"}
        return {"peptide": cid, "top_score": top_score,
                "convergence_pct": convergence_pct if convergence_pct is not None else "", "error": ""}

    def _existing_rows(self) -> Dict[str, dict]:
        if not os.path.isfile(self.csv_path):
            return {}
        with open(self.csv_path, newline="") as fh:
            return {row["peptide"]: row for row in csv.DictReader(fh)}

    def run(self, n_workers: Optional[int] = None) -> Dict[str, int]:
        """Refine every gate candidate not already completed (see --resume)."""
        create_folder(self.output_path)
        create_folder(self.jobs_dir)
        existing = {
            cid: row for cid, row in self._existing_rows().items()
            if row.get("top_score") not in (None, "")
        }
        pending = [cid for cid in self.gate_candidates if cid not in existing]

        rows: Dict[str, dict] = dict(existing)
        if pending:
            if n_workers is None:
                n_workers = min(4, cpu_count())
            n_workers = max(1, n_workers)

            # Auto-split cores across concurrent candidates unless the caller
            # pinned ncores explicitly -- same policy as LightDockExecution's
            # cores-per-job. Total cores used by this stage is
            # n_workers (candidates in flight) * ncores (CNS processes each),
            # so this keeps that product <= available cores instead of
            # silently running at HADDOCK3's own default of 4 per candidate
            # regardless of how many candidates run at once.
            ncores_mode = "explicit"
            if self.ncores is None:
                self.ncores = max(1, cpu_count() // n_workers)
                ncores_mode = "auto"

            logger.info(
                f"HADDOCK3 refinement: {len(pending)} candidate(s), {n_workers} worker(s), "
                f"{self.ncores} CNS core(s) each ({ncores_mode}; {cpu_count()} core(s) available)"
            )
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                results = list(step_bar_iter(
                    pool.map(self._refine_one, pending), "HADDOCK3 Refinement",
                    total=len(pending), unit="peptide", colour="blue",
                ))
            for result in results:
                rows[result["peptide"]] = result

        with open(self.csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["peptide", "top_score", "convergence_pct", "error"])
            writer.writeheader()
            for cid in self.gate_candidates:
                writer.writerow(rows.get(cid, {"peptide": cid, "top_score": "", "convergence_pct": "", "error": "not run"}))

        n_ok = sum(1 for cid in self.gate_candidates if rows.get(cid, {}).get("top_score") not in (None, ""))
        return {"refined": len(self.gate_candidates), "successful": n_ok, "csv": self.csv_path}
