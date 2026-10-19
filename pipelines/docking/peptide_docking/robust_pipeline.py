"""Robustness pipeline for peptide-protein docking: LightDock triage,
gated by score+distance against a known reference, followed by a
standardization audit, an independent PRODIGY cross-check and a combined
ranking. The reference itself is carried through every
cross-check stage alongside the gate candidates (see _validation_rows()),
so the combined ranking is an actual head-to-head against the reference on
every rigorous method, not just the cheap triage score used for the gate.

Formalizes this project's ad-hoc, hand-run validation workflow (see the
scripts under scripts/peptide_robustness/) into one reusable command.
Kept as a separate orchestrator from PeptideDockingPipeline rather than a
method on it: it adds a "gate" filter between the cheap triage stage and
the cross-validation stages after it, and per-stage resumability through
robust_config.json, neither of which PeptideDockingPipeline has any notion
of. PeptideDockingPipeline is
composed here for the triage stage instead of reimplemented.
"""

import csv
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .peptide_docking_pipeline import PeptideDockingPipeline
from .steps import ProdigyScoring, RankingCombiner, StandardizationAudit
from ....storage.file_manager import create_folder
from ....utils.logger import setup_logger

logger = setup_logger(__name__, level=logging.INFO)

STAGE_ORDER = ("triage", "gate", "audit", "prodigy", "combine")


@dataclass
class PeptideDockingRobustResult:
    """Structured result for the robustness pipeline."""

    triage: Dict[str, Any]
    gate: Dict[str, int]
    audit: Dict[str, int] = field(default_factory=dict)
    prodigy: Dict[str, int] = field(default_factory=dict)
    combined_ranking_csv: Optional[str] = None


class PeptideDockingRobustPipeline:
    """Compose LightDock triage with a gate filter and multi-tool cross-validation."""

    # Written by run_gate() (see _write_config), read back by from_config() --
    # lets the CLI's per-stage subcommands (`docking peptide robust
    # audit|prodigy|combine`) operate on an existing
    # --run-name without the caller repeating peptide_input_path/
    # receptor_pdb/restraints_file/reference_cid/gso_steps/glowworms on
    # every invocation -- only 'lightdock' (which creates the run) needs
    # them, everything downstream reads them back from disk.
    CONFIG_FILENAME = "robust_config.json"

    def __init__(
        self,
        peptide_input_path: str,
        receptor_pdb: str,
        output_path: str,
        restraints_file: str,
        reference_cid: str,
        max_candidates: Optional[int] = None,
        gso_steps: int = 10,
        fixed_distance: float = 15.0,
        glowworms: int = 20,
        anm_lig: int = 4,
        cores_per_job: Optional[int] = None,
        swarms: Optional[int] = None,
        standardize: bool = False,
        dedup: bool = False,
        prep_timeout_seconds: int = 180,
        prodigy_executable: Optional[str] = None,
        audit_timeout_seconds: int = 180,
    ):
        # Absolute: robust_config.json (see _write_config/from_config) persists
        # these across separate CLI process invocations (`robust lightdock` now,
        # `robust prodigy` later, possibly from a different cwd) -- a relative
        # path would silently break the moment it's re-resolved from elsewhere.
        self.peptide_input_path = os.path.abspath(peptide_input_path)
        self.receptor_pdb = os.path.abspath(receptor_pdb)
        self.output_path = os.path.abspath(output_path)
        self.restraints_file = os.path.abspath(restraints_file)
        self.reference_cid = reference_cid
        self.max_candidates = max_candidates
        self.gso_steps = gso_steps
        self.glowworms = glowworms
        self.prodigy_executable = prodigy_executable
        self.audit_timeout_seconds = audit_timeout_seconds

        # Stage 1 (triage) is the existing LightDock pipeline, reused as-is.
        self._triage_pipeline = PeptideDockingPipeline(
            peptide_input_path=peptide_input_path,
            receptor_pdb=receptor_pdb,
            output_path=output_path,
            restraints_file=restraints_file,
            gso_steps=gso_steps,
            fixed_distance=fixed_distance,
            glowworms=glowworms,
            anm_lig=anm_lig,
            cores_per_job=cores_per_job,
            swarms=swarms,
            standardize=standardize,
            dedup=dedup,
            prep_timeout_seconds=prep_timeout_seconds,
        )

        self.lightdock_results_dir = os.path.join(self.output_path, "lightdock_results")
        self.gate_csv_path = os.path.join(self.output_path, "gate_candidates.csv")

    # ── Shared config round-trip (see CONFIG_FILENAME) ──────────────────────

    def _write_config(self) -> None:
        """Persist the context every later stage needs but doesn't set itself.

        Called from run_gate() -- by then peptide_input_path/receptor_pdb/
        restraints_file/reference_cid/max_candidates/gso_steps/glowworms are
        all fixed for the rest of this run. gso_steps/glowworms matter
        downstream too, not just for triage itself: ProdigyScoring regenerates a candidate's best-pose PDB
        with lgd_generate_conformations.py if LightDock's own output is
        missing it, and that regeneration must use the same values LightDock
        was originally run with.
        """
        config = {
            "peptide_input_path": self.peptide_input_path,
            "receptor_pdb": self.receptor_pdb,
            "restraints_file": self.restraints_file,
            "reference_cid": self.reference_cid,
            "max_candidates": self.max_candidates,
            "gso_steps": self.gso_steps,
            "glowworms": self.glowworms,
        }
        create_folder(self.output_path)
        with open(os.path.join(self.output_path, self.CONFIG_FILENAME), "w") as fh:
            json.dump(config, fh, indent=2)

    @classmethod
    def from_config(cls, output_path: str, **overrides: Any) -> "PeptideDockingRobustPipeline":
        """Reconstruct a pipeline for an existing --run-name from its saved config.

        Used by the CLI's per-stage subcommands (`robust
        audit|prodigy|combine`), each a separate process
        invocation that only knows --run-name -- everything `robust
        lightdock` already fixed (see _write_config) is read back here
        instead of being repeated on every later command. `overrides` adds
        or replaces constructor kwargs (e.g. prodigy_executable); a `None`
        override is ignored rather than clobbering the saved value or the
        constructor's own default, so an unset CLI flag doesn't override
        anything.
        """
        config_path = os.path.join(output_path, cls.CONFIG_FILENAME)
        if not os.path.isfile(config_path):
            raise RuntimeError(
                f"{config_path} not found -- run 'docking peptide robust lightdock' "
                "for this --run-name first; every later stage reads its shared "
                "context (receptor, restraints, reference, ...) from there."
            )
        with open(config_path) as fh:
            config = json.load(fh)
        config["output_path"] = output_path
        config.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**config)

    # ── CSV round-trip helpers (so each stage can also be resumed/re-run
    #    independently, reading its inputs from disk rather than requiring
    #    the caller to have run_full_pipeline in the same process) ──────────

    def _read_gate_rows(self) -> List[dict]:
        if not os.path.isfile(self.gate_csv_path):
            raise RuntimeError(
                f"{self.gate_csv_path} not found -- run the gate stage before this one."
            )
        with open(self.gate_csv_path, newline="") as fh:
            return list(csv.DictReader(fh))

    def _read_audit_rows(self) -> Dict[str, dict]:
        audit_path = os.path.join(self.output_path, "standardization_audit.csv")
        if not os.path.isfile(audit_path):
            return {}
        with open(audit_path, newline="") as fh:
            return {row["peptide"]: row for row in csv.DictReader(fh)}

    # ── Stages ───────────────────────────────────────────────────────────

    def run_triage(self, peptide_workers: Optional[int] = None, docking_workers: Optional[int] = None):
        """Stage 1: peptide prep + LightDock docking + DFIRE ranking (existing pipeline)."""
        return self._triage_pipeline.run_full_pipeline(
            peptide_workers=peptide_workers, docking_workers=docking_workers,
            reference_cids={self.reference_cid},
        )

    def _read_ranking_rows(self) -> List[dict]:
        ranking_path = os.path.join(self.output_path, "ranking.csv")
        if not os.path.isfile(ranking_path):
            raise RuntimeError(f"{ranking_path} not found -- run the triage stage first.")
        with open(ranking_path, newline="") as fh:
            return list(csv.DictReader(fh))

    def _read_reference_row(self) -> dict:
        rows = self._read_ranking_rows()
        ref_row = next((r for r in rows if r["peptide"] == self.reference_cid), None)
        if ref_row is None:
            ranking_path = os.path.join(self.output_path, "ranking.csv")
            raise RuntimeError(
                f"Reference candidate '{self.reference_cid}' not found in {ranking_path}. "
                "Pass --reference with a peptide filename (without .sdf) actually in the batch."
            )
        return ref_row

    def _validation_rows(self) -> List[dict]:
        """Gate candidates plus the reference itself.

        Every cross-validation stage (audit, PRODIGY) needs to score the
        reference too, not just the candidates that beat it on cheap DFIRE
        triage -- otherwise the final combined ranking can never show
        whether a candidate still beats the reference on an independent
        method, only that it beat it on the triage score used for the gate
        itself.
        """
        return self._read_gate_rows() + [self._read_reference_row()]

    def run_gate(self) -> Dict[str, int]:
        """Stage 2: keep only candidates that beat the reference on BOTH score and site distance.

        The threshold is read from the reference's own ranking.csv row, not
        hardcoded -- this is what keeps the gate meaningful for any
        receptor/peptide batch/reference, not just this project's own data.
        """
        rows = self._read_ranking_rows()
        ref_row = self._read_reference_row()
        ref_score = float(ref_row["best_score"])
        ref_distance = float(ref_row["site_distance_A"]) if ref_row.get("site_distance_A") else None

        def passes(row: dict) -> bool:
            if row["peptide"] == self.reference_cid:
                return False
            try:
                score = float(row["best_score"])
            except (TypeError, ValueError):
                return False
            if score <= ref_score:
                return False
            if ref_distance is not None:
                if not row.get("site_distance_A"):
                    return False
                if float(row["site_distance_A"]) >= ref_distance:
                    return False
            return True

        candidates = sorted((r for r in rows if passes(r)), key=lambda r: float(r["best_score"]), reverse=True)
        if self.max_candidates:
            candidates = candidates[: self.max_candidates]

        create_folder(self.output_path)
        fieldnames = list(rows[0].keys()) if rows else [
            "peptide", "best_score", "best_swarm", "best_glowworm",
            "swarms_completed", "site_distance_A", "is_reference",
        ]
        with open(self.gate_csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(candidates)

        logger.info(
            f"Gate: {len(candidates)}/{len(rows) - 1} candidate(s) beat reference "
            f"'{self.reference_cid}' (score > {ref_score}"
            + (f", distance < {ref_distance:.2f} A" if ref_distance is not None else "") + ")"
        )
        self._write_config()
        return {"gate_candidates": len(candidates), "csv": self.gate_csv_path}

    def run_standardization_audit(self, n_workers: Optional[int] = None) -> Dict[str, int]:
        """Stage 3: which gate candidates (plus the reference) have non-canonical/modified chemistry."""
        validation_rows = self._validation_rows()
        return StandardizationAudit(
            peptide_input_path=self.peptide_input_path,
            candidates=[row["peptide"] for row in validation_rows],
            output_path=self.output_path,
            timeout_seconds=self.audit_timeout_seconds,
        ).run(n_workers=n_workers)

    def run_prodigy(self, n_workers: Optional[int] = None) -> Dict[str, int]:
        """Stage 4: PRODIGY binding-affinity cross-check (candidates + reference)."""
        validation_rows = self._validation_rows()
        return ProdigyScoring(
            lightdock_results_dir=self.lightdock_results_dir,
            gate_rows=validation_rows,
            output_path=self.output_path,
            gso_steps=self.gso_steps,
            glowworms=self.glowworms,
            executable=self.prodigy_executable,
        ).run(n_workers=n_workers)

    def run_combined_ranking(self) -> str:
        """Stage 5: merge every gathered signal (candidates + reference) into one rank-sum ranking."""
        validation_rows = self._validation_rows()
        return RankingCombiner(
            gate_candidates=[row["peptide"] for row in validation_rows],
            sources_dir=self.output_path,
            output_path=self.output_path,
            reference_cid=self.reference_cid,
        ).combine()

    def run_full_pipeline(
        self,
        peptide_workers: Optional[int] = None,
        docking_workers: Optional[int] = None,
        audit_workers: Optional[int] = None,
        prodigy_workers: Optional[int] = None,
    ) -> PeptideDockingRobustResult:
        """Run every stage in order: triage -> gate -> audit -> prodigy -> combine."""
        triage_result = self.run_triage(peptide_workers=peptide_workers, docking_workers=docking_workers)

        gate_stats = self.run_gate()
        if gate_stats.get("gate_candidates", 0) <= 0:
            raise RuntimeError(
                "No candidate beat the reference on both score and site distance, "
                "so there is nothing left to cross-validate. Check --reference and the "
                "restraints file, or widen the batch."
            )

        audit_stats = self.run_standardization_audit(n_workers=audit_workers)
        prodigy_stats = self.run_prodigy(n_workers=prodigy_workers)
        combined_ranking_csv = self.run_combined_ranking()

        return PeptideDockingRobustResult(
            triage={
                "peptide_preparation": triage_result.peptide_preparation,
                "lightdock_execution": triage_result.lightdock_execution,
            },
            gate=gate_stats,
            audit=audit_stats,
            prodigy=prodigy_stats,
            combined_ranking_csv=combined_ranking_csv,
        )
