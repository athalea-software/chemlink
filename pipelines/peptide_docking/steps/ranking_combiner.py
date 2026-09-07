"""Combined-ranking step for the robustness pipeline.

Merges the up-to-6 independent signals gathered for each gate candidate
(LightDock DFIRE score, restraint-site distance, PRODIGY dG, HADDOCK3 top
score, HADDOCK3 convergence, and mean Rosetta interface dG) into one rank-
sum ranking. No signal is treated as ground truth on its own -- see this
project's 7KI0 validation, where even a well-restrained LightDock top pose
sat ~40-50 A from the true structure -- so a candidate that scores well
across several independent methods is a more trustworthy pick than one
that only looks good by a single score.

A candidate missing a signal (HADDOCK3/Rosetta skipped, PRODIGY failed,
...) is never dropped from the table -- `combined_rank_mean` divides by
however many signals it actually has, instead of penalizing it with a
worst-case rank sum, so a candidate isn't punished for lacking data.

The reference (see robust_pipeline._validation_rows()) is included as a
row like any other and flagged via `is_reference` -- it is what makes
"N/M candidates beat the reference" a claim about the rigorous methods
themselves, not just the cheap triage score used for the stage-2 gate.
"""

import csv
import logging
import os
from typing import Dict, List, Optional

from ....storage.file_manager import create_folder
from ....utils.logger import setup_logger

logger = setup_logger(__name__, level=logging.INFO)

# (csv filename, value column, output column, higher_is_better)
_SIGNALS = (
    ("ranking.csv", "best_score", "dfire_best_score", True),
    ("ranking.csv", "site_distance_A", "site_distance_A", False),
    ("prodigy_scores.csv", "dg_kcal_mol", "prodigy_dg_kcal_mol", False),
    ("haddock3_results.csv", "top_score", "haddock3_top_score", False),
    ("haddock3_results.csv", "convergence_pct", "haddock3_convergence_pct", True),
    ("flexpepdock_results.csv", "mean_interface_dG", "rosetta_mean_interface_dG", False),
)


class RankingCombiner:
    """Combine every gathered signal into one rank-sum table."""

    def __init__(self, gate_candidates: List[str], sources_dir: str, output_path: str,
                 reference_cid: Optional[str] = None):
        self.gate_candidates = gate_candidates
        self.sources_dir = sources_dir
        self.output_path = output_path
        self.reference_cid = reference_cid
        self.csv_path = os.path.join(self.output_path, "combined_ranking.csv")

    @staticmethod
    def _read_column(path: str, key_col: str, value_col: str) -> Dict[str, float]:
        values: Dict[str, float] = {}
        if not os.path.isfile(path):
            return values
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                raw = (row.get(value_col) or "").strip()
                if not raw:
                    continue
                try:
                    values[row[key_col]] = float(raw)
                except ValueError:
                    continue
        return values

    @staticmethod
    def _rank_column(values: Dict[str, Optional[float]], higher_is_better: bool) -> Dict[str, int]:
        present = [(cid, v) for cid, v in values.items() if v is not None]
        present.sort(key=lambda item: item[1], reverse=higher_is_better)
        return {cid: rank for rank, (cid, _v) in enumerate(present, start=1)}

    def combine(self) -> str:
        create_folder(self.output_path)

        raw: Dict[str, Dict[str, Optional[float]]] = {
            cid: {out_col: None for _fn, _vc, out_col, _hib in _SIGNALS} for cid in self.gate_candidates
        }
        for filename, value_col, out_col, _hib in _SIGNALS:
            column = self._read_column(os.path.join(self.sources_dir, filename), "peptide", value_col)
            for cid in self.gate_candidates:
                if cid in column:
                    raw[cid][out_col] = column[cid]

        ranks: Dict[str, Dict[str, int]] = {cid: {} for cid in self.gate_candidates}
        for _filename, _value_col, out_col, higher_is_better in _SIGNALS:
            column_values = {cid: raw[cid][out_col] for cid in self.gate_candidates}
            column_ranks = self._rank_column(column_values, higher_is_better)
            for cid, rank in column_ranks.items():
                ranks[cid][out_col] = rank

        rows = []
        for cid in self.gate_candidates:
            n_signals = len(ranks[cid])
            combined_rank_mean = (sum(ranks[cid].values()) / n_signals) if n_signals else None
            row = {"peptide": cid, "is_reference": 1 if cid == self.reference_cid else 0,
                   "n_signals": n_signals,
                   "combined_rank_mean": f"{combined_rank_mean:.3f}" if combined_rank_mean is not None else ""}
            for _fn, _vc, out_col, _hib in _SIGNALS:
                row[out_col] = raw[cid][out_col] if raw[cid][out_col] is not None else ""
            rows.append(row)

        # Missing every signal sorts last regardless of its (absent) combined_rank_mean.
        rows.sort(key=lambda r: (r["n_signals"] == 0, float(r["combined_rank_mean"] or "inf")))

        fieldnames = ["combined_rank", "peptide", "is_reference", "n_signals", "combined_rank_mean"] + \
                     [out_col for _fn, _vc, out_col, _hib in _SIGNALS]
        with open(self.csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for combined_rank, row in enumerate(rows, start=1):
                writer.writerow({"combined_rank": combined_rank, **row})

        logger.info(f"Combined ranking written: {self.csv_path} ({len(rows)} candidate(s))")

        if self.reference_cid:
            ref_index = next((i for i, r in enumerate(rows) if r["peptide"] == self.reference_cid), None)
            if ref_index is None:
                logger.warning(f"Reference '{self.reference_cid}' has no row in the combined ranking "
                                f"(missing from every signal source).")
            else:
                ref_rank = ref_index + 1
                n_better = sum(1 for r in rows[:ref_index] if r["n_signals"] > 0)
                logger.info(f"Reference '{self.reference_cid}': combined rank {ref_rank}/{len(rows)} -- "
                            f"{n_better} candidate(s) beat it across the combined signals.")

        return self.csv_path
