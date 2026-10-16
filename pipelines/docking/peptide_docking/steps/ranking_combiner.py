"""Combined-ranking step for the robustness pipeline.

Merges the independent signals gathered for each gate candidate (LightDock
DFIRE score, restraint-site distance and PRODIGY dG) into one combined
ranking. No signal is treated as ground truth on its own -- see
this project's 7KI0 validation, where even a well-restrained LightDock top
pose sat ~40-50 A from the true structure -- so a candidate that scores
well across several independent methods is a more trustworthy pick than
one that only looks good by a single score.

Two combination methods are reported side by side:

- `combined_rank_mean`: average, across available signals, of each
  signal's independent rank (1 = best in that column). Ordinal only --
  "beats the next candidate by 0.1 kcal/mol" and "...by 10 kcal/mol" both
  count as "+1 rank", so a candidate that's merely slightly ahead in most
  signals can outrank one that dominates a couple of them by a wide
  margin.
- `combined_zscore` (the primary sort key): average robust z-score
  (`(value - median) / (MAD * 1.4826)`, sign-flipped so more negative is
  always better, across available signals). Median/MAD instead of
  mean/stdev because a single pathological candidate (this project has
  seen real cases: hung preparations, severe local clashes before this
  session's assembly-geometry fix) would otherwise inflate the stdev and
  compress everyone else's score. Unlike the rank mean, this reflects how
  FAR ahead or behind a candidate actually is in each signal's own units,
  not just its ordinal position -- found in practice to meaningfully
  re-order candidates the rank mean treated as close calls (see this
  project's semaglutide/MASTER_RANKING.csv comparison this session: the
  rank-based method placed it mid-table, the z-score pushed it further
  down once its two outlier-by-a-wide-margin signals were weighted by
  their actual magnitude, not just "worst in the column").

A candidate missing a signal (e.g. PRODIGY failed for it) is never dropped from the table -- both combined columns divide by
however many signals it actually has, instead of penalizing it with a
worst-case total, so a candidate isn't punished for lacking data.

The reference (see robust_pipeline._validation_rows()) is included as a
row like any other and flagged via `is_reference` -- it is what makes
"N/M candidates beat the reference" a claim about the independent
cross-checks too, not just the cheap triage score used for the stage-2 gate.
"""

import csv
import logging
import os
import statistics
from typing import Dict, List, Optional

from .....storage.file_manager import create_folder
from .....utils.logger import setup_logger

logger = setup_logger(__name__, level=logging.INFO)

_MAD_SCALE = 1.4826  # makes MAD comparable to a standard deviation under normality

# (csv filename, value column, output column, higher_is_better)
_SIGNALS = (
    ("ranking.csv", "best_score", "dfire_best_score", True),
    ("ranking.csv", "site_distance_A", "site_distance_A", False),
    ("prodigy_scores.csv", "dg_kcal_mol", "prodigy_dg_kcal_mol", False),
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

    @staticmethod
    def _zscore_column(values: Dict[str, Optional[float]], higher_is_better: bool) -> Dict[str, float]:
        """Robust z-score per candidate: (value - median) / (MAD * 1.4826),
        sign-flipped for a higher-is-better column so more negative always
        means better, matching every lower-is-better column's own sign.
        """
        present = {cid: v for cid, v in values.items() if v is not None}
        if not present:
            return {}
        med = statistics.median(present.values())
        mad = statistics.median(abs(v - med) for v in present.values()) * _MAD_SCALE
        if mad == 0:
            mad = 1e-9  # every candidate has the same value -- avoid dividing by zero
        z = {}
        for cid, v in present.items():
            zi = (v - med) / mad
            z[cid] = -zi if higher_is_better else zi
        return z

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
        zscores: Dict[str, Dict[str, float]] = {cid: {} for cid in self.gate_candidates}
        for _filename, _value_col, out_col, higher_is_better in _SIGNALS:
            column_values = {cid: raw[cid][out_col] for cid in self.gate_candidates}
            column_ranks = self._rank_column(column_values, higher_is_better)
            for cid, rank in column_ranks.items():
                ranks[cid][out_col] = rank
            column_z = self._zscore_column(column_values, higher_is_better)
            for cid, z in column_z.items():
                zscores[cid][out_col] = z

        rows = []
        for cid in self.gate_candidates:
            n_signals = len(ranks[cid])
            combined_rank_mean = (sum(ranks[cid].values()) / n_signals) if n_signals else None
            combined_zscore = (sum(zscores[cid].values()) / n_signals) if n_signals else None
            row = {"peptide": cid, "is_reference": 1 if cid == self.reference_cid else 0,
                   "n_signals": n_signals,
                   "combined_zscore": f"{combined_zscore:.3f}" if combined_zscore is not None else "",
                   "combined_rank_mean": f"{combined_rank_mean:.3f}" if combined_rank_mean is not None else ""}
            for _fn, _vc, out_col, _hib in _SIGNALS:
                row[out_col] = raw[cid][out_col] if raw[cid][out_col] is not None else ""
            rows.append(row)

        # Missing every signal sorts last regardless of its (absent) combined_zscore.
        # combined_zscore (not combined_rank_mean) is the primary sort key -- see module
        # docstring for why it's preferred: it reflects how far ahead/behind a candidate
        # actually is in each signal's own units, not just its ordinal position.
        rows.sort(key=lambda r: (r["n_signals"] == 0, float(r["combined_zscore"] or "inf")))

        fieldnames = ["combined_rank", "peptide", "is_reference", "n_signals",
                      "combined_zscore", "combined_rank_mean"] + \
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
