"""Peptide docking: robustness gate, combined ranking, PRODIGY parsing and CLI."""

import csv
import json
import subprocess

import pytest

from chemlink.adapters.prodigy.prodigy_adapter import ProdigyAdapter
from chemlink.cli.main import build_parser
from chemlink.pipelines.docking.peptide_docking import PeptideDockingRobustPipeline
from chemlink.pipelines.docking.peptide_docking.robust_pipeline import STAGE_ORDER
from chemlink.pipelines.docking.peptide_docking.steps.ranking_combiner import RankingCombiner

RANKING_FIELDS = ["peptide", "best_score", "best_swarm", "best_glowworm",
                  "swarms_completed", "site_distance_A", "is_reference"]


def _write_csv(path, fieldnames, rows):
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _ranking_row(peptide, score, distance, is_reference=0):
    return {"peptide": peptide, "best_score": score, "best_swarm": 1, "best_glowworm": 2,
            "swarms_completed": 80, "site_distance_A": distance, "is_reference": is_reference}


def _robust(tmp_path, **kwargs):
    for name in ("peptides", "receptor.pdb", "restraints.txt"):
        target = tmp_path / name
        target.mkdir() if name == "peptides" else target.write_text("")
    return PeptideDockingRobustPipeline(
        peptide_input_path=str(tmp_path / "peptides"),
        receptor_pdb=str(tmp_path / "receptor.pdb"),
        output_path=str(tmp_path / "run"),
        restraints_file=str(tmp_path / "restraints.txt"),
        reference_cid="ref",
        **kwargs,
    )


def test_stages_include_haddock3_and_rosetta():
    assert STAGE_ORDER == ("triage", "gate", "audit", "prodigy", "haddock3", "rosetta", "combine")


def test_gate_keeps_candidates_that_beat_the_reference_on_score_and_distance(tmp_path):
    pipeline = _robust(tmp_path)
    (tmp_path / "run").mkdir()
    _write_csv(tmp_path / "run" / "ranking.csv", RANKING_FIELDS, [
        _ranking_row("better_both", 12.0, 5.0),
        _ranking_row("better_score_far", 15.0, 20.0),
        _ranking_row("worse_score", 8.0, 3.0),
        _ranking_row("no_distance", 13.0, ""),
        _ranking_row("ref", 10.0, 10.0, is_reference=1),
    ])

    stats = pipeline.run_gate()

    with open(stats["csv"], newline="") as fh:
        kept = [row["peptide"] for row in csv.DictReader(fh)]
    assert kept == ["better_both"]
    assert stats["gate_candidates"] == 1
    config = json.loads((tmp_path / "run" / PeptideDockingRobustPipeline.CONFIG_FILENAME).read_text())
    assert config["reference_cid"] == "ref"


def test_from_config_restores_the_run_and_ignores_unset_overrides(tmp_path):
    pipeline = _robust(tmp_path, gso_steps=7)
    pipeline._write_config()

    restored = PeptideDockingRobustPipeline.from_config(str(tmp_path / "run"), prodigy_executable=None)

    assert restored.gso_steps == 7
    assert restored.reference_cid == "ref"
    assert restored.receptor_pdb == str(tmp_path / "receptor.pdb")


def test_combined_ranking_counts_only_the_signals_each_candidate_has(tmp_path):
    _write_csv(tmp_path / "ranking.csv", RANKING_FIELDS, [
        _ranking_row("a", 12.0, 5.0),
        _ranking_row("b", 11.0, 8.0),
        _ranking_row("ref", 10.0, 10.0, is_reference=1),
    ])
    _write_csv(tmp_path / "prodigy_scores.csv", ["peptide", "dg_kcal_mol", "error"], [
        {"peptide": "a", "dg_kcal_mol": -9.5, "error": ""},
        {"peptide": "ref", "dg_kcal_mol": -7.0, "error": ""},
        {"peptide": "b", "dg_kcal_mol": "", "error": "no contacts"},
    ])

    path = RankingCombiner(["a", "b", "ref"], str(tmp_path), str(tmp_path), reference_cid="ref").combine()

    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert [row["peptide"] for row in rows] == ["a", "b", "ref"]
    assert {row["peptide"]: row["n_signals"] for row in rows} == {"a": "3", "b": "2", "ref": "3"}
    assert rows[2]["is_reference"] == "1"
    assert rows[0]["haddock3_top_score"] == "" and rows[0]["rosetta_mean_interface_dG"] == ""


def test_prodigy_adapter_parses_the_predicted_affinity(tmp_path, monkeypatch):
    output = "[+] Predicted binding affinity (kcal.mol-1):     -9.1\n"
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, output, ""))
    adapter = ProdigyAdapter(executable="/bin/true")

    result = adapter.predict(str(tmp_path / "pose.pdb"), "R", "A", cwd=str(tmp_path))

    assert result.dg_kcal_mol == pytest.approx(-9.1)
    assert (tmp_path / "prodigy_pose.pdb.log").read_text() == output


def test_prodigy_adapter_reports_missing_affinity(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(
        a, 1, "", "Error processing model: No contacts found for selection"))
    adapter = ProdigyAdapter(executable="/bin/true")

    with pytest.raises(RuntimeError, match="No contacts found"):
        adapter.predict(str(tmp_path / "pose.pdb"), "R", "A", cwd=str(tmp_path))


def test_peptide_cli_exposes_every_robust_stage():
    parser = build_parser()

    full = parser.parse_args(["docking", "peptide", "full", "--reference", "x", "--reference", "y"])
    assert full.reference_cids == ["x", "y"] and full.docking_workers == 1

    run_all = parser.parse_args(["docking", "peptide", "robust", "all", "--reference", "ref",
                                 "--skip-haddock3"])
    assert run_all.reference_cid == "ref" and run_all.skip_haddock3 and not run_all.skip_rosetta
    assert run_all.haddock3_sampling == 500 and run_all.flexpepdock_trials == 10

    haddock3 = parser.parse_args(["docking", "peptide", "robust", "haddock3", "--haddock3-ncores", "4"])
    assert haddock3.haddock3_ncores == 4
    rosetta = parser.parse_args(["docking", "peptide", "robust", "rosetta", "--flexpepdock-trials", "3"])
    assert rosetta.flexpepdock_trials == 3
    with pytest.raises(SystemExit):
        parser.parse_args(["docking", "peptide", "robust", "all"])  # --reference is required
