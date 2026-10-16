"""Paths resolved from the working directory (cli/workspace.py) and the
per-command options they replace."""

import os
import time

import pytest

from chemlink.cli import workspace
from chemlink.cli.main import _bash_completion, _completion_tree, build_parser


def _pdb(path, resnames):
    lines = [
        f"ATOM  {i + 1:5d}  CA  {name:>3s} A{i + 1:4d}       0.000   0.000   0.000  1.00  0.00           C"
        for i, name in enumerate(resnames)
    ]
    path.write_text("\n".join(lines) + "\nEND\n")
    return path


def test_inputs_prefer_the_role_subfolder(tmp_path, monkeypatch):
    (tmp_path / "receptors").mkdir()
    _pdb(tmp_path / "loose.pdb", ["ALA", "GLY"])
    monkeypatch.chdir(tmp_path)

    source = workspace.find_inputs("receptors")

    assert source.origin == "subfolder"
    assert source.directory == str(tmp_path / "receptors")


def test_loose_ligands_are_linked_without_receptor_pdbs(tmp_path, monkeypatch):
    _pdb(tmp_path / "receptor.pdb", ["ALA", "GLY"])
    (tmp_path / "a.sdf").write_text("a")
    (tmp_path / "b.mol2").write_text("b")
    monkeypatch.chdir(tmp_path)

    source = workspace.find_inputs("ligands")
    staged = source.materialize(str(tmp_path / "run"))

    assert sorted(os.listdir(staged)) == ["a.sdf", "b.mol2"]
    assert os.path.islink(os.path.join(staged, "a.sdf"))


def test_copy_mode_stages_real_files(tmp_path, monkeypatch):
    (tmp_path / "a.sdf").write_text("a")
    monkeypatch.chdir(tmp_path)

    staged = workspace.find_inputs("ligands").materialize(str(tmp_path / "run"), copy=True)

    path = os.path.join(staged, "a.sdf")
    assert os.path.isfile(path) and not os.path.islink(path)


def test_missing_inputs_name_the_flag(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="-il DIR"):
        workspace.find_inputs("ligands")


def test_single_receptor_and_restraints_are_found(tmp_path, monkeypatch):
    _pdb(tmp_path / "target.pdb", ["ALA", "GLY"])
    (tmp_path / "restraints.txt").write_text("R A.ALA.1\n")
    monkeypatch.chdir(tmp_path)

    assert workspace.find_receptor_file() == str(tmp_path / "target.pdb")
    assert workspace.find_restraints() == str(tmp_path / "restraints.txt")


def test_ambiguous_receptor_and_missing_required_restraints(tmp_path, monkeypatch):
    _pdb(tmp_path / "a.pdb", ["ALA", "GLY"])
    _pdb(tmp_path / "b.pdb", ["ALA", "GLY"])
    monkeypatch.chdir(tmp_path)

    with pytest.raises(FileNotFoundError, match="-ir FILE"):
        workspace.find_receptor_file()
    assert workspace.find_restraints() is None
    with pytest.raises(FileNotFoundError, match="--restraints"):
        workspace.find_restraints(required=True)


def test_run_dir_is_the_cwd_or_the_latest_run(tmp_path, monkeypatch):
    base = tmp_path / "results" / "dynamics"
    for name in ("run_old", "run_new"):
        (base / name).mkdir(parents=True)
        (base / name / "run_config.json").write_text("{}")
        time.sleep(0.01)
    (base / "not_a_run").mkdir()
    monkeypatch.chdir(tmp_path)

    assert workspace.find_run_dir(None, "dynamics", ("run_config.json",)) == str(base / "run_new")
    monkeypatch.chdir(base / "run_old")
    assert workspace.find_run_dir(None, "dynamics", ("run_config.json",)) == str(base / "run_old")


def test_dynamics_detection_classifies_structures(tmp_path, monkeypatch):
    protein = _pdb(tmp_path / "protein.pdb", ["ALA"] * 120)
    peptide = _pdb(tmp_path / "peptide.pdb", ["GLY"] * 10)
    dna = _pdb(tmp_path / "dna.pdb", ["DA", "DT", "DG"])
    ligand = tmp_path / "ligand.sdf"
    ligand.write_text("ligand")
    monkeypatch.chdir(tmp_path)

    assert workspace.detect_dynamics_inputs("ppeptide", [], []) == ([str(protein)], [str(peptide)])
    assert workspace.detect_dynamics_inputs("pacid", [str(protein)], []) == ([str(protein)], [str(dna)])
    assert workspace.detect_dynamics_inputs("oligand", [], []) == ([], [str(ligand)])
    assert workspace.detect_dynamics_inputs("pligand", [str(protein)], []) == ([str(protein)], [str(ligand)])


def test_dynamics_detection_refuses_to_guess(tmp_path, monkeypatch):
    _pdb(tmp_path / "a.pdb", ["ALA"] * 50)
    _pdb(tmp_path / "b.pdb", ["ALA"] * 60)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="-ir/-il"):
        workspace.detect_dynamics_inputs("oprotein", [], [])


def test_binding_energy_options_match_the_simulation_type():
    parser = build_parser()
    oprotein = parser.parse_args(["dynamic", "oprotein"])
    ppeptide = parser.parse_args(["dynamic", "ppeptide"])
    ppligand = parser.parse_args(["dynamic", "ppligand"])

    assert not hasattr(oprotein, "mmpbsa_frames") and not hasattr(oprotein, "binding_energy")
    assert hasattr(ppeptide, "mmpbsa_frames") and not hasattr(ppeptide, "mmpbsa_method")
    assert hasattr(ppligand, "mmpbsa_cores") and not hasattr(ppligand, "mmpbsa_salt")


def test_removed_options_are_rejected():
    parser = build_parser()
    for argv in (
        ["docking", "full", "--fpocket-path", "/usr/bin/fpocket"],
        ["docking", "full", "--max-workers", "4"],
        ["docking", "full", "--rigid-receptor"],
        ["dynamic", "pligand", "-i", "protein.pdb"],
        ["hpc", "dynamics", "pligand", "--mpi"],
        ["update", "--branch", "main"],
    ):
        with pytest.raises(SystemExit):
            parser.parse_args(argv)


def test_completion_is_generated_from_the_parser():
    tree = _completion_tree(build_parser())
    script = _bash_completion(tree)

    assert "chemlink dynamic pligand" in tree
    assert "chemlink dynamic fep" in tree
    assert "chemlink docking peptide robust prodigy" in tree
    assert "chemlink docking peptide robust haddock3" in tree
    assert "chemlink docking peptide robust rosetta" in tree
    assert '["chemlink docking full -ir"]="1"' in script
    assert "--fpocket-path" not in script
