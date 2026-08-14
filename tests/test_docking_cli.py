from argparse import Namespace

import json
import os

from chemlink.cli.main import _docking_paths, _run_dir, _write_docking_input_config
from chemlink.storage.run_directory import indexed_run_name, validate_run_name


def test_docking_input_flags_are_resolved_and_take_priority(tmp_path):
    receptors = tmp_path / "selected-receptors"
    ligands = tmp_path / "selected-ligands"
    receptors.mkdir()
    ligands.mkdir()
    args = Namespace(
        receptor_input_dir="legacy-receptors",
        ligand_input_dir="legacy-ligands",
        output_dir="legacy-output",
        receptor_input_option=str(receptors),
        ligand_input_option=str(ligands),
        output_option=str(tmp_path / "results"),
    )

    receptor, ligand, output = _docking_paths(args)

    assert receptor == str(receptors.resolve())
    assert ligand == str(ligands.resolve())
    assert output == str((tmp_path / "results").resolve())


def test_docking_input_flags_reject_missing_directories(tmp_path):
    args = Namespace(
        receptor_input_dir=str(tmp_path / "missing-receptors"),
        ligand_input_dir=str(tmp_path / "missing-ligands"),
        output_dir=str(tmp_path / "results"),
        receptor_input_option=None,
        ligand_input_option=None,
        output_option=None,
    )

    try:
        _docking_paths(args)
    except FileNotFoundError as exc:
        assert "receptor directory" in str(exc)
    else:
        raise AssertionError("Missing docking inputs were not rejected")


def test_docking_input_paths_are_recorded(tmp_path):
    output = tmp_path / "run"
    _write_docking_input_config(output, "/inputs/receptors", "/inputs/ligands")

    config = json.loads((output / "docking_inputs.json").read_text(encoding="utf-8"))
    assert config["receptor_input_dir"] == "/inputs/receptors"
    assert config["ligand_input_dir"] == "/inputs/ligands"
    assert config["recorded_at"]


def test_custom_run_directory_and_metadata(tmp_path):
    output = _run_dir(str(tmp_path), "tnks1_gpu")
    _write_docking_input_config(output, "/inputs/receptors", "/inputs/ligands", "tnks1_gpu")

    assert output == str(tmp_path / "tnks1_gpu")
    config = json.loads((tmp_path / "tnks1_gpu/docking_inputs.json").read_text())
    assert config["run_name"] == "tnks1_gpu"
    assert config["directory_mode"] == "custom"


def test_custom_run_directory_refuses_existing_or_unsafe_names(tmp_path):
    _run_dir(str(tmp_path), "first-run")
    try:
        _run_dir(str(tmp_path), "first-run")
    except FileExistsError:
        pass
    else:
        raise AssertionError("Existing run directory was reused")

    for name in ("../escape", "/absolute", "bad name", "."):
        try:
            validate_run_name(name)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Unsafe run name was accepted: {name}")


def test_indexed_names_are_stable_for_multiple_dynamics_runs():
    assert indexed_run_name("experiment", 0, 1) == "experiment"
    assert indexed_run_name("experiment", 0, 2) == "experiment_001"
    assert indexed_run_name("experiment", 1, 2) == "experiment_002"


def test_short_input_names_are_searched_under_data_input(tmp_path):
    receptors = tmp_path / "data" / "input" / "my-receptors"
    ligands = tmp_path / "data" / "input" / "my-ligands"
    receptors.mkdir(parents=True)
    ligands.mkdir(parents=True)
    args = Namespace(
        receptor_input_dir="unused",
        ligand_input_dir="unused",
        output_dir="data/output",
        receptor_input_option="my-receptors",
        ligand_input_option="my-ligands",
        output_option=None,
    )

    previous = os.getcwd()
    try:
        os.chdir(tmp_path)
        receptor, ligand, _ = _docking_paths(args)
    finally:
        os.chdir(previous)

    assert receptor == str(receptors.resolve())
    assert ligand == str(ligands.resolve())
