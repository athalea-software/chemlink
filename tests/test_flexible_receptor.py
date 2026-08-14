import json
from pathlib import Path
from unittest.mock import patch

from chemlink.adapters.autodock_cpu.autodock_vina_adapter import AutoDockVinaAdapter
from chemlink.adapters.autodock_gpu.autodock_gpu_adapter import AutoDockGPUAdapter
from chemlink.pipelines.docking.steps.docking_execution import DockingExecution
from chemlink.pipelines.docking.steps.flexible_receptor import select_flexible_residues
from chemlink.pipelines.docking.steps.vina_execution import VinaExecution


def _atom(serial, name, residue, chain, number, x, y, z, element):
    return (
        f"ATOM  {serial:5d} {name:^4s} {residue:>3s} {chain:1s}{number:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00          {element:>2s}\n"
    )


def test_selects_by_heavy_sidechain_atom_distance(tmp_path):
    receptor = tmp_path / "receptor.pdb"
    receptor.write_text(
        _atom(1, "CA", "LYS", "A", 10, 50, 0, 0, "C")
        + _atom(2, "CB", "LYS", "A", 10, 10, 0, 0, "C")
        + _atom(3, "CB", "SER", "B", 20, 10.001, 0, 0, "C")
        + _atom(4, "CB", "ALA", "A", 30, 1, 0, 0, "C")
        + _atom(5, "CA", "GLY", "A", 31, 1, 0, 0, "C")
    )

    selected, excluded = select_flexible_residues(str(receptor), (0, 0, 0), 10.0)

    assert [item.meeko_id for item in selected] == ["A:10"]
    assert selected[0].distance == 10.0
    assert {item["name"] for item in excluded} == {"ALA", "GLY"}


def test_vina_adapter_adds_flex_argument(tmp_path):
    executable = tmp_path / "vina"
    executable.write_text("")
    executable.chmod(0o755)
    adapter = AutoDockVinaAdapter(str(executable))
    out_prefix = tmp_path / "result"

    def fake_run(cmd, **kwargs):
        Path(cmd[cmd.index("--out") + 1]).write_text("MODEL 1\nENDMDL\n")
        return type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    with patch("subprocess.run", side_effect=fake_run) as run:
        adapter.dock("rigid.pdbqt", "ligand.pdbqt", str(out_prefix), (0, 0, 0),
                     (20, 20, 20), flex_pdbqt="flex.pdbqt")

    cmd = run.call_args.args[0]
    assert cmd[cmd.index("--flex") + 1] == "flex.pdbqt"


def test_gpu_adapter_adds_flexres_argument(tmp_path):
    executable = tmp_path / "autodock-gpu"
    executable.write_text("")
    executable.chmod(0o755)
    adapter = AutoDockGPUAdapter(str(executable))

    def fake_run(cmd, **kwargs):
        (tmp_path / "result.dlg").write_text("DOCKED\n")
        return type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    with patch("subprocess.run", side_effect=fake_run) as run:
        adapter.dock("receptor.maps.fld", "ligand.pdbqt", str(tmp_path / "result"),
                     flexres_file="receptor_flex.pdbqt")

    cmd = run.call_args.args[0]
    assert cmd[cmd.index("--flexres") + 1] == "receptor_flex.pdbqt"


def test_execution_steps_ignore_flex_sidecar_as_receptor(tmp_path):
    receptors = tmp_path / "receptors"
    ligands = tmp_path / "ligands"
    receptors.mkdir()
    ligands.mkdir()
    (receptors / "target.pdbqt").write_text("ATOM\n")
    (receptors / "target_rigid.pdbqt").write_text("ATOM\n")
    (receptors / "target_flex.pdbqt").write_text("ROOT\n")
    (receptors / "target_rigid.gpf").write_text("npts 20 20 20\ngridcenter 0 0 0\n")
    (receptors / "target.maps.fld").write_text("grid\n")
    (ligands / "ligand.pdbqt").write_text("ROOT\n")

    vina = VinaExecution(str(receptors), str(ligands), str(tmp_path / "out"),
                         vina_executable="missing-vina")
    vina_jobs = vina._build_jobs()
    assert len(vina_jobs) == 1
    assert vina_jobs[0].receptor_file.endswith("target_rigid.pdbqt")
    assert vina_jobs[0].flex_file.endswith("target_flex.pdbqt")

    with patch.object(AutoDockGPUAdapter, "_resolve_executable", return_value="missing"):
        gpu = DockingExecution(str(receptors), str(ligands), str(tmp_path / "out"))
    gpu_jobs = gpu._build_docking_jobs()
    assert len(gpu_jobs) == 1
    assert gpu_jobs[0].flex_file.endswith("target_flex.pdbqt")
