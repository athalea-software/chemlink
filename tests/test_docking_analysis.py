import numpy as np
import csv

from chemlink.pipelines.docking.docking_pipeline import DockingPipeline
from chemlink.pipelines.docking.steps.docking_analysis import DockingAnalysis


def test_extract_pose_coords_accepts_records_without_chain(tmp_path):
    dlg = tmp_path / "pose.dlg"
    dlg.write_text(
        "Run:   404 / 1000\n"
        "DOCKED: MODEL        404\n"
        "DOCKED: ATOM      1  O   UNL     1     -27.682 -36.801  10.091 -0.42 +0.07    -0.109 OA\n"
        "DOCKED: HETATM    2  C   UNL     1     -29.110 -36.034  10.073 +0.27 -0.30    +0.750 C\n"
        "DOCKED: ENDMDL\n",
        encoding="utf-8",
    )

    coords = DockingAnalysis(str(tmp_path))._extract_pose_coords(str(dlg), 404)

    np.testing.assert_allclose(
        coords,
        [[-27.682, -36.801, 10.091], [-29.110, -36.034, 10.073]],
    )


def test_extract_pose_coords_accepts_records_with_chain(tmp_path):
    dlg = tmp_path / "pose.dlg"
    dlg.write_text(
        "Run: 7 / 10\n"
        "DOCKED: ATOM      1  C   LIG A   1      12.345  -6.789   0.125  1.00  0.00     0.000 C\n"
        "DOCKED: ENDMDL\n",
        encoding="utf-8",
    )

    coords = DockingAnalysis(str(tmp_path))._extract_pose_coords(str(dlg), 7)

    np.testing.assert_allclose(coords, [[12.345, -6.789, 0.125]])


def test_analysis_exports_dynamics_inputs(tmp_path):
    protein = "REC"
    ligand = "LIGAND"
    dlg_dir = tmp_path / "docking_results" / protein / "dlg"
    dlg_dir.mkdir(parents=True)
    (dlg_dir / f"{protein}_{ligand}.dlg").write_text(
        "Run:   1 / 1\n"
        "DOCKED: USER    Estimated Free Energy of Binding    =  -7.50 kcal/mol\n"
        "DOCKED: MODEL        1\n"
        "DOCKED: ATOM      1  C1  LIG     1      12.345  -6.789   0.125  1.00  0.00     0.000 C\n"
        "DOCKED: ENDMDL\n",
        encoding="utf-8",
    )
    receptor_dir = tmp_path / "clean_receptors"
    receptor_dir.mkdir()
    (receptor_dir / f"{protein}_h.pdb").write_text(
        "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00  0.00           C\n"
        "END\n",
        encoding="utf-8",
    )

    result = DockingAnalysis(str(tmp_path), pdb_export_limit=1, max_workers=1).run()

    dynamics_dir = tmp_path / "docking_analysis" / "dynamics_inputs"
    assert result["outputs"]["dynamics_inputs_dir"] == str(dynamics_dir)
    assert list((dynamics_dir / "receptors").glob("*.pdb"))
    assert list((dynamics_dir / "ligands").glob("*.pdb"))
    assert list((dynamics_dir / "complexes").glob("*.pdb"))
    assert (dynamics_dir / "manifest.csv").is_file()


def test_vina_analysis_parses_all_models(tmp_path):
    protein, ligand = "REC", "LIG"
    pose_dir = tmp_path / "docking_results" / protein / "poses"
    pose_dir.mkdir(parents=True)
    (pose_dir / f"{protein}_{ligand}_out.pdbqt").write_text(
        "MODEL 1\n"
        "REMARK VINA RESULT:    -7.500      0.000      0.000\n"
        "ATOM      1  C   UNL     1      10.000  11.000  12.000  1.00  0.00     0.000 C\n"
        "ENDMDL\n"
        "MODEL 2\n"
        "REMARK VINA RESULT:    -6.750      1.000      2.000\n"
        "ATOM      1  C   UNL     1      11.000  12.000  13.000  1.00  0.00     0.000 C\n"
        "ENDMDL\n",
        encoding="utf-8",
    )

    result = DockingAnalysis(str(tmp_path), engine="vina", max_workers=1).run()

    assert result["parsed_poses"] == 2
    assert result["analyzed_ligands"] == 1
    with open(result["outputs"]["ranking_csv"], newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["best_affinity_kcal_mol"] == "-7.50"


def test_dual_engine_analysis_writes_consensus(tmp_path):
    protein, ligand = "REC", "LIG"
    dlg_dir = tmp_path / "docking_results" / protein / "dlg"
    dlg_dir.mkdir(parents=True)
    (dlg_dir / f"{protein}_{ligand}.dlg").write_text(
        "Run: 1 / 1\n"
        "DOCKED: USER Estimated Free Energy of Binding = -7.50 kcal/mol\n"
        "DOCKED: MODEL 1\n"
        "DOCKED: ATOM      1  C1  LIG     1      12.345  -6.789   0.125  1.00  0.00     0.000 C\n"
        "DOCKED: ENDMDL\n",
        encoding="utf-8",
    )
    clean = tmp_path / "clean_receptors"
    clean.mkdir()
    (clean / f"{protein}_h.pdb").write_text(
        "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00  0.00           C\nEND\n"
    )
    vina_dir = tmp_path / "cpu_validation" / "docking_results" / protein / "poses"
    vina_dir.mkdir(parents=True)
    (vina_dir / f"{protein}_{ligand}_out.pdbqt").write_text(
        "MODEL 1\nREMARK VINA RESULT: -7.000 0.000 0.000\n"
        "ATOM      1  C   UNL     1      10.000  11.000  12.000  1.00  0.00     0.000 C\nENDMDL\n"
    )

    pipeline = DockingPipeline(str(tmp_path), str(tmp_path), str(tmp_path))
    result = pipeline.run_docking_analysis(engine="both", pdb_export_limit=1, max_workers=1)

    assert result["gpu"]["parsed_poses"] == 1
    assert result["cpu"]["parsed_poses"] == 1
    with open(result["comparison_file"], newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["engines"] == "2"
    assert row["gpu_rank"] == row["vina_rank"] == "1"
