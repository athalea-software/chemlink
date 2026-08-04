import numpy as np

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
