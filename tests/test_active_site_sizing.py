from chemlink.pipelines.docking.steps.active_site_detection import ActiveSiteDetection


def _atom(serial, x, y, z):
    return (
        f"ATOM  {serial:5d}  C   LIG     1    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00     0.000 C\n"
    )


def test_manual_center_sizes_grid_from_all_ligands(tmp_path):
    ligand_dir = tmp_path / "ligands"
    ligand_dir.mkdir()
    (ligand_dir / "wide_x.pdbqt").write_text(
        _atom(1, -7.5, -1, -2) + _atom(2, 7.5, 1, 2), encoding="utf-8"
    )
    (ligand_dir / "wide_yz.pdbqt").write_text(
        _atom(1, -2, -8, -6) + _atom(2, 2, 8, 6), encoding="utf-8"
    )
    step = ActiveSiteDetection(
        receptor_path=str(tmp_path / "receptors"),
        ligand_path=str(ligand_dir),
        output_path=str(tmp_path / "output"),
        manual_center=(1.0, 2.0, 3.0),
    )

    box = step._size_grid_from_ligands()

    assert box["ligand_max_spans"] == (15.0, 16.0, 12.0)
    assert box["requested_size"] == (45.0, 46.0, 42.0)
    assert box["npts"] == (120, 124, 112)
    assert box["ligand_count"] == 2


def test_manual_npts_without_center_is_rejected(tmp_path):
    try:
        ActiveSiteDetection(
            receptor_path=str(tmp_path),
            ligand_path=str(tmp_path),
            output_path=str(tmp_path),
            manual_npts=(40, 40, 40),
        )
    except ValueError as exc:
        assert "requires manual_center" in str(exc)
    else:
        raise AssertionError("manual_npts without manual_center was accepted")
