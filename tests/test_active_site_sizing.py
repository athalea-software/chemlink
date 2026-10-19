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
    assert box["requested_size"] == (25.0, 26.0, 22.0)
    assert box["npts"] == (68, 70, 60)
    assert box["ligand_count"] == 2


def _hetatm(serial, name, residue, chain, number, x, y, z, element):
    return (
        f"HETATM{serial:5d} {name:>4s} {residue:>3s} {chain}{number:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}\n"
    )


def test_cocrystal_ligand_supplies_center_only(tmp_path):
    originals = tmp_path / "originals"
    ligands = tmp_path / "ligands"
    originals.mkdir()
    ligands.mkdir()
    (originals / "target.pdb").write_text(
        _hetatm(1, "C1", "LIG", "B", 9, 10, 20, 30, "C")
        + _hetatm(2, "N1", "LIG", "B", 9, 14, 22, 32, "N")
        # Water must not become a second candidate.
        + _hetatm(3, "O", "HOH", "B", 10, 100, 100, 100, "O")
    )
    (ligands / "new.pdbqt").write_text(
        _atom(1, -2, -3, -4) + _atom(2, 2, 3, 4), encoding="utf-8"
    )
    step = ActiveSiteDetection(
        receptor_path=str(tmp_path / "prepared"),
        ligand_path=str(ligands),
        output_path=str(tmp_path / "output"),
        receptor_input_path=str(originals),
        ligand_box_padding=8.0,
        auto_cocrystal_center=True,
    )
    step._ligand_sized_box = step._size_grid_from_ligands()

    box = step._cocrystal_box_for("target_clean")

    assert box is not None
    assert box["mode"] == "cocrystal-centered"
    assert box["center"] == (12.0, 21.0, 31.0)
    assert box["center_source_ligands"] == ["LIG B:9"]
    assert box["requested_size"] == (12.0, 14.0, 16.0)
    assert box["ligand_max_spans"] == (4.0, 6.0, 8.0)


def test_more_than_two_cocrystal_candidates_is_ambiguous(tmp_path):
    originals = tmp_path / "originals"
    ligands = tmp_path / "ligands"
    originals.mkdir()
    ligands.mkdir()
    lines = "".join(
        _hetatm(i, "C1", f"L{i}", "A", i, i, i, i, "C")
        for i in range(1, 4)
    )
    (originals / "target.pdb").write_text(lines)
    (ligands / "new.pdbqt").write_text(
        _atom(1, 0, 0, 0) + _atom(2, 1, 1, 1), encoding="utf-8"
    )
    step = ActiveSiteDetection(
        receptor_path=str(tmp_path / "prepared"),
        ligand_path=str(ligands),
        output_path=str(tmp_path / "output"),
        receptor_input_path=str(originals),
        auto_cocrystal_center=True,
    )
    step._ligand_sized_box = step._size_grid_from_ligands()

    assert step._cocrystal_box_for("target_clean") is None


def test_two_equivalent_cocrystal_copies_use_combined_center(tmp_path):
    originals = tmp_path / "originals"
    ligands = tmp_path / "ligands"
    originals.mkdir()
    ligands.mkdir()
    (originals / "target.pdb").write_text(
        _hetatm(1, "C1", "LIG", "A", 1, 0, 0, 0, "C")
        + _hetatm(2, "N1", "LIG", "A", 1, 2, 0, 0, "N")
        + _hetatm(3, "C1", "LIG", "B", 1, 10, 0, 0, "C")
        + _hetatm(4, "N1", "LIG", "B", 1, 12, 0, 0, "N")
    )
    (ligands / "new.pdbqt").write_text(
        _atom(1, 0, 0, 0) + _atom(2, 1, 1, 1), encoding="utf-8"
    )
    step = ActiveSiteDetection(
        receptor_path=str(tmp_path / "prepared"),
        ligand_path=str(ligands),
        output_path=str(tmp_path / "output"),
        receptor_input_path=str(originals),
        auto_cocrystal_center=True,
    )
    step._ligand_sized_box = step._size_grid_from_ligands()

    box = step._cocrystal_box_for("target_clean")

    assert box is not None
    assert box["center"] == (6.0, 0.0, 0.0)
    assert box["center_source_ligands"] == ["LIG A:1", "LIG B:1"]


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


def test_cocrystal_center_is_opt_in(tmp_path):
    step = ActiveSiteDetection(
        receptor_path=str(tmp_path),
        ligand_path=str(tmp_path),
        output_path=str(tmp_path),
        receptor_input_path=str(tmp_path),
    )

    assert step.auto_cocrystal_center is False
