import json
from pathlib import Path
import tempfile
import unittest

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors

from cli.main import build_parser
from pipelines.dynamics.fep_pipeline import (
    FEPPreparationError,
    _distance,
    _is_rigid_bond,
    _merge_gro,
    _patch_complex_topology,
    _production_mdp,
    _split_ligand_topology,
    _solvent_topology,
    compare_ligands,
    materialize_boresch_restraints,
    prepare_fep_ligand,
    prepare_fep_run,
    select_boresch_anchors,
)


class FEPPipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _pdb_ligand(self, name: str, smiles: str) -> Path:
        molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
        self.assertEqual(AllChem.EmbedMolecule(molecule, randomSeed=7), 0)
        path = self.root / name
        Chem.MolToPDBFile(molecule, str(path))
        return path

    def test_cli_exposes_standalone_fep_options(self):
        args = build_parser().parse_args([
            "dynamic", "fep", "--protein", "protein.pdb",
            "--ligand-a", "a.pdb", "--ligand-b", "b.pdb",
            "--charge-a", "0", "--charge-b", "0", "--prepare-only",
        ])
        self.assertEqual(args.dyn_type, "fep")
        self.assertEqual(args.charge_a, 0)
        self.assertEqual(args.charge_mode, "auto")
        self.assertEqual(args.ligand_ph, 7.4)
        self.assertEqual(args.fep_type, "auto")
        self.assertTrue(args.prepare_only)
        self.assertFalse(args.build_only)

    def test_cli_exposes_build_only_flag(self):
        args = build_parser().parse_args([
            "dynamic", "fep", "--protein", "protein.pdb",
            "--ligand-a", "a.pdb", "--ligand-b", "b.pdb", "--build-only",
        ])
        self.assertTrue(args.build_only)
        self.assertFalse(args.prepare_only)

    def test_full_workflow_is_the_default(self):
        # Neither --prepare-only nor --build-only: the CLI help promises the
        # full prepare->build->pilot->produce->analyze chain runs by default.
        args = build_parser().parse_args([
            "dynamic", "fep", "--protein", "protein.pdb",
            "--ligand-a", "a.pdb", "--ligand-b", "b.pdb",
        ])
        self.assertFalse(args.prepare_only)
        self.assertFalse(args.build_only)

    def test_build_only_and_prepare_only_are_mutually_exclusive(self):
        from cli.main import _run_dynamic_fep

        with tempfile.TemporaryDirectory() as tmp:
            protein = Path(tmp) / "protein.pdb"
            protein.write_text("REMARK dummy\n", encoding="utf-8")
            ligand_a = Path(tmp) / "a.pdb"
            ligand_a.write_text("REMARK dummy\n", encoding="utf-8")
            ligand_b = Path(tmp) / "b.pdb"
            ligand_b.write_text("REMARK dummy\n", encoding="utf-8")
            args = build_parser().parse_args([
                "dynamic", "fep",
                "--protein", str(protein), "--ligand-a", str(ligand_a), "--ligand-b", str(ligand_b),
                "--prepare-only", "--build-only",
            ])
            # Fails on the mutual-exclusion check, before ever touching the
            # (deliberately invalid) dummy ligand/protein content above.
            self.assertEqual(_run_dynamic_fep(args), 1)

    def test_cli_exposes_fep_pilot_continuation(self):
        args = build_parser().parse_args([
            "dynamic", "fep-pilot", "--run-dir", "FEP_ABFE_24",
        ])
        self.assertEqual(args.dyn_type, "fep-pilot")
        self.assertEqual(args.run_dir, "FEP_ABFE_24")

    def test_pdb_is_converted_to_validated_sdf(self):
        pdb = self._pdb_ligand("methane.pdb", "C")
        output = self.root / "methane.sdf"

        prepared, molecule = prepare_fep_ligand(
            pdb, output, label="A", charge=0,
        )

        self.assertTrue(output.is_file())
        self.assertEqual(prepared.charge, 0)
        self.assertGreater(molecule.GetNumBonds(), 0)
        self.assertIn("DetermineBonds", prepared.method)

    def test_pdb_requires_explicit_charge(self):
        pdb = self._pdb_ligand("methane.pdb", "C")
        with self.assertRaisesRegex(FEPPreparationError, "--charge-a"):
            prepare_fep_ligand(pdb, self.root / "out.sdf", label="A", charge=None)

    def test_mislabeled_pdbqt_autodock_aromatic_type_is_normalized(self):
        pdb = self.root / "docked.pdb"
        pdb.write_text(
            "ROOT\n"
            "ATOM      1  C   UNL     1       0.000   0.000   0.000  0.00  0.00    +0.000 A \n"
            "ATOM      2  H   UNL     1       1.090   0.000   0.000  0.00  0.00    +0.000 HD\n"
            "ATOM      3  H   UNL     1      -1.090   0.000   0.000  0.00  0.00    +0.000 HD\n"
            "ATOM      4  H   UNL     1       0.000   1.090   0.000  0.00  0.00    +0.000 HD\n"
            "ATOM      5  H   UNL     1       0.000   0.000   1.090  0.00  0.00    +0.000 HD\n"
            "ENDROOT\nTORSDOF 0\n"
        )

        prepared, molecule = prepare_fep_ligand(
            pdb, self.root / "normalized.sdf", label="A", charge=0,
        )

        self.assertEqual(molecule.GetAtomWithIdx(0).GetSymbol(), "C")
        self.assertIn("PDBQT-formatted PDB", prepared.method)

    def test_pdbqt_conversion_restores_implicit_hydrogens(self):
        pdb = self.root / "fragment.pdb"
        pdb.write_text(
            "ROOT\n"
            "ATOM      1  C   UNL     1       0.000   0.000   0.000  0.00  0.00    +0.000 C \n"
            "ENDROOT\nTORSDOF 0\n"
        )
        _, molecule = prepare_fep_ligand(
            pdb, self.root / "fragment.sdf", label="A", charge=0,
        )
        self.assertEqual(rdMolDescriptors.CalcMolFormula(molecule), "CH4")

    def test_odd_electron_ligand_is_rejected_before_acpype(self):
        radical = Chem.MolFromSmiles("[CH3]")
        radical = Chem.AddHs(radical)
        self.assertEqual(AllChem.EmbedMolecule(radical, randomSeed=7), 0)
        source = self.root / "radical.sdf"
        writer = Chem.SDWriter(str(source))
        writer.write(radical)
        writer.close()
        with self.assertRaisesRegex(FEPPreparationError, "odd electron count"):
            prepare_fep_ligand(source, self.root / "out.sdf", label="A", charge=0)

    def test_initial_rbfe_rejects_charge_changes(self):
        neutral = Chem.MolFromSmiles("CC")
        charged = Chem.MolFromSmiles("C[NH3+]")
        with self.assertRaisesRegex(FEPPreparationError, "equal net charges"):
            compare_ligands(neutral, charged)

    def _ring_and_tail_ligand(self) -> Chem.Mol:
        """An aromatic ring with a rotatable ethyl tail on the ring-attachment carbon.

        Atom 0 (the ring carbon) neighbours atom 1 first (the tail, connected
        by an ordinary rotatable single bond) and atoms 3/7 second (both ring
        bonds) — this SMILES atom order is deliberately chosen so a "pick the
        first heavy neighbour" heuristic would grab the rotatable one.
        """
        molecule = Chem.AddHs(Chem.MolFromSmiles("c1(CC)ccccc1"))
        self.assertEqual(AllChem.EmbedMolecule(molecule, randomSeed=7), 0)
        return Chem.RemoveHs(molecule)

    def test_is_rigid_bond_distinguishes_ring_from_rotatable(self):
        ligand = self._ring_and_tail_ligand()
        self.assertEqual([n.GetIdx() for n in ligand.GetAtomWithIdx(0).GetNeighbors()], [1, 3, 7])
        self.assertFalse(_is_rigid_bond(ligand, 0, 1))  # ring carbon -> tail CH2: rotatable
        self.assertTrue(_is_rigid_bond(ligand, 0, 3))   # ring carbon -> ring carbon: rigid
        self.assertTrue(_is_rigid_bond(ligand, 0, 7))   # ring carbon -> ring carbon: rigid

    def test_select_boresch_anchors_prefers_a_rigid_l1_l2_bond(self):
        # Regression: an anchor chosen across a rotatable bond looks fine at
        # t=0 but drifts out of the valid angle/dihedral range once that bond
        # rotates during MD — reproduced running a real fep-pilot.
        ligand = self._ring_and_tail_ligand()
        conformer = ligand.GetConformer()
        attachment = conformer.GetAtomPosition(0)

        protein = self.root / "protein.pdb"
        # CA placed a hair from atom 0 itself (the ring-attachment carbon) so
        # it is unambiguously the closest candidate (L1) regardless of how
        # the embedded conformer happens to fold the ethyl tail in space —
        # this test only cares about the L1->L2 choice, not the resulting
        # distance's "valid"/"manual_review" status.
        ca = (attachment.x + 0.05, attachment.y, attachment.z)
        protein.write_text(
            f"ATOM      1  N   ALA A   1  {ca[0]-1.0:8.3f}{ca[1]:8.3f}{ca[2]:8.3f}  1.00  0.00           N\n"
            f"ATOM      2  CA  ALA A   1  {ca[0]:8.3f}{ca[1]:8.3f}{ca[2]:8.3f}  1.00  0.00           C\n"
            f"ATOM      3  C   ALA A   1  {ca[0]-0.5:8.3f}{ca[1]+1.3:8.3f}{ca[2]:8.3f}  1.00  0.00           C\n"
        )

        anchors = select_boresch_anchors(protein, ligand)

        self.assertEqual(anchors["ligand"][0]["index"], 0)
        self.assertIn(anchors["ligand"][1]["index"], (3, 7))  # a ring neighbour, not the tail (1)

    def test_omitted_lambda_windows_defaults_to_20_for_rbfe(self):
        protein = self.root / "protein.pdb"
        protein.write_text(
            "ATOM      1  N   ALA A   1       0.000   0.000   3.000  1.00  0.00           N\n"
            "ATOM      2  CA  ALA A   1       1.300   0.000   3.000  1.00  0.00           C\n"
            "ATOM      3  C   ALA A   1       1.800   1.300   3.000  1.00  0.00           C\n"
        )
        ligand_a = self._pdb_ligand("a.pdb", "CC")
        ligand_b = self._pdb_ligand("b.pdb", "CCO")
        report = prepare_fep_run({
            "work_dir": str(self.root / "rbfe_default"), "protein": str(protein),
            "ligand_a": str(ligand_a), "ligand_b": str(ligand_b),
            "charge_a": 0, "charge_b": 0, "smiles_a": None, "smiles_b": None,
            "fep_type": "auto", "replicas": 1,
            "equilibration_ns": 0.1, "production_ns": 0.2,
            "temperature_k": 300.0, "prepare_only": True,
        })
        self.assertEqual(report["method"], "RBFE")
        self.assertEqual(report["lambda_windows"], 20)

    def test_omitted_lambda_windows_defaults_to_30_for_abfe(self):
        protein = self.root / "protein.pdb"
        protein.write_text(
            "ATOM      1  N   ALA A   1       0.000   0.000   3.000  1.00  0.00           N\n"
            "ATOM      2  CA  ALA A   1       1.300   0.000   3.000  1.00  0.00           C\n"
            "ATOM      3  C   ALA A   1       1.800   1.300   3.000  1.00  0.00           C\n"
        )
        ligand_a = self._pdb_ligand("a.pdb", "CCCO")
        ligand_b = self._pdb_ligand("b.pdb", "c1ccccc1N")
        report = prepare_fep_run({
            "work_dir": str(self.root / "abfe_default"), "protein": str(protein),
            "ligand_a": str(ligand_a), "ligand_b": str(ligand_b),
            "charge_a": 0, "charge_b": 0, "smiles_a": None, "smiles_b": None,
            "fep_type": "auto", "replicas": 1,
            "equilibration_ns": 0.1, "production_ns": 0.2,
            "temperature_k": 300.0, "prepare_only": True,
        })
        self.assertEqual(report["method"], "ABFE")
        self.assertEqual(report["lambda_windows"], 30)

    def test_explicit_lambda_windows_overrides_the_method_default(self):
        protein = self.root / "protein.pdb"
        protein.write_text(
            "ATOM      1  N   ALA A   1       0.000   0.000   3.000  1.00  0.00           N\n"
            "ATOM      2  CA  ALA A   1       1.300   0.000   3.000  1.00  0.00           C\n"
            "ATOM      3  C   ALA A   1       1.800   1.300   3.000  1.00  0.00           C\n"
        )
        ligand_a = self._pdb_ligand("a.pdb", "CC")
        ligand_b = self._pdb_ligand("b.pdb", "CCO")
        report = prepare_fep_run({
            "work_dir": str(self.root / "rbfe_explicit"), "protein": str(protein),
            "ligand_a": str(ligand_a), "ligand_b": str(ligand_b),
            "charge_a": 0, "charge_b": 0, "smiles_a": None, "smiles_b": None,
            "fep_type": "auto", "replicas": 1, "lambda_windows": 6,
            "equilibration_ns": 0.1, "production_ns": 0.2,
            "temperature_k": 300.0, "prepare_only": True,
        })
        self.assertEqual(report["lambda_windows"], 6)

    def test_cli_lambda_windows_defaults_to_none(self):
        args = build_parser().parse_args([
            "dynamic", "fep", "--protein", "protein.pdb",
            "--ligand-a", "a.pdb", "--ligand-b", "b.pdb",
        ])
        self.assertIsNone(args.lambda_windows)

    def test_prepare_only_creates_two_leg_window_layout(self):
        protein = self.root / "protein.pdb"
        protein.write_text(
            "ATOM      1  N   ALA A   1       0.000   0.000   3.000  1.00  0.00           N\n"
            "ATOM      2  CA  ALA A   1       1.300   0.000   3.000  1.00  0.00           C\n"
            "ATOM      3  C   ALA A   1       1.800   1.300   3.000  1.00  0.00           C\n"
        )
        ligand_a = self._pdb_ligand("a.pdb", "CC")
        ligand_b = self._pdb_ligand("b.pdb", "CCO")
        run = self.root / "run"
        config = {
            "work_dir": str(run), "protein": str(protein),
            "ligand_a": str(ligand_a), "ligand_b": str(ligand_b),
            "charge_a": 0, "charge_b": 0, "smiles_a": None, "smiles_b": None,
            "replicas": 2, "lambda_windows": 3,
            "equilibration_ns": 0.1, "production_ns": 0.2,
            "temperature_k": 300.0, "prepare_only": True,
        }

        report = prepare_fep_run(config)

        self.assertEqual(report["state_count"], 12)
        self.assertEqual(report["status"], "prepared")
        self.assertTrue((run / "common/ligand_A_normalized.sdf").is_file())
        self.assertTrue((run / "complex/replica_01/lambda_00/production.mdp").is_file())
        self.assertTrue((run / "solvent/replica_02/lambda_02/state.json").is_file())
        mapping = json.loads((run / "common/atom_mapping.json").read_text())
        self.assertGreaterEqual(mapping["coverage"], 0.5)

    def test_auto_selects_two_independent_abfe_calculations(self):
        protein = self.root / "protein.pdb"
        protein.write_text(
            "ATOM      1  N   ALA A   1       0.000   0.000   3.000  1.00  0.00           N\n"
            "ATOM      2  CA  ALA A   1       1.300   0.000   3.000  1.00  0.00           C\n"
            "ATOM      3  C   ALA A   1       1.800   1.300   3.000  1.00  0.00           C\n"
        )
        ligand_a = self._pdb_ligand("a.pdb", "CCCO")
        ligand_b = self._pdb_ligand("b.pdb", "c1ccccc1N")
        run = self.root / "abfe"
        report = prepare_fep_run({
            "work_dir": str(run), "protein": str(protein),
            "ligand_a": str(ligand_a), "ligand_b": str(ligand_b),
            "charge_a": 0, "charge_b": 0, "smiles_a": None, "smiles_b": None,
            "fep_type": "auto", "replicas": 1, "lambda_windows": 3,
            "equilibration_ns": 0.1, "production_ns": 0.2,
            "temperature_k": 300.0, "prepare_only": True,
        })

        self.assertEqual(report["method"], "ABFE")
        self.assertEqual(report["independent_ligands"], ["A", "B"])
        self.assertTrue((run / "common/abfe_restraints.json").is_file())
        self.assertTrue((run / "common/ligand_A_boresch_anchors.json").is_file())
        self.assertTrue((run / "pilot_plan.json").is_file())
        self.assertTrue((run / "ligand_A/complex/replica_01/lambda_00/production.mdp").is_file())
        self.assertTrue((run / "ligand_B/solvent/replica_01/lambda_02/production.mdp").is_file())
        mdp = (run / "ligand_A/complex/replica_01/lambda_00/production.mdp").read_text()
        self.assertIn("bonded-lambdas", mdp)
        self.assertIn("coul-lambdas", mdp)
        self.assertIn("vdw-lambdas", mdp)

    def test_abfe_system_topologies_are_portable_and_include_ligand(self):
        source_top = self.root / "protein.top"
        source_top.write_text(
            '#include "amber99sb-ildn.ff/forcefield.itp"\n'
            '[ moleculetype ]\nProtein 3\n[ atoms ]\n1 CT 1 ALA CA 1 0 12\n'
            '#include "amber99sb-ildn.ff/tip3p.itp"\n'
            '#include "amber99sb-ildn.ff/ions.itp"\n\n'
            '[ system ]\nProtein\n\n[ molecules ]\nProtein 1\n'
        )
        complex_top = self.root / "complex.top"
        complex_top.write_text(source_top.read_text())
        _patch_complex_topology(complex_top, Path("ligand_atomtypes.itp"), Path("ligand.itp"))
        self.assertLess(
            complex_top.read_text().index('"ligand_atomtypes.itp"'),
            complex_top.read_text().index("[ moleculetype ]"),
        )
        self.assertIn('#include "ligand.itp"', complex_top.read_text())
        self.assertIn("LIG                 1", complex_top.read_text())

        solvent_top = self.root / "solvent.top"
        _solvent_topology(
            source_top, Path("ligand_atomtypes.itp"), Path("ligand.itp"), solvent_top,
        )
        self.assertIn('#include "ligand.itp"', solvent_top.read_text())
        self.assertNotIn(str(self.root), solvent_top.read_text())

    def test_acpype_atomtypes_are_split_from_ligand_molecule(self):
        source = self.root / "LIG_GMX.itp"
        source.write_text(
            "; generated\n[ atomtypes ]\nca ca 0 0 A 0.3 0.1\n\n"
            "[ moleculetype ]\nLIG 3\n[ atoms ]\n1 ca 1 LIG C1 1 0 12\n"
        )
        atomtypes = self.root / "ligand_atomtypes.itp"
        molecule = self.root / "ligand.itp"
        _split_ligand_topology(source, atomtypes, molecule)
        self.assertIn("[ atomtypes ]", atomtypes.read_text())
        self.assertNotIn("[ atomtypes ]", molecule.read_text())
        self.assertIn("[ moleculetype ]", molecule.read_text())

    def test_abfe_complex_gro_combines_protein_and_ligand_atoms(self):
        protein = self.root / "protein.gro"
        ligand = self.root / "ligand.gro"
        protein.write_text("Protein\n1\n    1ALA     CA    1   0.0   0.0   0.0\n1 1 1\n")
        ligand.write_text("Ligand\n1\n    1LIG     C1    1   0.1   0.1   0.1\n1 1 1\n")
        output = self.root / "complex.gro"
        _merge_gro(protein, ligand, output)
        self.assertEqual(output.read_text().splitlines()[1], "2")

    @staticmethod
    def _gro_line(resnum: int, resname: str, atomname: str, atomnum: int, xyz) -> str:
        x, y, z = xyz
        return f"{resnum:5d}{resname:<5s}{atomname:>5s}{atomnum:5d}{x:8.3f}{y:8.3f}{z:8.3f}\n"

    def _write_boresch_fixture(self) -> Path:
        """Build a synthetic (non-GROMACS) ABFE run directory for materialize_boresch_restraints."""
        root = self.root / "abfe_run"
        (root / "common" / "protein_topology").mkdir(parents=True)
        (root / "run_config.json").write_text(json.dumps({"temperature_k": 310.0}))

        # Protein anchors: N=1, CA=2, C=3 of ALA 1, in the order select_boresch_anchors
        # would emit them (CA, N, C).
        protein_xyz = {"N": (0.0, 0.0, 0.0), "CA": (0.15, 0.0, 0.0), "C": (0.15, 0.15, 0.0)}
        (root / "common" / "protein_topology" / "protein.gro").write_text(
            "protein\n3\n"
            + self._gro_line(1, "ALA", "N", 1, protein_xyz["N"])
            + self._gro_line(1, "ALA", "CA", 2, protein_xyz["CA"])
            + self._gro_line(1, "ALA", "C", 3, protein_xyz["C"])
            + "1.0 1.0 1.0\n"
        )

        ligand_xyz = {"L1": (0.35, 0.0, 0.0), "L2": (0.35, 0.15, 0.0), "L3": (0.5, 0.15, 0.05)}
        anchors = {
            "protein": [
                {"serial": 2, "name": "CA", "residue": "ALA", "chain": "A", "residue_number": "1"},
                {"serial": 1, "name": "N", "residue": "ALA", "chain": "A", "residue_number": "1"},
                {"serial": 3, "name": "C", "residue": "ALA", "chain": "A", "residue_number": "1"},
            ],
            "ligand": [
                {"index": 0, "name": "C", "atomic_number": 6},
                {"index": 1, "name": "C", "atomic_number": 6},
                {"index": 2, "name": "C", "atomic_number": 6},
            ],
            "force_constants": {
                "distance_kj_mol_nm2": 4184.0,
                "angle_kj_mol_rad2": 41.84,
                "dihedral_kj_mol_rad2": 41.84,
            },
        }

        for label in ("A", "B"):
            (root / "common" / f"ligand_{label}_boresch_anchors.json").write_text(
                json.dumps(anchors)
            )
            system = root / f"ligand_{label}" / "complex" / "system"
            system.mkdir(parents=True)
            (system / "topol.top").write_text(
                "[ moleculetype ]\nProtein 1\n[ atoms ]\n"
                "1 N 1 ALA N 1 0 14\n2 CT 1 ALA CA 2 0 12\n3 C 1 ALA C 3 0 12\n"
                "[ molecules ]\nProtein 1\nLIG 1\n"
            )
            (system / "input.gro").write_text(
                "complex\n6\n"
                + self._gro_line(1, "ALA", "N", 1, protein_xyz["N"])
                + self._gro_line(1, "ALA", "CA", 2, protein_xyz["CA"])
                + self._gro_line(1, "ALA", "C", 3, protein_xyz["C"])
                + self._gro_line(2, "LIG", "L1", 4, ligand_xyz["L1"])
                + self._gro_line(2, "LIG", "L2", 5, ligand_xyz["L2"])
                + self._gro_line(2, "LIG", "L3", 6, ligand_xyz["L3"])
                + "1.0 1.0 1.0\n"
            )
        return root

    def test_materialize_boresch_restraints_maps_global_atom_indices(self):
        root = self._write_boresch_fixture()
        result = materialize_boresch_restraints(root)

        self.assertEqual(result["status"], "materialized")
        self.assertEqual(result["temperature_k"], 310.0)
        for label in ("A", "B"):
            record = result["ligands"][label]
            # protein order in anchors is (CA, N, C) -> global indices (2, 1, 3)
            self.assertEqual(record["protein_indices"], [2, 1, 3])
            # ligand indices offset by the 3-atom protein.gro count: 3 + idx + 1
            self.assertEqual(record["ligand_indices"], [4, 5, 6])
            expected_distance = _distance((0.15, 0.0, 0.0), (0.35, 0.0, 0.0))
            self.assertAlmostEqual(record["geometry"]["distance_nm"], expected_distance)
            self.assertEqual(record["status"], "grompp_pending")

            topology_text = (root / f"ligand_{label}/complex/system/topol.top").read_text()
            self.assertIn("; ChemLink Boresch orientational restraints", topology_text)
            self.assertIn("[ intermolecular_interactions ]", topology_text)
            self.assertIn("[ bonds ]", topology_text)
            self.assertIn("[ angles ]", topology_text)
            self.assertIn("[ dihedrals ]", topology_text)
            self.assertTrue(
                (root / f"ligand_{label}/complex/system/boresch_restraints.json").is_file()
            )
        self.assertTrue((root / "boresch_mapping.json").is_file())

    def test_materialize_boresch_restraints_is_idempotent(self):
        root = self._write_boresch_fixture()
        materialize_boresch_restraints(root)
        materialize_boresch_restraints(root)

        for label in ("A", "B"):
            topology_text = (root / f"ligand_{label}/complex/system/topol.top").read_text()
            self.assertEqual(
                topology_text.count("; ChemLink Boresch orientational restraints"), 1,
            )

    def test_production_mdp_declares_a_tcoupl_group_grompp_requires(self):
        # Regression: grompp rejects `tcoupl`/`ref-t` without a matching
        # `tc-grps`/`tau_t` ("Invalid T coupling input: 0 groups, 1 ref-t
        # values and 0 tau-t values") — caught running a real fep-pilot.
        config = {"temperature_k": 300.0, "production_ns": 0.05}
        lambdas = [0.0, 0.5, 1.0]
        mdp = _production_mdp(1, lambdas, config, "ABFE")
        self.assertIn("tc-grps                  = System", mdp)
        self.assertIn("tau_t                    = 0.1", mdp)


if __name__ == "__main__":
    unittest.main()
