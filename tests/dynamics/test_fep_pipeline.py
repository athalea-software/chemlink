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
    _merge_gro,
    _patch_complex_topology,
    _split_ligand_topology,
    _solvent_topology,
    compare_ligands,
    prepare_fep_ligand,
    prepare_fep_run,
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
        self.assertIn("restraint-lambdas", mdp)
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


if __name__ == "__main__":
    unittest.main()
