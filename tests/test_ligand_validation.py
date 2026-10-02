from pathlib import Path
import tempfile
import unittest

from rdkit import Chem
from rdkit.Chem import AllChem

from pipelines.ligand_validation import (
    LigandValidationError,
    prepare_valid_ligand,
    require_valid_ligand,
    validate_ligand,
)


def _write_3d_mol(path: Path, smiles: str) -> Chem.Mol:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=42) == 0
    AllChem.UFFOptimizeMolecule(mol)
    Chem.MolToMolFile(mol, str(path))
    return mol


class LigandValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_accepts_connected_optimized_ligand(self):
        path = self.root / "ethanol.mol"
        _write_3d_mol(path, "CCO")
        result = require_valid_ligand(str(path), report_path=str(self.root / "report.json"))
        self.assertTrue(result.valid)
        self.assertEqual(result.fragments, 1)
        self.assertTrue((self.root / "report.json").is_file())

    def test_rejects_disconnected_fragments(self):
        path = self.root / "salt.mol"
        _write_3d_mol(path, "CC.[Na+]")
        result = validate_ligand(str(path))
        self.assertFalse(result.valid)
        self.assertTrue(any("disconnected" in error for error in result.errors))

    def test_rejects_broken_bond_distance(self):
        path = self.root / "broken.mol"
        mol = _write_3d_mol(path, "CCO")
        conf = mol.GetConformer()
        point = conf.GetAtomPosition(1)
        point.x += 8.0
        conf.SetAtomPosition(1, point)
        Chem.MolToMolFile(mol, str(path))
        result = validate_ligand(str(path))
        self.assertFalse(result.valid)
        self.assertTrue(any("bond length" in error for error in result.errors))

    def test_rejects_severe_nonbonded_clash(self):
        path = self.root / "clash.mol"
        mol = _write_3d_mol(path, "CCCC")
        conf = mol.GetConformer()
        conf.SetAtomPosition(3, conf.GetAtomPosition(0))
        Chem.MolToMolFile(mol, str(path))
        result = validate_ligand(str(path))
        self.assertFalse(result.valid)
        self.assertTrue(any("atomic clash" in error for error in result.errors))

    def test_pdbqt_content_under_a_pdb_extension_does_not_crash_rdkit(self):
        # Regression: docking tools often write PDBQT-typed atom records
        # (AutoDock types like "A" for aromatic carbon) to a file named
        # .pdb. RDKit's MolFromPDBBlock treats the last column as a literal
        # element symbol; "A" isn't one, and previously crashed the whole
        # process with a fatal RDKit C++ invariant violation instead of a
        # catchable validation error.
        path = self.root / "docked_output.pdb"
        path.write_text(
            "ROOT\n"
            "ATOM      1  C1  UNL     1       0.000   0.000   0.000  0.00  0.00    +0.000 A \n"
            "ATOM      2  N1  UNL     1       1.400   0.000   0.000  0.00  0.00    +0.000 NA\n"
            "ENDROOT\n"
            "TORSDOF 0\n",
            encoding="utf-8",
        )
        result = validate_ligand(str(path))
        self.assertEqual(result.atoms, 2)

    def test_pdbqt_macrocycle_glue_atoms_are_dropped_not_misread_as_element(self):
        # Regression: Meeko's flexible-macrocycle handling breaks a large
        # ring and inserts massless "glue" placeholder atoms (atom name "G",
        # AutoDock type exactly "G" + digits, e.g. "G0"/"G1") at the two
        # ends to keep them co-located during docking. That type isn't a
        # real element; routing it through the element map previously wrote
        # the literal, nonexistent symbol "G" into the rebuilt PDB block,
        # which made RDKit reject the whole structure even though the real
        # atoms being docked were perfectly fine. Real ring-opening atoms
        # (e.g. "CG0") keep their element prefix and already parsed
        # correctly via the atom-name fallback — only the bare "G0"/"G1"
        # dummy type is new here.
        path = self.root / "macrocycle.pdbqt"
        path.write_text(
            "ROOT\n"
            "ATOM      1  C1  UNL     1       0.000   0.000   0.000  0.00  0.00    +0.000 C \n"
            "ATOM      2  G   UNL     1      10.000  10.000  10.000  0.00  0.00    +0.000 G0\n"
            "ENDROOT\n"
            "BRANCH   1   3\n"
            "ATOM      3  C2  UNL     1       1.540   0.000   0.000  0.00  0.00    +0.000 C \n"
            "ATOM      4  C3  UNL     1       3.080   0.000   0.000  0.00  0.00    +0.000 C \n"
            "ATOM      5  C4  UNL     1       4.620   0.000   0.000  0.00  0.00    +0.000 C \n"
            "ATOM      6  G   UNL     1      20.000  20.000  20.000  0.00  0.00    +0.000 G0\n"
            "ENDBRANCH   1   3\n"
            "TORSDOF 2\n",
            encoding="utf-8",
        )
        result = validate_ligand(str(path))
        self.assertTrue(result.valid, result.errors)
        self.assertEqual(result.atoms, 4)
        self.assertEqual(result.fragments, 1)

    def _write_flat_pdb(self, smiles: str) -> Path:
        # A sketch exported to PDB: real graph, every z coordinate zero.
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        AllChem.Compute2DCoords(mol)
        path = self.root / "flat.pdb"
        Chem.MolToPDBFile(mol, str(path), flavor=4)
        return path

    def test_flat_drawing_with_sp3_centre_needs_3d_instead_of_angle_errors(self):
        path = self._write_flat_pdb("O=C(C=Cc1ccccc1)CC(=O)C")
        result = validate_ligand(str(path))
        self.assertTrue(result.valid, result.errors)
        self.assertTrue(result.needs_3d)

    def test_planar_aromatic_molecule_is_not_flagged_as_2d(self):
        path = self.root / "benzene.mol"
        _write_3d_mol(path, "c1ccccc1")
        result = validate_ligand(str(path))
        self.assertTrue(result.valid)
        self.assertFalse(result.needs_3d)

    def test_prepare_regenerates_3d_and_keeps_atom_order(self):
        path = self._write_flat_pdb("O=C(C=Cc1ccccc1)CC(=O)C")
        out_dir = self.root / "run"
        prepared, result = prepare_valid_ligand(str(path), str(out_dir))
        self.assertNotEqual(prepared, str(path))
        self.assertTrue(result.valid, result.errors)
        self.assertFalse(result.needs_3d)
        before = Chem.MolFromPDBFile(str(path), removeHs=False)
        after = Chem.MolFromPDBFile(prepared, removeHs=False)
        self.assertEqual(
            [a.GetSymbol() for a in before.GetAtoms()],
            [a.GetSymbol() for a in after.GetAtoms()],
        )
        self.assertGreater(max(abs(z) for _, _, z in after.GetConformer().GetPositions()), 0.3)

    def test_prepare_keeps_valid_3d_input_untouched(self):
        path = self.root / "ethanol.mol"
        _write_3d_mol(path, "CCO")
        prepared, _ = prepare_valid_ligand(str(path), str(self.root / "run"))
        self.assertEqual(prepared, str(path))

    def test_prepare_still_rejects_broken_chemistry(self):
        path = self.root / "salt.mol"
        _write_3d_mol(path, "CC.[Na+]")
        with self.assertRaises(LigandValidationError):
            prepare_valid_ligand(str(path), str(self.root / "run"))


if __name__ == "__main__":
    unittest.main()
