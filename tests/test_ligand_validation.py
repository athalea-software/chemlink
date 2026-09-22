from pathlib import Path
import tempfile
import unittest

from rdkit import Chem
from rdkit.Chem import AllChem

from pipelines.ligand_validation import require_valid_ligand, validate_ligand


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


if __name__ == "__main__":
    unittest.main()
