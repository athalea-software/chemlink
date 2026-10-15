from pathlib import Path
import tempfile
import unittest

from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

from chemlink.utils.receptor_cofactors import (
    _element,
    _max_matching,
    _perceive_framework,
    write_cofactor_pdbqt,
)
from chemlink.utils.receptor_processor import ReceptorProcessor, read_atom_residues

_HEME_PDB = Path(__file__).resolve().parents[1] / "data" / "test" / "cofactors" / "1tqn_heme.pdb"


def _heme_lines():
    return [line for line in _HEME_PDB.read_text().splitlines(keepends=True)
            if line.startswith("HETATM")]


class HemeTypingTests(unittest.TestCase):
    def test_framework_is_protoporphyrin_ix(self):
        organic = [line for line in _heme_lines() if _element(line) != "FE"]
        mol, _sp2 = _perceive_framework(organic)

        # Fe-free heme framework: two propionic acids, two vinyls, porphyrin.
        self.assertEqual(CalcMolFormula(mol), "C34H32N4O4")
        self.assertEqual(len(mol.GetSubstructMatches(Chem.MolFromSmarts("[CH2]=[CH]"))), 2)
        self.assertEqual(len(mol.GetSubstructMatches(Chem.MolFromSmarts("C(=O)[OH]"))), 2)

    def test_pdbqt_types_and_charges(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "heme.pdbqt"
            types = write_cofactor_pdbqt(str(_HEME_PDB), str(out))
            lines = out.read_text().splitlines()

        self.assertEqual(len(lines), 43)
        self.assertEqual(types, {"A", "C", "N", "OA", "Fe"})
        by_name = {line[12:16].strip(): line for line in lines}
        # The 16 pyrrole carbons are aromatic; meso carbons sit only in the
        # macrocycle, not in a 5/6-ring (as Open Babel typed them too).
        self.assertEqual(sum(line[77:79].strip() == "A" for line in lines), 16)
        self.assertEqual(by_name["CHA"][77:79].strip(), "C")
        self.assertEqual(float(by_name["FE"][70:76]), 0.0)
        for name in ("NA", "NB", "NC", "ND"):
            self.assertEqual(by_name[name][77:79].strip(), "N")
        # United-atom Gasteiger charges of a neutral framework sum to ~0.
        self.assertAlmostEqual(sum(float(line[70:76]) for line in lines), 0.0, delta=0.05)


class MaxMatchingTests(unittest.TestCase):
    def test_odd_cycle_needs_blossom(self):
        # 0-1 matched first leaves the 5-ring 1..5 needing an augmenting path
        # through an odd cycle to reach the perfect matching 0-1, 2-3, 4-5, 6-7.
        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 1), (3, 6), (6, 7)]
        matching = _max_matching(8, edges)
        self.assertEqual(len(matching), 4)
        covered = sorted(atom for pair in matching for atom in pair)
        self.assertEqual(covered, list(range(8)))


class ReadAtomResiduesTests(unittest.TestCase):
    def test_unusual_records_are_kept(self):
        records = (
            "ATOM      1  N   ALA A   1      11.104   6.134  -6.504  1.00  0.00           N\n"
            "HETATM    2  Q1  UNL B   5      12.000   6.000  -6.000  1.00  0.00           X\n"
            "HETATM    3  DU  DUM B   6      12.0x0   6.000  -6.000  1.00  0.00          Du\n"
            "HETATM    4 ZN    ZN B   7      12.000   6.000  -6.000\n"
            "ENDMDL\n"
            "ATOM      5  N   ALA A   2      11.104   6.134  -6.504  1.00  0.00           N\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "r.pdb"
            path.write_text(records)
            atoms = read_atom_residues(str(path))

            processor = ReceptorProcessor()
            processor._read_receptor(str(path))
            _, removed = processor._identify_atoms_to_remove()

        # Only the first model is read.
        self.assertEqual(atoms, [("ALA", "A", "1"), ("UNL", "B", "5"),
                                 ("DUM", "B", "6"), ("ZN", "B", "7")])
        self.assertEqual(removed, {"water": 0, "ligand": 2, "ion": 1, "other": 0})


if __name__ == "__main__":
    unittest.main()
