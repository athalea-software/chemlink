from pathlib import Path
import tempfile
import unittest

from chemlink.utils.cofactor_classifier import (
    CofactorError,
    classify_cofactor,
    find_ion_moleculetype,
    format_gro_atom_line,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CHARMM36_IONS_ITP = _REPO_ROOT / "forcefields" / "charmm36-jul2022.ff" / "ions.itp"


def _pdb_line(serial: int, name: str, resname: str, resseq: int,
              x: float, y: float, z: float, element: str) -> str:
    return (
        "HETATM"
        f"{serial:>5}"
        " "
        f"{name:<4}"
        " "
        f"{resname:>3}"
        " "
        "A"
        f"{resseq:>4}"
        "    "
        f"{x:>8.3f}{y:>8.3f}{z:>8.3f}"
        f"{1.00:>6.2f}{0.00:>6.2f}"
        "          "
        f"{element:>2}"
        "\n"
    )


class ClassifyCofactorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_single_heavy_atom_is_classified_as_ion(self):
        pdb = self.root / "zn.pdb"
        pdb.write_text(_pdb_line(1, "ZN", "ZN", 1, 10.0, 20.0, 30.0, "ZN"))

        result = classify_cofactor(str(pdb))

        self.assertEqual(result.kind, "ion")
        self.assertEqual(result.resname, "ZN")
        self.assertEqual(len(result.atoms), 1)
        self.assertEqual(result.atoms[0].element, "Zn")

    def test_multiple_heavy_atoms_are_classified_as_molecule(self):
        pdb = self.root / "so4.pdb"
        pdb.write_text(
            _pdb_line(1, "S", "SO4", 1, 0.0, 0.0, 0.0, "S")
            + _pdb_line(2, "O1", "SO4", 1, 1.5, 0.0, 0.0, "O")
            + _pdb_line(3, "O2", "SO4", 1, 0.0, 1.5, 0.0, "O")
            + _pdb_line(4, "O3", "SO4", 1, 0.0, 0.0, 1.5, "O")
        )

        result = classify_cofactor(str(pdb))

        self.assertEqual(result.kind, "molecule")
        self.assertEqual(result.resname, "SO4")
        self.assertEqual(len(result.atoms), 4)

    def test_hydrogen_only_atoms_are_rejected(self):
        pdb = self.root / "h_only.pdb"
        pdb.write_text(_pdb_line(1, "H1", "XXX", 1, 0.0, 0.0, 0.0, "H"))

        with self.assertRaises(CofactorError):
            classify_cofactor(str(pdb))

    def test_missing_file_is_rejected(self):
        with self.assertRaises(CofactorError):
            classify_cofactor(str(self.root / "missing.pdb"))


class FindIonMoleculetypeTests(unittest.TestCase):
    def test_exact_residue_name_match(self):
        itp = (
            "[ moleculetype ]\n"
            "; molname  nrexcl\n"
            "ZN         3\n\n"
            "[ atoms ]\n"
            "; id at_type res_nr res_name at_name cg_nr charge mass\n"
            "  1   ZN       1     ZN       ZN      1     2.000  65.37\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            itp_path = Path(directory) / "ions.itp"
            itp_path.write_text(itp)

            resname, atom_name = find_ion_moleculetype(itp_path, "ZN")

            self.assertEqual(resname, "ZN")
            self.assertEqual(atom_name, "ZN")

    def test_multi_atom_moleculetypes_are_never_matched(self):
        itp = (
            "[ moleculetype ]\n"
            "CO3        3\n\n"
            "[ atoms ]\n"
            "  1   CG2O6   1  CO3  C1  1  1.42  12.011\n"
            "  2   OG2D2   1  CO3  O1  1 -1.14  15.999\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            itp_path = Path(directory) / "ions.itp"
            itp_path.write_text(itp)

            with self.assertRaises(CofactorError):
                find_ion_moleculetype(itp_path, "CO3")

    def test_unknown_ion_raises_with_available_list(self):
        itp = (
            "[ moleculetype ]\n"
            "NA         3\n\n"
            "[ atoms ]\n"
            "  1   Na   1  NA  NA  1  1.000  22.98977\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            itp_path = Path(directory) / "ions.itp"
            itp_path.write_text(itp)

            with self.assertRaisesRegex(CofactorError, "NA"):
                find_ion_moleculetype(itp_path, "MN")

    @unittest.skipUnless(_CHARMM36_IONS_ITP.is_file(), "vendored CHARMM36 force field not present")
    def test_matches_against_the_shipped_charmm36_force_field(self):
        for resname, expected_atom_name in (("ZN", "ZN"), ("MG", "MG"), ("NA", "NA"), ("CL", "CL")):
            with self.subTest(resname=resname):
                resolved, atom_name = find_ion_moleculetype(_CHARMM36_IONS_ITP, resname)
                self.assertEqual(resolved, resname)
                self.assertEqual(atom_name, expected_atom_name)


class FormatGroAtomLineTests(unittest.TestCase):
    def test_line_has_fixed_gro_width_and_units(self):
        line = format_gro_atom_line(1, "ZN", "ZN", 42, 1.0, 2.0, 3.0)

        self.assertEqual(len(line.rstrip("\n")), 44)
        self.assertTrue(line.startswith(f"{1:5d}"))
        self.assertIn("ZN", line)
        self.assertIn(f"{42:5d}", line)
        self.assertIn("   1.000   2.000   3.000", line)
