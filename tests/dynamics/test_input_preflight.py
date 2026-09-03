from pathlib import Path
import json
import tempfile
import unittest

from pipelines.dynamics.input_preflight import (
    inspect_pdb,
    prepare_protein_pdb,
    validate_input_roles,
)


def atom_line(serial, name, residue, residue_number, element, record="ATOM  "):
    return (
        f"{record}{serial:5d} {name:>4s} {residue:>3s} A{residue_number:4d}    "
        f"{0.0:8.3f}{0.0:8.3f}{0.0:8.3f}{1.0:6.2f}{0.0:6.2f}          {element:>2s}\n"
    )


class InputPreflightTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_rejects_fad_as_protein_and_protein_as_ligand(self):
        fad = self.root / "Ligando.pdb"
        fad.write_text(atom_line(1, "P", "FAD", 600, "P", "HETATM"))
        protein = self.root / "Proteina.pdb"
        protein.write_text(
            atom_line(1, "N", "ALA", 1, "N")
            + atom_line(2, "N", "GLY", 2, "N")
        )
        report = self.root / "input_preflight.json"

        with self.assertRaisesRegex(ValueError, "no parece una proteína"):
            validate_input_roles("pligand", [str(fad), str(protein)], report)

        payload = json.loads(report.read_text())
        self.assertEqual(payload["inputs"][0]["classification"], "small_molecule")
        self.assertEqual(payload["inputs"][1]["classification"], "protein")
        self.assertEqual(len(payload["errors"]), 2)

    def test_accepts_protein_then_fad(self):
        protein = self.root / "Proteina.pdb"
        protein.write_text(
            atom_line(1, "N", "ALA", 1, "N", "HETATM")
            + atom_line(2, "N", "GLY", 2, "N", "HETATM")
        )
        fad = self.root / "Ligando.pdb"
        fad.write_text(atom_line(1, "P", "FAD", 600, "P", "HETATM"))

        result = validate_input_roles(
            "pligand", [str(protein), str(fad)], self.root / "report.json"
        )

        self.assertEqual(result["errors"], [])

    def test_preparation_removes_duplicate_hydrogens_and_normalizes_records(self):
        source = self.root / "yasara.pdb"
        source.write_text(
            atom_line(1, "N", "ARG", 36, "N", "HETATM")
            + atom_line(2, "1HH", "ARG", 36, "H", "HETATM")
            + atom_line(3, "1HH", "ARG", 36, "H", "HETATM")
        )
        destination = self.root / "prepared.pdb"

        before = inspect_pdb(source)
        result = prepare_protein_pdb(source, destination)

        self.assertEqual(before["duplicate_atom_names"], 1)
        self.assertEqual(before["duplicate_hydrogen_names"], 1)
        self.assertEqual(result["removed_hydrogens"], 2)
        self.assertEqual(result["converted_protein_records"], 1)
        self.assertTrue(destination.read_text().startswith("ATOM  "))
        self.assertNotIn("1HH", destination.read_text())


if __name__ == "__main__":
    unittest.main()
