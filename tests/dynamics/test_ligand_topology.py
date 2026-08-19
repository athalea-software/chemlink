import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from pipelines.dynamics.steps.ligand_topology import sanitize_zero_periodicity_dihedrals


class LigandDihedralSanitizationTests(TestCase):
    def test_normalizes_only_null_periodicity_zero_terms(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            topology = root / "LIG_GMX.itp"
            report_path = root / "report.json"
            topology.write_text(
                "[ dihedrals ] ; propers\n"
                "; i j k l func phase kd pn\n"
                "1 2 3 4 9 0.00 0.00000 0 ; null term\n"
                "1 2 3 4 9 0.00 1.04600 1 ; physical term\n"
                "[ pairs ]\n"
                "1 4 1\n",
                encoding="utf-8",
            )

            report = sanitize_zero_periodicity_dihedrals(topology, report_path)

            self.assertEqual(report["normalized_count"], 1)
            self.assertEqual(report["invalid_count"], 0)
            self.assertIn("0.00000 1 ; null term", topology.read_text())
            self.assertIn("1.04600 1 ; physical term", topology.read_text())
            self.assertEqual(json.loads(report_path.read_text())["normalized_count"], 1)

    def test_rejects_periodicity_zero_with_nonzero_force(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            topology = root / "LIG_GMX.itp"
            report_path = root / "report.json"
            original = "[ dihedrals ]\n1 2 3 4 9 0.00 2.50000 0\n"
            topology.write_text(original, encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "non-zero force constant"):
                sanitize_zero_periodicity_dihedrals(topology, report_path)

            self.assertEqual(topology.read_text(), original)
            report = json.loads(report_path.read_text())
            self.assertEqual(report["invalid_count"], 1)

    def test_ignores_other_sections_and_dihedral_functions(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            topology = root / "LIG_GMX.itp"
            report_path = root / "report.json"
            original = (
                "[ bonds ]\n1 2 3 4 9 0.00 0.00000 0\n"
                "[ dihedrals ]\n1 2 3 4 4 180.00 0.00000 0\n"
            )
            topology.write_text(original, encoding="utf-8")

            report = sanitize_zero_periodicity_dihedrals(topology, report_path)

            self.assertEqual(report["normalized_count"], 0)
            self.assertEqual(topology.read_text(), original)
