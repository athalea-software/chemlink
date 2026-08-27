import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from pipelines.dynamics.dihedral_sanitization import (
    sanitize_ligand_dihedrals as sanitize_zero_periodicity_dihedrals,
)


class LigandDihedralSanitizationTests(TestCase):
    def test_removes_only_null_periodicity_zero_terms(self):
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

            self.assertEqual(report["removed_null_count"], 1)
            self.assertEqual(report["invalid_count"], 0)
            self.assertNotIn("null term", topology.read_text())
            self.assertIn("1.04600 1 ; physical term", topology.read_text())
            saved_report = json.loads(report_path.read_text())
            self.assertEqual(saved_report["removed_null_count"], 1)
            self.assertEqual(
                saved_report["removed_null_terms"][0]["reason"],
                "zero periodicity and zero force constant",
            )

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

            self.assertEqual(report["removed_null_count"], 0)
            self.assertEqual(topology.read_text(), original)

    def test_null_term_does_not_conflict_with_physical_opposite_phase(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            topology = root / "LIG_GMX.itp"
            report_path = root / "report.json"
            topology.write_text(
                "[ dihedrals ]\n"
                "1 2 4 14 9 0.0 0.0 0 ; ACPYPE null term\n"
                "1 2 4 14 9 0.0 0.16736 2 ; physical n=2 term\n"
                "1 2 4 14 9 180.0 1.92464 1 ; physical n=1 term\n",
                encoding="utf-8",
            )

            report = sanitize_zero_periodicity_dihedrals(topology, report_path)

            self.assertEqual(report["removed_null_count"], 1)
            self.assertEqual(report["incompatible_count"], 0)
            sanitized = topology.read_text()
            self.assertNotIn("ACPYPE null term", sanitized)
            self.assertIn("0.16736 2", sanitized)
            self.assertIn("1.92464 1", sanitized)

    def test_removes_all_bace_style_null_terms_without_false_conflicts(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            topology = root / "LIG_GMX.itp"
            report_path = root / "report.json"
            quartets = ((1, 2, 4, 14), (3, 1, 2, 4), (3, 1, 2, 12), (12, 2, 4, 14))
            lines = ["[ dihedrals ]\n"]
            for atoms in quartets:
                prefix = " ".join(str(atom) for atom in atoms)
                lines.extend(
                    [
                        f"{prefix} 9 0.0 0.0 0 ; null\n",
                        f"{prefix} 9 0.0 0.16736 2 ; physical\n",
                        f"{prefix} 9 180.0 1.92464 1 ; physical\n",
                    ]
                )
            topology.write_text("".join(lines), encoding="utf-8")

            report = sanitize_zero_periodicity_dihedrals(topology, report_path)

            self.assertEqual(report["removed_null_count"], 4)
            self.assertEqual(report["incompatible_count"], 0)
            self.assertNotIn("; null", topology.read_text())

    def test_merges_equal_function_nine_terms_without_changing_potential(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            topology = root / "LIG_GMX.itp"
            report_path = root / "report.json"
            topology.write_text(
                "[ dihedrals ]\n"
                "1 2 3 4 9 180.0 1.25 2 ; first\n"
                "4 3 2 1 9 -180.0 2.75 2 ; equivalent reverse\n",
                encoding="utf-8",
            )

            report = sanitize_zero_periodicity_dihedrals(topology, report_path)

            self.assertEqual(report["merged_count"], 1)
            self.assertEqual(report["incompatible_count"], 0)
            self.assertIn("180.0 4", topology.read_text())
            self.assertNotIn("2.75", topology.read_text())

    def test_rejects_equal_periodicity_with_different_phases(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            topology = root / "LIG_GMX.itp"
            report_path = root / "report.json"
            original = (
                "[ dihedrals ]\n"
                "1 2 3 4 9 0.0 1.25 2\n"
                "1 2 3 4 9 180.0 2.75 2\n"
            )
            topology.write_text(original, encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "different phases"):
                sanitize_zero_periodicity_dihedrals(topology, report_path)

            self.assertEqual(topology.read_text(), original)
            self.assertEqual(json.loads(report_path.read_text())["incompatible_count"], 1)
