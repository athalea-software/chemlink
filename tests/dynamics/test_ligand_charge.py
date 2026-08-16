from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cli.main import build_parser

from pipelines.dynamics.ligand_charge import (
    detect_ligand_charge,
    detect_or_prepare_ligand_charge,
    estimate_unresolved_charge,
    explicit_charge_result,
    write_charge_provenance,
)


class LigandChargeDetectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_dynamic_ligand_charge_defaults_to_auto(self):
        args = build_parser().parse_args(["dynamic", "pligand", "protein.pdb", "ligand.sdf"])
        self.assertEqual(args.charge_mode, "auto")

    def write(self, name: str, content: str) -> Path:
        path = self.root / name
        path.write_text(content)
        return path

    def test_v3000_formal_charges(self):
        ligand = self.write(
            "ligand.sdf",
            "M  V30 1 N 0 0 0 0 CHG=1\nM  V30 2 O 0 0 0 0 CHG=-2\n",
        )
        result = detect_ligand_charge(ligand)
        self.assertEqual(result.charge, -1)
        self.assertEqual(result.confidence, "high")

    def test_v2000_m_chg_records(self):
        ligand = self.write(
            "ligand.mol",
            "Ligand\n  Test\n\n  0  0  0  0  0  0            999 V2000\n"
            "M  CHG  2   1   1   8  -1\nM  END\n",
        )
        result = detect_ligand_charge(ligand)
        self.assertEqual(result.charge, 0)
        self.assertEqual(result.confidence, "high")

    def test_v2000_atom_charge_code(self):
        ligand = self.write(
            "ligand.sdf",
            "Ligand\n  Test\n\n  1  0  0  0  0  0            999 V2000\n"
            "    0.0  0.0  0.0 N 0 3 0 0 0 0 0 0\nM  END\n$$$$\n",
        )
        result = detect_ligand_charge(ligand)
        self.assertEqual(result.charge, 1)
        self.assertEqual(result.confidence, "medium")

    def test_sdf_without_explicit_charge_is_not_assumed_neutral(self):
        ligand = self.write(
            "ligand.sdf",
            "Ligand\n  Test\n\n  0  0  0  0  0  0            999 V2000\nM  END\n$$$$\n",
        )
        result = detect_ligand_charge(ligand)
        self.assertIsNone(result.charge)
        self.assertEqual(result.confidence, "unknown")

    def test_unresolved_charge_gets_documented_neutral_estimate(self):
        ligand = self.write("ligand.sdf", "Ligand\n")
        result = estimate_unresolved_charge(detect_ligand_charge(ligand))
        self.assertEqual(result.charge, 0)
        self.assertEqual(result.confidence, "low")
        self.assertTrue(result.estimated)
        self.assertTrue(result.warnings)

        output = write_charge_provenance(result, self.root)
        provenance = output.read_text()
        self.assertIn('"estimated": true', provenance)
        self.assertIn('"warnings":', provenance)

    def test_estimator_preserves_resolved_charge(self):
        ligand = self.write(
            "charged.sdf",
            "Ligand\n  Test\n\n  0  0  0  0  0  0            999 V2000\n"
            "M  CHG  1   1  -1\nM  END\n",
        )
        result = estimate_unresolved_charge(detect_ligand_charge(ligand))
        self.assertEqual(result.charge, -1)
        self.assertEqual(result.confidence, "high")
        self.assertFalse(result.estimated)

    def test_mol2_partial_charges_round_to_integer(self):
        ligand = self.write(
            "ligand.mol2",
            "@<TRIPOS>MOLECULE\nLIG\n@<TRIPOS>ATOM\n"
            "1 C1 0 0 0 C.3 1 LIG -0.500\n"
            "2 O1 1 0 0 O.2 1 LIG -0.499\n@<TRIPOS>BOND\n",
        )
        result = detect_ligand_charge(ligand)
        self.assertEqual(result.charge, -1)
        self.assertAlmostEqual(result.raw_total, -0.999)

    def test_mol2_non_integer_sum_is_rejected(self):
        ligand = self.write(
            "ligand.mol2",
            "@<TRIPOS>ATOM\n1 C1 0 0 0 C.3 1 LIG -0.700\n@<TRIPOS>BOND\n",
        )
        result = detect_ligand_charge(ligand)
        self.assertIsNone(result.charge)

    def test_pdbqt_partial_charge_sum(self):
        ligand = self.write(
            "ligand.pdbqt",
            "ATOM 1 C1 LIG A 1 0 0 0 1.00 0.00 -0.500 C\n"
            "ATOM 2 O1 LIG A 1 1 0 0 1.00 0.00 -0.499 OA\n",
        )
        result = detect_ligand_charge(ligand)
        self.assertEqual(result.charge, -1)

    def test_pdb_formal_charge_columns(self):
        ligand = self.write(
            "charged.pdb",
            "HETATM    1  N1  LIG A   1       0.000   0.000   0.000  1.00  0.00           N1+\n"
            "HETATM    2  O1  LIG A   1       1.000   0.000   0.000  1.00  0.00           O2-\n",
        )
        result = detect_ligand_charge(ligand)
        self.assertEqual(result.charge, -1)
        self.assertEqual(result.confidence, "high")
        self.assertIn("PDB", result.method)

    @patch("pipelines.dynamics.ligand_charge.subprocess.run")
    def test_pdb_is_protonated_and_same_sdf_is_returned(self, run_mock):
        ligand = self.write("ligand.pdb", "ATOM      1  C1  LIG A   1\n")

        def fake_run(cmd, **kwargs):
            output = Path(cmd[cmd.index("-O") + 1])
            output.write_text(
                "Ligand\n  Test\n\n  0  0  0  0  0  0            999 V2000\n"
                "M  CHG  1   1  -1\nM  END\n"
            )
            return type("Completed", (), {"returncode": 0, "stderr": ""})()

        run_mock.side_effect = fake_run
        result = detect_or_prepare_ligand_charge(ligand, self.root, ph=7.4)
        self.assertEqual(result.charge, -1)
        self.assertEqual(Path(result.prepared_file).suffix, ".sdf")
        self.assertEqual(result.ph, 7.4)
        self.assertEqual(run_mock.call_args.args[0][1], "-ipdb")

    @patch("pipelines.dynamics.ligand_charge.subprocess.run")
    def test_mislabeled_pdbqt_is_converted_to_temporary_sdf(self, run_mock):
        ligand = self.write(
            "ligand.pdb",
            "ROOT\n"
            "ATOM 1 C1 LIG A 1 0 0 0 1.00 0.00 -0.500 C\n"
            "ATOM 2 O1 LIG A 1 1 0 0 1.00 0.00 -0.499 OA\n"
            "ENDROOT\nTORSDOF 0\n",
        )

        def fake_run(cmd, **kwargs):
            output = Path(cmd[cmd.index("-O") + 1])
            output.write_text(
                "Ligand\n  Test\n\n  0  0  0  0  0  0            999 V2000\n"
                "M  END\n$$$$\n"
            )
            return type("Completed", (), {"returncode": 0, "stderr": ""})()

        run_mock.side_effect = fake_run
        result = detect_or_prepare_ligand_charge(ligand, self.root)
        command = run_mock.call_args.args[0]
        self.assertEqual(command[1], "-ipdbqt")
        self.assertIn("-osdf", command)
        self.assertEqual(result.charge, -1)
        self.assertEqual(result.prepared_file, str(self.root / "ligand_pH74.sdf"))
        self.assertIn("PDBQT", result.method)

    def test_explicit_charge_and_provenance(self):
        ligand = self.write("ligand.pdb", "ATOM\n")
        result = explicit_charge_result(ligand, 2)
        output = write_charge_provenance(result, self.root)
        self.assertIn('"charge": 2', output.read_text())
        self.assertTrue(result.user_confirmed)


if __name__ == "__main__":
    unittest.main()
