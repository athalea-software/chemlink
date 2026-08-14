from pathlib import Path
import tempfile
import unittest

from pipelines.dynamics.steps.complex_builder import ComplexBuilderStep


class ComplexBuilderTests(unittest.TestCase):
    def test_normalizes_two_proteins_to_exactly_two_chains(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.pdb"
            second = root / "second.pdb"
            output = root / "complex.pdb"
            first.write_text(
                "ATOM      1  N   MET X   1      0.0  0.0  0.0\nTER\nEND\n"
            )
            second.write_text(
                "ATOM      2  N   ALA Y   1      0.0  0.0  0.0\nTER\nEND\n"
            )
            step = ComplexBuilderStep({
                "pdb_protein": str(first), "pdb_partner": str(second),
                "pdb_input": str(output), "work_dir": str(root),
            })
            step.run()
            lines = output.read_text().splitlines()
            atoms = [line for line in lines if line.startswith("ATOM")]
            self.assertEqual([line[21] for line in atoms], ["A", "B"])
            self.assertEqual(lines.count("TER"), 2)
            self.assertEqual(lines.count("END"), 1)


if __name__ == "__main__":
    unittest.main()
