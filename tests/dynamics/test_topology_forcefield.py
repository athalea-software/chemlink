from pathlib import Path
import tempfile
import unittest

from cli.main import build_parser
from pipelines.dynamics.steps.topology import TopologyStep, resolve_force_field


class TopologyForceFieldTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_chemfusion_alias_resolves_charmm36_package(self):
        (self.root / "charmm36-jul2022.ff").mkdir()
        name, directory = resolve_force_field("chemfusion", [self.root])
        self.assertEqual(name, "charmm36-jul2022")
        self.assertEqual(directory, self.root)

    def test_pprotein_cli_accepts_named_force_field(self):
        args = build_parser().parse_args([
            "dynamic", "pprotein", "protein1.pdb", "protein2.pdb",
            "--force-field", "charmm36-jul2022", "--force-field-dir", str(self.root),
        ])
        self.assertEqual(args.force_field, "charmm36-jul2022")
        self.assertEqual(args.water_model, "tip3p")

    def test_topology_command_uses_resolved_field_and_water(self):
        (self.root / "charmm36-jul2022.ff").mkdir()
        work = self.root / "run"
        work.mkdir()
        step = TopologyStep({
            "sim_type": "5",
            "pdb_input": str(self.root / "complex.pdb"),
            "work_dir": str(work),
            "force_field": "charmm36-jul2022",
            "force_field_dir": str(self.root),
            "water_model": "tip3p",
        }, "gmx")
        command = step._build_pdb2gmx_cmd("complex.pdb")
        self.assertIn("charmm36-jul2022", command)
        self.assertEqual(command[command.index("-water") + 1], "tip3p")
        self.assertEqual(step.config["resolved_force_field_dir"], str(self.root))


if __name__ == "__main__":
    unittest.main()
