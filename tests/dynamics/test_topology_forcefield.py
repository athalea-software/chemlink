from pathlib import Path
import tempfile
import unittest

from cli.main import build_parser
from unittest.mock import patch

from pipelines.dynamics.steps.topology import (
    TopologyStep, classify_pdb2gmx_error, resolve_force_field,
)
from utils.retry import SubprocessError


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
            "dynamic", "pprotein", "-ir", "protein1.pdb", "-il", "protein2.pdb",
            "--force-field", "charmm36-jul2022", "--force-field-dir", str(self.root),
        ])
        self.assertEqual(args.force_field, "charmm36-jul2022")
        self.assertEqual(args.water_model, "tip3p")

    def test_oprotein_cli_accepts_named_force_field(self):
        args = build_parser().parse_args([
            "dynamic", "oprotein", "-ir", "protein.pdb",
            "--force-field", "amber99sb-ildn", "--force-field-dir", str(self.root),
            "--water-model", "tip3p",
        ])
        self.assertEqual(args.force_field, "amber99sb-ildn")
        self.assertEqual(args.force_field_dir, str(self.root))
        self.assertEqual(args.water_model, "tip3p")

    def test_oprotein_topology_uses_selected_force_field(self):
        (self.root / "amber99sb-ildn.ff").mkdir()
        work = self.root / "run"
        work.mkdir()
        step = TopologyStep({
            "sim_type": "1",
            "pdb_input": str(self.root / "protein.pdb"),
            "work_dir": str(work),
            "force_field": "amber99sb-ildn",
            "force_field_dir": str(self.root),
            "water_model": "tip3p",
        }, "gmx")

        command = step._build_pdb2gmx_cmd("protein.pdb")

        self.assertEqual(command[command.index("-ff") + 1], "amber99sb-ildn")
        self.assertEqual(command[command.index("-water") + 1], "tip3p")

    def test_ppeptide_cli_accepts_named_force_field(self):
        args = build_parser().parse_args([
            "dynamic", "ppeptide", "-ir", "protein.pdb", "-il", "peptide.pdb",
            "--force-field", "charmm36-jul2022",
            "--force-field-dir", str(self.root),
            "--water-model", "tip3p",
        ])
        self.assertEqual(args.force_field, "charmm36-jul2022")
        self.assertEqual(args.force_field_dir, str(self.root))
        self.assertEqual(args.water_model, "tip3p")

    def test_ppeptide_topology_uses_selected_force_field(self):
        (self.root / "charmm36-jul2022.ff").mkdir()
        work = self.root / "run"
        work.mkdir()
        step = TopologyStep({
            "sim_type": "3",
            "pdb_input": str(self.root / "complex.pdb"),
            "work_dir": str(work),
            "force_field": "charmm36-jul2022",
            "force_field_dir": str(self.root),
            "water_model": "tip3p",
        }, "gmx")

        command = step._build_pdb2gmx_cmd("complex.pdb")

        self.assertEqual(command[command.index("-ff") + 1], "charmm36-jul2022")
        self.assertEqual(command[command.index("-water") + 1], "tip3p")

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

    def test_met_c1_error_is_terminal_patch_conflict(self):
        error = "atom C1 not found in buiding block 1MET while combining tdb and rtp"
        self.assertEqual(classify_pdb2gmx_error(error), "terminal_patch_conflict")

    @patch("pipelines.dynamics.steps.topology.run_subprocess")
    def test_charmm_uses_charged_termini_for_each_chain(self, run):
        ff_root = self.root / "ff"
        ff = ff_root / "charmm36-jul2022.ff"
        ff.mkdir(parents=True)
        (ff / "aminoacids.n.tdb").write_text(
            "[ None ]\n[ delete ]\n[ GLY-NH3+ ]\n[ add ]\n[ NH3+ ]\n[ replace ]\n"
        )
        (ff / "aminoacids.c.tdb").write_text("[ None ]\n[ COO- ]\n[ replace ]\n")
        work = self.root / "run"
        work.mkdir()
        pdb = self.root / "complex.pdb"
        pdb.write_text(
            "ATOM      1  N   MET A   1      0.0  0.0  0.0\n"
            "ATOM      2  N   ALA B   1      0.0  0.0  0.0\n"
        )
        step = TopologyStep({
            "sim_type": "5", "pdb_input": str(pdb), "work_dir": str(work),
            "force_field": "charmm36-jul2022", "force_field_dir": str(ff_root),
            "water_model": "tip3p",
        }, "gmx")
        step.run()
        kwargs = run.call_args.kwargs
        self.assertIn("-ter", run.call_args.args[0])
        self.assertEqual(kwargs["input_data"], "1\n0\n0\n0\n")
        metadata = (work / "topology_preflight.json").read_text()
        self.assertIn('"n": "NH3+"', metadata)
        self.assertIn('"c": "COO-"', metadata)

    @patch("pipelines.dynamics.steps.topology.run_subprocess")
    def test_terminal_conflict_does_not_invoke_pdbfixer(self, run):
        ff = self.root / "charmm36-jul2022.ff"
        ff.mkdir()
        (ff / "aminoacids.n.tdb").write_text("[ None ]\n[ NH3+ ]\n")
        (ff / "aminoacids.c.tdb").write_text("[ None ]\n[ COO- ]\n")
        work = self.root / "run"
        work.mkdir()
        pdb = self.root / "complex.pdb"
        pdb.write_text("ATOM      1  N   MET A   1      0.0  0.0  0.0\n")
        run.side_effect = SubprocessError(
            ["gmx", "pdb2gmx"], 1,
            "atom C1 not found in buiding block 1MET while combining tdb and rtp",
        )
        step = TopologyStep({
            "sim_type": "5", "pdb_input": str(pdb), "work_dir": str(work),
            "force_field": "charmm36-jul2022", "force_field_dir": str(self.root),
            "water_model": "tip3p",
        }, "gmx")
        with patch.object(step, "_repair_pdb") as repair:
            with self.assertRaisesRegex(RuntimeError, "MET1"):
                step.run()
        repair.assert_not_called()


if __name__ == "__main__":
    unittest.main()
