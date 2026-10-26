from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

import os

from cli.main import _resolve_dynamic_input_path, build_parser
from pipelines.dynamics.reanalysis import DynamicsReanalysisController


class DynamicsReanalysisTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "run_config.json").write_text(json.dumps({
            "sim_type": "5",
            "sim_type_label": "Protein + protein",
            "ns_time": 10,
            "threads": 8,
            "mmpbsa_cores": 4,
        }))
        for name in ("md.tpr", "md_center.xtc", "index.ndx", "topol.top"):
            (self.root / name).write_text("input")

    def tearDown(self):
        self.tmp.cleanup()

    def test_cli_parses_independent_binding_reanalysis(self):
        args = build_parser().parse_args([
            "dynamic", "analyze", str(self.root), "--only", "binding",
            "--mmpbsa-cores", "12", "--mmpbsa-frames", "500",
        ])
        self.assertEqual(args.only, "binding")
        self.assertEqual(args.mmpbsa_cores, 12)
        self.assertEqual(args.mmpbsa_frames, 500)

    def test_dynamic_i_accepts_an_existing_path(self):
        pdb = self.root / "Protein.pdb"
        pdb.write_text("ATOM\n")
        self.assertEqual(_resolve_dynamic_input_path(str(pdb), "protein"), str(pdb))

    def test_dynamic_i_resolves_a_bare_name_in_input_directory(self):
        self.assertEqual(
            _resolve_dynamic_input_path("Protein.pdb", "protein"),
            os.path.abspath("data/input/dynamics/Protein.pdb"),
        )

    @patch("pipelines.dynamics.reanalysis.check_gmx_installation", return_value="gmx")
    def test_overrides_saved_binding_configuration(self, _check):
        controller = DynamicsReanalysisController(
            str(self.root), selection="binding", mmpbsa_cores=12,
            mmpbsa_method="gb", mmpbsa_frames=500,
        )
        controller.validate()
        self.assertEqual(controller.config["mmpbsa_cores"], 12)
        self.assertEqual(controller.config["mmpbsa_method"], "gb")
        self.assertEqual(controller.config["mmpbsa_frames"], 500)

    @patch("pipelines.dynamics.reanalysis.check_gmx_installation", return_value="gmx")
    def test_general_reanalysis_does_not_require_topology(self, _check):
        (self.root / "topol.top").unlink()
        controller = DynamicsReanalysisController(str(self.root), selection="general")
        controller.validate()

    @patch("pipelines.dynamics.reanalysis.check_gmx_installation", return_value="gmx")
    def test_binding_reanalysis_requires_topology(self, _check):
        (self.root / "topol.top").unlink()
        controller = DynamicsReanalysisController(str(self.root), selection="binding")
        with self.assertRaisesRegex(FileNotFoundError, "topol.top"):
            controller.validate()


if __name__ == "__main__":
    unittest.main()
