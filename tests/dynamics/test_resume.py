import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pipelines.dynamics.resume import (
    DynamicsResumeController,
    detect_resume_stage,
    load_resume_config,
    phase_completed,
)


class DynamicsResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.config = {"sim_type": "2", "binding_energy": True}

    def tearDown(self):
        self.tmp.cleanup()

    def _complete(self, phase):
        (self.root / f"{phase}.gro").write_text("structure")
        (self.root / f"{phase}.log").write_text("Finished mdrun on rank 0")

    def test_detects_interrupted_nvt_checkpoint(self):
        (self.root / "nvt.cpt").write_text("checkpoint")
        (self.root / "nvt.tpr").write_text("run input")
        self.assertEqual(detect_resume_stage(self.root, self.config), "nvt")

    def test_detects_new_npt_after_completed_nvt(self):
        self._complete("nvt")
        self.assertEqual(detect_resume_stage(self.root, self.config), "npt-new")

    def test_detects_interrupted_production(self):
        self._complete("nvt")
        self._complete("npt")
        (self.root / "md.cpt").write_text("checkpoint")
        (self.root / "md.tpr").write_text("run input")
        self.assertEqual(detect_resume_stage(self.root, self.config), "md")

    def test_detects_analysis_only(self):
        for phase in ("nvt", "npt", "md"):
            self._complete(phase)
        for name in ("md_center.xtc", "md.pdb", "index.ndx"):
            (self.root / name).write_text("output")
        self.assertEqual(detect_resume_stage(self.root, self.config), "analysis")

    def test_detects_completed_run(self):
        for phase in ("nvt", "npt", "md"):
            self._complete(phase)
        for name in ("md_center.xtc", "md.pdb", "index.ndx", "SIMULATION_SUMMARY.txt"):
            (self.root / name).write_text("output")
        metadata = self.root / "binding_energy_analysis/gmx_MMPBSA"
        metadata.mkdir(parents=True)
        (metadata / "analysis_metadata.json").write_text(json.dumps({"status": "completed"}))
        self.assertEqual(detect_resume_stage(self.root, self.config), "complete")

    def test_legacy_ligand_config_infers_type_and_duration(self):
        (self.root / "ligand.itp").write_text("ligand")
        (self.root / "md.mdp").write_text("nsteps = 500000\ndt = 0.002\n")
        config = load_resume_config(self.root)
        self.assertEqual(config["sim_type"], "2")
        self.assertEqual(config["ns_time"], 1.0)

    def test_phase_requires_log_completion_marker(self):
        (self.root / "md.gro").write_text("partial")
        (self.root / "md.log").write_text("Step 100")
        self.assertFalse(phase_completed(self.root, "md"))

    @patch("pipelines.dynamics.resume.run_mdrun_with_fallback")
    @patch("pipelines.dynamics.resume.get_optimal_mdrun_flags")
    def test_resume_command_uses_checkpoint_and_append(self, optimizer, runner):
        optimizer.return_value = {"gpu": ["-nb", "gpu"], "cpu": []}
        (self.root / "md.mdp").write_text("nsteps = 500000\n")
        controller = DynamicsResumeController.__new__(DynamicsResumeController)
        controller.work_dir = self.root
        controller.config = {"gpu_ids": [0]}
        controller.gmx_bin = "gmx_mpi"
        controller._resume_mdrun("md")
        kwargs = runner.call_args.kwargs
        self.assertIn("-cpi", kwargs["cmd_gpu"])
        self.assertIn("md.cpt", kwargs["cmd_gpu"])
        self.assertIn("-append", kwargs["cmd_gpu"])


if __name__ == "__main__":
    unittest.main()
