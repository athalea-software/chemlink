import json
from pathlib import Path
import tempfile
import time
import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch

from cli.main import DEFAULT_MD_REPLICAS, build_parser
from pipelines.dynamics import dynamics_pipeline
from pipelines.dynamics.dynamics_pipeline import DynamicsPipeline
from pipelines.dynamics.replicas import (
    DEFAULT_REPLICAS,
    prepare_replica,
    write_replicas_summary,
)
from pipelines.dynamics.resume import DynamicsResumeController, detect_resume_stage


class ReplicaCliTests(unittest.TestCase):
    def _pligand(self, *extra):
        return build_parser().parse_args(
            ["dynamic", "pligand", "-ir", "protein.pdb", "-il", "ligand.sdf", *extra]
        )

    def test_pligand_defaults_to_two_replicas(self):
        self.assertEqual(DEFAULT_MD_REPLICAS, DEFAULT_REPLICAS)
        self.assertEqual(self._pligand().replicas, 2)

    def test_pligand_replicas_must_be_explicit_to_change(self):
        self.assertEqual(self._pligand("--replicas", "3").replicas, 3)
        self.assertEqual(self._pligand("--replicas", "1").replicas, 1)
        with self.assertRaises(SystemExit):
            self._pligand("--replicas", "0")

    def test_oprotein_defaults_to_one_replica(self):
        parser = build_parser()
        args = parser.parse_args(["dynamic", "oprotein", "-ir", "protein.pdb"])
        self.assertEqual(args.replicas, 1)
        args = parser.parse_args(["dynamic", "oprotein", "-ir", "protein.pdb", "--replicas", "3"])
        self.assertEqual(args.replicas, 3)
        with self.assertRaises(SystemExit):
            parser.parse_args(["dynamic", "oprotein", "-ir", "protein.pdb", "--replicas", "0"])

    def test_other_dynamics_types_have_no_replicas_option(self):
        args = build_parser().parse_args(["dynamic", "pprotein", "-ir", "a.pdb", "-ir", "b.pdb"])
        self.assertFalse(hasattr(args, "replicas"))

    def test_hpc_dynamics_replicas_for_pligand_and_oprotein(self):
        parser = build_parser()
        args = parser.parse_args(["hpc", "dynamics", "pligand", "-ir", "p.pdb", "-il", "l.sdf"])
        self.assertEqual(args.replicas, DEFAULT_MD_REPLICAS)
        args = parser.parse_args(["hpc", "dynamics", "oprotein", "-ir", "p.pdb", "--replicas", "2"])
        self.assertEqual(args.replicas, 2)
        args = parser.parse_args(["hpc", "dynamics", "pacid", "-ir", "p.pdb", "-ir", "n.pdb"])
        self.assertFalse(hasattr(args, "replicas"))


class ReplicaDirectoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_prepare_replica_inherits_prepared_system(self):
        for name in ("em.gro", "topol.top", "ligand.itp", "posre.itp"):
            (self.root / name).write_text(name)
        (self.root / "run_config.json").write_text("{}")
        (self.root / "simulation.log").write_text("parent log")
        (self.root / "acpype_work").mkdir()
        (self.root / "acpype_work" / "LIG.acpype").mkdir()

        config = prepare_replica(
            self.root, 2, {"work_dir": str(self.root), "replicas": 2, "_config_path": "x"}
        )

        replica = self.root / "replica_02"
        self.assertEqual(config["work_dir"], str(replica.resolve()))
        self.assertEqual(config["replica_index"], 2)
        self.assertNotIn("_config_path", config)
        self.assertEqual((replica / "topol.top").read_text(), "topol.top")
        self.assertTrue((replica / "acpype_work" / "LIG.acpype").is_dir())
        self.assertFalse((replica / "simulation.log").exists())
        saved = json.loads((replica / "run_config.json").read_text())
        self.assertEqual(saved["replica_index"], 2)

    def test_prepare_replica_requires_minimised_system(self):
        with self.assertRaises(FileNotFoundError):
            prepare_replica(self.root, 1, {"work_dir": str(self.root)})

    def test_summary_reports_mean_and_sd_across_replicas(self):
        for index, value in ((1, -30.0), (2, -34.0)):
            meta = self.root / f"replica_{index:02d}/binding_energy_analysis/gmx_MMPBSA"
            meta.mkdir(parents=True)
            (meta / "analysis_metadata.json").write_text(json.dumps(
                {"energies": {"GB": {"total_mean": value, "corrected_mean": value + 10}}}
            ))
            (meta.parents[1] / "SIMULATION_SUMMARY.txt").write_text("done")
        text = Path(write_replicas_summary(self.root, 2)).read_text()
        self.assertIn("MM-GBSA ΔH (kcal/mol) | -30.000 | -34.000 | -32.000 ± 2.828 (n=2)", text)
        self.assertIn("MM-GBSA ΔG (kcal/mol) | -20.000 | -24.000 | -22.000 ± 2.828 (n=2)", text)


class ReplicaPipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.calls = []

    def tearDown(self):
        self.tmp.cleanup()

    def _step(self, name):
        def factory(config, *args, **kwargs):
            def run():
                self.calls.append((name, Path(config["work_dir"]).name))
                if name == "EnergyMinStep":
                    (Path(config["work_dir"]) / "em.gro").write_text("minimised")
            step = MagicMock()
            step.run.side_effect = run
            return step
        return factory

    def _run_pipeline(self, replicas):
        config = {
            "sim_type": "2", "sim_type_label": "Protein + Ligand", "ns_time": 0.1,
            "work_dir": str(self.root), "replicas": replicas,
        }
        pipeline = DynamicsPipeline.__new__(DynamicsPipeline)
        pipeline.config = config
        pipeline.gmx_bin = "gmx"
        pipeline.total_steps = 6 + 4 * replicas
        pipeline.console = MagicMock()
        pipeline._started_at = time.monotonic()
        pipeline._started_wall = datetime.now()
        names = [
            "TopologyStep", "LigandTopologyStep", "SolvationStep", "IonsStep",
            "EnergyMinStep", "EquilibrationStep", "ProductionStep",
            "PostProcessingStep", "AnalysisStep",
        ]
        patches = [patch.object(dynamics_pipeline, n, side_effect=self._step(n)) for n in names]
        patches += [
            patch.object(dynamics_pipeline, "make_pipeline_bar"),
            patch.object(DynamicsPipeline, "_write_final_summary", return_value="summary.md"),
        ]
        for p in patches:
            p.start()
        try:
            pipeline.execute()
        finally:
            for p in patches:
                p.stop()

    def test_preparation_runs_once_and_md_once_per_replica(self):
        self._run_pipeline(2)
        root = self.root.name
        md_stages = ["EquilibrationStep", "ProductionStep", "PostProcessingStep", "AnalysisStep"]
        self.assertEqual(
            [name for name, _ in self.calls[:5]],
            ["TopologyStep", "LigandTopologyStep", "SolvationStep", "IonsStep", "EnergyMinStep"],
        )
        self.assertTrue(all(where == root for _, where in self.calls[:5]))
        self.assertEqual(
            self.calls[5:],
            [(n, "replica_01") for n in md_stages] + [(n, "replica_02") for n in md_stages],
        )
        self.assertTrue((self.root / "replicas_summary.md").is_file())

    def test_single_replica_keeps_flat_layout(self):
        self._run_pipeline(1)
        self.assertTrue(all(where == self.root.name for _, where in self.calls))
        self.assertFalse(list(self.root.glob("replica_*")))
        self.assertFalse((self.root / "replicas_summary.md").exists())


class ReplicaResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_fresh_replica_starts_from_new_nvt(self):
        (self.root / "em.gro").write_text("minimised")
        self.assertEqual(detect_resume_stage(self.root, {"sim_type": "2"}), "nvt-new")

    def test_parent_reports_first_incomplete_replica(self):
        (self.root / "em.gro").write_text("minimised")
        config = {"sim_type": "2", "ns_time": 0.1, "replicas": 2, "binding_energy": False}
        done = self.root / "replica_01"
        done.mkdir()
        (done / "run_config.json").write_text(json.dumps({**config, "replica_index": 1}))
        for phase in ("nvt", "npt", "md"):
            (done / f"{phase}.gro").write_text("structure")
            (done / f"{phase}.log").write_text("Finished mdrun")
        for name in ("md_center.xtc", "md.pdb", "index.ndx", "SIMULATION_SUMMARY.txt"):
            (done / name).write_text("output")

        controller = DynamicsResumeController.__new__(DynamicsResumeController)
        controller.work_dir = self.root
        controller.config = config
        with patch("pipelines.dynamics.resume.get_hardware_profile") as hw:
            hw.return_value.gpu_indices = []
            self.assertEqual(controller.detect_stage(), "replica_02: nvt-new")


if __name__ == "__main__":
    unittest.main()
