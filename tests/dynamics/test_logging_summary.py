import logging
import tempfile
import time
import unittest
from datetime import datetime
from pathlib import Path

from pipelines.dynamics.dynamics_pipeline import DynamicsPipeline
from pipelines.dynamics.logger import _ConsoleOnlyFilter


class DynamicsLoggingSummaryTests(unittest.TestCase):
    def test_file_only_records_are_hidden_from_console(self):
        record = logging.LogRecord("test", logging.INFO, __file__, 1, "detail", (), None)
        record.file_only = True
        self.assertFalse(_ConsoleOnlyFilter().filter(record))

    def test_final_summary_contains_resources_and_phase_logs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            (work_dir / "nvt.log").write_text("log")
            (work_dir / "nvt.gro").write_text("coordinates")

            pipeline = DynamicsPipeline.__new__(DynamicsPipeline)
            pipeline.config = {
                "work_dir": temp_dir,
                "gpu_ids": [0],
                "sim_type_label": "Protein + Ligand",
                "ns_time": 100,
            }
            pipeline.resources = {
                "gpu_name": "Test GPU",
                "gpu_arch": 86,
                "gpu_memory_total_mb": 4096,
                "gpu_memory_free_mb": 4000,
                "total_cpu_count": 12,
                "p_core_count": 12,
                "ntomp": 8,
                "max_concurrent_sims": 1,
                "gpu": ["-ntomp", "8", "-nb", "gpu"],
                "cpu": ["-ntomp", "8"],
                "nstlist": 20,
                "update_gpu": True,
                "diagnostics": ["GPU 0: Test GPU", "CPU: 12 cores"],
            }
            pipeline._started_at = time.monotonic()
            pipeline._started_wall = datetime.now()

            summary = Path(pipeline._write_final_summary("Completada", "analysis", None))
            text = summary.read_text()

            self.assertIn("Test GPU", text)
            self.assertIn("12 núcleos", text)
            self.assertIn("| Equilibración NVT | Completada | `nvt.log` |", text)
            self.assertIn("simulation.log", text)


if __name__ == "__main__":
    unittest.main()
