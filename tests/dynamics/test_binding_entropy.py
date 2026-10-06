"""Entropy integration: backend input, distinct estimates and recovery."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cli.main import build_parser
from pipelines.dynamics.mmpbsa_analysis import GMX_MMPBSA_Analyzer, calculate_sampling
from pipelines.dynamics.reanalysis import DynamicsReanalysisController
from pipelines.dynamics.run_mmpbsa_for_pp_with_ligand import MMPBSAPPLigandAnalyzer


class BindingEntropyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def analyzer(self, **kwargs):
        return GMX_MMPBSA_Analyzer(str(self.root), gmx_bin="gmx", **kwargs)

    def test_cli_entropy_and_saved_defaults(self):
        parser = build_parser()
        args = parser.parse_args(["dynamic", "analyze", str(self.root), "--only", "binding",
                                  "--mmpbsa-entropy", "ie", "--mmpbsa-temperature", "310",
                                  "--mmpbsa-ie-segment", "50"])
        self.assertEqual((args.mmpbsa_entropy, args.mmpbsa_temperature, args.mmpbsa_ie_segment),
                         ("ie", 310, 50))
        defaults = parser.parse_args(["dynamic", "analyze", str(self.root)])
        self.assertEqual(defaults.mmpbsa_entropy, "ie")
        self.assertIsNone(defaults.mmpbsa_temperature)

    def test_local_and_hpc_cli_accept_entropy(self):
        parser = build_parser()
        for command in (["dynamic", "pligand"], ["hpc", "dynamics", "pligand"]):
            with self.subTest(command=command):
                args = parser.parse_args(command + ["--mmpbsa-entropy", "c2",
                                                    "--mmpbsa-temperature", "310"])
                self.assertEqual(args.mmpbsa_entropy, "c2")
                self.assertEqual(args.mmpbsa_temperature, 310)
                defaults = parser.parse_args(command)
                self.assertEqual(defaults.mmpbsa_entropy, "ie")
                self.assertEqual(defaults.mmpbsa_temperature, 300)

    def test_analyzer_defaults_to_ie_and_cli_allows_explicit_disable(self):
        analyzer = self.analyzer()
        content = analyzer.create_input_file(calculate_sampling(100)).read_text()
        self.assertIn("interaction_entropy=1,", content)
        parser = build_parser()
        for command in (["dynamic", "pligand"], ["hpc", "dynamics", "pligand"],
                        ["dynamic", "analyze", str(self.root)]):
            with self.subTest(command=command):
                args = parser.parse_args(command + ["--mmpbsa-entropy", "none"])
                self.assertEqual(args.mmpbsa_entropy, "none")

    def test_input_enables_only_selected_method(self):
        for entropy in ("none", "ie", "c2"):
            with self.subTest(entropy=entropy):
                analyzer = self.analyzer(entropy=entropy, temperature=310, ie_segment=50)
                content = analyzer.create_input_file(calculate_sampling(100)).read_text()
                self.assertIn(f"interaction_entropy={int(entropy == 'ie')},", content)
                self.assertIn(f"c2_entropy={int(entropy == 'c2')},", content)
                self.assertIn("temperature=310.0,", content)
                self.assertIn("ie_segment=50,", content)

    def test_invalid_temperature_segment_and_method(self):
        for options in ({"temperature": 0}, {"temperature": float("nan")},
                        {"temperature": float("inf")}, {"ie_segment": 101},
                        {"ie_segment": 0}, {"ie_segment": 25.5}, {"entropy": "qh"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.analyzer(**options)

    def test_reads_base_and_corrected_estimates_without_double_counting(self):
        for entropy, title in (("ie", "Interaction"), ("c2", "C2")):
            analyzer = self.analyzer(entropy=entropy)
            (analyzer.output_dir / "FINAL_RESULTS_MMPBSA.dat").write_text(
                f"ENTROPY RESULTS\nGB {entropy.upper()} 2.0 10.0 1.0 0.1\n"
                "GENERALIZED BORN:\nComplex:\nTOTAL -100 1 2 3 4\n"
                "ΔTOTAL -35 9 4 3 2\n"
                f"Using {title} Entropy Approximation:\nΔG binding = -25.0 +/- 4.5\n"
                "POISSON BOLTZMANN:\nΔTOTAL -30 8 3 2 1\n"
                f"Using {title} Entropy Approximation:\nΔG binding = -20.0 +/- 3.5\n"
            )
            energies = analyzer.extract_energies()
            self.assertEqual(energies["GB"]["total_mean"], -35)
            self.assertEqual(energies["GB"]["corrected_mean"], -25)
            self.assertEqual(energies["GB"]["entropy_correction"], 10)
            self.assertEqual(energies["GB"]["interaction_energy_sigma"], 2)
            self.assertEqual(energies["PB"]["corrected_mean"], -20)
            self.assertEqual(analyzer.metadata["entropy_status"], "completed")
            analyzer._write_results(energies)
            report = (analyzer.output_dir / "SUMMARY_REPORT.txt").read_text()
            self.assertIn("without entropy = -35.00", report)
            self.assertIn("-TΔS = 10.00", report)
            self.assertTrue((analyzer.output_dir / "binding_energy.png").is_file())

    def test_missing_correction_retains_base_and_marks_incomplete(self):
        analyzer = self.analyzer(entropy="ie")
        (analyzer.output_dir / "FINAL_RESULTS_MMPBSA.dat").write_text(
            "WARNING: NOT RELIABLE\nGENERALIZED BORN:\nΔTOTAL -35 9 4 3 2\n")
        energies = analyzer.extract_energies()
        self.assertEqual(energies["GB"]["total_mean"], -35)
        self.assertNotIn("corrected_mean", energies["GB"])
        self.assertEqual(analyzer.metadata["entropy_status"], "incomplete")
        self.assertIn("WARNING: NOT RELIABLE", analyzer.metadata["entropy_diagnostics"])

    def test_corrected_only_output_is_never_treated_as_base(self):
        analyzer = self.analyzer(entropy="ie")
        (analyzer.output_dir / "FINAL_RESULTS_MMPBSA.dat").write_text(
            "GENERALIZED BORN:\nUsing Interaction Entropy Approximation:\nΔG binding = -25 +/- 4\n")
        self.assertEqual(analyzer.extract_energies()["GB"], {})

    @patch("pipelines.dynamics.reanalysis.check_gmx_installation", return_value="gmx")
    def test_reanalysis_enables_ie_and_allows_explicit_disable(self, _check):
        (self.root / "run_config.json").write_text(json.dumps({
            "sim_type": "5", "ns_time": 10, "threads": 2,
            "mmpbsa_entropy": "none", "mmpbsa_temperature": 310,
        }))
        controller = DynamicsReanalysisController(str(self.root), selection="binding")
        self.assertEqual(controller.config["mmpbsa_entropy"], "ie")
        self.assertEqual(controller.config["mmpbsa_temperature"], 310)
        controller = DynamicsReanalysisController(str(self.root), selection="binding",
                                                 mmpbsa_entropy="none", mmpbsa_temperature=300)
        self.assertEqual(controller.config["mmpbsa_entropy"], "none")
        self.assertEqual(controller.config["mmpbsa_temperature"], 300)

    @patch("pipelines.dynamics.run_mmpbsa_for_pp_with_ligand.GMX_MMPBSA_Analyzer")
    def test_cofactor_adapter_forwards_entropy(self, backend):
        adapter = MMPBSAPPLigandAnalyzer(str(self.root), entropy="ie", temperature=310)
        with patch.object(adapter, "_create_index_file", return_value=True):
            adapter.run_analysis()
        self.assertEqual(backend.call_args.kwargs["entropy"], "ie")
        self.assertEqual(backend.call_args.kwargs["temperature"], 310)

    def test_backend_failure_recovers_base_with_entropy_disabled(self):
        analyzer = self.analyzer(entropy="ie", method="gb", binding_system="protein-protein")
        calls = []
        def run(command, environment, frames):
            content = (analyzer.output_dir / "mmpbsa.in").read_text()
            calls.append(content)
            if len(calls) == 1:
                (analyzer.output_dir / "gmx_mmpbsa.err").write_text("Entropy failed")
                return 1
            (analyzer.output_dir / "FINAL_RESULTS_MMPBSA.dat").write_text(
                "GENERALIZED BORN:\nΔTOTAL -35 9 4 3 2\n")
            return 0
        with patch.object(analyzer, "validate_inputs", return_value=(True, [])), \
             patch.object(analyzer, "create_binding_index", return_value=(0, 1)), \
             patch.object(analyzer, "detect_available_frames", return_value=100), \
             patch.object(analyzer, "_command", return_value=["gmx_MMPBSA"]), \
             patch.object(analyzer, "_run_with_frame_progress", side_effect=run), \
             patch("pipelines.dynamics.mmpbsa_analysis.subprocess.run", side_effect=FileNotFoundError):
            self.assertTrue(analyzer.run_analysis())
        self.assertIn("interaction_entropy=1", calls[0])
        self.assertIn("interaction_entropy=0", calls[1])
        self.assertEqual(analyzer.metadata["entropy_status"], "incomplete")
        self.assertEqual(analyzer.metadata["status"], "completed_with_warnings")
        self.assertEqual(analyzer.metadata["energies"]["GB"]["total_mean"], -35)
        self.assertEqual((analyzer.output_dir / "entropy_attempt_gmx_mmpbsa.err").read_text(),
                         "Entropy failed")


if __name__ == "__main__":
    unittest.main()
