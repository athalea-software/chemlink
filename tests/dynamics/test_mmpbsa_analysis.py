from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pipelines.dynamics.mmpbsa_analysis import (
    GMX_MMPBSA_Analyzer,
    calculate_sampling,
    parse_gmx_frame_count,
    parse_gmx_trajectory_timing,
    parse_mmpbsa_frame_progress,
    parse_index,
)


class MMPBSAAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_sampling_discards_twenty_percent_and_caps_frames(self):
        result = calculate_sampling(5001, target_frames=1000, discard_percent=20)
        self.assertEqual(result["discarded_frames"], 1000)
        self.assertEqual(result["start_frame"], 1001)
        self.assertEqual(result["interval"], 5)
        self.assertEqual(result["analyzed_frames"], 801)

    def test_sampling_keeps_at_least_one_frame(self):
        result = calculate_sampling(1, target_frames=1000, discard_percent=20)
        self.assertEqual(result["start_frame"], 1)
        self.assertEqual(result["analyzed_frames"], 1)

    def test_parse_gromacs_frame_count(self):
        self.assertEqual(parse_gmx_frame_count("Coords 208\n"), 208)
        self.assertEqual(parse_gmx_frame_count("Last frame 207 time 10.0"), 208)

    def test_parse_gromacs_trajectory_timing(self):
        timing = parse_gmx_trajectory_timing(
            "Reading frame 0 time 5.000\rReading frame 1 time 15.000\r"
        )
        self.assertEqual(timing, {"start_ps": 5.0, "step_ps": 10.0})

    def test_parse_mmpbsa_frame_progress(self):
        self.assertEqual(
            parse_mmpbsa_frame_progress(" 46%|####5| 37/81 [elapsed: 03:06]"),
            (37, 81),
        )
        self.assertIsNone(parse_mmpbsa_frame_progress("calculating complex contribution"))

    @patch("pipelines.dynamics.mmpbsa_analysis.subprocess.Popen")
    def test_streams_frame_progress_with_rich_bar(self, popen):
        process = popen.return_value
        process.stderr = iter([
            "Beginning GB calculations\n",
            "calculating complex contribution\n",
            " 46%|####5| 37/81 [elapsed: 03:06]\n",
            "calculating receptor contribution\n",
            "100%|##########| 81/81 [elapsed: 00:48]\n",
        ])
        process.wait.return_value = 0
        process.poll.return_value = 0
        analyzer = GMX_MMPBSA_Analyzer(str(self.root), gmx_bin="gmx")
        result = analyzer._run_with_frame_progress(["gmx_MMPBSA"], {}, 81)
        self.assertEqual(result, 0)
        self.assertIn("37/81", (analyzer.output_dir / "gmx_mmpbsa.err").read_text())

    def test_create_binding_index_prefers_exact_mol_group(self):
        (self.root / "index.ndx").write_text(
            "[ System ]\n1 2 3 4 5\n"
            "[ Protein ]\n1 2 3\n"
            "[ Other ]\n4 5\n"
            "[ MOL ]\n4 5\n"
        )
        analyzer = GMX_MMPBSA_Analyzer(str(self.root), gmx_bin="gmx")
        protein, ligand = analyzer.create_binding_index()
        self.assertEqual((protein, ligand), (0, 1))
        groups = parse_index(analyzer.inputs["index"])
        self.assertEqual(groups, [("Protein", [1, 2, 3]), ("LIG", [4, 5])])

    def test_binding_index_rejects_overlap(self):
        (self.root / "index.ndx").write_text(
            "[ Protein ]\n1 2 3\n[ LIG ]\n3 4\n"
        )
        analyzer = GMX_MMPBSA_Analyzer(str(self.root), gmx_bin="gmx")
        with self.assertRaisesRegex(ValueError, "overlap"):
            analyzer.create_binding_index()

    def test_protein_protein_accepts_other_and_writes_two_macromolecule_groups(self):
        (self.root / "index.ndx").write_text(
            "[ Protein ]\n1 2 3\n[ Other ]\n4 5 6\n"
        )
        analyzer = GMX_MMPBSA_Analyzer(
            str(self.root), gmx_bin="gmx", method="both",
            binding_system="protein-protein",
        )
        self.assertEqual(analyzer.method, "gb")
        self.assertEqual(analyzer.create_binding_index(), (0, 1))
        self.assertEqual(
            parse_index(analyzer.inputs["index"]),
            [("Protein", [1, 2, 3]), ("Other", [4, 5, 6])],
        )

    def test_protein_protein_derives_groups_from_two_largest_pdb_chains(self):
        (self.root / "index.ndx").write_text("[ Protein ]\n1 2 3 4 5\n")
        (self.root / "md.pdb").write_text(
            "ATOM      1  CA  ALA A   1      0.0  0.0  0.0\n"
            "ATOM      2  CB  ALA A   1      0.0  0.0  0.0\n"
            "ATOM      3  CA  GLY B   1      0.0  0.0  0.0\n"
        )
        analyzer = GMX_MMPBSA_Analyzer(
            str(self.root), gmx_bin="gmx", binding_system="protein-protein"
        )
        analyzer.create_binding_index()
        self.assertEqual(
            parse_index(analyzer.inputs["index"]),
            [("Protein", [1, 2]), ("Other", [3])],
        )
        self.assertEqual(analyzer.metadata["chain_detection"]["receptor_chain"], "A")

    @patch("pipelines.dynamics.mmpbsa_analysis.find_mmpbsa_tool")
    def test_protein_protein_command_omits_small_molecule_topology(self, find_tool):
        find_tool.side_effect = lambda name, env: f"/env/bin/{name}"
        analyzer = GMX_MMPBSA_Analyzer(
            str(self.root), gmx_bin="gmx", binding_system="protein-protein"
        )
        command = analyzer._command(self.root / "in", None, 0, 1)
        self.assertNotIn("-lm", command)
        self.assertEqual(command[command.index("-cg") + 1:command.index("-cg") + 3], ["0", "1"])

    def test_extracts_gb_and_pb_results(self):
        analyzer = GMX_MMPBSA_Analyzer(str(self.root), gmx_bin="gmx")
        result_file = analyzer.output_dir / "FINAL_RESULTS_MMPBSA.dat"
        result_file.write_text(
            "GENERALIZED BORN\nΔTOTAL -35.21 9.99 4.83 3.00 2.00\n"
            "POISSON BOLTZMANN\nDELTA G binding -29.44 6.17\n"
        )
        energies = analyzer.extract_energies()
        self.assertEqual(energies["GB"]["total_mean"], -35.21)
        self.assertEqual(energies["PB"]["total_std"], 6.17)

    def test_input_file_supports_both_methods(self):
        analyzer = GMX_MMPBSA_Analyzer(
            str(self.root), gmx_bin="gmx", method="both", salt_concentration=0.15
        )
        path = analyzer.create_input_file(calculate_sampling(100))
        content = path.read_text()
        self.assertIn("&gb", content)
        self.assertIn("&pb", content)
        self.assertIn("saltcon=0.150", content)

    def test_writes_binding_energy_timeseries_csv(self):
        analyzer = GMX_MMPBSA_Analyzer(str(self.root), gmx_bin="gmx")
        analyzer.metadata["trajectory_timing"] = {"start_ps": 0.0, "step_ps": 10.0}
        (analyzer.output_dir / "energy_terms_per_frame.csv").write_text(
            "GENERALIZED BORN:\nDelta Energy Terms\n"
            "Frame #,VDWAALS,EEL,TOTAL\n21,-20.0,-30.0,-42.5\n"
            "22,-19.0,-29.0,-40.0\n\n"
            "POISSON BOLTZMANN:\nDelta Energy Terms\n"
            "Frame #,VDWAALS,EEL,TOTAL\n21,-20.0,-30.0,-35.5\n",
            encoding="utf-8",
        )
        self.assertEqual(analyzer._write_timeseries_outputs(), 3)
        content = (analyzer.output_dir / "binding_energy_timeseries.csv").read_text()
        self.assertIn("method,frame,time_ps,time_ns", content)
        self.assertIn("MM-GBSA,21,200.0,0.2", content)
        self.assertTrue((analyzer.output_dir / "binding_energy_timeseries.png").is_file())

    @patch("pipelines.dynamics.mmpbsa_analysis.find_mmpbsa_tool")
    def test_gmx_mpi_forces_serial_mmpbsa(self, find_tool):
        find_tool.side_effect = lambda name, env: f"/env/bin/{name}"
        analyzer = GMX_MMPBSA_Analyzer(str(self.root), gmx_bin="/usr/bin/gmx_mpi", n_cores=6)
        command = analyzer._command(self.root / "in", self.root / "lig.mol2", 0, 1)
        self.assertEqual(command[0], "/env/bin/gmx_MMPBSA")
        self.assertNotIn("-np", command)
        self.assertEqual(analyzer.metadata["mpi_cores"], 1)
        self.assertIn("running serially", analyzer.metadata["parallel_fallback"])

    def test_run_executes_gmx_mmpbsa_once(self):
        for filename in ("md.tpr", "md_center.xtc", "topol.top"):
            (self.root / filename).write_text("input")
        (self.root / "index.ndx").write_text(
            "[ Protein ]\n1 2 3\n[ MOL ]\n4 5\n"
        )
        acpype = self.root / "acpype_work/LIG.acpype"
        acpype.mkdir(parents=True)
        (acpype / "LIG_bcc_gaff2.mol2").write_text("mol2")
        analyzer = GMX_MMPBSA_Analyzer(str(self.root), gmx_bin="gmx", n_cores=1)
        with patch.object(analyzer, "detect_available_frames", return_value=100), \
             patch.object(analyzer, "_command", return_value=["gmx_MMPBSA"]), \
             patch.object(analyzer, "_run_with_frame_progress", return_value=0) as runner:
            self.assertTrue(analyzer.run_analysis())
        runner.assert_called_once()
        metadata = (analyzer.output_dir / "analysis_metadata.json").read_text()
        self.assertIn('"status": "completed_with_warnings"', metadata)


if __name__ == "__main__":
    unittest.main()
