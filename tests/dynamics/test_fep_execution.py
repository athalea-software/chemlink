import unittest

from cli.main import build_parser
from pathlib import Path

from pipelines.dynamics.fep_execution import (
    FEPExecutionError,
    _all_production_states,
    _anchor_index_file,
    _em_mdp,
    _evaluate_anchor_stability,
    _lambda_values,
    _npt_mdp,
    _nvt_mdp,
    _parse_traj_xvg,
    _pilot_mdp,
    compute_standard_state_correction,
)


class FEPExecutionTests(unittest.TestCase):
    def test_cli_exposes_fep_equilibrate(self):
        args = build_parser().parse_args([
            "dynamic", "fep-equilibrate", "--run-dir", "FEP_ABFE_24",
        ])
        self.assertEqual(args.dyn_type, "fep-equilibrate")
        self.assertEqual(args.run_dir, "FEP_ABFE_24")

    def test_lambda_values_span_zero_to_one(self):
        self.assertEqual(_lambda_values({"lambda_windows": 5}), [0.0, 0.25, 0.5, 0.75, 1.0])

    def test_em_mdp_is_free_of_dynamics_and_uses_lambda_state_zero(self):
        config = {"temperature_k": 300.0}
        mdp = _em_mdp(10, config, "ABFE", _lambda_values({"lambda_windows": 4}))
        self.assertIn("integrator               = steep", mdp)
        self.assertIn("init-lambda-state        = 0", mdp)
        self.assertIn("bonded-lambdas", mdp)
        self.assertIn("couple-moltype           = LIG", mdp)

    def test_nvt_mdp_randomizes_velocities_and_sizes_from_equilibration_ns(self):
        config = {"temperature_k": 305.0, "equilibration_ns": 0.5, "lambda_windows": 4}
        mdp = _nvt_mdp(12, config, "ABFE", _lambda_values(config))
        self.assertIn("gen_vel                  = yes", mdp)
        self.assertIn("gen_seed                 = -1", mdp)
        self.assertIn("continuation             = no", mdp)
        self.assertIn("pcoupl                   = no", mdp)
        self.assertIn("nsteps                   = 250000", mdp)  # 0.5 ns * 500000
        self.assertIn("ref_t                    = 305", mdp)

    def test_npt_mdp_continues_from_nvt_with_barostat(self):
        config = {"temperature_k": 300.0, "equilibration_ns": 0.2, "lambda_windows": 4}
        mdp = _npt_mdp(12, config, "ABFE", _lambda_values(config))
        self.assertIn("continuation             = yes", mdp)
        self.assertIn("gen_vel                  = no", mdp)
        self.assertIn("pcoupl                   = C-rescale", mdp)
        self.assertIn("nsteps                   = 100000", mdp)  # 0.2 ns * 500000

    def test_rbfe_mdp_uses_fep_lambdas_without_coupling_block(self):
        config = {"temperature_k": 300.0, "equilibration_ns": 0.1, "lambda_windows": 3}
        mdp = _nvt_mdp(10, config, "RBFE", _lambda_values(config))
        self.assertIn("fep-lambdas", mdp)
        self.assertNotIn("couple-moltype", mdp)

    def test_pilot_mdp_sizes_from_equilibration_ns_and_writes_trajectory(self):
        config = {"temperature_k": 300.0, "equilibration_ns": 0.05, "lambda_windows": 4}
        mdp = _pilot_mdp(2, _lambda_values(config), config, "ABFE")
        self.assertIn("nsteps                   = 25000", mdp)  # 0.05 ns * 500000
        self.assertIn("nstxout-compressed       = 500", mdp)
        self.assertIn("init-lambda-state        = 2", mdp)
        self.assertIn("bonded-lambdas", mdp)

    def test_pilot_mdp_declares_a_tcoupl_group_grompp_requires(self):
        # Regression: grompp rejects `tcoupl`/`ref-t` without a matching
        # `tc-grps`/`tau_t` ("Invalid T coupling input: 0 groups, 1 ref-t
        # values and 0 tau-t values") — caught running a real pilot.
        config = {"temperature_k": 300.0, "equilibration_ns": 0.05, "lambda_windows": 4}
        mdp = _pilot_mdp(0, _lambda_values(config), config, "ABFE")
        self.assertIn("tc-grps                  = System", mdp)
        self.assertIn("tau_t                    = 0.1", mdp)

    def test_anchor_index_file_is_a_single_group_in_anchor_order(self):
        # Must be exactly one group (see _anchor_index_file docstring): `gmx
        # traj -n` replaces gmx's default groups rather than extending them,
        # and only auto-selects without a prompt when there's just one.
        text = _anchor_index_file([2, 1, 3], [4, 5, 6])
        self.assertEqual(text, "[ ChemLink_Boresch_Anchors ]\n2 1 3 4 5 6\n")

    def test_parse_traj_xvg_skips_comments_and_headers(self):
        text = (
            "# This file was created by gmx traj\n"
            "@    title \"anchors\"\n"
            "0.000   1.0 2.0 3.0\n"
            "1.000   1.1 2.1 3.1\n"
        )
        rows = _parse_traj_xvg(text)
        self.assertEqual(rows, [(0.0, 1.0, 2.0, 3.0), (1.0, 1.1, 2.1, 3.1)])

    @staticmethod
    def _anchor_row(t, p1, p2, p3, l1, l2, l3):
        coords = (t,) + p1 + p2 + p3 + l1 + l2 + l3
        return tuple(float(value) for value in coords)

    def test_anchor_stability_is_stable_for_a_static_trajectory(self):
        row = self._anchor_row(
            0.0, (0, 0, 0), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 1, 0), (0, 1, 0),
        )
        result = _evaluate_anchor_stability([[row, row, row]])
        self.assertEqual(result["status"], "stable")
        self.assertEqual(result["problems"], [])
        self.assertEqual(result["frame_count"], 3)

    def test_anchor_stability_flags_near_linear_angle(self):
        # P2, P1, L1 colinear -> angle A == 180 deg, outside the 15-165 deg range.
        row = self._anchor_row(
            0.0, (0, 0, 0), (-1, 0, 0), (0, 0, 1), (1, 0, 0), (1, 1, 0), (1, 1, 1),
        )
        result = _evaluate_anchor_stability([[row]])
        self.assertEqual(result["status"], "unstable")
        self.assertTrue(any("angle A" in problem for problem in result["problems"]))

    def test_anchor_stability_flags_dihedral_flip_between_frames(self):
        # Only L3 changes between frames: dihedral C (P1,L1,L2,L3) flips cis (~0 deg)
        # -> trans (~180 deg), everything else held fixed and within valid bounds.
        common = dict(p1=(0, 0, 0), p2=(0, 1, 0), p3=(0, 1, 1), l1=(1, 0, 0), l2=(1, 1, 0))
        frame_cis = self._anchor_row(0.0, common["p1"], common["p2"], common["p3"],
                                      common["l1"], common["l2"], (0, 1, 0))
        frame_trans = self._anchor_row(1.0, common["p1"], common["p2"], common["p3"],
                                        common["l1"], common["l2"], (2, 1, 0))
        result = _evaluate_anchor_stability([[frame_cis, frame_trans]])
        self.assertEqual(result["status"], "unstable")
        self.assertTrue(any("dihedral C" in problem for problem in result["problems"]))

    def test_anchor_stability_ignores_a_jump_across_a_window_boundary(self):
        # Same two frames as the flip test above, but as separate pilot
        # windows: consecutive lambda windows are independent simulations,
        # so this must NOT be flagged (see _evaluate_anchor_stability).
        common = dict(p1=(0, 0, 0), p2=(0, 1, 0), p3=(0, 1, 1), l1=(1, 0, 0), l2=(1, 1, 0))
        frame_cis = self._anchor_row(0.0, common["p1"], common["p2"], common["p3"],
                                      common["l1"], common["l2"], (0, 1, 0))
        frame_trans = self._anchor_row(1.0, common["p1"], common["p2"], common["p3"],
                                        common["l1"], common["l2"], (2, 1, 0))
        result = _evaluate_anchor_stability([[frame_cis], [frame_trans]])
        self.assertEqual(result["status"], "stable")

    def test_anchor_stability_tolerates_a_rare_thermal_outlier(self):
        # Regression: a real pilot run had 2 frames out of 404 land ~0.1-0.7
        # deg past the 165 deg angle bound — ordinary harmonic-restraint
        # thermal noise, not instability. One outlier out of 200 frames
        # (0.5%) must stay under _STABILITY_OUTLIER_TOLERANCE (1%).
        stable_row = self._anchor_row(
            0.0, (0, 0, 0), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 1, 0), (0, 1, 0),
        )
        outlier_row = self._anchor_row(
            0.0, (0, 0, 0), (-1, 0, 0), (0, 0, 1), (1, 0, 0), (1, 1, 0), (1, 1, 1),
        )
        result = _evaluate_anchor_stability([[stable_row] * 199 + [outlier_row]])
        self.assertEqual(result["status"], "stable")

    def test_anchor_stability_flags_frequent_outliers(self):
        # Same shapes as the tolerance test, but outliers are now 20% of
        # frames — comfortably past _STABILITY_OUTLIER_TOLERANCE (1%).
        stable_row = self._anchor_row(
            0.0, (0, 0, 0), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 1, 0), (0, 1, 0),
        )
        outlier_row = self._anchor_row(
            0.0, (0, 0, 0), (-1, 0, 0), (0, 0, 1), (1, 0, 0), (1, 1, 0), (1, 1, 1),
        )
        result = _evaluate_anchor_stability([[stable_row] * 8 + [outlier_row] * 2])
        self.assertEqual(result["status"], "unstable")

    def test_anchor_stability_raises_on_empty_trajectory(self):
        with self.assertRaises(FEPExecutionError):
            _evaluate_anchor_stability([])
        with self.assertRaises(FEPExecutionError):
            _evaluate_anchor_stability([[]])

    def test_cli_exposes_fep_produce_with_resume_flag(self):
        args = build_parser().parse_args([
            "dynamic", "fep-produce", "--run-dir", "FEP_ABFE_24", "--no-resume",
        ])
        self.assertEqual(args.dyn_type, "fep-produce")
        self.assertEqual(args.run_dir, "FEP_ABFE_24")
        self.assertTrue(args.no_resume)

    def test_cli_fep_produce_resume_defaults_to_true(self):
        args = build_parser().parse_args([
            "dynamic", "fep-produce", "--run-dir", "FEP_ABFE_24",
        ])
        self.assertFalse(args.no_resume)

    def test_all_production_states_covers_every_ligand_leg_replica_lambda(self):
        config = {"replicas": 3, "lambda_windows": 24}
        states = _all_production_states(Path("/tmp/run"), config)
        self.assertEqual(len(states), 2 * 2 * 3 * 24)  # ligands x legs x replicas x windows
        self.assertEqual(states[0]["key"], "ligand_A_complex_replica_01_lambda_00")
        self.assertEqual(states[-1]["key"], "ligand_B_solvent_replica_03_lambda_23")
        first = states[0]
        self.assertEqual(
            first["directory"], Path("/tmp/run/ligand_A/complex/replica_01/lambda_00"),
        )
        self.assertEqual(
            first["equilibration_dir"], Path("/tmp/run/ligand_A/complex/replica_01/equilibration"),
        )

    def test_all_production_states_are_unique(self):
        config = {"replicas": 2, "lambda_windows": 4}
        states = _all_production_states(Path("/tmp/run"), config)
        keys = [state["key"] for state in states]
        self.assertEqual(len(keys), len(set(keys)))


class StandardStateCorrectionTests(unittest.TestCase):
    _FORCE_CONSTANTS = {
        "distance_kj_mol_nm2": 4184.0,
        "angle_kj_mol_rad2": 41.84,
        "dihedral_kj_mol_rad2": 41.84,
    }

    def test_typical_geometry_falls_in_the_literature_range(self):
        result = compute_standard_state_correction(
            {"distance_nm": 0.3, "angles_deg": [90.0, 90.0]},
            self._FORCE_CONSTANTS, 300.0,
        )
        # Boresch-restraint SSC values in the literature are typically -3 to -10 kcal/mol.
        self.assertTrue(-12.0 < result["delta_g_kcal_mol"] < -2.0)
        self.assertAlmostEqual(result["delta_g_kj_mol"], result["delta_g_kcal_mol"] * 4.184, places=6)

    def test_tighter_restraints_make_the_correction_more_negative(self):
        loose = compute_standard_state_correction(
            {"distance_nm": 0.3, "angles_deg": [90.0, 90.0]}, self._FORCE_CONSTANTS, 300.0,
        )
        tight_constants = dict(self._FORCE_CONSTANTS, distance_kj_mol_nm2=41840.0)
        tight = compute_standard_state_correction(
            {"distance_nm": 0.3, "angles_deg": [90.0, 90.0]}, tight_constants, 300.0,
        )
        self.assertLess(tight["delta_g_kj_mol"], loose["delta_g_kj_mol"])

    def test_degenerate_colinear_geometry_is_rejected(self):
        with self.assertRaises(FEPExecutionError):
            compute_standard_state_correction(
                {"distance_nm": 0.3, "angles_deg": [0.0, 90.0]}, self._FORCE_CONSTANTS, 300.0,
            )

    def test_zero_distance_is_rejected(self):
        with self.assertRaises(FEPExecutionError):
            compute_standard_state_correction(
                {"distance_nm": 0.0, "angles_deg": [90.0, 90.0]}, self._FORCE_CONSTANTS, 300.0,
            )


if __name__ == "__main__":
    unittest.main()
