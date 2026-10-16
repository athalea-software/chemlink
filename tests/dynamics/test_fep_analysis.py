import json
import tempfile
import unittest
from pathlib import Path

from pipelines.dynamics.fep_analysis import (
    _diagnose_ligand,
    _rank_ligands,
    combine_binding_free_energy,
    estimate_leg_free_energy,
    run_fep_analysis,
)
from pipelines.dynamics.fep_pipeline import FEPExecutionError

_TEMPERATURE = 300.0
_LAMBDAS = [0.0, 0.5, 1.0]
# Constant dH/dcoul across every state/frame so TI's trapezoidal integral has
# an exact, hand-computable answer: 100 kJ/mol * (1.0 - 0.0) = 100 kJ/mol.
_DHDL_COUL = 100.0


def _neighbors(state_index: int) -> list[int]:
    return [i for i in (state_index - 1, state_index, state_index + 1) if 0 <= i < len(_LAMBDAS)]


def _xvg_text(state_index: int) -> str:
    coul = _LAMBDAS[state_index]
    neighbor_indices = _neighbors(state_index)
    lines = [
        "# ChemLink test fixture (GROMACS dhdl.xvg format)",
        r'@    title "dH/d\xl\f{} and \xD\f{}H"',
        '@    xaxis  label "Time (ps)"',
        r'@    yaxis  label "dH/d\xl\f{} and \xD\f{}H (kJ/mol [\xl\f{}]\S-1\N)"',
        "@TYPE xy",
        f'@ subtitle "\\xl\\f{{}} state {state_index}: (coul-lambda, vdw-lambda) = ({coul:.4f}, 0.0000)"',
        "@ legend on",
        r'@ s0 legend "Total Energy (kJ/mol)"',
        f'@ s1 legend "dH/d\\xl\\f{{}} coul-lambda = {coul:.4f}"',
        r'@ s2 legend "dH/d\xl\f{} vdw-lambda = 0.0000"',
    ]
    for column, neighbor in enumerate(neighbor_indices):
        target = _LAMBDAS[neighbor]
        lines.append(
            f'@ s{3 + column} legend "\\xD\\f{{}}H \\xl\\f{{}} to ({target:.4f}, 0.0000)"'
        )
    # Per-frame fluctuation around the mean (average 0) so BAR/MBAR see a real
    # distribution instead of a degenerate, all-identical-samples case; the
    # mean dH/dcoul stays exactly 100.0, keeping TI's exact expected value.
    for frame_time, noise in ((0.0, -5.0), (0.002, 0.0), (0.004, 5.0)):
        dhdl_coul = _DHDL_COUL + noise
        deltas = [
            0.0 if neighbor == state_index else dhdl_coul * (_LAMBDAS[neighbor] - coul)
            for neighbor in neighbor_indices
        ]
        row = [frame_time, 0.0, dhdl_coul, 0.0] + deltas
        lines.append(" ".join(f"{value:.6f}" for value in row))
    return "\n".join(lines) + "\n"


def _write_completed_window(window_dir: Path, state_index: int) -> None:
    window_dir.mkdir(parents=True, exist_ok=True)
    (window_dir / "production.xvg").write_text(_xvg_text(state_index), encoding="utf-8")
    (window_dir / "production.gro").write_text("placeholder\n", encoding="utf-8")
    (window_dir / "production.log").write_text("...\nFinished mdrun on rank 0\n", encoding="utf-8")


def _build_leg(leg_dir: Path, replicas: int = 1) -> None:
    for replica in range(1, replicas + 1):
        for state_index in range(len(_LAMBDAS)):
            _write_completed_window(
                leg_dir / f"replica_{replica:02d}" / f"lambda_{state_index:02d}", state_index,
            )


class FEPAnalysisTests(unittest.TestCase):
    def test_estimate_leg_free_energy_ti_matches_exact_trapezoid(self):
        with tempfile.TemporaryDirectory() as tmp:
            leg_dir = Path(tmp) / "complex"
            _build_leg(leg_dir, replicas=1)
            result = estimate_leg_free_energy(leg_dir, replicas=1, windows=3, temperature=_TEMPERATURE)
            self.assertAlmostEqual(
                result["combined"]["TI"]["delta_g_kj_mol"], _DHDL_COUL * (1.0 - 0.0), places=3,
            )
            self.assertIn("BAR", result["combined"])
            self.assertEqual(result["replica_spread"]["replica_count"], 1)
            self.assertEqual(result["incomplete_replicas"], {})

    def test_mbar_is_unavailable_with_neighbor_only_foreign_energies(self):
        with tempfile.TemporaryDirectory() as tmp:
            leg_dir = Path(tmp) / "complex"
            _build_leg(leg_dir, replicas=1)
            result = estimate_leg_free_energy(leg_dir, replicas=1, windows=3, temperature=_TEMPERATURE)
            self.assertEqual(result["combined"]["MBAR"]["status"], "unavailable")

    def test_missing_replica_windows_are_reported_and_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            leg_dir = Path(tmp) / "complex"
            _build_leg(leg_dir, replicas=2)
            # Replica 2's last window never finished.
            import shutil
            shutil.rmtree(leg_dir / "replica_02" / "lambda_02")
            result = estimate_leg_free_energy(leg_dir, replicas=2, windows=3, temperature=_TEMPERATURE)
            self.assertEqual(result["replica_spread"]["replica_count"], 1)
            self.assertIn(2, result["incomplete_replicas"])
            self.assertIn("lambda_02", result["incomplete_replicas"][2])

    def test_raises_when_no_replica_is_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            leg_dir = Path(tmp) / "complex"
            # Nothing written at all: replica 1 has no completed windows.
            with self.assertRaises(FEPExecutionError):
                estimate_leg_free_energy(leg_dir, replicas=1, windows=3, temperature=_TEMPERATURE)

    def test_combine_binding_free_energy_applies_the_double_decoupling_formula(self):
        complex_leg = {"combined": {
            "TI": {"delta_g_kj_mol": 50.0, "error_kj_mol": 1.0},
            "BAR": {"delta_g_kj_mol": 48.0, "error_kj_mol": 0.5},
            "MBAR": {"status": "unavailable", "reason": "x"},
        }}
        solvent_leg = {"combined": {
            "TI": {"delta_g_kj_mol": 20.0, "error_kj_mol": 2.0},
            "BAR": {"delta_g_kj_mol": 19.0, "error_kj_mol": 0.3},
            "MBAR": {"status": "unavailable", "reason": "x"},
        }}
        restraint = {"delta_g_kj_mol": -10.0}
        binding = combine_binding_free_energy(complex_leg, solvent_leg, restraint)
        # delta_g_bind = delta_g_solvent - delta_g_complex + delta_g_restraint
        self.assertAlmostEqual(binding["TI"]["delta_g_kj_mol"], 20.0 - 50.0 - 10.0)
        self.assertAlmostEqual(binding["BAR"]["delta_g_kj_mol"], 19.0 - 48.0 - 10.0)
        self.assertAlmostEqual(binding["TI"]["error_kj_mol"], (1.0 ** 2 + 2.0 ** 2) ** 0.5)
        self.assertEqual(binding["MBAR"]["status"], "unavailable")

    def test_run_fep_analysis_end_to_end_writes_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "run_config.json").write_text(json.dumps({
                "replicas": 1, "lambda_windows": 3, "temperature_k": _TEMPERATURE,
            }), encoding="utf-8")
            (root / "fep_preflight.json").write_text(json.dumps({"method": "ABFE"}), encoding="utf-8")
            pilot_report = {
                "status": "passed",
                "ligands": {
                    label: {"standard_state_correction": {"delta_g_kj_mol": -10.0, "delta_g_kcal_mol": -2.39}}
                    for label in ("A", "B")
                },
            }
            (root / "pilot_report.json").write_text(json.dumps(pilot_report), encoding="utf-8")
            for label in ("A", "B"):
                for leg in ("complex", "solvent"):
                    _build_leg(root / f"ligand_{label}" / leg, replicas=1)

            result = run_fep_analysis(root)

            self.assertEqual(result["status"], "completed")
            self.assertTrue((root / "fep_analysis.json").is_file())
            for label in ("A", "B"):
                binding = result["ligands"][label]["binding_free_energy"]
                # Both legs are built from identical fixtures, so TI cancels to
                # exactly the (negative) standard-state correction.
                self.assertAlmostEqual(binding["TI"]["delta_g_kj_mol"], -10.0, places=3)

    def test_run_fep_analysis_requires_a_passed_pilot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "run_config.json").write_text(json.dumps({
                "replicas": 1, "lambda_windows": 3, "temperature_k": _TEMPERATURE,
            }), encoding="utf-8")
            (root / "fep_preflight.json").write_text(json.dumps({"method": "ABFE"}), encoding="utf-8")
            (root / "pilot_report.json").write_text(
                json.dumps({"status": "manual_review_required"}), encoding="utf-8",
            )
            with self.assertRaises(FEPExecutionError):
                run_fep_analysis(root)

    @staticmethod
    def _leg(replica_count=3, mean=0.0, std=0.0, incomplete=None):
        return {
            "replica_spread": {
                "replica_count": replica_count, "ti_mean_kj_mol": mean, "ti_std_kj_mol": std,
            },
            "incomplete_replicas": incomplete or {},
        }

    def test_diagnose_ligand_reliable_when_everything_looks_good(self):
        binding = {
            "TI": {"delta_g_kcal_mol": -8.0, "error_kcal_mol": 0.5},
            "BAR": {"delta_g_kcal_mol": -7.5, "error_kcal_mol": 0.4},
            "MBAR": {"status": "unavailable", "reason": "x"},
        }
        legs = {
            "complex": self._leg(replica_count=3, mean=-33.0, std=1.0),
            "solvent": self._leg(replica_count=3, mean=-25.0, std=1.0),
        }
        restraint = {"delta_g_kcal_mol": -7.0}
        result = _diagnose_ligand(binding, legs, restraint)
        self.assertEqual(result["status"], "reliable")
        self.assertTrue(all(c["status"] == "ok" for c in result["checks"]))

    def test_diagnose_ligand_flags_implausible_magnitude(self):
        binding = {
            "TI": {"delta_g_kcal_mol": -75.9, "error_kcal_mol": 1.3},
            "BAR": {"status": "unavailable", "reason": "x"},
        }
        legs = {"complex": self._leg(), "solvent": self._leg()}
        result = _diagnose_ligand(binding, legs, {"delta_g_kcal_mol": -7.0})
        self.assertEqual(result["status"], "needs_review")
        self.assertTrue(any(c["check"] == "binding_magnitude" and c["status"] == "warning"
                             for c in result["checks"]))

    def test_diagnose_ligand_flags_ti_bar_disagreement(self):
        binding = {
            "TI": {"delta_g_kcal_mol": -8.0, "error_kcal_mol": 0.3},
            "BAR": {"delta_g_kcal_mol": -12.0, "error_kcal_mol": 0.3},
        }
        legs = {"complex": self._leg(), "solvent": self._leg()}
        result = _diagnose_ligand(binding, legs, {"delta_g_kcal_mol": -7.0})
        self.assertTrue(any(c["check"] == "ti_bar_agreement" and c["status"] == "warning"
                             for c in result["checks"]))

    def test_diagnose_ligand_flags_nan_error_and_single_replica(self):
        binding = {"TI": {"delta_g_kcal_mol": -8.0, "error_kcal_mol": float("nan")}}
        legs = {
            "complex": self._leg(replica_count=1),
            "solvent": self._leg(replica_count=1),
        }
        result = _diagnose_ligand(binding, legs, {"delta_g_kcal_mol": -7.0})
        checks_by_name = {c["check"]: c for c in result["checks"]}
        self.assertEqual(checks_by_name["statistical_error"]["status"], "warning")
        self.assertEqual(checks_by_name["complex_replica_count"]["status"], "warning")
        self.assertEqual(checks_by_name["solvent_replica_count"]["status"], "warning")

    def test_diagnose_ligand_flags_replica_disagreement_and_dropped_replicas(self):
        binding = {"TI": {"delta_g_kcal_mol": -8.0, "error_kcal_mol": 0.3}}
        legs = {
            "complex": self._leg(replica_count=3, mean=-30.0, std=15.0),  # 50% spread
            "solvent": self._leg(replica_count=3, mean=-20.0, std=1.0, incomplete={2: ["lambda_03"]}),
        }
        result = _diagnose_ligand(binding, legs, {"delta_g_kcal_mol": -7.0})
        checks_by_name = {c["check"]: c for c in result["checks"]}
        self.assertEqual(checks_by_name["complex_replica_agreement"]["status"], "warning")
        self.assertEqual(checks_by_name["solvent_replica_completeness"]["status"], "warning")

    def test_diagnose_ligand_flags_standard_state_correction_out_of_range(self):
        binding = {"TI": {"delta_g_kcal_mol": -8.0, "error_kcal_mol": 0.3}}
        legs = {"complex": self._leg(), "solvent": self._leg()}
        result = _diagnose_ligand(binding, legs, {"delta_g_kcal_mol": -1.0})
        checks_by_name = {c["check"]: c for c in result["checks"]}
        self.assertEqual(checks_by_name["standard_state_correction_range"]["status"], "warning")

    def test_rank_ligands_picks_the_more_negative_ti(self):
        ligands = {
            "A": {"binding_free_energy": {"TI": {"delta_g_kcal_mol": -6.0}}},
            "B": {"binding_free_energy": {"TI": {"delta_g_kcal_mol": -9.0}}},
        }
        ranking = _rank_ligands(ligands)
        self.assertEqual(ranking["tighter_binder"], "B")
        self.assertAlmostEqual(ranking["delta_delta_g_kcal_mol"], -3.0)

    def test_rank_ligands_returns_none_without_both_ti_values(self):
        ligands = {
            "A": {"binding_free_energy": {"TI": {"delta_g_kcal_mol": -6.0}}},
            "B": {"binding_free_energy": {}},
        }
        self.assertIsNone(_rank_ligands(ligands))


if __name__ == "__main__":
    unittest.main()
