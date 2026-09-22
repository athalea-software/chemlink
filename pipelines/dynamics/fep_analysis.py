"""Fase 4: turn production dhdl.xvg files into a binding free energy.

`fep_pipeline.py` prepares ABFE inputs and `fep_execution.py` runs the
equilibration, pilot, and production MD. This module is the last step: it
reads the `production.xvg` (GROMACS dhdl) files every `replica_NN/lambda_NN/`
production window wrote, estimates each (ligand, leg) decoupling free energy
with TI, BAR, and MBAR (alchemlyb/pymbar), and combines the two legs with the
analytical standard-state correction already computed by `fep-pilot`
(`pilot_report.json`) into a per-ligand binding free energy:

    delta_g_bind = delta_g_solvent - delta_g_complex + delta_g_restraint

`delta_g_complex` is the free energy of turning the ligand's nonbonded
interactions off in the complex while the Boresch restraint turns on
(lambda 0: coupled, unrestrained -> lambda 1: decoupled, restrained);
`delta_g_solvent` is the same decoupling in bulk solvent, unrestrained
throughout; `delta_g_restraint` (Boresch et al. 2003, eq. 32) is the free
energy of releasing that restraint against the 1 M standard-state reference
volume. This is the standard Boresch double-decoupling cycle.

MBAR needs every lambda pair's foreign energy for each window; the
production `.mdp` this project writes (`calc-lambda-neighbors = 1`, see
`fep_pipeline._fep_lambda_schedule`) only records neighboring lambdas, which
is what BAR needs. MBAR is therefore reported only if the parsed data
happens to have no missing pairs, and marked "unavailable" otherwise — TI
and BAR are always available since they only need each window's own
dH/dlambda and neighbor energies.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from pipelines.dynamics.fep_pipeline import FEPExecutionError, dependency_status
from pipelines.dynamics.resume import phase_completed

_KJ_PER_KCAL = 4.184

# Heuristic thresholds for the automatic reliability checks in
# `_diagnose_ligand` — not physical constants, just the same rules of thumb
# a person would apply reading fep_analysis.json by hand (see docs/FEP.md,
# "Interpreting the result").
_PLAUSIBLE_MAGNITUDE_KCAL = (2.0, 20.0)  # typical drug-like ligand |delta_g_bind|
_TI_BAR_DISAGREEMENT_KCAL = 2.0
_ERROR_FRACTION_WARNING = 0.3  # statistical error vs |delta_g|
_REPLICA_STD_FRACTION_WARNING = 0.3  # inter-replica TI std vs |mean|
_SSC_PLAUSIBLE_RANGE_KCAL = (-12.0, -3.0)  # literature range for Boresch SSC


def _load_json(path: Path, error: str) -> dict:
    if not path.is_file():
        raise FEPExecutionError(error)
    return json.loads(path.read_text())


def _require_alchemlyb():
    tools = dependency_status()
    missing = [name for name in ("alchemlyb", "pymbar") if not tools.get(name)]
    if missing:
        raise FEPExecutionError("FEP analysis requires: " + ", ".join(missing))
    from alchemlyb.estimators import BAR, MBAR, TI
    from alchemlyb.parsing.gmx import extract_dHdl, extract_u_nk
    from alchemlyb.postprocessors.units import to_kJmol
    return extract_dHdl, extract_u_nk, TI, BAR, MBAR, to_kJmol


def _completed_xvg_files(leg_dir: Path, replica: int, windows: int) -> tuple[list[Path], list[str]]:
    """production.xvg paths for windows with a finished mdrun, and the missing ones' names."""
    files, missing = [], []
    for index in range(windows):
        window_dir = leg_dir / f"replica_{replica:02d}" / f"lambda_{index:02d}"
        if phase_completed(window_dir, "production"):
            files.append(window_dir / "production.xvg")
        else:
            missing.append(window_dir.name)
    return files, missing


def _parse_chain(paths: list[Path], temperature: float, extract_dHdl, extract_u_nk):
    import pandas as pd

    dhdl_frames = [extract_dHdl(str(path), T=temperature) for path in paths]
    u_nk_frames = [extract_u_nk(str(path), T=temperature) for path in paths]
    return pd.concat(dhdl_frames), pd.concat(u_nk_frames)


def _fit_estimators(dhdl, u_nk, temperature: float, TI, BAR, MBAR, to_kJmol) -> dict:
    """Fit TI and BAR always; MBAR only if the pooled u_nk matrix has no missing pairs."""
    ti = TI().fit(dhdl)
    ti_kj = to_kJmol(ti.delta_f_, T=temperature)
    ti_err_kj = to_kJmol(ti.d_delta_f_, T=temperature)
    results = {
        "TI": {
            "delta_g_kj_mol": float(ti_kj.iloc[0, -1]),
            "error_kj_mol": float(ti_err_kj.iloc[0, -1]),
        },
    }

    bar = BAR().fit(u_nk)
    bar_kj = to_kJmol(bar.delta_f_, T=temperature)
    bar_d_kj = to_kJmol(bar.d_delta_f_, T=temperature)
    state_count = len(bar.delta_f_.index)
    # BAR only reports adjacent-state errors (d_delta_f_[0, -1] is NaN); the
    # end-to-end error is the standard chain propagation over those steps.
    chain_error = math.sqrt(sum(
        float(bar_d_kj.iloc[i, i + 1]) ** 2 for i in range(state_count - 1)
    ))
    results["BAR"] = {
        "delta_g_kj_mol": float(bar_kj.iloc[0, -1]),
        "error_kj_mol": chain_error,
    }

    if u_nk.isna().any().any():
        results["MBAR"] = {
            "status": "unavailable",
            "reason": (
                "production.mdp uses calc-lambda-neighbors = 1 (neighbor-only foreign "
                "energies); full MBAR needs every lambda pair, so only TI and BAR are "
                "reported."
            ),
        }
    else:
        mbar = MBAR().fit(u_nk)
        mbar_kj = to_kJmol(mbar.delta_f_, T=temperature)
        mbar_err_kj = to_kJmol(mbar.d_delta_f_, T=temperature)
        results["MBAR"] = {
            "delta_g_kj_mol": float(mbar_kj.iloc[0, -1]),
            "error_kj_mol": float(mbar_err_kj.iloc[0, -1]),
        }
    return results


def estimate_leg_free_energy(leg_dir: Path, replicas: int, windows: int, temperature: float) -> dict:
    """Estimate one (ligand, leg)'s decoupling free energy from every completed replica.

    Pools every fully-completed replica's dhdl data into one combined
    TI/BAR/(MBAR) estimate (`combined`), and separately fits each replica on
    its own (`replica_estimates`) so reproducibility across independent
    replicas can be checked (`replica_spread`) — a replica whose TI estimate
    is far from the others usually means it needs a longer production run.
    """
    import pandas as pd

    extract_dHdl, extract_u_nk, TI, BAR, MBAR, to_kJmol = _require_alchemlyb()
    per_replica: dict[int, dict] = {}
    pooled_dhdl, pooled_u_nk = [], []
    incomplete: dict[int, list[str]] = {}

    for replica in range(1, replicas + 1):
        files, missing = _completed_xvg_files(leg_dir, replica, windows)
        if missing:
            incomplete[replica] = missing
            continue
        dhdl, u_nk = _parse_chain(files, temperature, extract_dHdl, extract_u_nk)
        pooled_dhdl.append(dhdl)
        pooled_u_nk.append(u_nk)
        per_replica[replica] = _fit_estimators(dhdl, u_nk, temperature, TI, BAR, MBAR, to_kJmol)

    if not pooled_dhdl:
        raise FEPExecutionError(
            f"No fully completed replica found in {leg_dir}; missing windows: {incomplete}. "
            "Run 'chemlink dynamic fep-produce' first."
        )

    combined = _fit_estimators(
        pd.concat(pooled_dhdl), pd.concat(pooled_u_nk), temperature, TI, BAR, MBAR, to_kJmol,
    )
    ti_values = [entry["TI"]["delta_g_kj_mol"] for entry in per_replica.values()]
    mean_ti = sum(ti_values) / len(ti_values)
    spread = {
        "replica_count": len(per_replica),
        "ti_mean_kj_mol": mean_ti,
        "ti_std_kj_mol": (
            math.sqrt(sum((value - mean_ti) ** 2 for value in ti_values) / len(ti_values))
            if len(ti_values) > 1 else 0.0
        ),
    }
    return {
        "combined": combined,
        "replica_estimates": per_replica,
        "replica_spread": spread,
        "incomplete_replicas": incomplete,
    }


def combine_binding_free_energy(complex_leg: dict, solvent_leg: dict, restraint: dict) -> dict:
    """delta_g_bind = delta_g_solvent - delta_g_complex + delta_g_restraint, per estimator."""
    binding: dict[str, dict[str, Any]] = {}
    for estimator in ("TI", "BAR", "MBAR"):
        complex_entry = complex_leg["combined"].get(estimator, {})
        solvent_entry = solvent_leg["combined"].get(estimator, {})
        if "delta_g_kj_mol" not in complex_entry or "delta_g_kj_mol" not in solvent_entry:
            binding[estimator] = {
                "status": "unavailable",
                "reason": complex_entry.get("reason") or solvent_entry.get("reason") or "missing leg estimate",
            }
            continue
        delta_g = (
            solvent_entry["delta_g_kj_mol"] - complex_entry["delta_g_kj_mol"]
            + restraint["delta_g_kj_mol"]
        )
        error = math.sqrt(complex_entry["error_kj_mol"] ** 2 + solvent_entry["error_kj_mol"] ** 2)
        binding[estimator] = {
            "delta_g_kj_mol": delta_g,
            "delta_g_kcal_mol": delta_g / _KJ_PER_KCAL,
            "error_kj_mol": error,
            "error_kcal_mol": error / _KJ_PER_KCAL,
        }
    return binding


def _check(name: str, ok: bool, message: str) -> dict:
    return {"check": name, "status": "ok" if ok else "warning", "message": message}


def _diagnose_ligand(binding: dict, legs: dict, restraint: dict) -> dict:
    """Automate the reliability checklist a person would apply by hand.

    Flags, per ligand: an implausible binding magnitude, TI/BAR disagreement,
    a large statistical error, replicas that disagree with each other (or
    were dropped), and a standard-state correction outside its usual
    literature range. None of these thresholds are hard physical limits —
    they mirror the same rules of thumb documented in docs/FEP.md, so a
    "needs_review" verdict means "look at this by hand", not "this is wrong".
    """
    checks: list[dict] = []
    ti = binding.get("TI", {})
    bar = binding.get("BAR", {})
    ti_value = ti.get("delta_g_kcal_mol")
    ti_error = ti.get("error_kcal_mol")
    bar_value = bar.get("delta_g_kcal_mol") if bar.get("status") != "unavailable" else None

    if ti_value is not None:
        magnitude = abs(ti_value)
        low, high = _PLAUSIBLE_MAGNITUDE_KCAL
        ok = low <= magnitude <= high
        checks.append(_check(
            "binding_magnitude", ok,
            f"|delta_g_bind| = {magnitude:.1f} kcal/mol is "
            + ("within" if ok else "outside")
            + f" the typical {low:.0f}-{high:.0f} kcal/mol range for a drug-like binder"
            + ("" if ok else "; treat this value as unconverged until checked"),
        ))

    if ti_value is not None and bar_value is not None:
        diff = abs(ti_value - bar_value)
        ok = diff <= _TI_BAR_DISAGREEMENT_KCAL
        checks.append(_check(
            "ti_bar_agreement", ok,
            f"TI and BAR differ by {diff:.2f} kcal/mol"
            + ("" if ok else
               f" (over the {_TI_BAR_DISAGREEMENT_KCAL:.1f} kcal/mol threshold); "
               "usually means poor overlap between neighboring lambda windows"),
        ))

    if ti_error is not None:
        if isinstance(ti_error, float) and math.isnan(ti_error):
            checks.append(_check("statistical_error", False, "TI error is NaN"))
        elif ti_value:
            ok = (ti_error / abs(ti_value)) <= _ERROR_FRACTION_WARNING
            checks.append(_check(
                "statistical_error", ok,
                f"TI error is +/-{ti_error:.2f} kcal/mol on a {ti_value:.1f} kcal/mol estimate"
                + ("" if ok else " — large relative to the estimate; run more production time"),
            ))

    for leg_name, leg in legs.items():
        if leg.get("incomplete_replicas"):
            dropped = len(leg["incomplete_replicas"])
            checks.append(_check(
                f"{leg_name}_replica_completeness", False,
                f"{leg_name} leg: {dropped} replica(s) excluded from the combined "
                "estimate (missing or failed production windows)",
            ))
        spread = leg.get("replica_spread", {})
        count = spread.get("replica_count", 0)
        mean, std = spread.get("ti_mean_kj_mol"), spread.get("ti_std_kj_mol")
        if count <= 1:
            checks.append(_check(
                f"{leg_name}_replica_count", False,
                f"{leg_name} leg: only {count} replica completed; independent "
                "replicas are how you check reproducibility, not just statistical error",
            ))
        elif mean:
            ok = abs(std / mean) <= _REPLICA_STD_FRACTION_WARNING
            checks.append(_check(
                f"{leg_name}_replica_agreement", ok,
                f"{leg_name} leg: {count} replicas, TI spread {abs(std / mean):.0%} of the mean"
                + ("" if ok else " — replicas disagree; needs longer production, not just more analysis"),
            ))

    ssc = restraint.get("delta_g_kcal_mol")
    if ssc is not None:
        low, high = _SSC_PLAUSIBLE_RANGE_KCAL
        ok = low <= ssc <= high
        checks.append(_check(
            "standard_state_correction_range", ok,
            f"Standard-state correction = {ssc:.2f} kcal/mol is "
            + ("within" if ok else "outside")
            + f" the typical {low:.0f} to {high:.0f} kcal/mol literature range"
            + ("" if ok else " — check the Boresch anchor geometry in common/*_boresch_anchors.json"),
        ))

    return {
        "status": "reliable" if all(item["status"] == "ok" for item in checks) else "needs_review",
        "checks": checks,
    }


def _rank_ligands(ligands: dict) -> dict | None:
    """Which ligand binds tighter, by TI (always available, unlike BAR/MBAR)."""
    values = {
        label: entry["binding_free_energy"].get("TI", {}).get("delta_g_kcal_mol")
        for label, entry in ligands.items()
    }
    if values.get("A") is None or values.get("B") is None:
        return None
    if values["A"] == values["B"]:
        tighter = "tie"
    else:
        tighter = "A" if values["A"] < values["B"] else "B"
    return {
        "estimator": "TI",
        "tighter_binder": tighter,
        "delta_delta_g_kcal_mol": values["B"] - values["A"],
        "note": "delta_delta_g_kcal_mol = delta_g_bind(B) - delta_g_bind(A); negative means B binds tighter",
    }


def run_fep_analysis(run_dir: str | Path) -> dict:
    """Estimate ΔG_bind for both ABFE ligands from every completed production window."""
    root = Path(run_dir).expanduser().resolve()
    config = _load_json(root / "run_config.json", f"Not a ChemLink FEP run directory: {root}")
    preflight = _load_json(
        root / "fep_preflight.json", f"No FEP preflight report found in {root}."
    )
    if preflight.get("method") != "ABFE":
        raise FEPExecutionError("FEP analysis is currently available only for ABFE runs.")
    pilot_report = _load_json(
        root / "pilot_report.json",
        f"No pilot_report.json found in {root}; run 'chemlink dynamic fep-pilot' first.",
    )
    if pilot_report.get("status") != "passed":
        raise FEPExecutionError(
            "The ABFE pilot has not passed yet "
            f'(pilot_report.json["status"] == {pilot_report.get("status")!r}); '
            "resolve the anchor/setup issues and re-run 'chemlink dynamic fep-pilot' first."
        )

    replicas = int(config["replicas"])
    windows = int(config["lambda_windows"])
    temperature = float(config["temperature_k"])

    ligands: dict[str, dict] = {}
    for label in ("A", "B"):
        legs = {
            leg: estimate_leg_free_energy(root / f"ligand_{label}" / leg, replicas, windows, temperature)
            for leg in ("complex", "solvent")
        }
        restraint = pilot_report["ligands"][label]["standard_state_correction"]
        binding = combine_binding_free_energy(legs["complex"], legs["solvent"], restraint)
        ligands[label] = {
            "legs": legs,
            "standard_state_correction": restraint,
            "binding_free_energy": binding,
            "diagnostics": _diagnose_ligand(binding, legs, restraint),
        }

    review_status = (
        "reliable" if all(entry["diagnostics"]["status"] == "reliable" for entry in ligands.values())
        else "needs_review"
    )
    report = {
        "status": "completed",
        "review_status": review_status,
        "temperature_k": temperature,
        "formula": "delta_g_bind = delta_g_solvent - delta_g_complex + delta_g_restraint",
        "ligands": ligands,
        "ranking": _rank_ligands(ligands),
    }
    (root / "fep_analysis.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
