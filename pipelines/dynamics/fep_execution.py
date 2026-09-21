"""Execution engine for prepared ABFE runs.

`fep_pipeline.py` prepares ligands, builds/solvates/neutralizes the four
ABFE systems, and validates the Boresch restraint topology with a throwaway
`grompp` check. It never calls `gmx mdrun`. This module runs the actual
molecular dynamics:

- Fase 1, replica equilibration (EM + NVT + NPT) at the lambda-0 reference
  state (restraints off, ligand fully coupled), which every replica's
  lambda-window run is seeded from.
- Fase 2, the real pilot: short MD at a lambda subset, anchor-stability
  verification from the resulting trajectories, and the analytical
  standard-state correction (Boresch et al. 2003) — the gate production
  needs before it unlocks.
"""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Any

from hpc.cluster.resource_detector import get_hardware_profile

from pipelines.dynamics.fep_pipeline import (
    FEPExecutionError,
    _angle,
    _dihedral,
    _distance,
    _fep_lambda_schedule,
    dependency_status,
)
from pipelines.dynamics.gmx_optimizer import get_cached_optimal_mdrun_flags
from pipelines.dynamics.logger import get_step_logger
from pipelines.dynamics.mdrun_runner import run_mdrun_with_fallback
from pipelines.dynamics.resume import phase_completed
from pipelines.dynamics.utils import get_system_threads
from utils.progress import job_bar
from utils.retry import SubprocessError, run_subprocess

_GROMPP_TIMEOUT = 180
_EM_STEPS = 50_000

# grompp raises utils.retry.SubprocessError; mdrun (via run_mdrun_with_fallback)
# raises the stdlib subprocess.CalledProcessError once GPU+CPU fallback is
# exhausted, or a bare RuntimeError on timeout. All three are the "this one
# GROMACS invocation failed" family every caller below needs to catch.
_MDRUN_FAILURES = (SubprocessError, subprocess.CalledProcessError, RuntimeError)

# Boresch et al. (2003, J. Phys. Chem. B 107, 9535) standard-state constants.
_STANDARD_STATE_VOLUME_NM3 = 1.66054  # 1 / (C_deg * N_A), C_deg = 1 mol/L
_GAS_CONSTANT_KJ_MOL_K = 0.0083144621  # kJ / (mol . K)

# Anchor-stability bounds: same acceptance range select_boresch_anchors()
# already uses for the *starting* geometry, now applied across the pilot
# trajectory. A distance/angle drifting past these means the restraint
# geometry is no longer well-defined (near-linear angle, dissociated pair).
_ANGLE_MIN_DEG = 15.0
_ANGLE_MAX_DEG = 165.0
_DISTANCE_MIN_NM = 0.15
_DISTANCE_MAX_NM = 1.5
_DIHEDRAL_JUMP_DEG = 150.0

# A harmonic restraint's thermal tail eventually samples past any fixed
# bound purely by chance, more so the longer the pilot runs; checking
# min()/max() against a hard cutoff over hundreds of frames flags that noise
# as "unstable" (observed on real data: 2 frames out of 404, ~0.07-0.7 deg
# past the angle bound). Tolerate a small fraction of such outliers and only
# fail once they're common enough to suggest the restrained geometry itself
# has shifted.
_STABILITY_OUTLIER_TOLERANCE = 0.01


def _load_json(path: Path, error: str) -> dict:
    if not path.is_file():
        raise FEPExecutionError(error)
    return json.loads(path.read_text())


def _lambda_values(config: dict) -> list[float]:
    windows = int(config["lambda_windows"])
    return [index / (windows - 1) for index in range(windows)]


def _em_mdp(nstlist: int, config: dict, method: str, lambdas: list[float]) -> str:
    lambda_parameters, coupling = _fep_lambda_schedule(lambdas, method)
    return f"""; ChemLink {method} equilibration EM (lambda-state 0)
integrator               = steep
emtol                    = 1000.0
emstep                   = 0.01
nsteps                   = {_EM_STEPS}
cutoff-scheme            = Verlet
nstlist                  = {nstlist}
coulombtype              = PME
vdwtype                  = Cut-off
rcoulomb                 = 1.0
rvdw                     = 1.0
pbc                      = xyz
free-energy              = yes
init-lambda-state        = 0
{lambda_parameters}calc-lambda-neighbors    = 1
sc-alpha                 = 0.5
sc-power                 = 1
sc-sigma                 = 0.3
{coupling}
"""


def _nvt_mdp(nstlist: int, config: dict, method: str, lambdas: list[float]) -> str:
    lambda_parameters, coupling = _fep_lambda_schedule(lambdas, method)
    nsteps = int(float(config["equilibration_ns"]) * 500000)
    temperature = float(config["temperature_k"])
    return f"""; ChemLink {method} equilibration NVT (lambda-state 0)
integrator               = md
dt                       = 0.002
nsteps                   = {nsteps}
nstxout                  = 5000
nstvout                  = 5000
nstenergy                = 5000
nstlog                   = 5000
continuation             = no
constraint_algorithm     = lincs
constraints              = h-bonds
cutoff-scheme            = Verlet
ns_type                  = grid
nstlist                  = {nstlist}
rcoulomb                 = 1.0
rvdw                     = 1.0
coulombtype              = PME
pme_order                = 4
fourierspacing           = 0.16
tcoupl                   = V-rescale
tc-grps                  = System
tau_t                    = 0.1
ref_t                    = {temperature:g}
pcoupl                   = no
pbc                      = xyz
gen_vel                  = yes
gen_temp                 = {temperature:g}
gen_seed                 = -1
free-energy              = yes
init-lambda-state        = 0
{lambda_parameters}calc-lambda-neighbors    = 1
sc-alpha                 = 0.5
sc-power                 = 1
sc-sigma                 = 0.3
{coupling}
"""


def _npt_mdp(nstlist: int, config: dict, method: str, lambdas: list[float]) -> str:
    lambda_parameters, coupling = _fep_lambda_schedule(lambdas, method)
    nsteps = int(float(config["equilibration_ns"]) * 500000)
    temperature = float(config["temperature_k"])
    return f"""; ChemLink {method} equilibration NPT (lambda-state 0)
integrator               = md
dt                       = 0.002
nsteps                   = {nsteps}
nstxout                  = 5000
nstvout                  = 5000
nstenergy                = 5000
nstlog                   = 5000
continuation             = yes
constraint_algorithm     = lincs
constraints              = h-bonds
cutoff-scheme            = Verlet
ns_type                  = grid
nstlist                  = {nstlist}
rcoulomb                 = 1.0
rvdw                     = 1.0
coulombtype              = PME
pme_order                = 4
fourierspacing           = 0.16
tcoupl                   = V-rescale
tc-grps                  = System
tau_t                    = 0.1
ref_t                    = {temperature:g}
pcoupl                   = C-rescale
pcoupltype               = isotropic
tau_p                    = 1.0
ref_p                    = 1.0
compressibility          = 4.5e-5
refcoord_scaling         = com
pbc                      = xyz
gen_vel                  = no
free-energy              = yes
init-lambda-state        = 0
{lambda_parameters}calc-lambda-neighbors    = 1
sc-alpha                 = 0.5
sc-power                 = 1
sc-sigma                 = 0.3
{coupling}
"""


def equilibrate_replica(
    leg_dir: Path, replica: int, config: dict, method: str, logger,
    *, system_label: str = "", show_progress: bool = True,
) -> dict:
    """Run EM -> NVT -> NPT for one replica of one ABFE system (ligand leg).

    `leg_dir` is e.g. `<run_dir>/ligand_A/complex`: it holds the already
    solvated/neutralized `system/` subdirectory built by
    `execute_abfe_system_setup`, and is the same parent that
    `_create_window_layout` uses for `replica_NN/lambda_NN`. Equilibration
    output lands in `replica_NN/equilibration/` as the seed every lambda
    window of that replica restarts from.

    `system_label` (e.g. "ligand_A/complex replica_01") disambiguates the
    per-phase console/log tag — `run_fep_equilibration` calls this once per
    system, so without it every system's phases would show up under the
    same generic "FEP equilibration NVT" name. `show_progress` controls
    whether each phase's own live step bar and PERF line are shown
    (`run_fep_equilibration` also shows an aggregate counter over all
    systems alongside it, same as `run_fep_pilot` does for pilot windows).
    """
    system_dir = leg_dir / "system"
    ionized = system_dir / "ionized.gro"
    topology = system_dir / "topol.top"
    if not ionized.is_file() or not topology.is_file():
        raise FEPExecutionError(
            f"ABFE system is not built yet in {system_dir}; "
            "run 'chemlink dynamic fep' without --prepare-only first."
        )

    work_dir = leg_dir / f"replica_{replica:02d}" / "equilibration"
    work_dir.mkdir(parents=True, exist_ok=True)

    gmx = str(dependency_status()["gromacs"])
    opt = get_cached_optimal_mdrun_flags(config)
    lambdas = _lambda_values(config)
    use_gpu = bool(config.get("gpu_ids"))
    mpi_tasks = int(config.get("mpi_tasks", 1))
    mpi_hosts = config.get("mpi_hosts", "")
    status: dict[str, str] = {}

    def _run_phase(phase: str, mdp_text: str, coord_args: list[str], nsteps: int) -> None:
        if phase_completed(work_dir, phase):
            status[phase] = "already_completed"
            return
        (work_dir / f"{phase}.mdp").write_text(mdp_text, encoding="utf-8")
        try:
            run_subprocess(
                [gmx, "grompp", "-f", f"{phase}.mdp", *coord_args,
                 "-p", str(topology), "-o", f"{phase}.tpr", "-maxwarn", "10"],
                timeout=_GROMPP_TIMEOUT, retries=2, cwd=str(work_dir), logger=logger,
            )
            base = [gmx, "mdrun", "-v", "-deffnm", phase]
            tag = f"FEP equilibration {system_label} {phase.upper()}".replace("  ", " ").strip()
            run_mdrun_with_fallback(
                cmd_gpu=base + opt["gpu"], cmd_cpu=base + opt["cpu"],
                work_dir=str(work_dir), phase=tag,
                total_steps=nsteps, logger=logger, use_gpu=use_gpu,
                timeout=None, mpi_tasks=mpi_tasks, mpi_hosts=mpi_hosts,
                show_progress=show_progress,
            )
        except _MDRUN_FAILURES as exc:
            raise FEPExecutionError(f"Equilibration {phase.upper()} failed in {work_dir}: {exc}") from exc
        status[phase] = "completed"

    _run_phase("em", _em_mdp(opt["nstlist"], config, method, lambdas),
               ["-c", str(ionized)], _EM_STEPS)
    equilibration_steps = int(float(config["equilibration_ns"]) * 500000)
    _run_phase("nvt", _nvt_mdp(opt["nstlist"], config, method, lambdas),
               ["-c", "em.gro"], equilibration_steps)
    _run_phase("npt", _npt_mdp(opt["nstlist"], config, method, lambdas),
               ["-c", "nvt.gro", "-t", "nvt.cpt", "-r", "nvt.gro"], equilibration_steps)

    if not phase_completed(work_dir, "npt"):
        raise FEPExecutionError(f"Equilibration finished but npt.gro was not found in {work_dir}.")

    (work_dir / "equilibration_status.json").write_text(
        json.dumps(status, indent=2) + "\n", encoding="utf-8",
    )
    return {"directory": str(work_dir), "status": status}


def run_fep_equilibration(run_dir: str | Path) -> dict:
    """Run independent EM+NVT+NPT equilibration for every ABFE replica.

    Each of the 3 replicas x 4 systems (ligand A/B x complex/solvent) gets
    its own equilibration with independently randomized initial velocities
    (`gen_vel = yes`, `gen_seed = -1` in NVT), so replicas stay statistically
    independent once production branches off from them.
    """
    root = Path(run_dir).expanduser().resolve()
    config = _load_json(root / "run_config.json", f"Not a ChemLink FEP run directory: {root}")
    preflight = _load_json(
        root / "fep_preflight.json", f"No FEP preflight report found in {root}."
    )
    if preflight.get("method") != "ABFE":
        raise FEPExecutionError("Dynamics execution is currently available only for ABFE runs.")
    if not dependency_status().get("gromacs"):
        raise FEPExecutionError("GROMACS is required to run FEP equilibration.")

    config.setdefault("gpu_ids", get_hardware_profile().gpu_indices)
    config.setdefault("threads", get_system_threads())
    logger = get_step_logger(__name__, str(root / "fep_execution.log"))

    replicas = int(config["replicas"])
    results: dict[str, Any] = {}
    total = 2 * 2 * replicas
    with job_bar("Equilibrating replicas", total, unit="system") as bar:
        for label in ("A", "B"):
            for leg in ("complex", "solvent"):
                leg_dir = root / f"ligand_{label}" / leg
                for replica in range(1, replicas + 1):
                    key = f"ligand_{label}_{leg}_replica_{replica:02d}"
                    results[key] = equilibrate_replica(
                        leg_dir, replica, config, "ABFE", logger,
                        system_label=f"ligand_{label}/{leg} replica_{replica:02d}",
                        show_progress=True,
                    )
                    bar.update(1)

    all_done = all(entry["status"].get("npt") in {"completed", "already_completed"} for entry in results.values())
    equilibration_report = {
        "status": "equilibrated" if all_done else "incomplete",
        "replicas": results,
    }
    (root / "equilibration_report.json").write_text(
        json.dumps(equilibration_report, indent=2) + "\n", encoding="utf-8",
    )
    return equilibration_report


# ── Fase 2: real pilot (anchor stability + standard-state correction) ──────


def _pilot_mdp(state: int, lambdas: list[float], config: dict, method: str) -> str:
    """Short pilot window at one lambda state, seeded from replica NPT.

    Unlike `_production_mdp` (sized by `production_ns`), this uses
    `equilibration_ns` for a cheap per-window run, and writes a compressed
    trajectory (`nstxout-compressed`) so `check_anchor_stability` can read
    the restraint atom positions back.
    """
    lambda_parameters, coupling = _fep_lambda_schedule(lambdas, method)
    nsteps = int(float(config["equilibration_ns"]) * 500000)
    return f"""; ChemLink {method} pilot window (lambda-state {state})
integrator               = md
dt                       = 0.002
nsteps                   = {nsteps}
nstxout-compressed       = 500
compressed-x-grps        = System
continuation             = yes
constraints              = h-bonds
cutoff-scheme            = Verlet
coulombtype              = PME
vdwtype                  = Cut-off
tcoupl                   = v-rescale
tc-grps                  = System
tau_t                    = 0.1
ref-t                    = {float(config['temperature_k']):g}
pcoupl                   = C-rescale
pcoupltype               = isotropic
tau_p                    = 1.0
ref_p                    = 1.0
compressibility          = 4.5e-5
free-energy              = yes
init-lambda-state        = {state}
{lambda_parameters}calc-lambda-neighbors    = 1
sc-alpha                 = 0.5
sc-power                 = 1
sc-sigma                 = 0.3
nstdhdl                  = 100
dhdl-print-energy        = total
separate-dhdl-file       = yes
{coupling}
"""


def run_pilot_lambda_state(
    leg_dir: Path, replica: int, lambda_index: int, config: dict, method: str, logger,
    *, system_label: str = "", show_progress: bool = True,
) -> dict:
    """Run one short pilot MD window, restarting from that replica's NPT endpoint."""
    equilibration_dir = leg_dir / f"replica_{replica:02d}" / "equilibration"
    if not phase_completed(equilibration_dir, "npt"):
        raise FEPExecutionError(
            f"Replica {replica} of {leg_dir} is not equilibrated yet; "
            "run 'chemlink dynamic fep-equilibrate' first."
        )
    work_dir = leg_dir / f"replica_{replica:02d}" / "pilot" / f"lambda_{lambda_index:02d}"
    work_dir.mkdir(parents=True, exist_ok=True)
    if phase_completed(work_dir, "pilot"):
        return {"directory": str(work_dir), "status": "already_completed"}

    gmx = str(dependency_status()["gromacs"])
    opt = get_cached_optimal_mdrun_flags(config)
    lambdas = _lambda_values(config)
    (work_dir / "pilot.mdp").write_text(
        _pilot_mdp(lambda_index, lambdas, config, method), encoding="utf-8",
    )
    nsteps = int(float(config["equilibration_ns"]) * 500000)
    try:
        run_subprocess(
            [gmx, "grompp", "-f", "pilot.mdp",
             "-c", str(equilibration_dir / "npt.gro"), "-t", str(equilibration_dir / "npt.cpt"),
             "-p", str(leg_dir / "system" / "topol.top"), "-o", "pilot.tpr", "-maxwarn", "10"],
            timeout=_GROMPP_TIMEOUT, retries=2, cwd=str(work_dir), logger=logger,
        )
        base = [gmx, "mdrun", "-v", "-deffnm", "pilot"]
        tag = f"FEP pilot {system_label} lambda {lambda_index:02d}".replace("  ", " ").strip()
        run_mdrun_with_fallback(
            cmd_gpu=base + opt["gpu"], cmd_cpu=base + opt["cpu"],
            work_dir=str(work_dir), phase=tag,
            total_steps=nsteps, logger=logger, use_gpu=bool(config.get("gpu_ids")),
            timeout=None, mpi_tasks=int(config.get("mpi_tasks", 1)), mpi_hosts=config.get("mpi_hosts", ""),
            show_progress=show_progress,
        )
    except _MDRUN_FAILURES as exc:
        raise FEPExecutionError(f"Pilot window lambda {lambda_index} failed in {work_dir}: {exc}") from exc
    return {"directory": str(work_dir), "status": "completed"}


def _anchor_index_file(protein_indices: list[int], ligand_indices: list[int]) -> str:
    """GROMACS .ndx text with one group holding all 6 Boresch anchors, in order.

    `gmx traj -ox` writes the per-atom XYZ of one selected group per frame,
    in the group's own atom order — so all 6 anchors (P1, P2, P3, L1, L2, L3)
    must be a single group for `_parse_traj_xvg`/`_evaluate_anchor_stability`
    to read them back as one row. It also has to be the *only* group: `-n`
    replaces gmx's auto-generated System/Protein/... groups rather than
    extending them, and a custom index file with exactly one group is
    auto-selected without an interactive prompt (checked against a real
    `gmx traj` run: giving it 6 one-atom groups instead makes it block on
    "Select a group:" with no stdin to answer it).
    """
    atoms = list(protein_indices) + list(ligand_indices)
    atom_line = " ".join(str(atom) for atom in atoms)
    return f"[ ChemLink_Boresch_Anchors ]\n{atom_line}\n"


def _parse_traj_xvg(text: str) -> list[tuple[float, ...]]:
    """Parse `gmx traj -ox` output: one row per frame, (time, x1,y1,z1, ..., x6,y6,z6)."""
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line[0] in "#@":
            continue
        rows.append(tuple(float(value) for value in line.split()))
    return rows


def _fraction_out_of_bounds(values: list[float], low: float, high: float) -> float:
    return sum(1 for value in values if value < low or value > high) / len(values)


def _evaluate_anchor_stability(windows: list[list[tuple[float, ...]]]) -> dict:
    """Recompute the 6 Boresch coordinates per frame and flag instability.

    `windows` is one list of frames per pilot lambda window, each frame
    `(time, x1,y1,z1, ..., x6,y6,z6)` for anchors (P1, P2, P3, L1, L2, L3) in
    that order. Distance/angle bounds are checked over every frame pooled
    together (a fixed geometric range doesn't care which window a frame came
    from), but dihedral "jumps" are computed only between consecutive frames
    *within* the same window: consecutive lambda windows are independent
    simulations seeded from the same NPT endpoint, so comparing the last
    frame of one to the first frame of the next would flag a jump that has
    nothing to do with restraint stability.

    A frame or two past a bound is expected thermal noise rather than
    instability — `_STABILITY_OUTLIER_TOLERANCE` absorbs that; see its
    definition for the real trajectory that motivated it. Pure function,
    unit-tested with synthetic rows instead of a real trajectory.
    """
    if not windows or not any(windows):
        raise FEPExecutionError("No pilot trajectory frames were extracted for anchor stability.")

    distances: list[float] = []
    angles_a: list[float] = []
    angles_b: list[float] = []
    jumps_by_dihedral: list[list[float]] = [[], [], []]
    for window in windows:
        dihedral_rows = []
        for row in window:
            coords = row[1:]
            if len(coords) < 18:
                continue
            p1, p2, p3, l1, l2, l3 = (tuple(coords[index * 3:index * 3 + 3]) for index in range(6))
            distances.append(_distance(p1, l1))
            angles_a.append(_angle(p2, p1, l1))
            angles_b.append(_angle(p1, l1, l2))
            dihedral_rows.append([
                _dihedral(p3, p2, p1, l1), _dihedral(p2, p1, l1, l2), _dihedral(p1, l1, l2, l3),
            ])
        for index in range(3):
            column = [row[index] for row in dihedral_rows]
            jumps_by_dihedral[index].extend(
                min(abs(current - previous), 360.0 - abs(current - previous))
                for previous, current in zip(column, column[1:])
            )
    if not distances:
        raise FEPExecutionError("Pilot trajectory frames did not contain 6 anchor atoms.")

    problems = []
    if _fraction_out_of_bounds(distances, _DISTANCE_MIN_NM, _DISTANCE_MAX_NM) > _STABILITY_OUTLIER_TOLERANCE:
        problems.append(
            f"distance left the {_DISTANCE_MIN_NM}-{_DISTANCE_MAX_NM} nm validated range"
        )
    for label, values in (("A", angles_a), ("B", angles_b)):
        if _fraction_out_of_bounds(values, _ANGLE_MIN_DEG, _ANGLE_MAX_DEG) > _STABILITY_OUTLIER_TOLERANCE:
            problems.append(f"angle {label} left the {_ANGLE_MIN_DEG}-{_ANGLE_MAX_DEG} deg validated range")
    for index, name in enumerate(("A", "B", "C")):
        jumps = jumps_by_dihedral[index]
        if jumps and sum(1 for jump in jumps if jump > _DIHEDRAL_JUMP_DEG) / len(jumps) > _STABILITY_OUTLIER_TOLERANCE:
            problems.append(f"dihedral {name} jumped more than {_DIHEDRAL_JUMP_DEG} deg between frames")

    return {
        "status": "stable" if not problems else "unstable",
        "problems": problems,
        "frame_count": len(distances),
        "distance_nm": {"min": min(distances), "max": max(distances)},
        "angle_a_deg": {"min": min(angles_a), "max": max(angles_a)},
        "angle_b_deg": {"min": min(angles_b), "max": max(angles_b)},
    }


def check_anchor_stability(
    leg_dir: Path, replica: int, lambda_indices: list[int], record: dict, logger,
) -> dict:
    """Extract the 6 anchor atoms' trajectories across the pilot windows and verify stability.

    Uses `gmx traj` to dump raw XYZ coordinates (one single-atom index group
    per anchor) instead of `gmx distance`/`gmx angle`, so the exact same
    `_distance`/`_angle`/`_dihedral` helpers that define the restraints are
    reused to recompute them from the trajectory.
    """
    gmx = str(dependency_status()["gromacs"])
    windows: list[list[tuple[float, ...]]] = []
    for lambda_index in lambda_indices:
        work_dir = leg_dir / f"replica_{replica:02d}" / "pilot" / f"lambda_{lambda_index:02d}"
        (work_dir / "anchors.ndx").write_text(
            _anchor_index_file(record["protein_indices"], record["ligand_indices"]), encoding="utf-8",
        )
        try:
            run_subprocess(
                [gmx, "traj", "-s", "pilot.tpr", "-f", "pilot.xtc", "-n", "anchors.ndx",
                 "-ox", "anchor_traj.xvg", "-nojump"],
                timeout=180, retries=1, cwd=str(work_dir), logger=logger,
            )
        except SubprocessError as exc:
            raise FEPExecutionError(f"Could not extract anchor trajectory in {work_dir}: {exc}") from exc
        windows.append(_parse_traj_xvg((work_dir / "anchor_traj.xvg").read_text()))
    return _evaluate_anchor_stability(windows)


def compute_standard_state_correction(
    geometry: dict, force_constants: dict, temperature_k: float,
) -> dict:
    """Analytical Boresch et al. (2003, JPCB 107:9535, eq. 32) standard-state correction.

    Closed-form Gaussian/harmonic approximation (valid when K >> RT, which
    holds for these restraint force constants) for the free energy of the
    6 Boresch restraints (1 distance + 2 angles + 3 dihedrals) relative to
    the standard-state (1 M) reference volume:

        dG = -RT * ln[ (8*pi^2*V0) / (r0^2 * sin(thetaA0) * sin(thetaB0))
                        * sqrt(Kr*KthetaA*KthetaB*KphiA*KphiB*KphiC)
                        / (2*pi*RT)^3 ]

    All inputs in GROMACS units (kJ/mol, nm, radians after conversion from
    the stored degrees), matching `select_boresch_anchors`/
    `materialize_boresch_restraints`. This is the standalone restraint free
    energy; how its sign enters the final double-decoupling ABFE cycle is
    assembled in the analysis phase, not here.
    """
    r0 = float(geometry["distance_nm"])
    angle_a_deg, angle_b_deg = geometry["angles_deg"]
    theta_a0, theta_b0 = math.radians(float(angle_a_deg)), math.radians(float(angle_b_deg))
    if r0 <= 0 or not (0 < theta_a0 < math.pi) or not (0 < theta_b0 < math.pi):
        raise FEPExecutionError(
            "Boresch geometry is degenerate (zero distance, or an angle at 0/180 deg); "
            "the standard-state correction is undefined."
        )

    k_r = float(force_constants["distance_kj_mol_nm2"])
    k_theta = float(force_constants["angle_kj_mol_rad2"])
    k_phi = float(force_constants["dihedral_kj_mol_rad2"])
    rt = _GAS_CONSTANT_KJ_MOL_K * float(temperature_k)

    product_k = k_r * k_theta * k_theta * k_phi * k_phi * k_phi
    geometric_term = (8 * math.pi ** 2 * _STANDARD_STATE_VOLUME_NM3) / (
        r0 ** 2 * math.sin(theta_a0) * math.sin(theta_b0)
    )
    thermal_term = math.sqrt(product_k) / (2 * math.pi * rt) ** 3
    delta_g = -rt * math.log(geometric_term * thermal_term)

    return {
        "delta_g_kj_mol": delta_g,
        "delta_g_kcal_mol": delta_g / 4.184,
        "temperature_k": float(temperature_k),
        "standard_volume_nm3": _STANDARD_STATE_VOLUME_NM3,
        "reference": (
            "Boresch, Tettinger, Leitgeb, Karplus. J. Phys. Chem. B 2003, 107, 9535 "
            "(eq. 32), harmonic closed-form approximation."
        ),
    }


def run_fep_pilot(run_dir: str | Path) -> dict:
    """Run the pilot lambda subset for both ABFE ligands' complex legs.

    Verifies restraint (anchor) stability from the resulting trajectories
    and computes the standard-state correction per ligand (its geometry is
    ligand-specific, so both A and B need their own value even though
    `pilot_plan.json` originally scoped the mechanical check to ligand A
    only). Production unlocks (`fep_preflight.json["production_locked"]`
    -> false) only once both ligands pass.
    """
    root = Path(run_dir).expanduser().resolve()
    config = _load_json(root / "run_config.json", f"Not a ChemLink FEP run directory: {root}")
    preflight = _load_json(
        root / "fep_preflight.json", f"No FEP preflight report found in {root}."
    )
    if preflight.get("method") != "ABFE":
        raise FEPExecutionError("The pilot run is currently available only for ABFE.")
    plan = _load_json(root / "pilot_plan.json", f"No pilot_plan.json found in {root}.")
    boresch_mapping = _load_json(
        root / "boresch_mapping.json",
        f"Boresch restraints are not mapped yet in {root}; run 'chemlink dynamic fep-pilot' first.",
    )
    if not dependency_status().get("gromacs"):
        raise FEPExecutionError("GROMACS is required to run the FEP pilot.")

    config.setdefault("gpu_ids", get_hardware_profile().gpu_indices)
    config.setdefault("threads", get_system_threads())
    logger = get_step_logger(__name__, str(root / "fep_execution.log"))

    replica = int(plan.get("replica", 1))
    lambda_indices = list(plan["lambda_states"])
    leg = "complex"

    ligand_results: dict[str, Any] = {}
    total = 2 * len(lambda_indices)
    with job_bar("Pilot windows", total, unit="window") as bar:
        for label in ("A", "B"):
            leg_dir = root / f"ligand_{label}" / leg
            for lambda_index in lambda_indices:
                run_pilot_lambda_state(
                    leg_dir, replica, lambda_index, config, "ABFE", logger,
                    system_label=f"ligand_{label}", show_progress=True,
                )
                bar.update(1)

    for label in ("A", "B"):
        leg_dir = root / f"ligand_{label}" / leg
        record = boresch_mapping["ligands"][label]
        stability = check_anchor_stability(leg_dir, replica, lambda_indices, record, logger)
        anchors_source = _load_json(
            root / "common" / f"ligand_{label}_boresch_anchors.json",
            f"Missing ligand {label} anchor definition in {root}/common.",
        )
        ssc = compute_standard_state_correction(
            record["geometry"], anchors_source["force_constants"], float(config["temperature_k"]),
        )
        ligand_results[label] = {"anchor_stability": stability, "standard_state_correction": ssc}

    passed = all(entry["anchor_stability"]["status"] == "stable" for entry in ligand_results.values())
    pilot_report = {
        "status": "passed" if passed else "manual_review_required",
        "replica": replica, "leg": leg, "lambda_states": lambda_indices,
        "ligands": ligand_results,
    }
    (root / "pilot_report.json").write_text(json.dumps(pilot_report, indent=2) + "\n", encoding="utf-8")

    preflight["production_locked"] = not passed
    preflight["pilot"] = pilot_report
    (root / "fep_preflight.json").write_text(json.dumps(preflight, indent=2) + "\n", encoding="utf-8")
    return pilot_report


# ── Fase 3: production across every lambda window ──────────────────────────


def _all_production_states(root: Path, config: dict) -> list[dict]:
    """Deterministic (ligand, leg, replica, lambda) ordering of every production window.

    Mirrors the directory layout `_create_window_layout` already wrote:
    `<root>/ligand_<label>/<leg>/replica_NN/lambda_NN/`. Pure path
    construction — no filesystem access — so it is unit-testable on its own.
    """
    replicas = int(config["replicas"])
    windows = int(config["lambda_windows"])
    states = []
    for label in ("A", "B"):
        for leg in ("complex", "solvent"):
            leg_dir = root / f"ligand_{label}" / leg
            for replica in range(1, replicas + 1):
                equilibration_dir = leg_dir / f"replica_{replica:02d}" / "equilibration"
                for lambda_index in range(windows):
                    directory = leg_dir / f"replica_{replica:02d}" / f"lambda_{lambda_index:02d}"
                    states.append({
                        "key": f"ligand_{label}_{leg}_replica_{replica:02d}_lambda_{lambda_index:02d}",
                        "label": label, "leg": leg, "replica": replica, "lambda_index": lambda_index,
                        "leg_dir": leg_dir, "directory": directory,
                        "equilibration_dir": equilibration_dir,
                    })
    return states


def run_fep_production(run_dir: str | Path, *, resume: bool = True) -> dict:
    """Run production MD across every prepared lambda window, sequentially and resumably.

    Every window in a replica is seeded independently from that replica's
    own `npt.gro`/`npt.cpt` (Fase 1) rather than chained window-to-window,
    so one failed or interrupted window never blocks the rest, and
    re-running this function (the default, `resume=True`) only touches
    windows that are not `"Finished mdrun"` yet. A failed window is
    recorded in `production_progress.json["failed"]` and the loop
    continues instead of aborting the remaining ~288 states.
    """
    root = Path(run_dir).expanduser().resolve()
    config = _load_json(root / "run_config.json", f"Not a ChemLink FEP run directory: {root}")
    preflight = _load_json(
        root / "fep_preflight.json", f"No FEP preflight report found in {root}."
    )
    if preflight.get("method") != "ABFE":
        raise FEPExecutionError("Production execution is currently available only for ABFE runs.")
    if preflight.get("production_locked", True):
        raise FEPExecutionError(
            "Production is locked: run 'chemlink dynamic fep-pilot' and confirm "
            'pilot_report.json["status"] == "passed" first.'
        )
    if not dependency_status().get("gromacs"):
        raise FEPExecutionError("GROMACS is required to run FEP production.")

    config.setdefault("gpu_ids", get_hardware_profile().gpu_indices)
    config.setdefault("threads", get_system_threads())
    logger = get_step_logger(__name__, str(root / "fep_execution.log"))
    gmx = str(dependency_status()["gromacs"])
    opt = get_cached_optimal_mdrun_flags(config)
    nsteps = int(float(config["production_ns"]) * 500000)
    use_gpu = bool(config.get("gpu_ids"))
    mpi_tasks = int(config.get("mpi_tasks", 1))
    mpi_hosts = config.get("mpi_hosts", "")

    states = _all_production_states(root, config)
    completed = 0
    failed: list[str] = []

    def _write_progress(status: str) -> dict:
        progress = {
            "status": status, "total": len(states), "completed": completed,
            "failed": failed, "remaining": len(states) - completed - len(failed),
        }
        (root / "production_progress.json").write_text(
            json.dumps(progress, indent=2) + "\n", encoding="utf-8",
        )
        return progress

    with job_bar("Production windows", len(states), unit="window") as bar:
        for entry in states:
            try:
                window_dir = entry["directory"]
                state_path = window_dir / "state.json"
                state = json.loads(state_path.read_text()) if state_path.is_file() else dict(entry)
                state.pop("leg_dir", None)
                state.pop("directory", None)
                state.pop("equilibration_dir", None)

                def _finish(status: str, error: str | None = None) -> None:
                    state["status"] = status
                    if error is not None:
                        state["error"] = error
                    state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")

                if resume and phase_completed(window_dir, "production"):
                    _finish("completed")
                    completed += 1
                    continue
                if not phase_completed(entry["equilibration_dir"], "npt"):
                    _finish("failed", "replica is not equilibrated (run fep-equilibrate first)")
                    failed.append(entry["key"])
                    continue

                _finish("running")
                try:
                    run_subprocess(
                        [gmx, "grompp", "-f", "production.mdp",
                         "-c", str(entry["equilibration_dir"] / "npt.gro"),
                         "-t", str(entry["equilibration_dir"] / "npt.cpt"),
                         "-p", str(entry["leg_dir"] / "system" / "topol.top"),
                         "-o", "production.tpr", "-maxwarn", "10"],
                        timeout=_GROMPP_TIMEOUT, retries=2, cwd=str(window_dir), logger=logger,
                    )
                    base = [gmx, "mdrun", "-v", "-deffnm", "production"]
                    run_mdrun_with_fallback(
                        cmd_gpu=base + opt["gpu"], cmd_cpu=base + opt["cpu"],
                        work_dir=str(window_dir), phase=f"FEP production {entry['key']}",
                        total_steps=nsteps, logger=logger, use_gpu=use_gpu,
                        timeout=None, mpi_tasks=mpi_tasks, mpi_hosts=mpi_hosts,
                        show_progress=False,
                    )
                except _MDRUN_FAILURES as exc:
                    logger.error("[%s] production failed: %s", entry["key"], exc)
                    _finish("failed", str(exc))
                    failed.append(entry["key"])
                    _write_progress("incomplete")
                    continue

                _finish("completed")
                completed += 1
                _write_progress("incomplete")
            finally:
                bar.update(1)

    status = "completed" if not failed and completed == len(states) else "incomplete"
    return _write_progress(status)
