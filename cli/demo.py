"""``chemlink demo`` — drive the real interface with simulated work.

Nothing here computes anything or calls an external tool: it exercises the
same progress display, logger, cards, tables and error panels a real run
uses, with synthetic timings. It exists so the interface can be checked on
any machine (a laptop without a GPU, GROMACS or AutoDock), and so changes to
the UI can be reviewed without a multi-hour simulation.
"""
from __future__ import annotations

import logging
import os
import random
import tempfile
import time
from typing import Callable

try:
    from ..utils.logger import attach_file, setup_logger
    from ..utils.progress import (
        checklist, make_analysis_progress, make_dynamics_progress, pipeline_bar,
        step_bar, step_bar_iter,
    )
    from ..utils.style import (
        log_line, print_card, record_block, section, stats_block, status_line,
    )
    from ..utils.theme import format_duration, get_console
except ImportError:  # pragma: no cover
    from utils.logger import attach_file, setup_logger  # type: ignore
    from utils.progress import (  # type: ignore
        checklist, make_analysis_progress, make_dynamics_progress, pipeline_bar,
        step_bar, step_bar_iter,
    )
    from utils.style import (  # type: ignore
        log_line, print_card, record_block, section, stats_block, status_line,
    )
    from utils.theme import format_duration, get_console  # type: ignore


class _Clock:
    """Scaled sleeps so ``--speed 3`` plays the whole demo three times faster."""

    def __init__(self, speed: float) -> None:
        self.speed = max(0.05, speed)

    def __call__(self, seconds: float) -> None:
        time.sleep(seconds / self.speed)


def _demo_logger(name: str, log_dir: str) -> logging.Logger:
    logger = setup_logger(f"chemlink.demo.{name}", level=logging.DEBUG)
    attach_file(logger, os.path.join(log_dir, "simulation.log"))
    return logger


# ── Docking ───────────────────────────────────────────────────────────────────

def _docking(sleep: Callable[[float], None], fail: bool) -> None:
    con = get_console()
    rng = random.Random(7)
    receptors = ["1hsg", "6lu7", "3ptb"]
    ligands = ["indinavir", "nirmatrelvir", "ibuprofen", "aspirin", "caffeine", "benzamidine"]

    print_card(con, [
        ("Receptors", f"{len(receptors)}  [cl.muted]{', '.join(receptors)}[/]"),
        ("Ligands", f"{len(ligands)}  [cl.muted]data/test/ligands[/]"),
        ("Engine", "cpu · AutoDock Vina  [cl.muted](no GPU detected)[/]"),
        ("Exhaustiveness", 8),
        ("Output", "data/output/demo_docking"),
    ], title="Docking campaign", subtitle="simulated")
    section(con, "Preparation")

    for name in step_bar_iter(receptors, "Receptor preparation", unit="receptor", colour="blue"):
        sleep(0.55)
        con.print(log_line("INFO", f"Prepared receptor {name} · waters and ligands stripped", phase="receptor"))

    for name in step_bar_iter(ligands, "Ligand preparation", unit="ligand", colour="magenta"):
        sleep(0.3)
        if name == "nirmatrelvir":
            con.print(log_line("WARN", f"{name}: 2 unassigned stereocentres — keeping RDKit's embedding", phase="ligand"))

    jobs = len(receptors) * len(ligands)
    start = time.monotonic()
    with checklist() as cl:
        with cl.step("Detecting active site", "fpocket") as item:
            for i in range(1, len(receptors) + 1):
                sleep(0.45)
                item.update(f"{i}/{len(receptors)} receptors")
            item.finish(f"{len(receptors)} pockets · 22 Å boxes")
        with cl.step("Running CPU search", "AutoDock Vina") as item:
            for i in range(1, jobs + 1):
                sleep(rng.uniform(0.12, 0.3))
                item.update(f"{i:,} / {jobs:,} ligands docked")
                if fail and i == jobs // 2:
                    raise RuntimeError(
                        "Vina exited with code -11 (segmentation fault) for 6lu7 × nirmatrelvir\n"
                        "Command: vina --receptor 6lu7.pdbqt --ligand nirmatrelvir.pdbqt --exhaustiveness 8\n"
                        "free(): invalid pointer"
                    )
            item.finish(f"{jobs} / {jobs} ligands docked · 17 ok · 1 failed")
        cl.complete(f"Campaign complete in {format_duration(time.monotonic() - start)}")

    with step_bar("Docking analysis", 3, unit="report") as bar:
        for _ in range(3):
            sleep(0.35)
            bar.update(1)

    stats_block(con, {
        "Run directory": "data/output/demo_docking",
        "Receptors": {"total": 3, "successful": 3, "failed": 0},
        "Ligands": {"total": 6, "successful": 6, "failed": 0, "warnings": 1},
        "Docking execution": {"jobs": jobs, "successful": jobs - 1, "failed": 1},
        "Docking analysis": {"poses": 162, "exported pdb": 10},
    }, title="Pipeline results")
    record_block(con, ["Rank", "Receptor", "Ligand", "ΔG (kcal/mol)", "Pose"], [
        ["1", "1hsg", "indinavir", "-11.8", "data/output/demo_docking/top/1hsg_indinavir.pdb"],
        ["2", "6lu7", "nirmatrelvir", "-8.9", "data/output/demo_docking/top/6lu7_nirmatrelvir.pdb"],
        ["3", "3ptb", "benzamidine", "-6.4", "data/output/demo_docking/top/3ptb_benzamidine.pdb"],
    ], title="Top hits")


# ── Dynamics ──────────────────────────────────────────────────────────────────

def _md_phase(sleep, name: str, steps: int, ns_day: float, seconds: float, *, fail_at: float = 2.0,
              note: str = "") -> None:
    con = get_console()
    progress = make_dynamics_progress()
    with progress:
        task = progress.add_task(name, total=steps, ns_day=0.0, eta_s=0.0, dt_ps=0.002)
        ticks = 40
        rng = random.Random(name)
        for i in range(1, ticks + 1):
            sleep(seconds / ticks)
            progress.advance(task, steps / ticks)
            progress.update(task, ns_day=ns_day * rng.uniform(0.93, 1.05))
            if note and i == 4:
                con.print(log_line("NOTE", note, phase=name))
            if i / ticks >= fail_at:
                raise RuntimeError(
                    f"[{name}] mdrun exited with code 1\n"
                    "Fatal error:\n"
                    "Step 31200: The total potential energy is nan, which is not finite.\n"
                    "The LJ and electrostatic contributions to the energy are 0 and -183920.\n"
                    "A non-finite potential energy can be caused by overlapping interactions\n"
                    "in bonded interactions or very large or NaN coordinate values."
                )


def _dynamics(sleep, fail: bool, log_dir: str) -> None:
    con = get_console()
    log = _demo_logger("topology", log_dir)
    ions = _demo_logger("ions", log_dir)
    mdrun = _demo_logger("mdrun-runner", log_dir)

    print_card(con, [
        ("Simulation", "Protein + Ligand"),
        ("Duration", "0.1 ns"),
        ("Replicas", 1),
        ("Ligand charge", "[cl.num]+1[/]  [cl.muted](high confidence)[/]"),
        ("Output", "data/output/dynamics/demo_pligand"),
    ], title="Simulation parameters", subtitle="simulated")
    print_card(con, [
        ("GPU", "[cl.warn.soft]none detected[/] · running on CPU"),
        ("CPU", f"{os.cpu_count() or 8} cores · {os.cpu_count() or 8} threads per simulation"),
    ], title="Compute resources")
    section(con, "Molecular dynamics", "Protein + Ligand · 0.1 ns · 1 replica")
    con.print()

    with pipeline_bar(10, desc="MD pipeline") as pb:
        pb.set_stage("topology")
        with step_bar("Topology", 3, unit="stage") as bar:
            for msg in ("pdb2gmx · AMBER99SB-ILDN / TIP3P", "Protonation states assigned (pH 7.4)", "Disulfide bridges: 0"):
                sleep(0.4)
                log.info(msg)
                bar.update(1)
        pb.update(1)

        pb.set_stage("ligand topology")
        with step_bar("Ligand topology", 2, unit="stage") as bar:
            sleep(0.6)
            log.info("ACPYPE · GAFF2 · AM1-BCC charges (net +1)")
            bar.update(1)
            sleep(0.5)
            bar.update(1)
        pb.update(1)

        pb.set_stage("solvation")
        with step_bar("Solvation", 2, unit="stage") as bar:
            sleep(0.5); bar.update(1)
            sleep(0.4); bar.update(1)
        log.info("Dodecahedron box · 1.2 nm padding · 9,842 waters")
        pb.update(1)

        pb.set_stage("ions")
        with step_bar("Ions", 2, unit="stage") as bar:
            sleep(0.4); bar.update(1)
            ions.warning("System net charge is +1.000 — neutralising with 1 CL and adding 0.15 M NaCl")
            sleep(0.3); bar.update(1)
        pb.update(1)

        pb.set_stage("energy minimisation")
        with step_bar("Energy minimisation", 2, unit="stage") as bar:
            sleep(0.4); bar.update(1)
            sleep(0.9); bar.update(1)
        log.info("Converged in 1,284 steps · Fmax 812.4 < 1000 kJ/mol/nm")
        pb.update(1)

        pb.set_stage("nvt")
        _md_phase(sleep, "NVT equilibration", 50_000, 41.0, 2.4,
                  note="NOTE: GPU-based PME is not supported with steepest descent; using CPU")
        pb.update(1)
        pb.set_stage("npt")
        _md_phase(sleep, "NPT equilibration", 50_000, 38.5, 2.4)
        pb.update(1)
        pb.set_stage("production")
        try:
            _md_phase(sleep, "Production", 50_000, 37.2, 3.2, fail_at=0.62 if fail else 2.0)
        except RuntimeError as exc:
            mdrun.error("%s", exc)
            raise
        pb.update(1)

        pb.set_stage("post-processing")
        with step_bar("Post-processing", 3, unit="stage") as bar:
            for _ in range(3):
                sleep(0.35); bar.update(1)
        pb.update(1)

        pb.set_stage("analysis")
        frames = make_analysis_progress()
        with frames:
            task = frames.add_task("Binding energy · MM-GBSA", total=200)
            for _ in range(40):
                sleep(0.05)
                frames.advance(task, 5)
        log.info("ΔG bind (MM-GBSA) = -42.7 ± 3.1 kcal/mol · -TΔS (IE) = 12.4 kcal/mol")
        pb.update(1)

    con.print()
    con.print(status_line("ok", "Simulation finished"))
    con.print(status_line("info", "Summary   [cl.path]data/output/dynamics/demo_pligand/simulation_summary.md[/]"))


# ── Logs ──────────────────────────────────────────────────────────────────────

def _logs(sleep, log_dir: str) -> None:
    con = get_console()
    section(con, "Log levels", "console format · the file keeps every line")
    logger = _demo_logger("energy_min", log_dir)
    logger.debug("Hidden unless you pass -v (debug level)")
    logger.info("grompp · 1 warning suppressed (-maxwarn 10)")
    sleep(0.2)
    logger.warning("GPU energy minimisation failed — retrying on CPU only")
    sleep(0.2)
    logger.error(
        "Energy Minimization failed:\n" + "\n".join(
            f"  step {n * 100:>5}: Fmax {1e6 / (n + 1):,.1f} kJ/mol/nm on atom {300 + n}" for n in range(25)
        )
    )
    logger.info("Full diagnostics only in the file", extra={"file_only": True})


# ── Entry ─────────────────────────────────────────────────────────────────────

def run_demo(scenario: str = "all", *, speed: float = 1.0, fail: bool = False) -> int:
    con = get_console()
    sleep = _Clock(speed)
    log_dir = tempfile.mkdtemp(prefix="chemlink-demo-")
    try:
        if scenario in ("all", "docking"):
            section(con, "Demo · docking", "simulated — no tools are executed")
            _docking(sleep, fail and scenario == "docking")
        if scenario in ("all", "dynamics"):
            section(con, "Demo · molecular dynamics", "simulated — no tools are executed")
            _dynamics(sleep, fail, log_dir)
        if scenario in ("all", "logs"):
            _logs(sleep, log_dir)
    finally:
        log_file = os.path.join(log_dir, "simulation.log")
        if os.path.isfile(log_file):
            con.print(status_line("info", f"Demo log file  [cl.path]{log_file}[/]"))
    return 0
