"""ChemLink CLI — unified molecular simulation toolkit entry point."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
from argparse import ArgumentParser, ArgumentTypeError, HelpFormatter, Namespace
from datetime import datetime
from typing import Optional, Tuple
from uuid import uuid4
try:
    from ..storage.run_directory import (
        create_named_run_directory,
        indexed_run_name,
        validate_run_name,
    )
except ImportError:
    from storage.run_directory import (  # type: ignore
        create_named_run_directory,
        indexed_run_name,
        validate_run_name,
    )

from rich import box as rich_box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console     = Console()
err_console = Console(stderr=True)

# ── ANSI helpers (kept for argparse formatter only) ────────────────────────────
_TTY = sys.stdout.isatty() and not os.environ.get("NO_COLOR")

def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _TTY else text

def bold(t: str) -> str:        return _c("1",    t)
def dim(t: str) -> str:         return _c("2",    t)
def cyan(t: str) -> str:        return _c("36",   t)
def bold_cyan(t: str) -> str:   return _c("1;36", t)
def bold_red(t: str) -> str:    return _c("1;31", t)
def yellow(t: str) -> str:      return _c("33",   t)


# ── Logo ───────────────────────────────────────────────────────────────────────
LOGO = """\
    
    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣠⣴⣾⠷⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
    ⠀⠀⠀⠀⠀⠀⣀⣤⣶⡿⠿⠛⠉⣀⣤⣶⣤⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
    ⠀⠀⣀⣤⣶⡿⠟⠋⢁⣠⣤⣶⣿⠿⠛⠉⠛⠿⢿⣶⣦⣄⡀⠀⠀⠀⠀⠀⠀⠀
    ⠲⡿⠟⠋⢁⣠⣴⣾⡿⠟⠋⢁⣠⣴⣾⣿⣷⣦⣄⡈⠙⠻⢿⣷⣦⣄⡀⠀⠀⠀
    ⠀⠀⢰⣾⡿⠟⠋⢁⣠⣴⣾⣿⣿⠿⠛⠉⠙⠻⢿⣿⣿⣶⣤⣈⠙⠻⠋⠀⢠⣤
    ⠀⠀⢸⣿⡇⠀⣿⣿⡿⠟⢻⣿⣷⠀⠀⠀⠀⠀⠀⠈⠙⠻⢿⣿⣿⡆⠀⠀⢸⣿
    ⠀⠀⢸⣿⡇⠀⣿⣿⡇⠀⢸⣿⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⣿⡇⠀⠀⢸⣿
    ⠀⠀⢸⣿⡇⠀⣿⣿⡇⠀⢸⣿⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⣿⡇⠀⠀⢸⣿
    ⠀⠀⢸⣿⡇⠀⣿⣿⡇⠀⢸⣿⣷⣤⣄⡀⠀⠀⠀⠀⠀⠀⢸⣿⣿⡇⠀⠀⢸⣿
    ⠀⠀⢸⣿⡇⠀⣿⣿⣷⣤⣀⠈⠙⠻⠿⣿⣷⣶⣤⣀⣀⣤⣾⣿⣿⠇⠀⠀⢸⣿
    ⠀⠀⠸⢿⣷⣦⣄⡉⠛⠿⣿⣿⣷⣦⣄⡀⣉⣽⣿⣿⣿⡿⠟⢋⣡⣄⠀⠀⠸⠿
    ⢾⣶⣤⣀⡈⠙⠻⢿⣷⣦⣄⡉⠛⠿⣿⣿⣿⠿⠛⢉⣠⣴⣾⡿⠟⠋⠁⠀⠀⠀
    ⠀⠈⠙⠻⠿⣷⣦⣄⡈⠙⠻⢿⣷⣦⣄⣉⣠⣴⣾⡿⠟⠋⠁⠀⠀⠀⠀⠀⠀⠀
    ⠀⠀⠀⠀⠀⠀⠉⠛⠿⣷⣦⣄⡈⠙⠻⠿⠟⠋⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠛⠿⣷⡦⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
\
"""

VERSION        = "1.0.0"
DOCTOR_VERSION = "1.0"

# ── Default paths ──────────────────────────────────────────────────────────────
_REC_IN  = "data/input/receptors"
_LIG_IN  = "data/input/ligands"
_OUT     = "data/output"
_REC_OUT = f"{_OUT}/prepared_receptors_pdbqt"
_LIG_OUT = f"{_OUT}/prepared_ligands_pdbqt"
PIPELINE_STEPS = ("receptor", "ligand", "active-site", "execution", "analysis")


# ── Output helpers ─────────────────────────────────────────────────────────────

def print_ok(msg: str)   -> None: console.print(f"  [bold green]✓[/]  {msg}")
def print_err(msg: str)  -> None: console.print(f"  [bold red]✗[/]  {msg}")
def print_warn(msg: str) -> None: console.print(f"  [bold yellow]![/]  {msg}")
def print_info(msg: str) -> None: console.print(f"  [cyan]·[/]  {msg}")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise ArgumentTypeError("must be an integer greater than zero")
    return parsed


def print_stats(stats: dict, title: str = "Results") -> None:
    t = Table(
        title=f"[bold cyan]{title}[/]",
        box=rich_box.ROUNDED,
        border_style="cyan",
        header_style="bold cyan",
        show_header=True,
        min_width=40,
    )
    t.add_column("Metric", style="dim")
    t.add_column("Value",  style="bold")
    for k, v in stats.items():
        t.add_row(str(k), str(v))
    console.print()
    console.print(t)


# ── Custom argparse formatter ──────────────────────────────────────────────────

class _Fmt(HelpFormatter):
    def __init__(self, prog: str):
        super().__init__(prog, max_help_position=32, width=88)

    def _fill_text(self, text: str, width: int, indent: str) -> str:
        return "\n".join(indent + line for line in text.splitlines())

    def start_section(self, heading: Optional[str]) -> None:
        super().start_section(bold_cyan(heading.upper()) if heading else heading)

    def _format_action_invocation(self, action) -> str:
        return cyan(super()._format_action_invocation(action))

    def _format_usage(self, usage, actions, groups, prefix) -> str:
        return super()._format_usage(
            usage, actions, groups,
            bold("Usage: ") if prefix is None else prefix,
        )


# ── Custom parser with coloured errors ────────────────────────────────────────

_TOP_CMDS = ["docking", "dynamic", "doctor", "completion", "hpc"]


def _suggest(word: str) -> list:
    if not word or word.startswith("-"):
        return []
    return [c for c in _TOP_CMDS if c.startswith(word) or word in c][:3]


class _Parser(ArgumentParser):
    def error(self, message: str) -> None:
        err_console.print(f"\n[bold red]Error:[/] {message}")
        if len(sys.argv) > 1:
            matches = _suggest(sys.argv[1])
            if matches:
                suggestions = ", ".join(f"[cyan]{m}[/]" for m in matches)
                err_console.print(f"  Did you mean: {suggestions}?")
        err_console.print(f"\n  Run [cyan]chemlink --help[/] to see all commands.\n")
        sys.exit(2)


# ── Misc helpers ───────────────────────────────────────────────────────────────

def _manual_params(args: Namespace) -> Tuple[Optional[tuple], Optional[tuple]]:
    if args.manual_npts is not None and args.manual_center is None:
        err_console.print(
            "[bold red]Error:[/] --manual-npts requires --manual-center."
        )
        sys.exit(1)
    center = tuple(args.manual_center) if args.manual_center else None
    npts   = tuple(args.manual_npts)   if args.manual_npts   else None
    return center, npts


def _run_dir(base: str, run_name: Optional[str] = None) -> str:
    if run_name is not None:
        return create_named_run_directory(base, run_name)
    os.makedirs(base, exist_ok=True)
    name = f"run_{datetime.now():%Y%m%dT%H%M%S}_{uuid4().hex[:8]}"
    path = os.path.join(base, name)
    os.makedirs(path)
    return path


def _docking_paths(args: Namespace) -> Tuple[str, str, str]:
    """Resolve and validate docking input directories once at CLI entry.

    Explicit ``-ir``/``-il``/``-o`` options take priority over the legacy
    positional paths.  Absolute paths are passed through every pipeline stage
    so a run always keeps tracking the originally selected input directories.
    """
    receptor = getattr(args, "receptor_input_option", None) or args.receptor_input_dir
    ligand = getattr(args, "ligand_input_option", None) or args.ligand_input_dir
    output = getattr(args, "output_option", None) or args.output_dir

    def resolve_input(path: str) -> str:
        expanded = os.path.expanduser(path)
        if os.path.isabs(expanded) or os.path.isdir(expanded):
            return os.path.abspath(expanded)
        # Short names such as ``receptors`` and ``ligands`` are searched
        # relative to ChemLink's conventional input root.
        return os.path.abspath(os.path.join("data", "input", expanded))

    receptor = resolve_input(receptor)
    ligand = resolve_input(ligand)
    output = os.path.abspath(os.path.expanduser(output))

    missing = []
    if not os.path.isdir(receptor):
        missing.append(f"receptor directory: {receptor}")
    if not os.path.isdir(ligand):
        missing.append(f"ligand directory: {ligand}")
    if missing:
        raise FileNotFoundError("Input directory not found — " + "; ".join(missing))
    return receptor, ligand, output


def _write_docking_input_config(
    output_dir: str,
    receptor: str,
    ligand: str,
    run_name: Optional[str] = None,
) -> None:
    """Persist the exact input directories used by a docking run."""
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, "docking_inputs.json")
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "receptor_input_dir": receptor,
                "ligand_input_dir": ligand,
                "run_name": run_name or os.path.basename(os.path.abspath(output_dir)),
                "run_directory": os.path.abspath(output_dir),
                "directory_mode": "custom" if run_name else "automatic",
                "recorded_at": datetime.now().isoformat(timespec="seconds"),
            },
            handle,
            indent=2,
        )


# ── Docking command handlers ───────────────────────────────────────────────────

def _run_receptor_preparation(args: Namespace) -> int:
    try:
        from ..pipelines.docking.steps import ReceptorPreparation
    except ImportError:
        from chemlink.pipelines.docking.steps import ReceptorPreparation  # type: ignore
    step = ReceptorPreparation(
        input_path=args.input_dir,
        output_path=args.output_dir,
    )
    print_stats(step.prepare(n_workers=args.workers), "Receptor Preparation")
    return 0


def _run_ligand_preparation(args: Namespace) -> int:
    try:
        from ..pipelines.docking.steps import LigandPreparation
    except ImportError:
        from chemlink.pipelines.docking.steps import LigandPreparation  # type: ignore
    step = LigandPreparation(input_path=args.input_dir, output_path=args.output_dir)
    print_stats(step.prepare(n_workers=args.workers), "Ligand Preparation")
    return 0


def _run_active_site(args: Namespace) -> int:
    try:
        from ..pipelines.docking.steps import ActiveSiteDetection, load_box_config
    except ImportError:
        from chemlink.pipelines.docking.steps import ActiveSiteDetection, load_box_config  # type: ignore
    center, npts = _manual_params(args)
    box_config = getattr(args, "box_config", None)
    manual_boxes = load_box_config(box_config) if box_config else None
    step = ActiveSiteDetection(
        receptor_path=args.receptor_dir,
        ligand_path=args.ligand_dir,
        output_path=args.output_dir,
        fpocket_path=args.fpocket_path,
        manual_center=center,
        manual_npts=npts,
        manual_boxes=manual_boxes,
        flex_radius=getattr(args, "flex_radius", 0.0),
    )
    print_stats(step.prepare(n_workers=args.workers), "Active Site Detection")
    return 0


def _run_docking_execution(args: Namespace) -> int:
    workers = getattr(args, "workers", getattr(args, "docking_workers", 1))
    engine = getattr(args, "engine", "gpu")

    if engine == "cpu":
        try:
            from ..pipelines.docking.steps import VinaExecution
        except ImportError:
            from chemlink.pipelines.docking.steps import VinaExecution  # type: ignore
        step = VinaExecution(
            receptor_dir=args.prepared_receptors_dir,
            ligand_dir=args.prepared_ligands_dir,
            output_path=args.output_dir,
            box_config_path=getattr(args, "box_config", None),
            vina_executable=getattr(args, "vina_executable", None),
            exhaustiveness=getattr(args, "exhaustiveness", 8),
            num_modes=getattr(args, "num_modes", 9),
            cpu_per_job=getattr(args, "vina_cpu", None),
            seed=getattr(args, "seed", None),
        )
        print_stats(step.run(n_workers=workers), "Docking Execution (Vina / CPU)")
        return 0

    try:
        from ..pipelines.docking.steps import DockingExecution
    except ImportError:
        from chemlink.pipelines.docking.steps import DockingExecution  # type: ignore
    step = DockingExecution(
        protein_maps_dir=args.prepared_receptors_dir,
        ligand_dir=args.prepared_ligands_dir,
        output_path=args.output_dir,
        autogrid_executable=args.autogrid_executable,
        autodock_gpu_executable=args.autodock_gpu_executable,
    )
    print_stats(step.run(n_workers=workers), "Docking Execution")
    return 0


def _run_docking_analysis(args: Namespace) -> int:
    try:
        from ..pipelines.docking.steps import DockingAnalysis
    except ImportError:
        from chemlink.pipelines.docking.steps import DockingAnalysis  # type: ignore
    step = DockingAnalysis(
        output_path=args.output_dir,
        pdb_export_limit=args.pdb_export_limit,
        max_workers=args.max_workers,
    )
    results = step.run()
    print_stats(
        {"Poses analyzed": results["parsed_poses"], "Ligands analyzed": results["analyzed_ligands"]},
        "Docking Analysis",
    )
    outputs = results.get("outputs", {})
    if outputs:
        console.print(f"\n  [bold]Output files:[/]")
        for name, path in outputs.items():
            print_ok(f"[dim]{name}[/]: {path}")
    else:
        print_warn(f"No DLG files found under {args.output_dir}/docking_results/**/dlg/*.dlg")
    return 0


def _run_docking_flow(args: Namespace) -> int:
    try:
        from ..pipelines.docking import DockingPipeline
    except ImportError:
        from chemlink.pipelines.docking import DockingPipeline  # type: ignore
    center, npts = _manual_params(args)
    receptor_input, ligand_input, output_dir = _docking_paths(args)
    run_name = getattr(args, "run_name", None)
    run_output = _run_dir(output_dir, run_name)
    _write_docking_input_config(run_output, receptor_input, ligand_input, run_name)
    result = DockingPipeline(
        receptor_input_path=receptor_input,
        ligand_input_path=ligand_input,
        output_path=run_output,
        fpocket_path=args.fpocket_path,
        manual_center=center,
        manual_npts=npts,
        box_config_path=getattr(args, "box_config", None),
        flex_radius=getattr(args, "flex_radius", 0.0),
    ).run_step_range(
        from_step=args.from_step,
        to_step=args.to_step,
        receptor_workers=args.receptor_workers,
        ligand_workers=args.ligand_workers,
        active_site_workers=args.active_site_workers,
        docking_workers=args.docking_workers,
        autogrid_executable=args.autogrid_executable,
        autodock_gpu_executable=args.autodock_gpu_executable,
        pdb_export_limit=args.pdb_export_limit,
        max_workers=args.max_workers,
        engine=getattr(args, "engine", "gpu"),
        exhaustiveness=getattr(args, "exhaustiveness", 8),
        num_modes=getattr(args, "num_modes", 9),
        vina_cpu=getattr(args, "vina_cpu", 1),
        vina_executable=getattr(args, "vina_executable", None),
        seed=getattr(args, "seed", None),
    )
    stats: dict = {"Run directory": run_output, "Steps": ", ".join(result.executed_steps)}
    if "receptor"    in result.executed_steps: stats["Receptors"]         = result.receptor_preparation
    if "ligand"      in result.executed_steps: stats["Ligands"]           = result.ligand_preparation
    if "active-site" in result.executed_steps: stats["Active sites"]      = result.active_site_detection
    if "execution"   in result.executed_steps and result.docking_execution:
        stats["Docking execution"] = result.docking_execution
    if "analysis"    in result.executed_steps and result.docking_analysis:
        stats["Docking analysis"]  = result.docking_analysis
    print_stats(stats, "Pipeline Results")
    return 0


def _run_docking_pipeline(args: Namespace, *, full: bool) -> int:
    try:
        from ..pipelines.docking import DockingPipeline
    except ImportError:
        from chemlink.pipelines.docking import DockingPipeline  # type: ignore
    center, npts = _manual_params(args)
    receptor_input, ligand_input, output_dir = _docking_paths(args)
    run_name = getattr(args, "run_name", None)
    out = _run_dir(output_dir, run_name) if (full or run_name) else output_dir
    _write_docking_input_config(out, receptor_input, ligand_input, run_name)
    pipeline = DockingPipeline(
        receptor_input_path=receptor_input,
        ligand_input_path=ligand_input,
        output_path=out,
        fpocket_path=args.fpocket_path,
        manual_center=center,
        manual_npts=npts,
        box_config_path=getattr(args, "box_config", None),
        flex_radius=getattr(args, "flex_radius", 0.0),
    )
    result = (
        pipeline.run_full_pipeline(
            receptor_workers=args.receptor_workers,
            ligand_workers=args.ligand_workers,
            active_site_workers=args.active_site_workers,
            docking_workers=args.docking_workers,
            autogrid_executable=args.autogrid_executable,
            autodock_gpu_executable=args.autodock_gpu_executable,
            pdb_export_limit=args.pdb_export_limit,
            engine=getattr(args, "engine", "gpu"),
            exhaustiveness=getattr(args, "exhaustiveness", 8),
            num_modes=getattr(args, "num_modes", 9),
            vina_cpu=getattr(args, "vina_cpu", None),
            vina_executable=getattr(args, "vina_executable", None),
            seed=getattr(args, "seed", None),
        )
        if full
        else pipeline.run_preparation_pipeline(
            receptor_workers=args.receptor_workers,
            ligand_workers=args.ligand_workers,
            active_site_workers=args.active_site_workers,
        )
    )
    stats: dict = {}
    if full or run_name:
        stats["Run directory"] = out
    stats.update({
        "Receptors":    result.receptor_preparation,
        "Ligands":      result.ligand_preparation,
        "Active sites": result.active_site_detection,
    })
    if result.docking_execution: stats["Docking execution"] = result.docking_execution
    if result.docking_analysis:  stats["Docking analysis"]  = result.docking_analysis
    print_stats(stats, "Pipeline Results")
    return 0


# ── Dynamics command handler ───────────────────────────────────────────────────

def _resolve_dynamic_input_path(filename: str, role: str) -> str:
    """Resolve a dynamics input using the shared and role-specific roots."""
    expanded = os.path.expanduser(filename)
    if os.path.isabs(expanded) or os.path.exists(expanded):
        return os.path.abspath(expanded)
    candidates = [
        os.path.join("data", "input", "dynamics", expanded),
        os.path.join("data", "input", "dynamics", role, expanded),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    return os.path.abspath(candidates[0])


_DYN_SPEC = {
    #  name        sim_id  label                              min_files
    "oprotein":  ("1", "Protein",                               1),
    "pligand":   ("2", "Protein + Ligand",                      2),
    "ppeptide":  ("3", "Protein + Peptide",                     2),
    "pacid":     ("4", "Protein + Nucleic Acid",                2),
    "pprotein":  ("5", "Protein + Protein",                     2),
    "ppligand":  ("6", "Protein + Protein + Ligand/Cofactor",   3),
}

_DYN_FILE_LABELS = {
    "oprotein":  ["protein"],
    "pligand":   ["protein", "ligand"],
    "ppeptide":  ["protein", "peptide"],
    "pacid":     ["protein", "nucleic_acid"],
    "pprotein":  ["protein1", "protein2"],
    "ppligand":  ["protein1", "protein2", "ligand"],
}

_DYN_INPUT_COUNTS = {
    "oprotein":  (1, 0),
    "pligand":   (1, 1),
    "ppeptide":  (1, 1),
    "pacid":     (1, 1),
    "pprotein":  (1, 1),
    "ppligand":  (2, 1),
}


def _resolve_dynamics_input_sets(
    dyn_type: str,
    receptor_inputs: Optional[list[str]] = None,
    ligand_inputs: Optional[list[str]] = None,
    legacy_inputs: Optional[list[str]] = None,
    *,
    allow_multiple_ligands: bool = False,
) -> Tuple[list[list[str]], str]:
    """Validate and order dynamics inputs for local and HPC execution."""
    receptors = list(receptor_inputs or [])
    ligands = list(ligand_inputs or [])
    legacy = list(legacy_inputs or [])
    uses_standard = bool(receptors or ligands)
    if uses_standard and legacy:
        raise ValueError(
            "Do not mix -ir/-il (or --protein/--ligands) with legacy -i or positional inputs."
        )

    expected_receptors, expected_ligands = _DYN_INPUT_COUNTS[dyn_type]
    if not uses_standard:
        expected_total = expected_receptors + expected_ligands
        if len(legacy) != expected_total:
            labels = ", ".join(_DYN_FILE_LABELS[dyn_type])
            raise ValueError(
                f"{dyn_type} requires exactly {expected_total} legacy input file(s) "
                f"in this order: {labels}. Received {len(legacy)}."
            )
        resolved = [
            _resolve_dynamic_input_path(path, "receptors" if index < expected_receptors else "ligands")
            for index, path in enumerate(legacy)
        ]
        return [resolved], "legacy"

    ligand_count_valid = (
        dyn_type == "pligand" and allow_multiple_ligands and len(ligands) >= 1
    ) or len(ligands) == expected_ligands
    if len(receptors) != expected_receptors or not ligand_count_valid:
        ligand_requirement = ">=1" if dyn_type == "pligand" and allow_multiple_ligands else str(expected_ligands)
        raise ValueError(
            f"{dyn_type} requires {expected_receptors} receptor input(s) and "
            f"{ligand_requirement} ligand/partner input(s). Received "
            f"{len(receptors)} receptor(s) and {len(ligands)} ligand/partner(s)."
        )

    resolved_receptors = [
        _resolve_dynamic_input_path(path, "receptors") for path in receptors
    ]
    resolved_ligands = [
        _resolve_dynamic_input_path(path, "ligands") for path in ligands
    ]
    if dyn_type == "pligand" and allow_multiple_ligands:
        return [resolved_receptors + [ligand] for ligand in resolved_ligands], "standardized"
    return [resolved_receptors + resolved_ligands], "standardized"


def _validate_dynamics_ligand(ligand_path: str, work_dir: str) -> None:
    try:
        from ..pipelines.ligand_validation import require_valid_ligand
    except ImportError:
        try:
            from chemlink.pipelines.ligand_validation import require_valid_ligand  # type: ignore
        except ImportError:
            from pipelines.ligand_validation import require_valid_ligand  # type: ignore
    require_valid_ligand(
        ligand_path,
        report_path=os.path.join(work_dir, "ligand_validation.json"),
    )


def _resolve_ligand_charge_for_run(
    ligand_path: str,
    work_dir: str,
    *,
    explicit_charge: Optional[int],
    mode: str,
    ph: float,
    interactive: bool,
):
    try:
        from ..pipelines.dynamics.ligand_charge import (
            LigandChargeError,
            detect_or_prepare_ligand_charge,
            explicit_charge_result,
            write_charge_provenance,
        )
    except ImportError:
        try:
            from chemlink.pipelines.dynamics.ligand_charge import (  # type: ignore
                LigandChargeError,
                detect_or_prepare_ligand_charge,
                explicit_charge_result,
                write_charge_provenance,
            )
        except ImportError:
            from pipelines.dynamics.ligand_charge import (  # type: ignore
                LigandChargeError,
                detect_or_prepare_ligand_charge,
                explicit_charge_result,
                write_charge_provenance,
            )

    if explicit_charge is not None:
        result = explicit_charge_result(ligand_path, explicit_charge)
    else:
        result = detect_or_prepare_ligand_charge(ligand_path, work_dir, ph=ph)

        console.print(
            f"[cyan]Ligand charge detection:[/] "
            f"{result.charge if result.charge is not None else 'unresolved'}  "
            f"[dim]({result.method}; confidence {result.confidence})[/]"
        )

        if mode == "strict":
            if result.charge is None or result.confidence != "high":
                raise LigandChargeError(
                    "Strict charge mode requires an explicit --charge or a high-confidence "
                    "formal charge from SDF/MOL."
                )
        elif mode == "auto":
            if result.charge is None or result.confidence not in {"high", "medium"}:
                raise LigandChargeError(
                    "Ligand charge could not be resolved automatically; provide --charge INTEGER."
                )
        elif mode == "prompt":
            if not interactive:
                raise LigandChargeError(
                    "Charge confirmation requires an interactive terminal; use --charge INTEGER "
                    "or --charge-mode auto."
                )
            default = "" if result.charge is None else f" [Enter={result.charge}]"
            while True:
                try:
                    entered = input(f"Ligand net charge{default}: ").strip()
                except EOFError as exc:
                    raise LigandChargeError("Ligand charge confirmation was cancelled.") from exc
                if not entered and result.charge is not None:
                    result.user_confirmed = True
                    break
                if re.fullmatch(r"[+-]?\d+", entered):
                    result.charge = int(entered)
                    result.method = f"user-confirmed override; detected by {result.method}"
                    result.confidence = "explicit"
                    result.user_confirmed = True
                    break
                console.print("[red]Enter an integer charge, for example 0, +1, or -1.[/]")

    if result.charge is None:
        raise LigandChargeError("Ligand net charge is unresolved; provide --charge INTEGER.")
    write_charge_provenance(result, work_dir)
    return result


def _run_dynamic(args: Namespace) -> int:
    try:
        from ..pipelines.dynamics.dynamics_pipeline import DynamicsPipeline
        from ..pipelines.dynamics.utils import get_system_threads, setup_work_directory
    except ImportError:
        try:
            from chemlink.pipelines.dynamics.dynamics_pipeline import DynamicsPipeline  # type: ignore
            from chemlink.pipelines.dynamics.utils import get_system_threads, setup_work_directory  # type: ignore
        except ImportError:
            from pipelines.dynamics.dynamics_pipeline import DynamicsPipeline  # type: ignore
            from pipelines.dynamics.utils import get_system_threads, setup_work_directory  # type: ignore

    dyn_type = args.dyn_type
    sim_id, label, min_files = _DYN_SPEC[dyn_type]

    legacy_inputs = list(args.input_files or []) + list(getattr(args, "files", None) or [])
    try:
        input_sets, input_interface = _resolve_dynamics_input_sets(
            dyn_type,
            receptor_inputs=getattr(args, "input_receptors", None),
            ligand_inputs=getattr(args, "input_ligands", None),
            legacy_inputs=legacy_inputs,
        )
    except ValueError as exc:
        err_console.print(f"\n[bold red]Input error:[/] {exc}\n")
        return 1
    all_files = input_sets[0]

    missing_files = [path for path in all_files if not os.path.isfile(path)]
    if missing_files:
        err_console.print("\n[bold red]Input file not found:[/]")
        for path in missing_files:
            err_console.print(f"  {os.path.abspath(path)}")
        err_console.print(
            "Use positional paths for files in the current directory, or place names "
            "passed with -i inside data/input/dynamics."
        )
        return 1

    run_name = getattr(args, "run_name", None)
    work_dir = setup_work_directory("data/output/dynamics", dyn_type, run_name=run_name)
    threads  = get_system_threads()
    config: dict = {
        "sim_type":       sim_id,
        "sim_type_label": label,
        "ns_time":        args.time,
        "threads":        threads,
        "work_dir":       work_dir,
        "run_name":       run_name or os.path.basename(work_dir),
        "directory_mode": "custom" if run_name else "automatic",
        "receptor_inputs": all_files[:_DYN_INPUT_COUNTS[dyn_type][0]],
        "ligand_inputs": all_files[_DYN_INPUT_COUNTS[dyn_type][0]:],
        "input_interface": input_interface,
        "binding_energy": getattr(args, "binding_energy", True),
        "mmpbsa_method":  getattr(args, "mmpbsa_method", "both"),
        "mmpbsa_frames":  getattr(args, "mmpbsa_frames", 1000),
        "mmpbsa_discard": getattr(args, "mmpbsa_discard", 20.0),
        "mmpbsa_cores":   getattr(args, "mmpbsa_cores", threads),
        "mmpbsa_salt":    getattr(args, "mmpbsa_salt", 0.150),
        "force_field":     getattr(args, "force_field", "default"),
        "force_field_dir": getattr(args, "force_field_dir", None),
        "water_model":     getattr(args, "water_model", "tip3p"),
    }

    if dyn_type == "oprotein":
        config["pdb_input"] = all_files[0]
    elif dyn_type == "pligand":
        try:
            _validate_dynamics_ligand(all_files[1], work_dir)
            charge = _resolve_ligand_charge_for_run(
                all_files[1], work_dir,
                explicit_charge=args.charge,
                mode=args.charge_mode,
                ph=args.ligand_ph,
                interactive=sys.stdin.isatty(),
            )
        except Exception as exc:
            err_console.print(f"\n[bold red]Ligand charge error:[/] {exc}\n")
            return 1
        config.update(
            pdb_input=all_files[0], ligand_pdb=all_files[1],
            ligand_charge=charge.charge,
            ligand_prepared_file=charge.prepared_file,
            ligand_charge_method=charge.method,
            ligand_charge_confidence=charge.confidence,
            ligand_charge_source=charge.source_file,
            ligand_ph=charge.ph,
        )
    elif dyn_type in ("ppeptide", "pacid", "pprotein"):
        config.update(pdb_protein=all_files[0], pdb_partner=all_files[1],
                      pdb_input=os.path.join(work_dir, "complex.pdb"))
    elif dyn_type == "ppligand":
        try:
            _validate_dynamics_ligand(all_files[2], work_dir)
            charge = _resolve_ligand_charge_for_run(
                all_files[2], work_dir,
                explicit_charge=args.charge,
                mode=args.charge_mode,
                ph=args.ligand_ph,
                interactive=sys.stdin.isatty(),
            )
        except Exception as exc:
            err_console.print(f"\n[bold red]Ligand charge error:[/] {exc}\n")
            return 1
        config.update(
            pdb_protein=all_files[0], pdb_partner=all_files[1],
            pdb_input=os.path.join(work_dir, "complex.pdb"),
            ligand_pdb=all_files[2], ligand_charge=charge.charge,
            ligand_prepared_file=charge.prepared_file,
            ligand_charge_method=charge.method,
            ligand_charge_confidence=charge.confidence,
            ligand_charge_source=charge.source_file,
            ligand_ph=charge.ph,
        )

    t = Table(
        title=f"[bold cyan]Simulation Parameters[/]",
        box=rich_box.ROUNDED,
        border_style="cyan",
        show_header=False,
        min_width=40,
    )
    t.add_column(style="dim",  no_wrap=True)
    t.add_column(style="bold")
    t.add_row("Simulation", label)
    t.add_row("Duration",   f"{args.time} ns")
    t.add_row("Threads",    str(threads))
    if config["force_field"] != "default":
        t.add_row("Force field", config["force_field"])
        t.add_row("Water model", config["water_model"])
    t.add_row("Output",     work_dir)
    console.print()
    console.print(t)
    console.print()

    with open(os.path.join(work_dir, "run_config.json"), "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    DynamicsPipeline(config).execute()
    return 0


def _run_dynamic_resume(args: Namespace) -> int:
    try:
        from ..pipelines.dynamics.resume import DynamicsResumeController
    except ImportError:
        try:
            from chemlink.pipelines.dynamics.resume import DynamicsResumeController  # type: ignore
        except ImportError:
            from pipelines.dynamics.resume import DynamicsResumeController  # type: ignore
    try:
        controller = DynamicsResumeController(
            args.run_dir, sim_type_name=args.sim_type, ns_time=args.time,
        )
        stage = controller.detect_stage()
        console.print(f"\n[bold cyan]Recovery directory:[/] {controller.work_dir}")
        console.print(f"[bold cyan]Detected stage:[/] {stage}\n")
        if args.dry_run:
            return 0
        result = controller.run()
        if result == "complete":
            print_ok("The simulation and its analyses are already complete.")
        else:
            print_ok(f"Recovery completed from stage: {result}")
        return 0
    except Exception as exc:
        err_console.print(f"\n[bold red]Resume error:[/] {exc}\n")
        return 1


def _run_dynamic_analyze(args: Namespace) -> int:
    try:
        from ..pipelines.dynamics.reanalysis import DynamicsReanalysisController
    except ImportError:
        try:
            from chemlink.pipelines.dynamics.reanalysis import DynamicsReanalysisController  # type: ignore
        except ImportError:
            from pipelines.dynamics.reanalysis import DynamicsReanalysisController  # type: ignore
    try:
        controller = DynamicsReanalysisController(
            args.run_dir,
            selection=args.only,
            sim_type_name=args.sim_type,
            ns_time=args.time,
            mmpbsa_cores=args.mmpbsa_cores,
            mmpbsa_method=args.mmpbsa_method,
            mmpbsa_frames=args.mmpbsa_frames,
            mmpbsa_discard=args.mmpbsa_discard,
            mmpbsa_salt=args.mmpbsa_salt,
            plot_only=args.plot_only,
        )
        console.print(f"\n[bold cyan]Results directory:[/] {controller.work_dir}")
        console.print(f"[bold cyan]Analysis selection:[/] {controller.selection}\n")
        controller.run()
        print_ok(f"{controller.selection.capitalize()} analysis completed.")
        return 0
    except Exception as exc:
        err_console.print(f"\n[bold red]Analysis error:[/] {exc}\n")
        return 1


# ── Doctor ─────────────────────────────────────────────────────────────────────

def _run_doctor(args: Namespace) -> int:
    from hpc.cluster.resource_detector import get_hardware_profile

    # ── helpers ────────────────────────────────────────────────────────────────

    def _chk(
        name: str, cat: str, level: str, detail: str,
        suggestion: Optional[str] = None,
    ) -> dict:
        return {"name": name, "cat": cat, "level": level,
                "detail": detail, "suggestion": suggestion}

    def _has_module(name: str) -> bool:
        return importlib.util.find_spec(name) is not None

    def _module_version(name: str) -> Optional[str]:
        try:
            from importlib.metadata import version as _v
            return _v(name)
        except Exception:
            return None

    def _find_binary(*candidates: str, fallback_paths: Optional[list] = None) -> Optional[str]:
        for name in candidates:
            found = shutil.which(name)
            if found:
                return found
        for path in (fallback_paths or []):
            if path and os.path.isfile(path) and os.access(path, os.X_OK):
                return path
        return None

    def _gmx_version(binary: str) -> str:
        try:
            r = subprocess.run(
                [binary, "--version"], capture_output=True, text=True, timeout=8,
            )
            m = re.search(r"GROMACS\s+version[:\s]+(\S+)", r.stdout + r.stderr, re.IGNORECASE)
            return m.group(1).strip() if m else "found"
        except Exception:
            return "found"

    # ── collect checks ─────────────────────────────────────────────────────────

    checks: list = []

    # -- software ---------------------------------------------------------------

    py_ok = sys.version_info >= (3, 9)
    checks.append(_chk(
        "Python ≥ 3.9", "software",
        "ok" if py_ok else "error",
        sys.version.split()[0],
        None if py_ok else "Upgrade Python to 3.9 or later.",
    ))

    for pkg, required in [
        ("tqdm",     True),
        ("numpy",    True),
        ("rdkit",    False),
        ("pdbfixer", False),
        ("meeko",    False),
    ]:
        found = _has_module(pkg)
        ver   = _module_version(pkg) or ("installed" if found else "not found")
        level = ("error" if required else "warning") if not found else "ok"
        checks.append(_chk(
            pkg, "software", level, ver,
            f"pip install {pkg}" if not found else None,
        ))

    gmx_serial = _find_binary(
        os.environ.get("CHEMLINK_GMX_BIN", ""), "gmx",
        fallback_paths=["/usr/local/gromacs/bin/gmx", "/opt/gromacs/bin/gmx"],
    )
    gmx_mpi = _find_binary(
        os.environ.get("CHEMLINK_GMX_MPI_BIN", ""), "gmx_mpi",
        fallback_paths=["/usr/local/gromacs/bin/gmx_mpi", "/opt/gromacs/bin/gmx_mpi"],
    )
    gmx_bin = gmx_mpi or gmx_serial
    checks.append(_chk(
        "GROMACS dynamics (gmx_mpi/gmx)", "software",
        "ok" if gmx_bin else "warning",
        f"{gmx_bin} ({_gmx_version(gmx_bin)})" if gmx_bin else "not found",
        "Install GROMACS and ensure `gmx_mpi` or `gmx` is on PATH." if not gmx_bin else None,
    ))
    checks.append(_chk(
        "GROMACS serial (parallel MM/PBSA)", "software",
        "ok" if gmx_serial else "warning",
        f"{gmx_serial} ({_gmx_version(gmx_serial)})" if gmx_serial else "not found",
        (
            "Install the serial/thread-MPI `gmx` executable or set CHEMLINK_GMX_BIN; "
            "use --mmpbsa-cores 1 meanwhile."
            if not gmx_serial else None
        ),
    ))

    mk_prepare_bin = shutil.which("mk_prepare_receptor.py")
    checks.append(_chk(
        "Meeko (mk_prepare_receptor.py)", "software",
        "ok" if mk_prepare_bin else "warning",
        mk_prepare_bin or "not found",
        "pip install meeko" if not mk_prepare_bin else None,
    ))

    fpocket_bin = shutil.which("fpocket")
    checks.append(_chk(
        "fpocket", "software",
        "ok" if fpocket_bin else "warning",
        fpocket_bin or "not found",
        "Install fpocket: https://github.com/Discngine/fpocket" if not fpocket_bin else None,
    ))

    ag4_bin = shutil.which("autogrid4")
    checks.append(_chk(
        "AutoGrid4", "software",
        "ok" if ag4_bin else "warning",
        ag4_bin or "not found",
        "Install AutoDock4/AutoGrid4 and add to PATH." if not ag4_bin else None,
    ))

    _autodock_env = os.environ.get("AUTODOCK_GPU_BIN", "")
    autodock_bin  = _find_binary(
        "autodock-gpu", "autodock_gpu",
        fallback_paths=[
            _autodock_env,
            "/nfs/chemlink/software/autodock-gpu/bin/autodock-gpu",
            "/usr/local/bin/autodock-gpu",
        ],
    )
    checks.append(_chk(
        "AutoDock-GPU", "software",
        "ok" if autodock_bin else "warning",
        autodock_bin or "not found",
        "Install AutoDock-GPU or set AUTODOCK_GPU_BIN=/path/to/binary." if not autodock_bin else None,
    ))

    acpype_bin = _find_binary(
        "acpype",
        fallback_paths=[os.path.join(os.path.dirname(sys.executable), "acpype")],
    )
    checks.append(_chk(
        "acpype", "software",
        "ok" if acpype_bin else "warning",
        acpype_bin or "not found",
        "pip install acpype" if not acpype_bin else None,
    ))

    mmpbsa_bin = _find_binary(
        "gmx_MMPBSA",
        fallback_paths=[
            os.environ.get("CHEMLINK_MMPBSA_BIN", ""),
            os.path.expanduser("~/.conda/envs/gmx_mmpbsa/bin/gmx_MMPBSA"),
            "/opt/conda/envs/gmx_mmpbsa/bin/gmx_MMPBSA",
            "/opt/miniconda/envs/gmx_mmpbsa/bin/gmx_MMPBSA",
        ],
    )
    checks.append(_chk(
        "gmx_MMPBSA", "software",
        "ok" if mmpbsa_bin else "warning",
        mmpbsa_bin or "not found (binding-energy analysis will be skipped)",
        "Install gmx-MMPBSA or set CHEMLINK_MMPBSA_BIN." if not mmpbsa_bin else None,
    ))

    mpirun_bin = _find_binary(
        "mpirun", "mpiexec",
        fallback_paths=[
            os.environ.get("CHEMLINK_MMPBSA_MPIRUN", ""),
            os.path.expanduser("~/.conda/envs/gmx_mmpbsa/bin/mpirun"),
            "/opt/conda/envs/gmx_mmpbsa/bin/mpirun",
            "/opt/miniconda/envs/gmx_mmpbsa/bin/mpirun",
        ],
    )
    checks.append(_chk(
        "OpenMPI (mpirun)", "software",
        "ok" if mpirun_bin else "warning",
        mpirun_bin or "not found (gmx_MMPBSA will run serially)",
        "Install OpenMPI or set CHEMLINK_MMPBSA_MPIRUN." if not mpirun_bin else None,
    ))

    mpi4py_found = _has_module("mpi4py")
    mpi4py_detail = _module_version("mpi4py") or ("installed" if mpi4py_found else "not found")
    if not mpi4py_found:
        for env_python in (
            os.path.expanduser("~/.conda/envs/gmx_mmpbsa/bin/python"),
            "/opt/conda/envs/gmx_mmpbsa/bin/python",
            "/opt/miniconda/envs/gmx_mmpbsa/bin/python",
        ):
            if not os.path.isfile(env_python):
                continue
            probe = subprocess.run(
                [env_python, "-c", "import mpi4py; print(mpi4py.__version__)"],
                capture_output=True, text=True, timeout=8,
            )
            if probe.returncode == 0:
                mpi4py_found = True
                mpi4py_detail = f"{probe.stdout.strip()} ({env_python})"
                break
    checks.append(_chk(
        "mpi4py", "software",
        "ok" if mpi4py_found else "warning",
        mpi4py_detail,
        "Install mpi4py in the gmx_MMPBSA environment." if not mpi4py_found else None,
    ))

    amber_bin = _find_binary(
        "antechamber", "MMPBSA.py",
        fallback_paths=[
            os.path.expanduser("~/.conda/envs/gmx_mmpbsa/bin/antechamber"),
            "/opt/conda/envs/gmx_mmpbsa/bin/antechamber",
            "/opt/miniconda/envs/gmx_mmpbsa/bin/antechamber",
        ],
    )
    checks.append(_chk(
        "AmberTools", "software",
        "ok" if amber_bin else "warning",
        amber_bin or "not found (required by gmx_MMPBSA)",
        "Install AmberTools in the gmx_MMPBSA environment." if not amber_bin else None,
    ))

    # -- hardware ---------------------------------------------------------------

    hw = get_hardware_profile()

    if hw.has_gpu:
        gpu_detail = "  ".join(
            f"GPU {g.index}: {g.name} ({g.memory_total_gb} GB, {g.memory_free_gb} GB free)"
            for g in hw.gpus
        )
        if len(hw.gpus) > 1:
            gpu_detail = f"{hw.gpu_count} GPUs — " + gpu_detail
        checks.append(_chk("GPU (NVIDIA)", "hardware", "ok", gpu_detail))
    else:
        checks.append(_chk(
            "GPU (NVIDIA)", "hardware", "warning",
            "nvidia-smi not available — MD and docking will run on CPU.",
            "Install NVIDIA drivers and nvidia-smi, or accept CPU-only execution.",
        ))

    ram_ok = hw.ram_total_gb >= 8.0
    ram_detail = (
        f"{hw.ram_total_gb} GB total · {hw.ram_available_gb} GB available"
        if hw.ram_total_gb > 0 else "unavailable"
    )
    checks.append(_chk(
        "RAM", "hardware",
        "ok" if ram_ok else "warning",
        ram_detail,
        "Free memory or add RAM; recommended ≥ 8 GB for MD simulations." if not ram_ok else None,
    ))

    cpu_detail = (
        f"{hw.cpu_cores} cores · {hw.cpu_threads} threads"
        if hw.cpu_cores else "unavailable"
    )
    checks.append(_chk("CPU", "hardware", "ok", cpu_detail))

    # -- config / env -----------------------------------------------------------

    autodock_env_set = bool(_autodock_env)
    checks.append(_chk(
        "AUTODOCK_GPU_BIN", "config",
        "ok" if autodock_env_set else "warning",
        _autodock_env if autodock_env_set else "not set (using PATH + fallback locations)",
        f"export AUTODOCK_GPU_BIN={autodock_bin or '/path/to/autodock-gpu'}" if not autodock_env_set else None,
    ))

    gmxlib = os.environ.get("GMXLIB", "")
    checks.append(_chk(
        "GMXLIB", "config", "ok",
        gmxlib if gmxlib else "not set (GROMACS uses built-in force fields)",
    ))

    slurm_job = os.environ.get("SLURM_JOB_ID", "")
    checks.append(_chk(
        "SLURM_JOB_ID", "config", "ok",
        f"running inside SLURM job {slurm_job}" if slurm_job else "not set (running locally)",
    ))

    # -- paths ------------------------------------------------------------------

    for rel_path, label in [
        ("data/input/receptors", "Receptor inputs"),
        ("data/input/ligands",   "Ligand inputs"),
        ("data/input/dynamics",  "Dynamics inputs"),
    ]:
        abs_path = os.path.abspath(rel_path)
        exists   = os.path.isdir(abs_path)
        checks.append(_chk(
            label, "paths",
            "ok" if exists else "warning",
            abs_path if exists else f"{abs_path}  (not found)",
            f"mkdir -p {rel_path}" if not exists else None,
        ))

    out_abs = os.path.abspath("data/output")
    if os.path.exists(out_abs):
        writable = os.access(out_abs, os.W_OK)
        checks.append(_chk(
            "Output directory", "paths",
            "ok" if writable else "error",
            f"{out_abs}  ({'writable' if writable else 'NOT writable'})",
            f"chmod 755 {out_abs}" if not writable else None,
        ))
    else:
        checks.append(_chk(
            "Output directory", "paths", "warning",
            f"{out_abs}  (not found — will be created on first run)",
            "mkdir -p data/output",
        ))

    # ── summary counts ─────────────────────────────────────────────────────────

    n_ok     = sum(1 for c in checks if c["level"] == "ok")
    n_warn   = sum(1 for c in checks if c["level"] == "warning")
    n_errors = sum(1 for c in checks if c["level"] == "error")

    # ── JSON output ────────────────────────────────────────────────────────────

    if getattr(args, "json", False):
        payload = {
            "doctor_version":   DOCTOR_VERSION,
            "chemlink_version": VERSION,
            "timestamp":        datetime.utcnow().isoformat() + "Z",
            "summary":          {"ok": n_ok, "warnings": n_warn, "errors": n_errors},
            "checks":           checks,
        }
        print(json.dumps(payload, indent=2))
        return 1 if n_errors else 0

    # ── Rich display ───────────────────────────────────────────────────────────

    _CAT_TITLES = {
        "software": "Software & Tools",
        "hardware": "Hardware",
        "config":   "Config & Environment",
        "paths":    "Paths",
    }
    _LEVEL_CELL = {
        "ok":      "[bold green]✓  OK[/]",
        "warning": "[bold yellow]⚠  Warning[/]",
        "error":   "[bold red]✗  Error[/]",
    }

    console.print()
    for cat in ("software", "hardware", "config", "paths"):
        cat_checks = [c for c in checks if c["cat"] == cat]
        if not cat_checks:
            continue
        t = Table(
            title=f"[bold cyan]{_CAT_TITLES[cat]}[/]",
            box=rich_box.ROUNDED,
            border_style="cyan",
            header_style="bold cyan",
        )
        t.add_column("Item",   style="bold")
        t.add_column("Status", justify="center")
        t.add_column("Detail", style="dim")
        for c in cat_checks:
            t.add_row(c["name"], _LEVEL_CELL[c["level"]], c["detail"])
        console.print(t)
        console.print()

    # Issues + suggestions block
    issues = [c for c in checks if c["level"] in ("error", "warning")]
    if issues:
        console.rule("[bold yellow]Issues & Suggestions[/]", style="yellow")
        console.print()
        for c in issues:
            prefix = "[bold red]✗[/]" if c["level"] == "error" else "[bold yellow]⚠[/]"
            console.print(f"  {prefix}  [bold]{c['name']}[/]  —  {c['detail']}")
            if c["suggestion"]:
                console.print(f"     [dim]→[/]  [cyan]{c['suggestion']}[/]")
        console.print()

    # Summary line
    if n_errors == 0 and n_warn == 0:
        print_ok(f"All {len(checks)} checks passed.")
    elif n_errors == 0:
        print_warn(f"{n_warn} warning(s) — no blockers.")
    else:
        print_err(f"{n_errors} error(s), {n_warn} warning(s).")

    console.print(
        f"  [dim]Doctor version: {DOCTOR_VERSION} · ChemLink: {VERSION}[/]\n"
    )
    return 1 if n_errors else 0


# ── HPC command handler ────────────────────────────────────────────────────────

def _auto_batch_size(n_ligands: int) -> int:
    if n_ligands <= 100:
        return n_ligands
    if n_ligands <= 500:
        return 100
    if n_ligands <= 2000:
        return 200
    return 500


def _run_hpc_docking(args: Namespace) -> int:
    import glob

    ligand_dir = args.ligand_dir
    exts = ("*.sdf", "*.mol2", "*.pdb", "*.mol", "*.pdbqt")
    ligand_files = []
    for ext in exts:
        ligand_files.extend(glob.glob(os.path.join(ligand_dir, ext)))
    n_ligands = len(ligand_files)

    if n_ligands == 0 and not args.dry_run:
        err_console.print(
            f"\n[bold red]Error:[/] No ligand files found in [cyan]{ligand_dir}[/]\n"
            f"  Supported formats: {', '.join(e.lstrip('*') for e in exts)}\n"
        )
        return 1

    manual_center = getattr(args, "manual_center", None)
    manual_npts   = getattr(args, "manual_npts",   None)
    if manual_npts is not None and manual_center is None:
        err_console.print(
            "[bold red]Error:[/] --manual-npts requires --manual-center."
        )
        return 1

    box_config = getattr(args, "box_config", None)
    box_config_abs: Optional[str] = None
    if box_config:
        try:
            from ..pipelines.docking.steps import load_box_config
        except ImportError:
            from chemlink.pipelines.docking.steps import load_box_config  # type: ignore
        # Fail fast on the submission host before anything is queued; the
        # SLURM job itself re-resolves the path from shared storage.
        load_box_config(box_config)
        box_config_abs = os.path.abspath(box_config)

    batch_size = args.batch_size if args.batch_size else _auto_batch_size(max(n_ligands, 1))
    total_batches = max(1, (n_ligands + batch_size - 1) // batch_size) if n_ligands else 1
    prep_tasks    = args.prep_tasks
    array_range   = f"0-{prep_tasks - 1}"

    if total_batches > 100:
        print_warn(
            f"{total_batches} docking batches will be submitted. "
            "Consider increasing --batch-size to reduce queue pressure."
        )

    mode = args.mode
    script_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "hpc", "slurm", mode, "run_multinode_pipeline.sh",
    )

    env_overrides: dict = {
        "INPUT_LIGANDS_DIR":        ligand_dir,
        "INPUT_RECEPTORS_DIR":      args.receptor_dir,
        "BATCH_SIZE":               str(batch_size),
        "MAX_GPU_CONCURRENCY":      str(args.max_gpu_concurrency),
        "PREP_ARRAY_RANGE":         array_range,
        "RECEPTOR_WORKERS":         str(args.receptor_workers),
        "LIGAND_WORKERS":           str(args.ligand_workers),
        "ACTIVE_SITE_WORKERS":      str(args.active_site_workers),
        "DOCKING_WORKERS":          str(args.docking_workers),
        "FLEX_RADIUS":              str(args.flex_radius),
    }
    if args.run_name:
        env_overrides["RUN_ID"] = args.run_name
        env_overrides["RUN_NAME_CUSTOM"] = "1"
    if args.partition:
        env_overrides["SLURM_PARTITION"] = args.partition
    if args.gres:
        env_overrides["DOCKING_GRES"] = args.gres
    if args.nodes:
        env_overrides["SLURM_NODELIST"] = args.nodes
    if mode == "container" and args.container_image:
        env_overrides["CONTAINER_IMAGE"] = args.container_image
    if manual_center:
        env_overrides["MANUAL_CENTER"] = " ".join(f"{v:g}" for v in manual_center)
        if manual_npts:
            env_overrides["MANUAL_NPTS"] = " ".join(str(v) for v in manual_npts)
    if box_config_abs:
        env_overrides["BOX_CONFIG_PATH"] = box_config_abs

    t = Table(
        title="[bold cyan]HPC Docking Configuration[/]",
        box=rich_box.ROUNDED,
        border_style="cyan",
        show_header=False,
        min_width=48,
    )
    t.add_column(style="dim",  no_wrap=True)
    t.add_column(style="bold")
    t.add_row("Mode",               mode)
    t.add_row("Ligand directory",   ligand_dir)
    t.add_row("Receptor directory", args.receptor_dir)
    t.add_row("Ligands detected",   str(n_ligands) if n_ligands else "0  (dry-run)")
    t.add_row("Batch size",         str(batch_size))
    t.add_row("Total batches",      str(total_batches))
    t.add_row("Prep array",         array_range)
    t.add_row("Max GPU concurrency",str(args.max_gpu_concurrency))
    t.add_row("Receptor workers",   str(args.receptor_workers))
    t.add_row("Ligand workers",     str(args.ligand_workers))
    t.add_row("Active-site workers",str(args.active_site_workers))
    t.add_row("Docking workers",    str(args.docking_workers))
    t.add_row("Flexible radius",    f"{args.flex_radius:g} A" if args.flex_radius else "disabled")
    if args.run_name:
        t.add_row("Run name",           args.run_name)
    if manual_center:
        t.add_row("Grid center (manual)", " ".join(f"{v:g}" for v in manual_center))
        t.add_row(
            "Grid npts",
            " ".join(str(v) for v in manual_npts) if manual_npts else "auto from ligands",
        )
    if box_config_abs:
        t.add_row("Box config", box_config_abs)
    if args.partition:
        t.add_row("SLURM partition", args.partition)
    if args.gres:
        t.add_row("GRES",            args.gres)
    if args.nodes:
        t.add_row("Nodes",           args.nodes)
    if mode == "container" and args.container_image:
        t.add_row("Container image", args.container_image)
    console.print()
    console.print(t)
    console.print()

    if args.dry_run:
        console.print("[bold yellow]Dry-run mode — command that would be executed:[/]")
        env_str = " ".join(f"{k}={v}" for k, v in env_overrides.items())
        console.print(f"  [dim]{env_str}[/]")
        console.print(f"  [cyan]bash {script_path}[/]\n")
        return 0

    if not os.path.isfile(script_path):
        err_console.print(
            f"\n[bold red]Error:[/] Pipeline script not found: [cyan]{script_path}[/]\n"
        )
        return 1

    env = {**os.environ, **env_overrides}
    result = subprocess.run(["bash", script_path], env=env)
    return result.returncode


# ── HPC Dynamics command handler ──────────────────────────────────────────────

def _count_pdb_atoms(pdb_path: str) -> int:
    try:
        count = 0
        with open(pdb_path) as fh:
            for line in fh:
                if line.startswith(("ATOM  ", "HETATM")):
                    count += 1
        return count
    except OSError:
        return 0


def _run_hpc_dynamics(args: Namespace) -> int:
    try:
        from ..pipelines.dynamics.utils import setup_work_directory
    except ImportError:
        try:
            from chemlink.pipelines.dynamics.utils import setup_work_directory  # type: ignore
        except ImportError:
            from pipelines.dynamics.utils import setup_work_directory  # type: ignore

    try:
        from ..hpc.cluster.network_detector import get_multinode_recommendation
    except ImportError:
        try:
            from hpc.cluster.network_detector import get_multinode_recommendation  # type: ignore
        except ImportError:
            from chemlink.hpc.cluster.network_detector import get_multinode_recommendation  # type: ignore

    dyn_type = args.dyn_type
    sim_id, label, min_files = _DYN_SPEC[dyn_type]

    receptor_inputs = list(getattr(args, "input_receptors", None) or [])
    ligand_inputs = list(getattr(args, "input_ligands", None) or [])
    ligand_inputs.extend(getattr(args, "ligand_files", None) or [])
    try:
        sim_input_sets, input_interface = _resolve_dynamics_input_sets(
            dyn_type,
            receptor_inputs=receptor_inputs,
            ligand_inputs=ligand_inputs,
            legacy_inputs=getattr(args, "input_files", None),
            allow_multiple_ligands=True,
        )
    except ValueError as exc:
        err_console.print(f"\n[bold red]Input error:[/] {exc}\n")
        return 1

    missing_files = [
        path for files in sim_input_sets for path in files if not os.path.isfile(path)
    ]
    if missing_files:
        err_console.print("\n[bold red]Input file not found:[/]")
        for path in sorted(set(missing_files)):
            err_console.print(f"  {path}")
        return 1

    n_nodes = len(args.nodes.split(",")) if args.nodes else 1

    # --mpi: single job spanning all nodes — warn and fall back if only 1 node
    use_mpi = getattr(args, "mpi", False) and n_nodes > 1
    if getattr(args, "mpi", False) and not use_mpi:
        print_warn("--mpi requires --nodes with more than one node; falling back to normal mode.")

    if use_mpi:
        # Force a single simulation config — MPI handles parallelism internally
        n_sims = 1
        sim_input_sets = sim_input_sets[:1]
    else:
        # --replicate: submit the same single simulation to every node in --nodes
        if getattr(args, "replicate", False) and n_nodes > 1 and len(sim_input_sets) == 1:
            sim_input_sets = sim_input_sets * n_nodes

        n_sims = len(sim_input_sets)

    # --- Estimate atom count from first simulation's first PDB ---
    raw_atoms = _count_pdb_atoms(sim_input_sets[0][0]) if sim_input_sets else 0
    estimated_atoms = max(raw_atoms * 10, 1)

    rec = get_multinode_recommendation(estimated_atoms, n_nodes)
    if rec["use_multinode"]:
        print_info(f"Network: {rec['speed_gbps']:.0f} GbE — multi-node recommended")
    else:
        print_warn(f"Network: {rec['reason']}")

    # --- Build per-simulation config dicts and write individual JSON files ---
    timestamp = f"{datetime.now():%Y%m%dT%H%M%S}"
    configs: list[dict] = []
    node_list_for_label = args.nodes.split(",") if args.nodes else []
    is_replicate = getattr(args, "replicate", False)
    requested_run_name = getattr(args, "run_name", None)
    custom_run_names = [
        indexed_run_name(requested_run_name, index, n_sims)
        for index in range(n_sims)
    ]
    if requested_run_name:
        existing = [
            os.path.abspath(os.path.join("data/output/dynamics", name))
            for name in custom_run_names
            if name and os.path.exists(os.path.join("data/output/dynamics", name))
        ]
        if existing:
            err_console.print(
                "\n[bold red]Error:[/] Run directory already exists:\n  "
                + "\n  ".join(existing)
                + "\nChoose another --run-name.\n"
            )
            return 1

    for idx, files in enumerate(sim_input_sets):
        # When replicating, embed the node name in the dir so each replica
        # gets its own work directory even within the same second.
        custom_name = custom_run_names[idx]
        if custom_name:
            work_dir = setup_work_directory(
                "data/output/dynamics", dyn_type, run_name=custom_name
            )
        elif is_replicate and node_list_for_label:
            node_tag = node_list_for_label[idx % len(node_list_for_label)]
            work_dir = os.path.abspath(
                os.path.join("data/output/dynamics",
                             f"run_{dyn_type}_{timestamp}_{node_tag}")
            )
            os.makedirs(work_dir, exist_ok=True)
        else:
            work_dir = setup_work_directory("data/output/dynamics", dyn_type)
        config: dict = {
            "sim_type":       sim_id,
            "sim_type_label": label,
            "ns_time":        args.time,
            "threads":        args.cpus,
            "work_dir":       work_dir,
            "run_name":       custom_name or os.path.basename(work_dir),
            "run_name_base":  requested_run_name,
            "directory_mode": "custom" if requested_run_name else "automatic",
            "receptor_inputs": files[:_DYN_INPUT_COUNTS[dyn_type][0]],
            "ligand_inputs": files[_DYN_INPUT_COUNTS[dyn_type][0]:],
            "input_interface": input_interface,
            "binding_energy": getattr(args, "binding_energy", True),
            "mmpbsa_method":  getattr(args, "mmpbsa_method", "both"),
            "mmpbsa_frames":  getattr(args, "mmpbsa_frames", 1000),
            "mmpbsa_discard": getattr(args, "mmpbsa_discard", 20.0),
            "mmpbsa_cores":   getattr(args, "mmpbsa_cores", args.cpus),
            "mmpbsa_salt":    getattr(args, "mmpbsa_salt", 0.150),
        }

        if dyn_type == "oprotein":
            config["pdb_input"] = files[0]
        elif dyn_type == "pligand":
            try:
                _validate_dynamics_ligand(files[1], work_dir)
                charge = _resolve_ligand_charge_for_run(
                    files[1], work_dir,
                    explicit_charge=args.charge,
                    mode=args.charge_mode,
                    ph=args.ligand_ph,
                    interactive=False,
                )
            except Exception as exc:
                err_console.print(f"\n[bold red]Ligand charge error:[/] {exc}\n")
                return 1
            config.update(
                pdb_input=files[0], ligand_pdb=files[1],
                ligand_charge=charge.charge,
                ligand_prepared_file=charge.prepared_file,
                ligand_charge_method=charge.method,
                ligand_charge_confidence=charge.confidence,
                ligand_charge_source=charge.source_file,
                ligand_ph=charge.ph,
            )
        elif dyn_type in ("ppeptide", "pacid", "pprotein"):
            config.update(pdb_protein=files[0], pdb_partner=files[1],
                          pdb_input=os.path.join(work_dir, "complex.pdb"))
        elif dyn_type == "ppligand":
            try:
                _validate_dynamics_ligand(files[2], work_dir)
                charge = _resolve_ligand_charge_for_run(
                    files[2], work_dir,
                    explicit_charge=args.charge,
                    mode=args.charge_mode,
                    ph=args.ligand_ph,
                    interactive=False,
                )
            except Exception as exc:
                err_console.print(f"\n[bold red]Ligand charge error:[/] {exc}\n")
                return 1
            config.update(
                pdb_protein=files[0], pdb_partner=files[1],
                pdb_input=os.path.join(work_dir, "complex.pdb"),
                ligand_pdb=files[2], ligand_charge=charge.charge,
                ligand_prepared_file=charge.prepared_file,
                ligand_charge_method=charge.method,
                ligand_charge_confidence=charge.confidence,
                ligand_charge_source=charge.source_file,
                ligand_ph=charge.ph,
            )

        cfg_path = os.path.join(work_dir, "hpc_config.json")
        config["_config_path"] = cfg_path
        with open(cfg_path, "w") as fh:
            json.dump(config, fh, indent=2)

        configs.append(config)

    master_json = f"/tmp/chemlink_dyn_configs_{timestamp}.json"
    with open(master_json, "w") as fh:
        json.dump(configs, fh, indent=2)

    # --- Summary table ---
    t = Table(
        title="[bold cyan]HPC Dynamics Configuration[/]",
        box=rich_box.ROUNDED,
        border_style="cyan",
        show_header=False,
        min_width=52,
    )
    t.add_column(style="dim",  no_wrap=True)
    t.add_column(style="bold")
    t.add_row("Simulation type",   label)
    t.add_row("Duration",          f"{args.time} ns")
    t.add_row("Simulations",       str(n_sims))
    t.add_row("CPUs per job",      str(args.cpus))
    t.add_row("Memory per job",    args.mem)
    t.add_row("Wall time",         args.time_limit)
    t.add_row("Interconnect",      f"{rec['speed_gbps']:.0f} GbE")
    if args.partition:
        t.add_row("SLURM partition", args.partition)
    if args.nodes:
        t.add_row("Nodes",           args.nodes)
    if use_mpi:
        t.add_row("Mode", f"[bold magenta]MPI ({n_nodes} nodes)[/] — domain decomposition")
    elif getattr(args, "replicate", False):
        t.add_row("Mode",            f"[bold yellow]replicate[/bold yellow] — same sim on each node")
    t.add_row("Config list",       master_json)
    console.print()
    console.print(t)
    console.print()

    if args.dry_run:
        console.print("[bold yellow]Dry-run mode — jobs that would be submitted:[/]")
        for i, cfg in enumerate(configs):
            console.print(f"  Sim {i + 1}: [cyan]{cfg['work_dir']}[/]")
        console.print()
        return 0

    _native_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "hpc", "slurm", "native",
    )

    if use_mpi:
        # ── MPI path: submit one multi-node sbatch job directly ──────────────
        mpi_slurm = os.path.join(_native_dir, "dynamics_mpi.slurm")
        if not os.path.isfile(mpi_slurm):
            err_console.print(
                f"\n[bold red]Error:[/] MPI SLURM script not found: [cyan]{mpi_slurm}[/]\n"
            )
            return 1

        cfg = configs[0]
        sbatch_cmd = [
            "sbatch",
            f"--nodes={n_nodes}",
            f"--ntasks={n_nodes}",
            "--ntasks-per-node=1",
        ]
        if args.partition:
            sbatch_cmd.append(f"--partition={args.partition}")
        if args.nodes:
            sbatch_cmd.append(f"--nodelist={args.nodes}")
        sbatch_cmd += [
            f"--time={args.time_limit}",
            f"--mem={args.mem}",
            f"--cpus-per-task={args.cpus}",
            "--export=ALL,"
            f"DYN_CONFIG_JSON={cfg['_config_path']},"
            f"DYN_WORK_DIR={cfg['work_dir']},"
            f"DYN_TIME_LIMIT={args.time_limit}",
            mpi_slurm,
        ]
        result = subprocess.run(sbatch_cmd)

        rt = Table(
            title="[bold cyan]Submitted MPI Job[/]",
            box=rich_box.ROUNDED,
            border_style="cyan",
            header_style="bold cyan",
            min_width=60,
        )
        rt.add_column("#",        style="dim",  justify="right")
        rt.add_column("Type",     style="bold")
        rt.add_column("Mode",     style="magenta")
        rt.add_column("Work dir", style="dim")
        rt.add_row("1", label, f"MPI ({n_nodes} nodes)", cfg["work_dir"])

        console.print()
        console.print(rt)
        console.print()
        return result.returncode

    else:
        # ── Normal path: submit N independent jobs via shell script ──────────
        script_path = os.path.join(_native_dir, "run_dynamics_pipeline.sh")

        if not os.path.isfile(script_path):
            err_console.print(
                f"\n[bold red]Error:[/] Pipeline script not found: [cyan]{script_path}[/]\n"
            )
            return 1

        env_overrides: dict = {
            "DYN_TYPE":         dyn_type,
            "DYN_NS_TIME":      str(args.time),
            "DYN_CHARGE":       "" if args.charge is None else str(args.charge),
            "DYN_CONFIGS_JSON": master_json,
            "DYN_TIME_LIMIT":   args.time_limit,
            "DYN_MEM":          args.mem,
            "DYN_CPUS":         str(args.cpus),
        }
        if args.partition:
            env_overrides["SLURM_PARTITION"] = args.partition
        if args.nodes:
            env_overrides["SLURM_NODELIST"] = args.nodes

        env = {**os.environ, **env_overrides}
        result = subprocess.run(["bash", script_path], env=env)

        # --- Results table ---
        rt = Table(
            title="[bold cyan]Submitted Jobs[/]",
            box=rich_box.ROUNDED,
            border_style="cyan",
            header_style="bold cyan",
            min_width=60,
        )
        rt.add_column("#",         style="dim",  justify="right")
        rt.add_column("Type",      style="bold")
        rt.add_column("Work dir",  style="dim")
        if args.nodes:
            rt.add_column("Node", style="cyan")

        node_list = args.nodes.split(",") if args.nodes else []
        for i, cfg in enumerate(configs):
            row = [str(i + 1), label, cfg["work_dir"]]
            if args.nodes:
                row.append(node_list[i % len(node_list)])
            rt.add_row(*row)

        console.print()
        console.print(rt)
        console.print()

        return result.returncode


# ── Shell completion ───────────────────────────────────────────────────────────

_BASH_SCRIPT = """\
# ChemLink bash completion
# Add to ~/.bashrc:  source <(chemlink completion bash)

_chemlink() {
    local cur="${COMP_WORDS[COMP_CWORD]}"
    local cmd="${COMP_WORDS[1]}"
    local sub="${COMP_WORDS[2]}"
    local depth=$COMP_CWORD

    case "$cmd" in
        docking)
            [[ $depth -eq 2 ]] && COMPREPLY=( $(compgen -W "prepare prep full flow run analyze analysis receptor ligand active-site" -- "$cur") )
            if [[ $depth -ge 3 && "$sub" =~ ^(prepare|prep|full|flow)$ ]]; then
                COMPREPLY=( $(compgen -W "-ir --input-receptors -il --input-ligands -o --output --run-name --manual-center --manual-npts --box-config --receptor-workers --ligand-workers --active-site-workers --docking-workers" -- "$cur") )
            fi;;
        dynamic)
            [[ $depth -eq 2 ]] && COMPREPLY=( $(compgen -W "oprotein pligand ppeptide pacid pprotein ppligand resume analyze" -- "$cur") )
            [[ $depth -ge 3 && "$sub" =~ ^(oprotein|pligand|ppeptide|pacid|pprotein|ppligand)$ ]] && COMPREPLY=( $(compgen -W "-ir --input-receptors -il --input-ligands --run-name -i --input-dir -t --time" -- "$cur") );;
        completion)
            [[ $depth -eq 2 ]] && COMPREPLY=( $(compgen -W "bash zsh fish install" -- "$cur") );;
        doctor)
            COMPREPLY=( $(compgen -W "--json" -- "$cur") );;
        hpc)
            [[ $depth -eq 2 ]] && COMPREPLY=( $(compgen -W "docking dynamics" -- "$cur") )
            if [[ $depth -ge 3 && "$sub" == "docking" ]]; then
                COMPREPLY=( $(compgen -W "--run-name --ligand-dir --receptor-dir --nodes --partition --gres --batch-size --prep-tasks --receptor-workers --ligand-workers --active-site-workers --docking-workers --max-gpu-concurrency --manual-center --manual-npts --mode --container-image --dry-run" -- "$cur") )
            fi
            if [[ $depth -ge 3 && "$sub" == "dynamics" ]]; then
                COMPREPLY=( $(compgen -W "oprotein pligand ppeptide pacid pprotein ppligand -ir --input-receptors -il --input-ligands --run-name --protein --ligands --time --charge --charge-mode --ligand-ph --nodes --partition --time-limit --mem --cpus --dry-run" -- "$cur") )
            fi;;
        *)
            COMPREPLY=( $(compgen -W "docking dynamic doctor completion hpc --help --version" -- "$cur") );;
    esac
}
complete -F _chemlink chemlink
"""

_ZSH_SCRIPT = """\
#compdef chemlink
# ChemLink zsh completion
# source <(chemlink completion zsh)

# ── Styles (scoped to chemlink only) ─────────────────────────────────────────
zstyle ':completion:*:chemlink:*:descriptions' format '%F{cyan}%B%d%b%f'
zstyle ':completion:*:chemlink:*:messages'     format '%F{yellow}%d%f'
zstyle ':completion:*:chemlink:*'              group-name ''
zstyle ':completion:*:chemlink:*'              list-colors '=:=90'

_chemlink() {
    local -a top dock dyn shells hpccmds hpcdock hpcdyn

    top=(
        'docking:Molecular docking workflows'
        'dynamic:Molecular dynamics simulations'
        'hpc:Submit pipelines to SLURM cluster'
        'doctor:Check environment and tool prerequisites'
        'completion:Generate shell tab-completion scripts'
    )
    dock=(
        'prepare:Preparation only  (receptor · ligand · active-site)'
        'prep:Alias for prepare'
        'full:Full pipeline  (preparation + execution + analysis)'
        'flow:Run a contiguous subset of pipeline steps'
        'run:Docking execution on already-prepared files'
        'analyze:Analyze DLG results and generate reports'
        'analysis:Alias for analyze'
        'receptor:Prepare receptor PDB files to PDBQT'
        'ligand:Prepare ligand files to PDBQT'
        'active-site:Detect binding sites from PDBQT files'
    )
    dyn=(
        'oprotein:Protein-only simulation'
        'pligand:Protein + small-molecule ligand'
        'ppeptide:Protein + peptide complex'
        'pacid:Protein + nucleic acid complex'
        'pprotein:Protein + protein complex'
        'analyze:Re-run analysis from an existing dynamics result directory'
        'ppligand:Protein + protein + ligand/cofactor'
        'resume:Resume an interrupted run from a checkpoint'
    )
    shells=(
        'bash:Bash completion script'
        'zsh:Zsh completion script'
        'fish:Fish completion script'
        'install:Auto-detect shell and install'
    )
    hpccmds=(
        'docking:Submit full docking pipeline to SLURM'
        'dynamics:Submit molecular dynamics simulations to SLURM'
    )

    case $words[2] in
        docking)
            (( CURRENT == 3 )) && _describe -t dock 'docking subcommand' dock
            if (( CURRENT >= 4 )) && [[ $words[3] == (prepare|prep|full|flow) ]]; then
                _arguments \
                    '-ir[Directory to search for receptors]:dir:_files -/' \
                    '--input-receptors[Directory to search for receptors]:dir:_files -/' \
                    '-il[Directory to search for ligands]:dir:_files -/' \
                    '--input-ligands[Directory to search for ligands]:dir:_files -/' \
                    '-o[Base output directory]:dir:_files -/' \
                    '--output[Base output directory]:dir:_files -/' \
                    '--run-name[Explicit run directory name]:name:'
            fi
            ;;
        dynamic)
            (( CURRENT == 3 )) && _describe -t dyn 'simulation type' dyn
            if (( CURRENT >= 4 )) && [[ $words[3] == (oprotein|pligand|ppeptide|pacid|pprotein|ppligand) ]]; then
                _arguments \
                    '-ir[Receptor/protein input file]:file:_files' \
                    '--input-receptors[Receptor/protein input file]:file:_files' \
                    '-il[Ligand or partner input file]:file:_files' \
                    '--input-ligands[Ligand or partner input file]:file:_files' \
                    '--run-name[Explicit run directory name]:name:'
            fi
            ;;
        completion)
            (( CURRENT == 3 )) && _describe -t shells 'target shell' shells
            ;;
        doctor)
            _arguments '--json[Output check results as JSON for CI]'
            ;;
        hpc)
            if (( CURRENT == 3 )); then
                _describe -t hpccmds 'hpc subcommand' hpccmds
            elif (( CURRENT >= 4 )) && [[ $words[3] == docking ]]; then
                _arguments \\
                    '--ligand-dir[Ligand input directory]:dir:_files -/' \\
                    '--receptor-dir[Receptor PDB directory]:dir:_files -/' \\
                    '--nodes[SLURM nodelist]:nodelist:' \\
                    '--partition[SLURM partition]:partition:' \\
                    '--gres[SLURM generic resource (e.g. gpu\\:1)]:gres:' \\
                    '--batch-size[Ligands per docking batch]:N:' \\
                    '--prep-tasks[Array tasks for prep steps]:N:' \\
                    '--receptor-workers[Workers per receptor-prep task]:N:' \\
                    '--ligand-workers[Workers per ligand-prep task]:N:' \\
                    '--active-site-workers[Workers for active-site detection]:N:' \\
                    '--docking-workers[Workers per docking task]:N:' \\
                    '--max-gpu-concurrency[Max concurrent docking tasks]:N:' \\
                    '--run-name[Explicit run directory name]:name:' \\
                    '--mode[Script set to use]:mode:(native container)' \\
                    '--container-image[Container image path]:image:_files' \\
                    '--dry-run[Print config without submitting]'
            elif (( CURRENT >= 4 )) && [[ $words[3] == dynamics ]]; then
                _arguments \\
                    ':simulation type:(oprotein pligand ppeptide pacid pprotein ppligand)' \\
                    '-i[File inside data/input/dynamics]:file:_files' \\
                    '-ir[Receptor/protein input file]:file:_files' \\
                    '--input-receptors[Receptor/protein input file]:file:_files' \\
                    '-il[Ligand or partner input file]:file:_files' \\
                    '--input-ligands[Ligand or partner input file]:file:_files' \\
                    '--protein[Protein PDB file]:file:_files' \\
                    '--ligands[Ligand PDB files]:file:_files' \\
                    '--time[Simulation time in ns]:NS:' \\
                    '--charge[Ligand net charge]:Q:' \\
                    '--charge-mode[Charge resolution policy]:mode:(auto strict)' \\
                    '--ligand-ph[pH used to prepare PDB/PDBQT ligands]:PH:' \\
                    '--nodes[SLURM nodelist]:nodelist:' \\
                    '--partition[SLURM partition]:partition:' \\
                    '--time-limit[Wall time per job HH:MM:SS]:HH\\:MM\\:SS:' \\
                    '--mem[Memory per job]:MEM:' \\
                    '--cpus[CPUs per job]:N:' \\
                    '--run-name[Base name for dynamics run directories]:name:' \\
                    '--dry-run[Print config without submitting]'
            fi
            ;;
        *)
            _describe -t cmds 'command' top
            ;;
    esac
}

compdef _chemlink chemlink
"""


def _run_completion(args: Namespace) -> int:
    shell = args.shell
    if shell == "bash":
        print(_BASH_SCRIPT)
    elif shell == "zsh":
        print(_ZSH_SCRIPT)
    elif shell == "fish":
        err_console.print("[yellow]Fish completion is not yet available. Use bash or zsh.[/]")
        return 1
    elif shell == "install":
        _install_completion()
    return 0


def _install_completion() -> None:
    shell_bin = os.environ.get("SHELL", "")
    is_zsh    = "zsh" in shell_bin
    rc        = os.path.expanduser("~/.zshrc" if is_zsh else "~/.bashrc")
    line      = f'source <(chemlink completion {"zsh" if is_zsh else "bash"})'
    print_info(f"Appending completion hook to {rc}")
    try:
        with open(rc, "a") as fh:
            fh.write(f"\n# ChemLink autocompletion\n{line}\n")
        print_ok(f"Done. Restart your shell or run:  [cyan]source {rc}[/]")
    except OSError as exc:
        print_err(f"Could not write to {rc}: {exc}")


# ── Argument parser ────────────────────────────────────────────────────────────

def _common_pipeline_args(p: ArgumentParser) -> None:
    p.add_argument("receptor_input_dir", nargs="?", default=_REC_IN, metavar="RECEPTOR_DIR",
                   help=f"Receptor PDB directory (legacy positional)  [{_REC_IN}]")
    p.add_argument("ligand_input_dir", nargs="?", default=_LIG_IN, metavar="LIGAND_DIR",
                   help=f"Ligand input directory (legacy positional)  [{_LIG_IN}]")
    p.add_argument("output_dir", nargs="?", default=_OUT, metavar="OUTPUT_DIR",
                   help=f"Base output directory (legacy positional)  [{_OUT}]")
    p.add_argument("-ir", "--input-receptors", dest="receptor_input_option", metavar="PATH",
                   help=f"Receptor directory or name under data/input  [{_REC_IN}]")
    p.add_argument("-il", "--input-ligands", dest="ligand_input_option", metavar="PATH",
                   help=f"Ligand directory or name under data/input  [{_LIG_IN}]")
    p.add_argument("-o", "--output", dest="output_option", metavar="PATH",
                   help=f"Base directory for run outputs  [{_OUT}]")
    p.add_argument("--fpocket-path", "--fpocket", dest="fpocket_path", metavar="PATH",
                   help="fpocket binary path")
    p.add_argument("--manual-center", nargs=3, type=float, metavar=("X", "Y", "Z"),
                   help="Manual grid center; without --manual-npts, grid dimensions "
                        "are estimated from the largest prepared ligands")
    p.add_argument("--manual-npts", nargs=3, type=int, metavar=("NX", "NY", "NZ"),
                   help="Manual AutoDock grid dimensions (requires --manual-center)")
    p.add_argument("--box-config", dest="box_config", metavar="PATH",
                   help="JSON file mapping receptor name -> {center, npts} for a "
                        "per-receptor grid box  (overrides --manual-center/--manual-npts "
                        "and fpocket auto-detection for the receptors it lists)")
    p.add_argument("--flex-radius", type=float, default=0.0, metavar="ANGSTROM",
                   help="Flex side chains with a heavy atom within this distance of "
                        "the box center  [disabled; rigid receptor]")
    p.add_argument("--rigid-receptor", dest="flex_radius", action="store_const", const=0.0,
                   help="Disable automatic flexible side-chain selection")
    p.add_argument("--receptor-workers", "-r", type=int, metavar="N",
                   help="Parallel workers for receptor preparation")
    p.add_argument("--ligand-workers", "-l", type=int, metavar="N",
                   help="Parallel workers for ligand preparation")
    p.add_argument("--active-site-workers", "-a", type=int, metavar="N",
                   help="Parallel workers for active-site detection")


def _exec_args(p: ArgumentParser) -> None:
    p.add_argument("--docking-workers", "-d", type=int, default=1, metavar="N",
                   help="Concurrent docking jobs  [1]  (cpu engine: cores are split "
                        "across these; total threads = workers x vina-cpu)")
    p.add_argument("--engine", choices=("gpu", "cpu"), default="gpu", metavar="ENGINE",
                   help="Docking engine: gpu (AutoDock-GPU) or cpu (AutoDock Vina)  [gpu]")
    p.add_argument("--autogrid-executable", "--autogrid", dest="autogrid_executable", metavar="EXE",
                   help="AutoGrid4 executable path or name  (gpu engine)")
    p.add_argument("--autodock-gpu-executable", "--autodock", dest="autodock_gpu_executable", metavar="EXE",
                   help="AutoDock-GPU executable path or name  (gpu engine)")
    p.add_argument("--vina-bin", dest="vina_executable", metavar="EXE",
                   help="AutoDock Vina executable path or name  (cpu engine)")
    p.add_argument("--exhaustiveness", type=int, default=8, metavar="N",
                   help="Vina search exhaustiveness  (cpu engine)  [8]")
    p.add_argument("--num-modes", dest="num_modes", type=int, default=9, metavar="N",
                   help="Max poses Vina reports per ligand  (cpu engine)  [9]")
    p.add_argument("--vina-cpu", dest="vina_cpu", type=int, default=None, metavar="N",
                   help="CPU threads per Vina job  (cpu engine)  "
                        "[auto = cores / docking-workers]")
    p.add_argument("--seed", type=int, default=None, metavar="N",
                   help="Vina random seed for reproducible runs  (cpu engine)")
    p.add_argument("--pdb-export-limit", type=int, default=10, metavar="N",
                   help="Top candidates to export as PDB  [10]")
    p.add_argument("--max-workers", type=int, default=4, metavar="N",
                   help="Max parallel threads for PDB export  [4]")


def _step_range_args(p: ArgumentParser) -> None:
    choices_str = " | ".join(PIPELINE_STEPS)
    p.add_argument("--from-step", choices=PIPELINE_STEPS, default="receptor", metavar="STEP",
                   help=f"First pipeline step  [receptor]  ({choices_str})")
    p.add_argument("--to-step", choices=PIPELINE_STEPS, default="analysis", metavar="STEP",
                   help=f"Last pipeline step   [analysis]  ({choices_str})")


def _run_name_arg(p: ArgumentParser) -> None:
    def run_name_value(value: str) -> str:
        try:
            return validate_run_name(value)
        except ValueError as exc:
            raise ArgumentTypeError(str(exc)) from exc

    p.add_argument(
        "--run-name", type=run_name_value, metavar="NAME",
        help="Name the run directory explicitly; default uses the automatic timestamped name",
    )


def _dyn_parent() -> ArgumentParser:
    p = ArgumentParser(add_help=False)
    p.add_argument("-ir", "--input-receptors", action="append", dest="input_receptors",
                   metavar="FILE", help="Receptor/protein input file (repeat for ppligand)")
    p.add_argument("-il", "--input-ligands", action="append", dest="input_ligands",
                   metavar="FILE", help="Ligand or partner input file")
    p.add_argument("-i", "--input-dir", action="append", dest="input_files", metavar="FILE",
                   help="Legacy ordered input file (cannot be mixed with -ir/-il)")
    p.add_argument("files", nargs="*", metavar="PATH",
                   help="Absolute or relative file paths")
    p.add_argument("-t", "--time", type=float, default=0.1, metavar="NS",
                   help="Simulation time in nanoseconds  [0.1]")
    _run_name_arg(p)
    return p


def build_parser() -> _Parser:
    epilog = (
        f"\n  {bold('Quick examples:')}\n"
        f"    {cyan('chemlink docking prepare')}  data/receptors  data/ligands\n"
        f"    {cyan('chemlink docking full')}     data/receptors  data/ligands\n"
        f"    {cyan('chemlink dynamic oprotein')} -i protein.pdb  -t 0.5\n"
        f"    {cyan('chemlink dynamic pligand')}  -i prot.pdb -i lig.pdbqt -c 0\n"
        f"    {cyan('chemlink doctor')}\n"
        f"    {cyan('chemlink completion bash')}  >> ~/.bashrc  &&  source ~/.bashrc\n"
    )

    parser = _Parser(
        prog="chemlink",
        formatter_class=_Fmt,
        epilog=epilog,
    )
    parser.add_argument("--version", "-V", action="version", version=f"chemlink {VERSION}")

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # ── docking ────────────────────────────────────────────────────────────────
    dock_desc = (
        f"  {bold_cyan('docking')}  — Run molecular docking pipelines.\n\n"
        f"  Pipeline steps (in order):\n"
        f"    {cyan('receptor')}  →  {cyan('ligand')}  →  {cyan('active-site')}  "
        f"→  {cyan('execution')}  →  {cyan('analysis')}"
    )
    dock = sub.add_parser("docking", help="Molecular docking workflows",
                          description=dock_desc, formatter_class=_Fmt)
    dock_sub = dock.add_subparsers(dest="docking_command", metavar="SUBCOMMAND")

    prep = dock_sub.add_parser("prepare", aliases=["prep"], formatter_class=_Fmt,
                               help="Preparation only  (receptor + ligand + active-site)")
    _common_pipeline_args(prep)
    _run_name_arg(prep)
    prep.set_defaults(handler=lambda a: _run_docking_pipeline(a, full=False))

    full_p = dock_sub.add_parser("full", formatter_class=_Fmt,
                                 help="Full pipeline  (preparation + execution + analysis)")
    _common_pipeline_args(full_p)
    _run_name_arg(full_p)
    _exec_args(full_p)
    full_p.set_defaults(handler=lambda a: _run_docking_pipeline(a, full=True))

    flow_p = dock_sub.add_parser("flow", formatter_class=_Fmt,
                                 help="Run any contiguous subset of pipeline steps")
    _common_pipeline_args(flow_p)
    _run_name_arg(flow_p)
    _exec_args(flow_p)
    _step_range_args(flow_p)
    flow_p.set_defaults(handler=_run_docking_flow)

    run_p = dock_sub.add_parser("run", formatter_class=_Fmt,
                                help="Docking execution on already-prepared files")
    run_p.add_argument("prepared_receptors_dir", nargs="?", default=_REC_OUT, metavar="RECEPTOR_DIR",
                       help=f"Directory with .maps.fld files  [{_REC_OUT}]")
    run_p.add_argument("prepared_ligands_dir", nargs="?", default=_LIG_OUT, metavar="LIGAND_DIR",
                       help=f"Directory with prepared ligands  [{_LIG_OUT}]")
    run_p.add_argument("output_dir", nargs="?", default=_OUT, metavar="OUTPUT_DIR",
                       help=f"Base output directory  [{_OUT}]")
    run_p.add_argument("--workers", "-w", type=int, default=1, metavar="N",
                       help="Concurrent docking jobs  [1]  (cpu engine: cores split across these)")
    run_p.add_argument("--engine", choices=("gpu", "cpu"), default="gpu", metavar="ENGINE",
                       help="Docking engine: gpu (AutoDock-GPU) or cpu (AutoDock Vina)  [gpu]")
    run_p.add_argument("--autogrid-executable", "--autogrid", dest="autogrid_executable", metavar="EXE")
    run_p.add_argument("--autodock-gpu-executable", "--autodock", dest="autodock_gpu_executable", metavar="EXE")
    run_p.add_argument("--vina-bin", dest="vina_executable", metavar="EXE",
                       help="AutoDock Vina executable path or name  (cpu engine)")
    run_p.add_argument("--box-config", dest="box_config", metavar="PATH",
                       help="JSON receptor -> {center, npts}  (cpu engine; else GPF files are used)")
    run_p.add_argument("--exhaustiveness", type=int, default=8, metavar="N",
                       help="Vina search exhaustiveness  (cpu engine)  [8]")
    run_p.add_argument("--num-modes", dest="num_modes", type=int, default=9, metavar="N",
                       help="Max poses Vina reports per ligand  (cpu engine)  [9]")
    run_p.add_argument("--vina-cpu", dest="vina_cpu", type=int, default=None, metavar="N",
                       help="CPU threads per Vina job  (cpu engine)  [auto = cores / workers]")
    run_p.add_argument("--seed", type=int, default=None, metavar="N",
                       help="Vina random seed  (cpu engine)")
    run_p.set_defaults(handler=_run_docking_execution)

    ana_p = dock_sub.add_parser("analyze", aliases=["analysis"], formatter_class=_Fmt,
                                help="Analyze DLG results and generate reports")
    ana_p.add_argument("output_dir", help="Base output directory")
    ana_p.add_argument("--pdb-export-limit", type=int, default=10, metavar="N",
                       help="Top candidates to export as PDB  [10]")
    ana_p.add_argument("--max-workers", type=int, default=4, metavar="N",
                       help="Max threads for PDB export  [4]")
    ana_p.set_defaults(handler=_run_docking_analysis)

    rec_p = dock_sub.add_parser("receptor", formatter_class=_Fmt,
                                help="Prepare receptor PDB files to PDBQT format")
    rec_p.add_argument("input_dir", nargs="?", default=_REC_IN, metavar="INPUT_DIR",
                       help=f"Receptor PDB directory  [{_REC_IN}]")
    rec_p.add_argument("output_dir", nargs="?", default="data/input", metavar="OUTPUT_DIR",
                       help="Output base directory  [data/input]")
    rec_p.add_argument("--workers", "-w", type=int, default=1, metavar="N",
                       help="Parallel workers  [1]")
    rec_p.set_defaults(handler=_run_receptor_preparation)

    lig_p = dock_sub.add_parser("ligand", formatter_class=_Fmt,
                                help="Prepare ligand files to PDBQT format")
    lig_p.add_argument("input_dir", nargs="?", default=_LIG_IN, metavar="INPUT_DIR",
                       help=f"Ligand input directory  [{_LIG_IN}]")
    lig_p.add_argument("output_dir", nargs="?", default="data/input", metavar="OUTPUT_DIR",
                       help="Output base directory  [data/input]")
    lig_p.add_argument("--workers", "-w", type=int, default=1, metavar="N",
                       help="Parallel workers  [1]")
    lig_p.set_defaults(handler=_run_ligand_preparation)

    as_p = dock_sub.add_parser("active-site", formatter_class=_Fmt,
                               help="Detect binding sites from prepared PDBQT files")
    as_p.add_argument("receptor_dir", nargs="?", default=_REC_OUT, metavar="RECEPTOR_DIR",
                      help=f"Prepared receptors directory  [{_REC_OUT}]")
    as_p.add_argument("ligand_dir", nargs="?", default=_LIG_OUT, metavar="LIGAND_DIR",
                      help=f"Prepared ligands directory  [{_LIG_OUT}]")
    as_p.add_argument("output_dir", nargs="?", default=_OUT, metavar="OUTPUT_DIR",
                      help=f"Base output directory  [{_OUT}]")
    as_p.add_argument("--fpocket-path", "--fpocket", dest="fpocket_path", metavar="PATH",
                      help="fpocket binary path")
    as_p.add_argument("--workers", "-w", type=int, default=1, metavar="N",
                      help="Parallel workers  [1]")
    as_p.add_argument("--manual-center", nargs=3, type=float, metavar=("X", "Y", "Z"),
                      help="Manual grid center; omit --manual-npts to size from ligands")
    as_p.add_argument("--manual-npts", nargs=3, type=int, metavar=("NX", "NY", "NZ"),
                      help="Manual AutoDock grid dimensions applied to every receptor "
                           "without a --box-config entry  (requires --manual-center)")
    as_p.add_argument("--box-config", dest="box_config", metavar="PATH",
                      help="JSON file mapping receptor name -> {center, npts} for a "
                           "per-receptor grid box  (overrides --manual-center/--manual-npts "
                           "and fpocket auto-detection for the receptors it lists)")
    as_p.add_argument("--flex-radius", type=float, default=0.0, metavar="ANGSTROM",
                      help="Flex side chains within this radius of the box center  "
                           "[disabled; rigid receptor]")
    as_p.add_argument("--rigid-receptor", dest="flex_radius", action="store_const", const=0.0,
                      help="Disable automatic flexible side-chain selection")
    as_p.set_defaults(handler=_run_active_site)

    # ── dynamic ────────────────────────────────────────────────────────────────
    dyn_desc = (
        f"  {bold_cyan('dynamic')}  — Run GROMACS-based molecular dynamics simulations.\n\n"
        f"  Simulation types:\n\n"
        f"    {cyan('oprotein')}   Protein only\n"
        f"               Full simulation and analysis flow.\n\n"
        f"    {cyan('pligand')}    Protein + small-molecule ligand  (requires {bold('-c')} charge)\n"
        f"               Adds ligand topology analysis; exclusive energy analysis.\n\n"
        f"    {cyan('ppeptide')}   Protein + peptide\n"
        f"               Assembles complex first; exclusive energy interaction analysis.\n\n"
        f"    {cyan('pacid')}      Protein + nucleic acid\n"
        f"               Assembles complex first; RMSD and DNA structural fitting analysis.\n\n"
        f"    {cyan('pprotein')}   Protein + protein\n"
        f"               Assembles complex first; exclusive energy interaction analysis.\n\n"
        f"    {cyan('ppligand')}   Protein + protein + ligand  (requires {bold('-c')} charge)\n"
        f"               Combines pligand + pprotein steps; exclusive combined analysis."
    )
    parent = _dyn_parent()
    dyn = sub.add_parser("dynamic", help="Molecular dynamics simulations",
                         description=dyn_desc, formatter_class=_Fmt)
    dyn_sub = dyn.add_subparsers(dest="dyn_type", metavar="TYPE")

    _dyn_types = [
        (
            "oprotein",
            "Protein-only simulation",
            (
                f"  {bold_cyan('oprotein')}  — Protein-only GROMACS simulation.\n\n"
                f"  Runs the full simulation and analysis flow for a single protein structure.\n"
                f"  Input: 1 PDB file."
            ),
            False,
        ),
        (
            "pligand",
            "Protein + small-molecule ligand  (automatic charge detection)",
            (
                f"  {bold_cyan('pligand')}  — Protein + small-molecule ligand simulation.\n\n"
                f"  Extends the general flow with a ligand topology analysis step.\n"
                f"  The energy analysis includes exclusive modifications for protein-ligand systems.\n"
                f"  Input: 2 files (protein PDB, ligand SDF/MOL/MOL2/PDB/PDBQT).\n"
                f"  The ligand charge is detected automatically; {cyan('-c')} provides an explicit override."
            ),
            True,
        ),
        (
            "ppeptide",
            "Protein + peptide complex",
            (
                f"  {bold_cyan('ppeptide')}  — Protein + peptide complex simulation.\n\n"
                f"  The two structures are merged into a complex before the simulation starts.\n"
                f"  Post-processing adapts labels specific to peptide systems.\n"
                f"  Analysis includes an exclusive energy interaction study.\n"
                f"  Input: 2 PDB files  (protein, peptide)."
            ),
            False,
        ),
        (
            "pacid",
            "Protein + nucleic acid complex",
            (
                f"  {bold_cyan('pacid')}  — Protein + nucleic acid complex simulation.\n\n"
                f"  The two structures are merged into a complex before the simulation starts.\n"
                f"  Post-processing adapts labels specific to nucleic acid systems.\n"
                f"  Analysis shows RMSD structural variation and DNA conformational fitting plots.\n"
                f"  Input: 2 PDB files  (protein, nucleic acid)."
            ),
            False,
        ),
        (
            "pprotein",
            "Protein + protein complex",
            (
                f"  {bold_cyan('pprotein')}  — Protein + protein complex simulation.\n\n"
                f"  The two structures are merged into a complex before the simulation starts.\n"
                f"  Post-processing adapts labels specific to protein-protein systems.\n"
                f"  Analysis includes an exclusive energy interaction study.\n"
                f"  Input: 2 PDB files  (protein 1, protein 2)."
            ),
            False,
        ),
        (
            "ppligand",
            "Protein + protein + ligand  (automatic charge detection)",
            (
                f"  {bold_cyan('ppligand')}  — Protein + protein + ligand/cofactor simulation.\n\n"
                f"  Combines the steps of {cyan('pligand')} and {cyan('pprotein')}:\n"
                f"  the two proteins are merged into a complex, ligand topology is analysed,\n"
                f"  and the final analysis runs the exclusive combined protocol for this system.\n"
                f"  Input: 3 files (protein 1 PDB, protein 2 PDB, ligand SDF/MOL/MOL2/PDB/PDBQT).\n"
                f"  The ligand charge is detected automatically; {cyan('-c')} provides an explicit override."
            ),
            True,
        ),
    ]

    for name, help_text, desc, needs_charge in _dyn_types:
        dp = dyn_sub.add_parser(name, help=help_text, description=desc,
                                parents=[parent], formatter_class=_Fmt)
        if needs_charge:
            dp.add_argument("-c", "--charge", type=int, metavar="Q",
                            help="Explicit ligand net charge (overrides detection)")
            dp.add_argument("--charge-mode", choices=("prompt", "auto", "strict"),
                            default="auto",
                            help="Charge resolution policy  [auto]")
            dp.add_argument("--ligand-ph", type=float, default=7.4, metavar="PH",
                            help="pH for OpenBabel preparation of PDB/PDBQT ligands  [7.4]")
        dp.add_argument("--no-binding-energy", dest="binding_energy", action="store_false",
                        help="Skip automatic MM-GBSA/MM-PBSA analysis")
        dp.add_argument("--mmpbsa-method", choices=("gb", "pb", "both"), default="both",
                        help="Implicit-solvent binding-energy method  [both]")
        dp.add_argument("--mmpbsa-frames", type=int, default=1000, metavar="N",
                        help="Maximum uniformly sampled trajectory frames  [1000]")
        dp.add_argument("--mmpbsa-discard", type=float, default=20.0, metavar="PCT",
                        help="Initial trajectory percentage to discard  [20]")
        dp.add_argument("--mmpbsa-cores", type=_positive_int, default=6, metavar="N",
                        help="Exact number of mpirun processes for gmx_MMPBSA  [6]")
        dp.add_argument("--mmpbsa-salt", type=float, default=0.150, metavar="M",
                        help="GB/PB salt concentration in molar units  [0.150]")
        if name == "pprotein":
            dp.add_argument(
                "--force-field", default="default", metavar="NAME",
                help="GROMACS force-field name, e.g. charmm36-jul2022  [default]",
            )
            dp.add_argument(
                "--force-field-dir", metavar="DIR",
                help="Directory containing the requested NAME.ff package",
            )
            dp.add_argument(
                "--water-model", default="tip3p", metavar="NAME",
                help="GROMACS water model used with the selected field  [tip3p]",
            )
        dp.set_defaults(handler=_run_dynamic, dyn_type=name)

    resume_p = dyn_sub.add_parser(
        "resume", formatter_class=_Fmt,
        help="Resume an interrupted dynamics run from its latest checkpoint",
        description=(
            f"  {bold_cyan('resume')} — Continue an interrupted ChemLink dynamics run.\n\n"
            "  Detects NVT, NPT, production, post-processing, or analysis state;\n"
            "  validates checkpoint/TPR pairs and skips completed stages."
        ),
    )
    resume_p.add_argument("run_dir", metavar="RUN_DIR",
                          help="Existing data/output/dynamics/run_* directory")
    resume_p.add_argument("--type", dest="sim_type", choices=list(_DYN_SPEC.keys()),
                          help="Simulation type (needed only for ambiguous legacy runs)")
    resume_p.add_argument("--time", type=float, metavar="NS",
                          help="Original target duration (needed only if md.mdp/config is absent)")
    resume_p.add_argument("--dry-run", action="store_true",
                          help="Validate the run and show the detected recovery stage")
    resume_p.set_defaults(handler=_run_dynamic_resume)

    analyze_p = dyn_sub.add_parser(
        "analyze", formatter_class=_Fmt,
        help="Re-run analyses from an existing dynamics result directory",
        description=(
            f"  {bold_cyan('analyze')} — Reprocess an existing ChemLink dynamics run.\n\n"
            "  Runs general trajectory analysis, binding-energy analysis, or both\n"
            "  without repeating topology preparation or molecular dynamics."
        ),
    )
    analyze_p.add_argument("run_dir", metavar="RUN_DIR",
                           help="Existing data/output/dynamics/run_* directory")
    analyze_p.add_argument("--only", choices=("general", "binding", "all"), default="all",
                           help="Analysis component to execute  [all]")
    analyze_p.add_argument("--type", dest="sim_type", choices=list(_DYN_SPEC.keys()),
                           help="Simulation type (only for ambiguous legacy runs)")
    analyze_p.add_argument("--time", type=float, metavar="NS",
                           help="Original duration (only if absent from run configuration)")
    analyze_p.add_argument("--plot-only", action="store_true",
                           help="Regenerate general-analysis plots from existing data")
    analyze_p.add_argument("--mmpbsa-method", choices=("gb", "pb", "both"), default=None,
                           help="Override the saved binding-energy method")
    analyze_p.add_argument("--mmpbsa-frames", type=_positive_int, default=None, metavar="N",
                           help="Override maximum sampled trajectory frames")
    analyze_p.add_argument("--mmpbsa-discard", type=float, default=None, metavar="PCT",
                           help="Override initial trajectory percentage to discard")
    analyze_p.add_argument("--mmpbsa-cores", type=_positive_int, default=None, metavar="N",
                           help="Override the exact number of mpirun processes")
    analyze_p.add_argument("--mmpbsa-salt", type=float, default=None, metavar="M",
                           help="Override GB/PB salt concentration")
    analyze_p.set_defaults(handler=_run_dynamic_analyze)

    # ── doctor ─────────────────────────────────────────────────────────────────
    doc = sub.add_parser("doctor", formatter_class=_Fmt,
                         help="Check environment and tool prerequisites")
    doc.add_argument(
        "--json", action="store_true", default=False,
        help=f"Output results as JSON (doctor version: {DOCTOR_VERSION})",
    )
    doc.set_defaults(handler=_run_doctor)

    # ── completion ─────────────────────────────────────────────────────────────
    comp_desc = (
        f"  {bold_cyan('completion')}  — Generate shell tab-completion scripts.\n\n"
        f"  Quick setup:\n"
        f"    bash  →  {cyan('source <(chemlink completion bash)')}\n"
        f"    zsh   →  {cyan('source <(chemlink completion zsh)')}\n"
        f"    auto  →  {cyan('chemlink completion install')}  (appends to your shell rc)"
    )
    comp = sub.add_parser("completion", help="Generate shell tab-completion scripts",
                          description=comp_desc, formatter_class=_Fmt)
    comp.add_argument("shell", choices=["bash", "zsh", "fish", "install"], metavar="SHELL",
                      help="Target shell: bash · zsh · fish · install (auto-detect)")
    comp.set_defaults(handler=_run_completion)

    # ── hpc ────────────────────────────────────────────────────────────────────
    hpc_desc = (
        f"  {bold_cyan('hpc')}  — Submit molecular docking to a SLURM cluster.\n\n"
        f"  Automatically calculates batch size from ligand count,\n"
        f"  shows a configuration summary, then submits the full\n"
        f"  7-step pipeline (receptor prep → ligand prep → active-site\n"
        f"  → make batches → docking array → merge → analysis)."
    )
    hpc = sub.add_parser("hpc", help="Submit pipelines to SLURM cluster",
                         description=hpc_desc, formatter_class=_Fmt)
    hpc_sub = hpc.add_subparsers(dest="hpc_command", metavar="SUBCOMMAND")

    hpc_dock = hpc_sub.add_parser("docking", formatter_class=_Fmt,
                                  help="Submit full docking pipeline to SLURM")
    hpc_dock.add_argument("--ligand-dir", "--ligands", dest="ligand_dir",
                          default="/nfs/chemlink/chemlink/data/input/ligands", metavar="DIR",
                          help="Ligand input directory")
    hpc_dock.add_argument("--receptor-dir", "--receptors", dest="receptor_dir",
                          default="/nfs/chemlink/chemlink/data/input/receptors", metavar="DIR",
                          help="Receptor PDB input directory")
    hpc_dock.add_argument("--nodes", metavar="NODELIST",
                          help="SLURM nodelist to restrict submission (e.g. node[01-04])")
    hpc_dock.add_argument("--partition", metavar="PARTITION",
                          help="SLURM partition to submit to")
    hpc_dock.add_argument("--gres", metavar="GRES",
                          help="SLURM generic resource (e.g. gpu:1)")
    hpc_dock.add_argument("--batch-size", type=int, metavar="N",
                          help="Ligands per docking batch  [auto: ≤100→all, ≤500→100, ≤2000→200, >2000→500]")
    hpc_dock.add_argument("--prep-tasks", type=int, default=6, metavar="N",
                          help="Number of array tasks for receptor/ligand prep  [6]")
    hpc_dock.add_argument("--receptor-workers", type=int, default=4, metavar="N",
                          help="Parallel workers per receptor-prep task  [4]")
    hpc_dock.add_argument("--ligand-workers", type=int, default=8, metavar="N",
                          help="Parallel workers per ligand-prep task  [8]")
    hpc_dock.add_argument("--manual-center", nargs=3, type=float, metavar=("X", "Y", "Z"),
                          help="Manual grid center in Å applied to every receptor without a "
                               "--box-config entry — skips fpocket  (requires --manual-npts)")
    hpc_dock.add_argument("--manual-npts", nargs=3, type=int, metavar=("NX", "NY", "NZ"),
                          help="Manual AutoDock grid dimensions applied to every receptor "
                               "without a --box-config entry  (requires --manual-center)")
    hpc_dock.add_argument("--box-config", dest="box_config", metavar="PATH",
                          help="JSON file mapping receptor name -> {center, npts} for a "
                               "per-receptor grid box  (overrides --manual-center/--manual-npts "
                               "and fpocket auto-detection for the receptors it lists; must be "
                               "on storage reachable from the SLURM compute nodes)")
    hpc_dock.add_argument("--flex-radius", type=float, default=0.0, metavar="ANGSTROM",
                          help="Flex side chains within this radius of the box center  "
                               "[disabled; rigid receptor]")
    hpc_dock.add_argument("--rigid-receptor", dest="flex_radius", action="store_const", const=0.0,
                          help="Disable automatic flexible side-chain selection")
    hpc_dock.add_argument("--active-site-workers", type=int, default=4, metavar="N",
                          help="Parallel workers for active-site detection  [4]")
    hpc_dock.add_argument("--docking-workers", type=int, default=1, metavar="N",
                          help="Parallel workers per docking task  [1]")
    hpc_dock.add_argument("--max-gpu-concurrency", type=int, default=6, metavar="N",
                          help="Max concurrent docking array tasks  [6]")
    hpc_dock.add_argument("--mode", choices=["native", "container"], default="native",
                          help="SLURM script set to use  [native]")
    hpc_dock.add_argument("--container-image", metavar="IMAGE",
                          help="Container image path (required when --mode=container)")
    hpc_dock.add_argument("--dry-run", action="store_true",
                          help="Print configuration and command without submitting")
    _run_name_arg(hpc_dock)
    hpc_dock.set_defaults(handler=_run_hpc_docking)

    hpc_dyn = hpc_sub.add_parser("dynamics", formatter_class=_Fmt,
                                 help="Submit molecular dynamics simulations to SLURM")
    hpc_dyn.add_argument("dyn_type", choices=list(_DYN_SPEC.keys()), metavar="TYPE",
                         help="Simulation type: " + " | ".join(_DYN_SPEC.keys()))
    hpc_dyn.add_argument("-i", "--input-files", dest="input_files", action="append",
                         metavar="FILE",
                         help="Legacy ordered input file (cannot be mixed with -ir/-il)")
    hpc_dyn.add_argument("-ir", "--input-receptors", "--protein", dest="input_receptors",
                         action="append", metavar="FILE",
                         help="Receptor/protein input file; --protein is a compatible alias")
    hpc_dyn.add_argument("-il", "--input-ligands", dest="input_ligands",
                         action="append", metavar="FILE",
                         help="Ligand or partner input file (repeat for multi-ligand pligand)")
    hpc_dyn.add_argument("--ligands", dest="ligand_files", nargs="+", metavar="PDB",
                         help="Compatible multi-ligand alias for repeated -il")
    hpc_dyn.add_argument("--time", "-t", type=float, default=100.0, metavar="NS",
                         help="Simulation time in nanoseconds  [100.0]")
    hpc_dyn.add_argument("--charge", type=int, default=None, metavar="Q",
                         help="Explicit ligand net charge (overrides detection)")
    hpc_dyn.add_argument("--charge-mode", choices=("auto", "strict"), default="strict",
                         help="Non-interactive charge policy  [strict]")
    hpc_dyn.add_argument("--ligand-ph", type=float, default=7.4, metavar="PH",
                         help="pH for OpenBabel preparation of PDB/PDBQT ligands  [7.4]")
    hpc_dyn.add_argument("--no-binding-energy", dest="binding_energy", action="store_false",
                         help="Skip automatic MM-GBSA/MM-PBSA analysis")
    hpc_dyn.add_argument("--mmpbsa-method", choices=("gb", "pb", "both"), default="both",
                         help="Implicit-solvent binding-energy method  [both]")
    hpc_dyn.add_argument("--mmpbsa-frames", type=int, default=1000, metavar="N",
                         help="Maximum uniformly sampled trajectory frames  [1000]")
    hpc_dyn.add_argument("--mmpbsa-discard", type=float, default=20.0, metavar="PCT",
                         help="Initial trajectory percentage to discard  [20]")
    hpc_dyn.add_argument("--mmpbsa-cores", type=_positive_int, default=6, metavar="N",
                         help="MPI processes for gmx_MMPBSA  [6]")
    hpc_dyn.add_argument("--mmpbsa-salt", type=float, default=0.150, metavar="M",
                         help="GB/PB salt concentration in molar units  [0.150]")
    hpc_dyn.add_argument("--nodes", metavar="NODELIST",
                         help="SLURM nodelist (comma-separated) — distributes jobs across nodes")
    hpc_dyn.add_argument("--partition", metavar="PARTITION",
                         help="SLURM partition to submit to")
    hpc_dyn.add_argument("--time-limit", dest="time_limit", default="72:00:00",
                         metavar="HH:MM:SS",
                         help="Wall time per job  [72:00:00]")
    hpc_dyn.add_argument("--mem", default="0", metavar="MEM",
                         help="Memory per job — use '0' for all available on the node  [0]")
    hpc_dyn.add_argument("--cpus", type=int, default=8, metavar="N",
                         help="CPUs per job  [8]")
    hpc_dyn.add_argument("--replicate", action="store_true",
                         help="Submit the same simulation to every node in --nodes (benchmark/comparison mode)")
    hpc_dyn.add_argument("--mpi", action="store_true",
                         help="Single simulation across all --nodes via MPI domain decomposition (test 1 GbE overhead)")
    hpc_dyn.add_argument("--dry-run", action="store_true",
                         help="Print configuration without submitting jobs")
    _run_name_arg(hpc_dyn)
    hpc_dyn.set_defaults(handler=_run_hpc_dynamics)

    return parser


# ── Entry point ────────────────────────────────────────────────────────────────

def _print_banner() -> None:
    logo = textwrap.dedent(LOGO).strip()
    console.print(Text(logo, style="cyan"), justify="center")
    console.print()
    console.rule(
        f"[bold cyan]ChemLink[/]  [dim]v{VERSION}[/]  [white]Molecular Simulation Toolkit[/]",
        style="cyan",
    )
    console.print(
        "[dim]Docking · Molecular Dynamics · Analysis[/]\n",
        justify="center",
    )


def main() -> int:
    try:
        import argcomplete  # type: ignore
        _has_argcomplete = True
    except ImportError:
        _has_argcomplete = False

    parser = build_parser()

    if _has_argcomplete:
        argcomplete.autocomplete(parser)

    if len(sys.argv) == 1:
        _print_banner()
        parser.print_help()
        return 0

    if sys.argv[1] in ("-h", "--help"):
        _print_banner()

    args = parser.parse_args()

    if not hasattr(args, "handler"):
        parser.parse_args([args.command, "--help"])
        return 0

    try:
        return args.handler(args) or 0
    except KeyboardInterrupt:
        err_console.print(f"\n[yellow]Interrupted.[/]")
        return 130
    except Exception as exc:
        err_console.print(f"\n[bold red]Error:[/] {exc}")
        if os.environ.get("CHEMLINK_DEBUG"):
            import traceback
            traceback.print_exc()
        else:
            err_console.print(f"  [dim]Set CHEMLINK_DEBUG=1 for a full traceback.[/]\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
