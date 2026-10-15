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
import time
from argparse import SUPPRESS, ArgumentParser, ArgumentTypeError, HelpFormatter, Namespace
from datetime import datetime, timezone
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
from rich.console import Group
from rich.padding import Padding
from rich.table import Table
from rich.markup import escape
from rich.text import Text

try:
    from ..utils.style import (
        GLYPH, STYLE, error_panel, indent, kv_block, print_card,
        record_block, section, stats_block, status_block, status_line,
    )
    from ..utils.theme import (
        BRAND, CYAN, format_duration, get_console, get_err_console,
        gradient_text, set_color_enabled,
    )
except ImportError:
    from utils.style import (  # type: ignore
        GLYPH, STYLE, error_panel, indent, kv_block, print_card,
        record_block, section, stats_block, status_block, status_line,
    )
    from utils.theme import (  # type: ignore
        BRAND, CYAN, format_duration, get_console, get_err_console,
        gradient_text, set_color_enabled,
    )

try:
    from ..utils.progress import busy, checklist
except ImportError:
    from utils.progress import busy, checklist  # type: ignore

try:
    from .banner import TAGLINE
except ImportError:
    from banner import TAGLINE  # type: ignore

try:
    from . import workspace
except ImportError:
    import workspace  # type: ignore

console     = get_console()
err_console = get_err_console()

# ── ANSI helpers (argparse descriptions are plain strings) ─────────────────────
# These emit the theme's true-colour codes; the Rich help renderer and the TUI
# convert them back into styled text with Text.from_ansi().
_TTY = (
    sys.stdout.isatty()
    and not os.environ.get("NO_COLOR")
    and "--no-color" not in sys.argv
)


def _rgb(hex_color: str) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return f"38;2;{r};{g};{b}"


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _TTY else text

def bold(t: str) -> str:        return _c("1",    t)
def dim(t: str) -> str:         return _c("2",    t)
def cyan(t: str) -> str:        return _c(_rgb(CYAN), t)
def bold_cyan(t: str) -> str:   return _c("1;" + _rgb(BRAND), t)
def bold_red(t: str) -> str:    return _c("1;31", t)
def yellow(t: str) -> str:      return _c("33",   t)


VERSION        = "1.0.0"
DOCTOR_VERSION = "1.2"

PIPELINE_STEPS = ("receptor", "ligand", "active-site", "execution", "analysis")


# ── Output helpers ─────────────────────────────────────────────────────────────

def print_ok(msg: str)   -> None: console.print(status_line("ok", msg), soft_wrap=True)
def print_err(msg: str)  -> None: console.print(status_line("err", msg), soft_wrap=True)
def print_warn(msg: str) -> None: console.print(status_line("warn", msg), soft_wrap=True)
def print_info(msg: str) -> None: console.print(status_line("info", msg), soft_wrap=True)


def print_error(title: str, message, *, hint_text: Optional[str] = None, details: Optional[str] = None) -> None:
    """A boxed, red error on stderr: what failed, why, and what to try next."""
    err_console.print()
    err_console.print(Padding(error_panel(title, message, hint_text=hint_text, details=details), (0, 0, 0, 2)))
    err_console.print()


def print_ligand_warning_summary(stats: dict) -> None:
    """Print a deferred ligand-warning summary after the command's final results."""
    count = stats.get("warnings", 0)
    report = stats.get("warning_report")
    if count and report:
        print_warn(f"{count} ligand warning(s) — details in [cl.path]{report}[/]")


def visible_ligand_stats(stats: dict) -> dict:
    """Hide deferred-display metadata from result tables."""
    return {key: value for key, value in stats.items() if key != "warning_report"}


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise ArgumentTypeError("must be an integer greater than zero")
    return parsed


def print_stats(stats: dict, title: str = "Results") -> None:
    stats_block(console, stats, title=title)


# ── Help rendering ─────────────────────────────────────────────────────────────

class _Fmt(HelpFormatter):
    """Plain-text formatter — used for ``format_help()`` (non-TTY consumers);
    interactive ``--help`` goes through the Rich renderer below."""

    def __init__(self, prog: str):
        super().__init__(prog, max_help_position=32, width=88)

    def _fill_text(self, text: str, width: int, indent: str) -> str:
        return "\n".join(indent + line for line in text.splitlines())

    def start_section(self, heading: Optional[str]) -> None:
        super().start_section(heading.upper() if heading else heading)

    def _format_usage(self, usage, actions, groups, prefix) -> str:
        return super()._format_usage(usage, actions, groups, "Usage: " if prefix is None else prefix)


_COMMAND_GROUPS = {
    "Pipelines": ("docking", "peptide-docking", "dynamic"),
    "Cluster":   ("hpc",),
    "Tools":     ("doctor", "demo", "update", "completion"),
}

_DEFAULT_RE = re.compile(r"(\s*\[[^\[\]]*\])")


def _ansi_text(text: str) -> Text:
    return Text.from_ansi(text) if "\x1b[" in text else Text(text)


def _help_text(help_str: str) -> Text:
    """Help text with trailing ``[default]`` brackets muted."""
    text = _ansi_text(help_str)
    for m in _DEFAULT_RE.finditer(text.plain):
        text.stylize("cl.muted", m.start(), m.end())
    return text


def _invocation(action) -> Text:
    out = Text()
    if action.option_strings:
        for i, opt in enumerate(action.option_strings):
            if i:
                out.append(", ", style="cl.faint")
            out.append(opt, style="cl.ref")
        if action.nargs != 0:
            metavar = action.metavar or action.dest.upper()
            if isinstance(metavar, tuple):
                metavar = " ".join(metavar)
            elif action.choices and not action.metavar:
                metavar = "{" + ",".join(map(str, action.choices)) + "}"
            out.append(f" {metavar}", style="cl.muted")
    else:
        out.append(str(action.metavar or action.dest).upper() if action.metavar is None else str(action.metavar), style="cl.accent")
    return out


def _sub_action(parser):
    import argparse
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return action
    return None


def render_help(parser) -> Group:
    """The ``--help`` screen: header, usage, commands, arguments, options, examples."""
    import argparse
    parts: list = []
    words = parser.prog.split()
    is_root = len(words) == 1
    title = Text("  ")
    title.append(f"{GLYPH['brand']} ", style="cl.brand")
    title.append_text(gradient_text(words[0], bold=True))
    if len(words) > 1:
        title.append(" " + " ".join(words[1:]), style="bold")
    title.append(f"   v{VERSION}", style="cl.faint")
    if not is_root:  # the root's banner already shows name, version and tagline
        parts.append(title)

    description = (parser.description or "").strip("\n")
    if description and not is_root:
        parts.append(Padding(_ansi_text(textwrap.dedent(description)), (1, 0, 0, 2)))

    sub = _sub_action(parser)
    positionals = [a for a in parser._actions if not a.option_strings and not isinstance(a, argparse._SubParsersAction) and a.help != argparse.SUPPRESS]
    optionals = [a for a in parser._actions if a.option_strings and a.help != argparse.SUPPRESS]

    usage = Text("  ")
    usage.append(parser.prog, style="bold")
    if sub is not None:
        usage.append(" <command>", style="cl.accent")
    if optionals:
        usage.append(" [options]", style="cl.muted")
    for a in positionals:
        name = str(a.metavar or a.dest)
        usage.append(f" [{name}]" if a.nargs in ("?", "*") else f" {name}", style="cl.accent")

    def heading(label: str) -> Text:
        return Text(f"\n  {label.upper()}", style="cl.brand")

    parts.append(heading("Usage"))
    parts.append(Padding(usage, (0, 0, 0, 2)))

    def grid() -> Table:
        g = Table.grid(padding=(0, 3))
        g.add_column(no_wrap=True, min_width=14, max_width=38)
        g.add_column(overflow="fold")
        return g

    if sub is not None:
        choices = [c for c in sub._choices_actions if c.help != argparse.SUPPRESS]
        by_name = {c.dest: c for c in choices}
        groups = []
        if parser.prog == "chemlink":
            seen = set()
            for label, names in _COMMAND_GROUPS.items():
                members = [by_name[n] for n in names if n in by_name]
                seen.update(n for n in names if n in by_name)
                if members:
                    groups.append((label, members))
            rest = [c for c in choices if c.dest not in seen]
            if rest:
                groups.append(("Commands", rest))
        else:
            groups.append(("Commands", choices))
        for label, members in groups:
            parts.append(heading(label))
            g = grid()
            for c in members:
                name = Text(str(c.metavar or c.dest), style="cl.ref")
                g.add_row(name, _help_text(c.help or ""))
            parts.append(Padding(g, (0, 0, 0, 4)))

    if positionals:
        parts.append(heading("Arguments"))
        g = grid()
        for a in positionals:
            g.add_row(_invocation(a), _help_text(a.help or ""))
        parts.append(Padding(g, (0, 0, 0, 4)))

    if optionals:
        parts.append(heading("Options"))
        g = grid()
        for a in optionals:
            g.add_row(_invocation(a), _help_text(a.help or ""))
        parts.append(Padding(g, (0, 0, 0, 4)))

    if parser.epilog:
        parts.append(Padding(_ansi_text(parser.epilog.strip("\n")), (1, 0, 0, 0)))

    footer = Text("\n  ")
    if sub is not None:
        footer.append(f"{parser.prog} <command> --help", style="cl.ref")
        footer.append("  for details on a command", style="cl.hint")
    else:
        footer.append("Docs: ", style="cl.hint")
        footer.append("docs/Desarrollo.md", style="cl.ref")
    parts.append(footer)
    parts.append(Text(""))
    return Group(*parts)


# ── Custom parser: Rich help, helpful errors ──────────────────────────────────

_TOP_CMDS = ["docking", "peptide-docking", "dynamic", "doctor", "demo", "completion", "update", "hpc"]


def _suggest(word: str, choices=None) -> list:
    import difflib
    if not word or word.startswith("-"):
        return []
    pool = list(choices or _TOP_CMDS)
    close = difflib.get_close_matches(word, pool, n=3, cutoff=0.5)
    prefixed = [c for c in pool if c.startswith(word) or word in c]
    return list(dict.fromkeys(close + prefixed))[:3]


class CLIUsageError(Exception):
    """Raised instead of printing when ``_Parser.raise_errors`` is set (the TUI)."""


class _Parser(ArgumentParser):
    #: The TUI sets this so validation errors come back as exceptions it can
    #: show inline, instead of a panel printed under its full-screen frame.
    raise_errors = False
    last_error: Optional[str] = None

    def print_help(self, file=None) -> None:  # noqa: D401 — argparse API
        if file is None or file is sys.stdout:
            console.print(render_help(self))
        else:
            super().print_help(file)

    #: argv being parsed (set by ``main``) — lets errors raised on the root
    #: parser find the subcommand the user was actually typing.
    current_argv: list = []

    def _leaf(self) -> "ArgumentParser":
        parser = self
        for word in _Parser.current_argv:
            sub = _sub_action(parser)
            if sub is None or word not in sub.choices:
                break
            parser = sub.choices[word]
        return parser

    def error(self, message: str) -> None:
        import difflib
        _Parser.last_error = message
        if _Parser.raise_errors:
            raise CLIUsageError(message)
        leaf = self._leaf() if self.prog == "chemlink" else self
        title, text, tips = "Invalid command line", message[:1].upper() + message[1:], []
        m = re.search(r"invalid choice: '([^']*)' \(choose from (.*)\)", message)
        if m:
            bad, choices = m.group(1), re.findall(r"'([^']*)'", m.group(2))
            title = "Unknown command" if "COMMAND" in message else "Invalid choice"
            arg = re.search(r"argument ([^:]+): invalid choice", message)
            prefix = f"{arg.group(1).split('/')[-1]}: " if arg and "COMMAND" not in arg.group(1) else ""
            text = f"{prefix}'{bad}' is not one of: " + ", ".join(choices)
            matches = difflib.get_close_matches(bad, choices, n=2, cutoff=0.6) or _suggest(bad, choices)[:2]
            if matches:
                tips.append("Did you mean " + " or ".join(f"[cl.ref]{x}[/]" for x in matches) + "?")
        m = re.search(r"unrecognized arguments: (.*)", message)
        if m:
            title = "Unknown option"
            known = [o for o in leaf._option_string_actions if o.startswith("--")]
            for bad in m.group(1).split():
                if bad.startswith("-"):
                    close = difflib.get_close_matches(bad.split("=")[0], known, n=1, cutoff=0.6)
                    if close:
                        tips.append(f"Did you mean [cl.ref]{close[0]}[/] instead of [cl.ref]{bad}[/]?")
        if "required" in message and "arguments are required" in message:
            title = "Missing required argument"
        tips.append(f"See [cl.ref]{leaf.prog} --help[/]")
        print_error(title, text, hint_text="\n".join(tips))
        sys.exit(2)


# ── Misc helpers ───────────────────────────────────────────────────────────────

def _manual_params(args: Namespace) -> Tuple[Optional[tuple], Optional[tuple]]:
    if args.manual_npts is not None and args.manual_center is None:
        print_error("Invalid grid box", "--manual-npts requires --manual-center.",
                    hint_text="Pass [cl.ref]--manual-center X Y Z[/] as well, or drop [cl.ref]--manual-npts[/].")
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


def _docking_inputs(args: Namespace) -> Tuple["workspace.InputSource", "workspace.InputSource", str]:
    """Resolve docking inputs and the output base once at CLI entry.

    Omitted paths come from the working directory (cli/workspace.py); the
    resolved absolute paths are passed through every pipeline stage so a run
    always keeps tracking the inputs it started from.
    """
    receptors = workspace.find_inputs("receptors", getattr(args, "receptors", None))
    ligands = workspace.find_inputs("ligands", getattr(args, "ligands", None))
    return receptors, ligands, workspace.output_base("docking", getattr(args, "output", None))


def _prepared_dirs(args: Namespace) -> Tuple[str, str, str]:
    """Prepared receptor/ligand directories and the output base for the
    steps that start from PDBQT files (active-site, run)."""
    output = workspace.output_base("docking", getattr(args, "output", None))

    def pick(explicit: Optional[str], name: str) -> str:
        return os.path.abspath(os.path.expanduser(explicit)) if explicit else os.path.join(output, name)

    return (pick(getattr(args, "receptors", None), "prepared_receptors_pdbqt"),
            pick(getattr(args, "ligands", None), "prepared_ligands_pdbqt"),
            output)


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
    source = workspace.find_inputs("receptors", args.receptors)
    output = workspace.output_base("docking", args.output)
    step = ReceptorPreparation(
        input_path=source.materialize(output),
        output_path=output,
        keep_cofactors=getattr(args, "keep_cofactors", False),
    )
    print_stats(step.prepare(n_workers=args.workers), "Receptor Preparation")
    return 0


def _run_ligand_preparation(args: Namespace) -> int:
    try:
        from ..pipelines.docking.steps import LigandPreparation
    except ImportError:
        from chemlink.pipelines.docking.steps import LigandPreparation  # type: ignore
    source = workspace.find_inputs("ligands", args.ligands)
    output = workspace.output_base("docking", args.output)
    step = LigandPreparation(input_path=source.materialize(output), output_path=output)
    stats = step.prepare(n_workers=args.workers)
    print_stats(visible_ligand_stats(stats), "Ligand Preparation")
    print_ligand_warning_summary(stats)
    return 0


def _run_active_site(args: Namespace) -> int:
    try:
        from ..pipelines.docking.steps import ActiveSiteDetection, load_box_config
    except ImportError:
        from chemlink.pipelines.docking.steps import ActiveSiteDetection, load_box_config  # type: ignore
    center, npts = _manual_params(args)
    box_config = getattr(args, "box_config", None)
    manual_boxes = load_box_config(box_config) if box_config else None
    receptor_dir, ligand_dir, output = _prepared_dirs(args)
    redocking = getattr(args, "redocking_center", None)
    source_receptors = None
    if redocking:
        source_receptors = workspace.find_inputs(
            "receptors", None if redocking == "auto" else redocking
        ).directory
    step = ActiveSiteDetection(
        receptor_path=receptor_dir,
        ligand_path=ligand_dir,
        output_path=output,
        manual_center=center,
        manual_npts=npts,
        manual_boxes=manual_boxes,
        flex_radius=getattr(args, "flex_radius", 0.0),
        receptor_input_path=source_receptors,
        ligand_box_padding=getattr(args, "ligand_box_padding", 10.0),
        auto_cocrystal_center=bool(redocking),
    )
    print_stats(step.prepare(n_workers=args.workers), "Active Site Detection")
    return 0


def _run_docking_execution(args: Namespace) -> int:
    workers = args.workers
    engine = getattr(args, "engine", "both")
    receptor_dir, ligand_dir, output_dir = _prepared_dirs(args)

    try:
        from ..pipelines.docking.steps import DockingExecution, VinaExecution
        from ..pipelines.docking.docking_pipeline import DockingPipeline
    except ImportError:
        from chemlink.pipelines.docking.steps import DockingExecution, VinaExecution  # type: ignore
        from chemlink.pipelines.docking.docking_pipeline import DockingPipeline  # type: ignore

    def _cpu_step(output_path: str) -> VinaExecution:
        return VinaExecution(
            receptor_dir=receptor_dir,
            ligand_dir=ligand_dir,
            output_path=output_path,
            box_config_path=getattr(args, "box_config", None),
            exhaustiveness=getattr(args, "exhaustiveness", 8),
            num_modes=getattr(args, "num_modes", 9),
            cpu_per_job=getattr(args, "vina_cpu", None),
            seed=getattr(args, "seed", None),
            # Shared prep lives next to output_dir even in engine="both",
            # where output_path here may be the isolated cpu_validation/ folder.
            clean_receptors_dir=os.path.join(output_dir, "clean_receptors"),
        )

    def _gpu_step(output_path: str) -> DockingExecution:
        return DockingExecution(
            protein_maps_dir=receptor_dir,
            ligand_dir=ligand_dir,
            output_path=output_path,
            run_count=getattr(args, "gpu_nrun", 50),
            nev=getattr(args, "gpu_nev", 2_500_000),
            timeout_seconds=getattr(args, "gpu_timeout", 600),
        )

    if engine == "both":
        from concurrent.futures import ThreadPoolExecutor

        # Isolated subfolder so Vina's docking_config.txt/log don't overwrite the GPU
        # run's, which is also what makes running both at once (different hardware:
        # GPU device vs. CPU cores) safe instead of racing on the same files.
        cpu_output = os.path.join(output_dir, "cpu_validation")
        with ThreadPoolExecutor(max_workers=2) as executor:
            gpu_future = executor.submit(_gpu_step(output_dir).run, n_workers=workers)
            cpu_future = executor.submit(_cpu_step(cpu_output).run, n_workers=workers)
            gpu_stats, gpu_error = DockingPipeline._future_result(gpu_future)
            cpu_stats, cpu_error = DockingPipeline._future_result(cpu_future)

        if gpu_error is not None and cpu_error is not None:
            print_err(f"Both GPU and CPU docking failed. GPU: {gpu_error}. CPU: {cpu_error}.")
            return 1

        comparison_file = DockingPipeline._write_dual_engine_comparison(
            output_dir, output_dir, cpu_output
        )
        print_stats(
            {
                "gpu": gpu_stats if gpu_error is None else {"error": str(gpu_error)},
                "cpu": cpu_stats if cpu_error is None else {"error": str(cpu_error)},
            },
            "Docking Execution (GPU + CPU, parallel)",
        )
        print_ok(f"Comparación GPU vs CPU: {comparison_file}")
        return 0

    if engine == "cpu":
        start = time.monotonic()
        with checklist() as cl:
            stats = _cpu_step(output_dir).run(n_workers=workers, checklist=cl)
            cl.complete(f"Docking complete in {format_duration(time.monotonic() - start)}")
        print_stats(stats, "Docking Execution (Vina / CPU)")
        return 0

    start = time.monotonic()
    with checklist() as cl:
        stats = _gpu_step(output_dir).run(n_workers=workers, checklist=cl)
        cl.complete(f"Docking complete in {format_duration(time.monotonic() - start)}")
    print_stats(stats, "Docking Execution")
    return 0


def _run_docking_analysis(args: Namespace) -> int:
    try:
        from ..pipelines.docking.steps import DockingAnalysis
    except ImportError:
        from chemlink.pipelines.docking.steps import DockingAnalysis  # type: ignore
    run_dir = workspace.find_run_dir(args.run_dir, "docking", ("docking_results", "docking_inputs.json"))
    step = DockingAnalysis(output_path=run_dir, pdb_export_limit=args.pdb_export_limit)
    results = step.run()
    print_stats(
        {"Poses analyzed": results["parsed_poses"], "Ligands analyzed": results["analyzed_ligands"]},
        "Docking Analysis",
    )
    outputs = results.get("outputs", {})
    if outputs:
        stats_block(console, {name: path for name, path in outputs.items()}, title="Output files")
    else:
        print_warn(f"No DLG files found under [cl.path]{run_dir}/docking_results/**/dlg/*.dlg[/]")
    return 0


def _selected_pipeline_steps(from_step: str, to_step: str) -> Tuple[str, ...]:
    return PIPELINE_STEPS[PIPELINE_STEPS.index(from_step):PIPELINE_STEPS.index(to_step) + 1]


def _docking_pipeline(args: Namespace, run_output: str, receptor_src, ligand_src):
    """DockingPipeline for ``run_output``; loose working-directory inputs are
    linked into the run first."""
    try:
        from ..pipelines.docking import DockingPipeline
    except ImportError:
        from chemlink.pipelines.docking import DockingPipeline  # type: ignore
    center, npts = _manual_params(args)
    receptor_input = receptor_src.materialize(run_output)
    ligand_input = ligand_src.materialize(run_output)
    _write_docking_input_config(run_output, receptor_input, ligand_input, getattr(args, "run_name", None))
    return DockingPipeline(
        receptor_input_path=receptor_input,
        ligand_input_path=ligand_input,
        output_path=run_output,
        manual_center=center,
        manual_npts=npts,
        box_config_path=getattr(args, "box_config", None),
        flex_radius=getattr(args, "flex_radius", 0.0),
        ligand_box_padding=getattr(args, "ligand_box_padding", 10.0),
        auto_cocrystal_center=getattr(args, "redocking_center", False),
        keep_cofactors=getattr(args, "keep_cofactors", False),
    )


def _run_docking_flow(args: Namespace) -> int:
    _manual_params(args)
    receptor_src, ligand_src, output_dir = _docking_inputs(args)
    run_output = _run_dir(output_dir, getattr(args, "run_name", None))
    pipeline = _docking_pipeline(args, run_output, receptor_src, ligand_src)
    requested_steps = _selected_pipeline_steps(args.from_step, args.to_step)
    engine = getattr(args, "engine", "both")
    # A shared checklist covers active-site + execution -- the two phases that
    # otherwise print almost nothing today. Skipped for the dual-engine "both"
    # path (two threads animating the same Checklist isn't a combination this
    # design covers), and skipped entirely when neither phase is in range.
    use_checklist = engine != "both" and (
        "active-site" in requested_steps or "execution" in requested_steps
    )
    start = time.monotonic()
    cl = checklist() if use_checklist else None
    if cl is not None:
        cl.__enter__()
    try:
        result = pipeline.run_step_range(
            from_step=args.from_step,
            to_step=args.to_step,
            receptor_workers=args.receptor_workers,
            ligand_workers=args.ligand_workers,
            active_site_workers=args.active_site_workers,
            docking_workers=args.docking_workers,
            pdb_export_limit=args.pdb_export_limit,
            engine=engine,
            exhaustiveness=getattr(args, "exhaustiveness", 8),
            num_modes=getattr(args, "num_modes", 9),
            vina_cpu=getattr(args, "vina_cpu", 1),
            seed=getattr(args, "seed", None),
            gpu_nrun=getattr(args, "gpu_nrun", 50),
            gpu_nev=getattr(args, "gpu_nev", 2_500_000),
            gpu_timeout=getattr(args, "gpu_timeout", 600),
            checklist=cl,
        )
    finally:
        if cl is not None:
            cl.__exit__(None, None, None)
    if cl is not None:
        verb = "Campaign" if "execution" in result.executed_steps else "Preparation"
        cl.complete(f"{verb} complete in {format_duration(time.monotonic() - start)}")
    stats: dict = {"Run directory": run_output, "Steps": ", ".join(result.executed_steps)}
    if "receptor"    in result.executed_steps: stats["Receptors"]         = result.receptor_preparation
    if "ligand"      in result.executed_steps: stats["Ligands"]           = visible_ligand_stats(result.ligand_preparation)
    if "active-site" in result.executed_steps: stats["Active sites"]      = result.active_site_detection
    if "execution"   in result.executed_steps and result.docking_execution:
        stats["Docking execution"] = result.docking_execution
    if "analysis"    in result.executed_steps and result.docking_analysis:
        stats["Docking analysis"]  = result.docking_analysis
    print_stats(stats, "Pipeline Results")
    print_ligand_warning_summary(result.ligand_preparation)
    return 0


def _run_docking_pipeline(args: Namespace, *, full: bool) -> int:
    _manual_params(args)
    receptor_src, ligand_src, output_dir = _docking_inputs(args)
    run_name = getattr(args, "run_name", None)
    out = _run_dir(output_dir, run_name) if (full or run_name) else output_dir
    pipeline = _docking_pipeline(args, out, receptor_src, ligand_src)
    engine = getattr(args, "engine", "both")
    # "prepare" only ever runs active-site (no execution, so no dual-engine thread
    # concern); "full" also runs execution, so the checklist is skipped for the
    # dual-engine "both" path the same way _run_docking_flow skips it.
    use_checklist = (engine != "both") if full else True
    start = time.monotonic()
    cl = checklist() if use_checklist else None
    if cl is not None:
        cl.__enter__()
    try:
        result = (
            pipeline.run_full_pipeline(
                receptor_workers=args.receptor_workers,
                ligand_workers=args.ligand_workers,
                active_site_workers=args.active_site_workers,
                docking_workers=args.docking_workers,
                pdb_export_limit=args.pdb_export_limit,
                engine=engine,
                exhaustiveness=getattr(args, "exhaustiveness", 8),
                num_modes=getattr(args, "num_modes", 9),
                vina_cpu=getattr(args, "vina_cpu", None),
                seed=getattr(args, "seed", None),
                gpu_nrun=getattr(args, "gpu_nrun", 50),
                gpu_nev=getattr(args, "gpu_nev", 2_500_000),
                gpu_timeout=getattr(args, "gpu_timeout", 600),
                checklist=cl,
            )
            if full
            else pipeline.run_preparation_pipeline(
                receptor_workers=args.receptor_workers,
                ligand_workers=args.ligand_workers,
                active_site_workers=args.active_site_workers,
                checklist=cl,
            )
        )
    finally:
        if cl is not None:
            cl.__exit__(None, None, None)
    if cl is not None:
        verb = "Campaign" if full else "Preparation"
        cl.complete(f"{verb} complete in {format_duration(time.monotonic() - start)}")
    stats: dict = {}
    if full or run_name:
        stats["Run directory"] = out
    stats.update({
        "Receptors":    result.receptor_preparation,
        "Ligands":      visible_ligand_stats(result.ligand_preparation),
        "Active sites": result.active_site_detection,
    })
    if result.docking_execution: stats["Docking execution"] = result.docking_execution
    if result.docking_analysis:  stats["Docking analysis"]  = result.docking_analysis
    print_stats(stats, "Pipeline Results")
    print_ligand_warning_summary(result.ligand_preparation)
    return 0


def _peptide_inputs(args: Namespace, *, restraints_required: bool):
    """Peptide directory, receptor PDB, restraints file and output base,
    resolved from the working directory when not given (cli/workspace.py)."""
    peptides = workspace.find_inputs("peptides", args.peptides)
    receptor = workspace.find_receptor_file(args.receptor)
    restraints = workspace.find_restraints(args.restraints_file, required=restraints_required)
    return peptides, receptor, restraints, workspace.output_base("peptide-docking", args.output)


def _run_peptide_docking(args: Namespace) -> int:
    try:
        from ..pipelines.docking.peptide_docking import PeptideDockingPipeline
    except ImportError:
        from chemlink.pipelines.docking.peptide_docking import PeptideDockingPipeline  # type: ignore

    peptides, receptor, restraints, output = _peptide_inputs(args, restraints_required=False)
    out = _run_dir(output, getattr(args, "run_name", None))
    pipeline = PeptideDockingPipeline(
        peptide_input_path=peptides.materialize(out),
        receptor_pdb=receptor,
        output_path=out,
        restraints_file=restraints,
        gso_steps=args.gso_steps,
        glowworms=args.glowworms,
        anm_lig=args.anm_lig,
        cores_per_job=args.cores_per_job,
        swarms=args.swarms,
        standardize=args.standardize,
        dedup=args.dedup,
    )
    result = pipeline.run_full_pipeline(
        peptide_workers=args.prep_workers,
        docking_workers=args.docking_workers,
        reference_cids=set(getattr(args, "reference_cids", None) or []),
    )
    stats = {
        "Run directory": out,
        "Peptide preparation": result.peptide_preparation,
        "LightDock execution": result.lightdock_execution,
    }
    print_stats(stats, "Peptide Docking Results")
    return 0


def _import_robust_pipeline():
    try:
        from ..pipelines.docking.peptide_docking import PeptideDockingRobustPipeline
    except ImportError:
        from chemlink.pipelines.docking.peptide_docking import PeptideDockingRobustPipeline  # type: ignore
    return PeptideDockingRobustPipeline


def _robust_new_run_dir(args: Namespace, output: str) -> str:
    """Create (or, with --resume, reuse) the run directory 'lightdock' writes into.

    Deliberate exception to _run_dir's fail-if-exists convention: this
    pipeline runs for hours across three external tools, so re-running the
    same --run-name must resume rather than refuse to start. Without
    --resume it behaves like every other command (fails if the name is
    already used).
    """
    run_name = getattr(args, "run_name", None)
    if args.resume and run_name:
        out = os.path.join(output, validate_run_name(run_name))
        os.makedirs(out, exist_ok=True)
        return out
    return _run_dir(output, run_name)


def _robust_run_dir(args: Namespace) -> str:
    """Existing run for a per-stage `robust` subcommand."""
    PeptideDockingRobustPipeline = _import_robust_pipeline()
    return workspace.find_run_dir(
        args.run_dir, "peptide-docking", (PeptideDockingRobustPipeline.CONFIG_FILENAME,)
    )


def _robust_pipeline_kwargs(args: Namespace, peptides, receptor: str, restraints: str, out: str) -> dict:
    return dict(
        peptide_input_path=peptides.materialize(out),
        receptor_pdb=receptor,
        output_path=out,
        restraints_file=restraints,
        reference_cid=args.reference_cid,
        max_candidates=args.max_candidates,
        gso_steps=args.gso_steps,
        glowworms=args.glowworms,
        anm_lig=args.anm_lig,
        cores_per_job=args.cores_per_job,
        swarms=args.swarms,
        standardize=args.standardize,
        dedup=args.dedup,
    )


def _run_peptide_docking_robust_lightdock(args: Namespace) -> int:
    """Stage 1+2: LightDock triage + gate. Creates the run and writes
    robust_config.json (see PeptideDockingRobustPipeline._write_config) so
    every later `robust <stage>` subcommand can find this run on its own."""
    PeptideDockingRobustPipeline = _import_robust_pipeline()
    peptides, receptor, restraints, output = _peptide_inputs(args, restraints_required=True)
    out = _robust_new_run_dir(args, output)

    pipeline = PeptideDockingRobustPipeline(
        **_robust_pipeline_kwargs(args, peptides, receptor, restraints, out)
    )
    pipeline.run_triage(peptide_workers=args.prep_workers, docking_workers=args.docking_workers)
    gate_stats = pipeline.run_gate()

    print_stats(
        {"Run directory": out, "Gate candidates": gate_stats.get("gate_candidates"),
         "Gate CSV": gate_stats.get("csv")},
        "Peptide Docking Robust -- LightDock Triage + Gate",
    )
    if gate_stats.get("gate_candidates", 0) <= 0:
        print_warn(
            "No candidate beat the reference on both score and site distance — there is "
            "nothing yet for audit/prodigy/haddock3/rosetta to cross-validate."
        )
        console.print(f"{indent(2)}[cl.muted]Check [cl.ref]--reference[/] and the restraints file, or widen the "
                      f"batch, then re-run [cl.ref]lightdock --resume[/].[/]")
    return 0


def _run_peptide_docking_robust_audit(args: Namespace) -> int:
    """Stage 3: standardization audit of an existing run."""
    PeptideDockingRobustPipeline = _import_robust_pipeline()
    out = _robust_run_dir(args)
    pipeline = PeptideDockingRobustPipeline.from_config(out)
    stats = pipeline.run_standardization_audit(n_workers=args.audit_workers)
    print_stats({"Run directory": out, **stats}, "Peptide Docking Robust -- Standardization Audit")
    return 0


def _run_peptide_docking_robust_prodigy(args: Namespace) -> int:
    """Stage 4: PRODIGY cross-check of an existing run."""
    PeptideDockingRobustPipeline = _import_robust_pipeline()
    out = _robust_run_dir(args)
    pipeline = PeptideDockingRobustPipeline.from_config(out)
    stats = pipeline.run_prodigy(n_workers=args.prodigy_workers)
    print_stats({"Run directory": out, **stats}, "Peptide Docking Robust -- PRODIGY Scoring")
    return 0


def _run_peptide_docking_robust_haddock3(args: Namespace) -> int:
    """Stage 5: HADDOCK3 refinement of an existing run."""
    PeptideDockingRobustPipeline = _import_robust_pipeline()
    out = _robust_run_dir(args)
    pipeline = PeptideDockingRobustPipeline.from_config(
        out,
        haddock3_sampling=args.haddock3_sampling,
        haddock3_ncores=args.haddock3_ncores,
    )
    stats = pipeline.run_haddock3(n_workers=args.haddock3_workers)
    print_stats({"Run directory": out, **stats}, "Peptide Docking Robust -- HADDOCK3 Refinement")
    return 0


def _run_peptide_docking_robust_rosetta(args: Namespace) -> int:
    """Stage 6: PyRosetta/FlexPepDock refinement of an existing run."""
    PeptideDockingRobustPipeline = _import_robust_pipeline()
    out = _robust_run_dir(args)
    pipeline = PeptideDockingRobustPipeline.from_config(out, flexpepdock_trials=args.flexpepdock_trials)
    stats = pipeline.run_rosetta(n_workers=args.rosetta_workers)
    print_stats({"Run directory": out, **stats}, "Peptide Docking Robust -- PyRosetta/FlexPepDock Refinement")
    return 0


def _run_peptide_docking_robust_combine(args: Namespace) -> int:
    """Stage 7: merge every signal already on disk into combined_ranking.csv."""
    PeptideDockingRobustPipeline = _import_robust_pipeline()
    out = _robust_run_dir(args)
    pipeline = PeptideDockingRobustPipeline.from_config(out)
    combined_csv = pipeline.run_combined_ranking()
    print_stats({"Run directory": out, "Combined ranking": combined_csv},
                "Peptide Docking Robust -- Combined Ranking")
    return 0


def _run_peptide_docking_robust_all(args: Namespace) -> int:
    """Every stage chained in one call, as a single long-running command."""
    PeptideDockingRobustPipeline = _import_robust_pipeline()
    peptides, receptor, restraints, output = _peptide_inputs(args, restraints_required=True)
    out = _robust_new_run_dir(args, output)

    pipeline = PeptideDockingRobustPipeline(
        **_robust_pipeline_kwargs(args, peptides, receptor, restraints, out),
        skip_haddock3=args.skip_haddock3,
        skip_rosetta=args.skip_rosetta,
        haddock3_sampling=args.haddock3_sampling,
        haddock3_ncores=args.haddock3_ncores,
        flexpepdock_trials=args.flexpepdock_trials,
    )
    result = pipeline.run_full_pipeline(
        peptide_workers=args.prep_workers,
        docking_workers=args.docking_workers,
        audit_workers=args.audit_workers,
        prodigy_workers=args.prodigy_workers,
        haddock3_workers=args.haddock3_workers,
        rosetta_workers=args.rosetta_workers,
    )

    def _stage_stat(stage: dict, key: str) -> str:
        if stage.get("skipped"):
            return "skipped"
        return str(stage.get(key, "n/a"))

    stats = {
        "Run directory": out,
        "Gate candidates": result.gate.get("gate_candidates"),
        "Standardization audit": _stage_stat(result.audit, "successful"),
        "PRODIGY scored": _stage_stat(result.prodigy, "successful"),
        "HADDOCK3 refined": _stage_stat(result.haddock3, "successful"),
        "Rosetta refined": _stage_stat(result.rosetta, "successful"),
        "Combined ranking": result.combined_ranking_csv,
    }
    print_stats(stats, "Peptide Docking Robustness Results")
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


# Default independent MD replicas for pligand; mirrors
# pipelines.dynamics.replicas.DEFAULT_REPLICAS (kept local to avoid importing
# the pipeline package at CLI start-up).
DEFAULT_MD_REPLICAS = 2

_DYN_SPEC = {
    #  name        sim_id  label                              min_files
    "oprotein":  ("1", "Protein",                               1),
    "pligand":   ("2", "Protein + Ligand",                      2),
    "ppeptide":  ("3", "Protein + Peptide",                     2),
    "pacid":     ("4", "Protein + Nucleic Acid",                2),
    "pprotein":  ("5", "Protein + Protein",                     2),
    "ppligand":  ("6", "Protein + Protein + Ligand/Cofactor",   3),
    "oligand":   ("7", "Ligand only",                           1),
}

_DYN_FILE_LABELS = {
    "oprotein":  ["protein"],
    "pligand":   ["protein", "ligand"],
    "ppeptide":  ["protein", "peptide"],
    "pacid":     ["protein", "nucleic_acid"],
    "pprotein":  ["protein1", "protein2"],
    "ppligand":  ["protein1", "protein2", "ligand"],
    "oligand":   ["ligand"],
}

_DYN_INPUT_COUNTS = {
    "oprotein":  (1, 0),
    "pligand":   (1, 1),
    "ppeptide":  (1, 1),
    "pacid":     (1, 1),
    "pprotein":  (1, 1),
    "ppligand":  (2, 1),
    "oligand":   (0, 1),
}


def _resolve_dynamics_input_sets(
    dyn_type: str,
    receptor_inputs: Optional[list[str]] = None,
    ligand_inputs: Optional[list[str]] = None,
    cofactor_inputs: Optional[list[str]] = None,
    *,
    allow_multiple_ligands: bool = False,
) -> Tuple[list[list[str]], str]:
    """Validate and order dynamics inputs for local and HPC execution.

    Inputs not given with -ir/-il are detected in the working directory
    (cli/workspace.py). A cofactor (ion or small inorganic molecule) is only
    supported for 'pligand', is always optional, and -- when present -- is
    appended after the ligand in every returned input set.
    """
    cofactors = [_resolve_dynamic_input_path(path, "cofactors") for path in cofactor_inputs or []]
    if cofactors and dyn_type != "pligand":
        raise ValueError("A cofactor input is only supported for the 'pligand' simulation type.")
    if len(cofactors) > 1:
        raise ValueError("Only one cofactor input file is supported.")

    receptors = [_resolve_dynamic_input_path(path, "receptors") for path in receptor_inputs or []]
    ligands = [_resolve_dynamic_input_path(path, "ligands") for path in ligand_inputs or []]
    explicit = (bool(receptors) or not _DYN_INPUT_COUNTS[dyn_type][0]) and (
        bool(ligands) or not _DYN_INPUT_COUNTS[dyn_type][1]
    )
    receptors, ligands = workspace.detect_dynamics_inputs(
        dyn_type, receptors, ligands, exclude=cofactors, multiple_ligands=allow_multiple_ligands,
    )

    expected_receptors, expected_ligands = _DYN_INPUT_COUNTS[dyn_type]
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

    interface = "explicit" if explicit else "auto-detected"
    if dyn_type == "pligand" and allow_multiple_ligands:
        return [receptors + [ligand] + cofactors for ligand in ligands], interface
    return [receptors + ligands + cofactors], interface


def _prepare_dynamics_structure(path: str, work_dir: str, report_name: str, label: str) -> str:
    """Validate a ligand/cofactor; return the path to simulate.

    A 2D drawing (e.g. a sketch exported to PDB with z = 0) is replaced by a
    generated 3D conformer inside *work_dir*; chemically broken input raises.
    """
    try:
        from ..pipelines.ligand_validation import prepare_valid_ligand
    except ImportError:
        try:
            from chemlink.pipelines.ligand_validation import prepare_valid_ligand  # type: ignore
        except ImportError:
            from pipelines.ligand_validation import prepare_valid_ligand  # type: ignore
    prepared, _ = prepare_valid_ligand(
        path, work_dir, report_path=os.path.join(work_dir, report_name),
    )
    if os.path.abspath(prepared) != os.path.abspath(path):
        print_warn(
            f"{label} [bold]{os.path.basename(path)}[/] has 2D coordinates — "
            f"using a generated 3D conformer: [cl.path]{prepared}[/]"
        )
    return prepared


def _validate_dynamics_ligand(ligand_path: str, work_dir: str) -> str:
    return _prepare_dynamics_structure(ligand_path, work_dir, "ligand_validation.json", "Ligand")


def _classify_cofactor(cofactor_path: str):
    try:
        from ..utils.cofactor_classifier import classify_cofactor
    except ImportError:
        try:
            from chemlink.utils.cofactor_classifier import classify_cofactor  # type: ignore
        except ImportError:
            from utils.cofactor_classifier import classify_cofactor  # type: ignore
    return classify_cofactor(cofactor_path)


def _resolve_cofactor_for_run(
    cofactor_path: str,
    work_dir: str,
    *,
    explicit_charge: Optional[int],
    ph: float,
) -> dict:
    """Classify a cofactor input and, for polyatomic species, resolve its net charge.

    Monoatomic ions are matched to a ready-made force-field moleculetype at
    pipeline time (CofactorTopologyStep) and never need a net charge here.
    Charge/validation artifacts are written under cofactor-specific names
    (and, for charge detection, an isolated subdirectory) so they never
    collide with the ligand's own ligand_charge.json / ligand_pH*.sdf.
    """
    classification = _classify_cofactor(cofactor_path)
    if classification.kind == "ion":
        print_info(f"Cofactor  monoatomic ion [bold]{classification.resname}[/]")
        return {"cofactor_kind": "ion", "cofactor_pdb": cofactor_path}

    try:
        from ..pipelines.dynamics.ligand_charge import (
            detect_or_prepare_ligand_charge,
            estimate_unresolved_charge,
            explicit_charge_result,
        )
    except ImportError:
        try:
            from chemlink.pipelines.dynamics.ligand_charge import (  # type: ignore
                detect_or_prepare_ligand_charge,
                estimate_unresolved_charge,
                explicit_charge_result,
            )
        except ImportError:
            from pipelines.dynamics.ligand_charge import (  # type: ignore
                detect_or_prepare_ligand_charge,
                estimate_unresolved_charge,
                explicit_charge_result,
            )

    cofactor_path = _prepare_dynamics_structure(
        cofactor_path, work_dir, "cofactor_validation.json", "Cofactor",
    )
    if explicit_charge is not None:
        result = explicit_charge_result(cofactor_path, explicit_charge)
    else:
        prep_dir = os.path.join(work_dir, "cofactor_charge_prep")
        result = detect_or_prepare_ligand_charge(cofactor_path, prep_dir, ph=ph)
        if result.charge is None:
            result = estimate_unresolved_charge(result)

    with open(os.path.join(work_dir, "cofactor_charge.json"), "w", encoding="utf-8") as handle:
        json.dump(result.to_dict(), handle, indent=2)

    print_info(f"Cofactor charge  [cl.num]{result.charge:+d}[/]" if isinstance(result.charge, int) else f"Cofactor charge  {result.charge}")
    return {
        "cofactor_kind": "molecule",
        "cofactor_pdb": cofactor_path,
        "cofactor_charge": result.charge,
        "cofactor_prepared_file": result.prepared_file,
        "cofactor_charge_method": result.method,
        "cofactor_charge_confidence": result.confidence,
    }


def _print_resource_card(config: dict) -> None:
    """Show the same GPU/CPU resource panel DynamicsPipeline prints for pligand."""
    try:
        from ..hpc.cluster.resource_detector import get_hardware_profile
        from ..pipelines.dynamics.gmx_optimizer import get_cached_optimal_mdrun_flags
    except ImportError:
        try:
            from chemlink.hpc.cluster.resource_detector import get_hardware_profile  # type: ignore
            from chemlink.pipelines.dynamics.gmx_optimizer import (  # type: ignore
                get_cached_optimal_mdrun_flags,
            )
        except ImportError:
            from hpc.cluster.resource_detector import get_hardware_profile  # type: ignore
            from pipelines.dynamics.gmx_optimizer import get_cached_optimal_mdrun_flags  # type: ignore

    config.setdefault("gpu_ids", get_hardware_profile().gpu_indices)
    resources = get_cached_optimal_mdrun_flags(config)
    if resources.get("gpu_name"):
        memory_gb = resources["gpu_memory_total_mb"] / 1024
        gpu = (
            f"GPU {config['gpu_ids'][0]} · {resources['gpu_name']} · "
            f"{memory_gb:.1f} GB · sm_{resources['gpu_arch']}"
        )
    else:
        gpu = "[cl.warn.soft]none detected[/] · running on CPU"
    cpu = (
        f"{resources['total_cpu_count']} cores · "
        f"{resources['p_core_count']} P-cores · {resources['ntomp']} threads per simulation"
    )
    print_card(console, [("GPU", gpu), ("CPU", cpu)], title="Compute resources")


def _print_ligand_charge_result(result, *, label: Optional[str] = None) -> None:
    """Print a detected ligand charge.

    Detection method/confidence are deliberately not echoed here — they
    still land in ligand_charge.json for anyone who needs the provenance.
    """
    prefix = f"Ligand {label} charge" if label else "Ligand net charge"
    if result.charge is None:
        print_warn(f"{prefix}  [bold]unresolved[/]")
    else:
        confidence = getattr(result, "confidence", "") or ""
        tail = f"  [cl.muted]({confidence} confidence)[/]" if confidence and confidence != "explicit" else ""
        print_info(f"{prefix}  [cl.num]{result.charge:+d}[/]{tail}")


def _resolve_ligand_charge_for_run(
    ligand_path: str,
    work_dir: str,
    *,
    explicit_charge: Optional[int],
    mode: str,
    ph: float,
    interactive: bool,
    announce: bool = True,
):
    try:
        from ..pipelines.dynamics.ligand_charge import (
            LigandChargeError,
            detect_or_prepare_ligand_charge,
            estimate_unresolved_charge,
            explicit_charge_result,
            write_charge_provenance,
        )
    except ImportError:
        try:
            from chemlink.pipelines.dynamics.ligand_charge import (  # type: ignore
                LigandChargeError,
                detect_or_prepare_ligand_charge,
                estimate_unresolved_charge,
                explicit_charge_result,
                write_charge_provenance,
            )
        except ImportError:
            from pipelines.dynamics.ligand_charge import (  # type: ignore
                LigandChargeError,
                detect_or_prepare_ligand_charge,
                estimate_unresolved_charge,
                explicit_charge_result,
                write_charge_provenance,
            )

    if explicit_charge is not None:
        result = explicit_charge_result(ligand_path, explicit_charge)
    else:
        result = detect_or_prepare_ligand_charge(ligand_path, work_dir, ph=ph)

        if announce:
            _print_ligand_charge_result(result)

        if mode == "strict":
            if result.charge is None or result.confidence != "high":
                raise LigandChargeError(
                    "Strict charge mode requires an explicit --charge or a high-confidence "
                    "formal charge from SDF/MOL."
                )
        elif mode == "auto":
            if result.charge is None:
                result = estimate_unresolved_charge(result)
            if announce and result.confidence == "low":
                print_warn(result.warnings[-1] if result.warnings else "Using a low-confidence ligand charge estimate.")
        elif mode == "prompt":
            if not interactive:
                raise LigandChargeError(
                    "Charge confirmation requires an interactive terminal; use --charge INTEGER "
                    "or --charge-mode auto."
                )
            default = "" if result.charge is None else f" [Enter={result.charge}]"
            while True:
                try:
                    entered = console.input(f"  [cl.brand]?[/] Ligand net charge[cl.muted]{default}[/]: ").strip()
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
                print_err("Enter an integer charge, for example 0, +1, or -1.")

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

    try:
        input_sets, input_interface = _resolve_dynamics_input_sets(
            dyn_type,
            receptor_inputs=getattr(args, "input_receptors", None),
            ligand_inputs=getattr(args, "input_ligands", None),
            cofactor_inputs=getattr(args, "input_cofactor", None),
        )
    except ValueError as exc:
        print_error("Input error", exc)
        return 1
    all_files = input_sets[0]
    expected_file_count = sum(_DYN_INPUT_COUNTS[dyn_type])
    has_cofactor = dyn_type == "pligand" and len(all_files) == expected_file_count + 1

    missing_files = [path for path in all_files if not os.path.isfile(path)]
    if missing_files:
        print_error(
            "Input file not found",
            f"{len(missing_files)} input file(s) do not exist:",
            details="\n".join(os.path.abspath(path) for path in missing_files),
            hint_text="Paths given with [cl.ref]-ir/-il[/] are relative to the current directory.",
        )
        return 1

    run_name = getattr(args, "run_name", None)
    work_dir = setup_work_directory(
        workspace.output_base("dynamics", getattr(args, "output", None)), dyn_type, run_name=run_name
    )
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
        "ligand_inputs": all_files[_DYN_INPUT_COUNTS[dyn_type][0]:expected_file_count],
        "input_interface": input_interface,
        "binding_energy": getattr(args, "binding_energy", True),
        "mmpbsa_method":  getattr(args, "mmpbsa_method", "both"),
        "mmpbsa_frames":  getattr(args, "mmpbsa_frames", 1000),
        "mmpbsa_discard": getattr(args, "mmpbsa_discard", 20.0),
        "mmpbsa_cores":   getattr(args, "mmpbsa_cores", threads),
        "mmpbsa_salt":    getattr(args, "mmpbsa_salt", 0.150),
        "mmpbsa_entropy": getattr(args, "mmpbsa_entropy", "ie"),
        # The MD runs at 300 K (equilibration.py), so the entropy term must too.
        "mmpbsa_temperature": 300.0,
        "mmpbsa_ie_segment": 25,
        "force_field":     getattr(args, "force_field", "default"),
        "force_field_dir": getattr(args, "force_field_dir", None),
        "water_model":     getattr(args, "water_model", "tip3p"),
    }

    if dyn_type == "oprotein":
        config["pdb_input"] = all_files[0]
    elif dyn_type == "oligand":
        try:
            all_files[0] = _validate_dynamics_ligand(all_files[0], work_dir)
            charge = _resolve_ligand_charge_for_run(
                all_files[0], work_dir,
                explicit_charge=args.charge,
                mode=args.charge_mode,
                ph=args.ligand_ph,
                interactive=sys.stdin.isatty(),
            )
        except Exception as exc:
            print_error("Ligand charge error", exc)
            return 1
        config.update(
            ligand_pdb=all_files[0],
            ligand_charge=charge.charge,
            ligand_prepared_file=charge.prepared_file,
            ligand_charge_method=charge.method,
            ligand_charge_confidence=charge.confidence,
            ligand_charge_source=charge.source_file,
            ligand_ph=charge.ph,
        )
    elif dyn_type == "pligand":
        try:
            all_files[1] = _validate_dynamics_ligand(all_files[1], work_dir)
            charge = _resolve_ligand_charge_for_run(
                all_files[1], work_dir,
                explicit_charge=args.charge,
                mode=args.charge_mode,
                ph=args.ligand_ph,
                interactive=sys.stdin.isatty(),
            )
        except Exception as exc:
            print_error("Ligand charge error", exc)
            return 1
        config.update(
            pdb_input=all_files[0], ligand_pdb=all_files[1],
            ligand_charge=charge.charge,
            ligand_prepared_file=charge.prepared_file,
            ligand_charge_method=charge.method,
            ligand_charge_confidence=charge.confidence,
            ligand_charge_source=charge.source_file,
            ligand_ph=charge.ph,
            replicas=getattr(args, "replicas", DEFAULT_MD_REPLICAS),
        )
        if has_cofactor:
            try:
                cofactor_plan = _resolve_cofactor_for_run(
                    all_files[2], work_dir,
                    explicit_charge=getattr(args, "cofactor_charge", None),
                    ph=args.ligand_ph,
                )
            except Exception as exc:
                print_error("Cofactor error", exc)
                return 1
            config.update(
                # Distinct from cofactor_plan's own "cofactor_charge_method" key
                # (how the net charge was *detected*): this is ACPYPE's GAFF2
                # partial-charge scheme (bcc/gas) used to build the cofactor.
                cofactor_gaff_charge_method=getattr(args, "cofactor_charge_method", "bcc"),
                **cofactor_plan,
            )
            cofactor_timeout = getattr(args, "cofactor_charge_timeout", None)
            if cofactor_timeout is not None:
                config["cofactor_charge_timeout"] = cofactor_timeout
    elif dyn_type in ("ppeptide", "pacid", "pprotein"):
        config.update(pdb_protein=all_files[0], pdb_partner=all_files[1],
                      pdb_input=os.path.join(work_dir, "complex.pdb"))
    elif dyn_type == "ppligand":
        try:
            all_files[2] = _validate_dynamics_ligand(all_files[2], work_dir)
            charge = _resolve_ligand_charge_for_run(
                all_files[2], work_dir,
                explicit_charge=args.charge,
                mode=args.charge_mode,
                ph=args.ligand_ph,
                interactive=sys.stdin.isatty(),
            )
        except Exception as exc:
            print_error("Ligand charge error", exc)
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

    sim_rows = [
        ("Simulation", label),
        ("Duration", f"{args.time} ns"),
    ]
    if "replicas" in config:
        sim_rows.append(("Replicas", str(config["replicas"])))
    sim_rows.append(("Threads", str(threads)))
    if config["force_field"] != "default":
        sim_rows += [("Force field", config["force_field"]), ("Water model", config["water_model"])]
    if config.get("cofactor_pdb"):
        cofactor_desc = config.get("cofactor_resname") or config["cofactor_kind"]
        sim_rows.append(("Cofactor", f"{config['cofactor_kind']} ({cofactor_desc})"))
    sim_rows.append(("Output", work_dir))
    kv_block(console, sim_rows, title="Simulation Parameters")
    console.print()

    with open(os.path.join(work_dir, "run_config.json"), "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    DynamicsPipeline(config).execute()
    return 0


def _continue_fep_workflow(args: Namespace, work_dir: str) -> int:
    """Run pilot -> production -> analysis against an existing ABFE run directory.

    Used both by `chemlink dynamic fep`'s default full-workflow tail and by
    `--resume` (continuing a run whose pilot/production was interrupted or
    failed, without re-running prepare/build).
    """
    section(console, "FEP · pilot", "stage 1 of 3 · equilibration + anchor stability")
    args.run_dir = work_dir
    result = _run_dynamic_fep_pilot(args)
    if result != 0:
        return result
    with open(os.path.join(work_dir, "pilot_report.json"), encoding="utf-8") as handle:
        pilot_report = json.load(handle)
    if pilot_report.get("status") != "passed":
        print_warn(
            f"Workflow stopped — the pilot did not pass (status [bold]{pilot_report.get('status')}[/]); "
            "production stays locked."
        )
        console.print(f"{indent(2)}[cl.muted]Inspect [cl.path]{os.path.join(work_dir, 'pilot_report.json')}[/], then run "
                      f"[cl.ref]chemlink dynamic fep --resume {work_dir}[/][/]")
        return 1

    section(console, "FEP · production", "stage 2 of 3 · every λ window")
    args.no_resume = False
    result = _run_dynamic_fep_produce(args)
    if result != 0:
        return result

    section(console, "FEP · analysis", "stage 3 of 3 · ΔG estimators")
    return _run_dynamic_fep_analyze(args)


def _run_dynamic_fep(args: Namespace) -> int:
    """Prepare FEP inputs and optionally build the ABFE systems."""
    try:
        from ..pipelines.dynamics.fep_pipeline import (
            FEPExecutionError, FEPPreparationError, execute_abfe_system_setup,
            prepare_fep_run, validate_boresch_topologies,
        )
        from ..pipelines.dynamics.logger import get_step_logger
        from ..pipelines.dynamics.utils import setup_work_directory
    except ImportError:
        try:
            from chemlink.pipelines.dynamics.fep_pipeline import (  # type: ignore
                FEPExecutionError, FEPPreparationError, execute_abfe_system_setup,
                prepare_fep_run, validate_boresch_topologies,
            )
            from chemlink.pipelines.dynamics.logger import get_step_logger  # type: ignore
            from chemlink.pipelines.dynamics.utils import setup_work_directory  # type: ignore
        except ImportError:
            from pipelines.dynamics.fep_pipeline import (  # type: ignore
                FEPExecutionError, FEPPreparationError, execute_abfe_system_setup,
                prepare_fep_run, validate_boresch_topologies,
            )
            from pipelines.dynamics.logger import get_step_logger  # type: ignore
            from pipelines.dynamics.utils import setup_work_directory  # type: ignore
    try:
        from ..utils.progress import job_bar
    except ImportError:
        try:
            from chemlink.utils.progress import job_bar  # type: ignore
        except ImportError:
            from utils.progress import job_bar  # type: ignore

    if args.resume:
        if args.prepare_only or args.build_only:
            print_error("FEP input error", "--resume is mutually exclusive with --prepare-only/--build-only.")
            return 1
        try:
            run_dir = workspace.find_run_dir(
                None if args.resume == "auto" else args.resume, "fep", ("abfe_setup.json",)
            )
        except FileNotFoundError as exc:
            print_error("FEP input error", exc)
            return 1
        return _continue_fep_workflow(args, run_dir)

    explicit_ligands = [path for path in (args.ligand_a, args.ligand_b) if path]
    try:
        receptors, ligands = workspace.detect_dynamics_inputs(
            "pligand",
            [_resolve_dynamic_input_path(args.protein, "receptors")] if args.protein else [],
            [_resolve_dynamic_input_path(path, "ligands") for path in explicit_ligands],
            multiple_ligands=True,
        )
    except ValueError as exc:
        print_error("FEP input error", exc)
        return 1
    if len(ligands) != 2:
        print_error(
            "FEP input error",
            f"FEP needs exactly 2 ligands; found {len(ligands)}: "
            + ", ".join(os.path.basename(path) for path in ligands),
            hint_text="Choose them with [cl.ref]-a FILE -b FILE[/].",
        )
        return 1
    protein, (ligand_a, ligand_b) = receptors[0], ligands
    missing = [path for path in (protein, ligand_a, ligand_b) if not os.path.isfile(path)]
    if missing:
        print_error("FEP input file not found", f"{len(missing)} input file(s) do not exist:", details="\n".join(missing))
        return 1
    if not protein.lower().endswith(".pdb"):
        print_error("FEP input error", "--protein must be a PDB file.")
        return 1
    if args.prepare_only and args.build_only:
        print_error("FEP input error", "--prepare-only and --build-only are mutually exclusive.")
        return 1

    equilibration_time = (
        args.equilibration_time if args.equilibration_time is not None else 0.5
    )

    try:
        work_dir = setup_work_directory(
            workspace.output_base("fep", args.output), "fep", run_name=getattr(args, "run_name", None)
        )
        config = {
            "simulation_type": "fep", "fep_type": args.fep_type,
            "protein": protein,
            "ligand_a_source": ligand_a, "ligand_b_source": ligand_b,
            "smiles_a": args.smiles_a, "smiles_b": args.smiles_b,
            "replicas": args.replicas, "lambda_windows": args.lambda_windows,
            "equilibration_ns": equilibration_time,
            "production_ns": args.production_time,
            "temperature_k": args.temperature, "force_field": args.force_field,
            "water_model": args.water_model, "prepare_only": args.prepare_only,
            "work_dir": work_dir, "run_name": args.run_name or os.path.basename(work_dir),
        }

        fep_rows = [
            ("Simulation", "Free-energy perturbation"),
            ("Method", args.fep_type),
            ("Replicas", args.replicas),
        ]
        if args.lambda_windows is not None:
            fep_rows.append(("Lambda windows", args.lambda_windows))
        if args.equilibration_time is not None:
            fep_rows.append(("Equilibration", f"{equilibration_time} ns"))
        fep_rows += [
            ("Production", f"{args.production_time} ns"),
            ("Temperature", f"{args.temperature} K"),
            ("Force field", args.force_field),
            ("Water model", args.water_model),
            ("Output", work_dir),
        ]
        print_card(console, fep_rows, title="Simulation parameters")
        _print_resource_card(config)

        section(console, "Ligand charges")
        if args.charge_mode == "prompt":
            charge_a = _resolve_ligand_charge_for_run(
                ligand_a, os.path.join(work_dir, "common", "charge_A"),
                explicit_charge=args.charge_a, mode=args.charge_mode,
                ph=args.ligand_ph, interactive=sys.stdin.isatty(),
            )
            charge_b = _resolve_ligand_charge_for_run(
                ligand_b, os.path.join(work_dir, "common", "charge_B"),
                explicit_charge=args.charge_b, mode=args.charge_mode,
                ph=args.ligand_ph, interactive=sys.stdin.isatty(),
            )
        else:
            with job_bar("Ligand charge detection", 2, unit="ligand") as bar:
                charge_a = _resolve_ligand_charge_for_run(
                    ligand_a, os.path.join(work_dir, "common", "charge_A"),
                    explicit_charge=args.charge_a, mode=args.charge_mode,
                    ph=args.ligand_ph, interactive=sys.stdin.isatty(), announce=False,
                )
                bar.update(1)
                charge_b = _resolve_ligand_charge_for_run(
                    ligand_b, os.path.join(work_dir, "common", "charge_B"),
                    explicit_charge=args.charge_b, mode=args.charge_mode,
                    ph=args.ligand_ph, interactive=sys.stdin.isatty(), announce=False,
                )
                bar.update(1)
            _print_ligand_charge_result(charge_a, label="A")
            if charge_a.warnings:
                print_warn(charge_a.warnings[-1])
            _print_ligand_charge_result(charge_b, label="B")
            if charge_b.warnings:
                print_warn(charge_b.warnings[-1])

        config.update({
            "ligand_a": charge_a.prepared_file, "ligand_b": charge_b.prepared_file,
            "charge_a": charge_a.charge, "charge_b": charge_b.charge,
            "charge_provenance": {"A": charge_a.to_dict(), "B": charge_b.to_dict()},
        })

        report = prepare_fep_run(config)
        setup = None
        boresch = None
        if not args.prepare_only:
            if report["method"] != "ABFE":
                raise FEPExecutionError(
                    "Automatic system construction is currently available only for ABFE; "
                    "use --prepare-only for RBFE."
                )
            setup = execute_abfe_system_setup(config, report)
            boresch = validate_boresch_topologies(work_dir)
    except (FEPPreparationError, FEPExecutionError, FileExistsError, RuntimeError, ValueError) as exc:
        print_error("FEP error", exc)
        return 1

    prep_logger = get_step_logger(__name__, os.path.join(work_dir, "fep_execution.log"))
    file_only = {"extra": {"file_only": True}}
    prep_logger.info("FEP preparation completed.", **file_only)
    prep_logger.info("Run directory: %s", work_dir, **file_only)
    prep_rows = [("Heavy-atom coverage", f"{report['mapping']['coverage']:.1%}")]
    if args.lambda_windows is None:
        prep_rows.append(("Lambda windows", f"{report['lambda_windows']}  [cl.muted](default for {report['method']})[/]"))
    prep_rows.append(("Thermodynamic states", report["state_count"]))
    if setup:
        prep_rows.append(("ABFE systems", "4 built, solvated and neutralized"))
    if boresch:
        prep_rows.append(("Boresch anchors", "mapped · topologies validated with grompp"))
    kv_block(console, prep_rows, title="FEP preparation")
    if report["missing_dependencies"]:
        prep_logger.info(
            "Execution dependencies not installed: %s",
            ", ".join(report["missing_dependencies"]), **file_only,
        )
    if setup:
        prep_logger.info("Four ABFE systems built, solvated and neutralized.", **file_only)
    if boresch:
        prep_logger.info("Boresch anchors mapped and topologies validated with grompp.", **file_only)
        prep_logger.info("Pilot MD remains locked pending standard-state correction.", **file_only)

    if args.prepare_only or args.build_only:
        return 0

    return _continue_fep_workflow(args, work_dir)


def _run_dynamic_fep_pilot(args: Namespace) -> int:
    """Validate Boresch topologies, equilibrate replicas, and run the real pilot."""
    try:
        from ..pipelines.dynamics.fep_execution import run_fep_equilibration, run_fep_pilot
        from ..pipelines.dynamics.fep_pipeline import FEPExecutionError, validate_boresch_topologies
    except ImportError:
        try:
            from chemlink.pipelines.dynamics.fep_execution import (  # type: ignore
                run_fep_equilibration, run_fep_pilot,
            )
            from chemlink.pipelines.dynamics.fep_pipeline import (  # type: ignore
                FEPExecutionError, validate_boresch_topologies,
            )
        except ImportError:
            from pipelines.dynamics.fep_execution import (  # type: ignore
                run_fep_equilibration, run_fep_pilot,
            )
            from pipelines.dynamics.fep_pipeline import (  # type: ignore
                FEPExecutionError, validate_boresch_topologies,
            )
    run_dir = os.path.abspath(os.path.expanduser(args.run_dir))
    try:
        validate_boresch_topologies(run_dir)
        equilibration = run_fep_equilibration(run_dir)
        pilot = run_fep_pilot(run_dir)
    except (FEPExecutionError, FileNotFoundError, ValueError) as exc:
        print_error("FEP pilot error", exc)
        return 1
    passed = pilot["status"] == "passed"
    rows = [
        ("Run directory", run_dir),
        ("Complex topologies", "[cl.ok.soft]✓ grompp validated[/]"),
        ("Replicas equilibrated", len(equilibration["replicas"])),
        ("Anchor stability", "[cl.ok.soft]✓ passed — production unlocked[/]" if passed
         else f"[cl.warn.soft]▲ {pilot['status']} — inspect pilot_report.json[/]"),
    ]
    for label, entry in pilot["ligands"].items():
        ssc = entry["standard_state_correction"]["delta_g_kcal_mol"]
        rows.append((f"Ligand {label} SSC", f"{ssc:+.2f} kcal/mol"))
    kv_block(console, rows, title="FEP pilot results")
    console.print()
    return 0


def _run_dynamic_fep_produce(args: Namespace) -> int:
    """Run production MD across every lambda window of a pilot-approved ABFE run."""
    try:
        from ..pipelines.dynamics.fep_execution import run_fep_production
        from ..pipelines.dynamics.fep_pipeline import FEPExecutionError
    except ImportError:
        try:
            from chemlink.pipelines.dynamics.fep_execution import run_fep_production  # type: ignore
            from chemlink.pipelines.dynamics.fep_pipeline import FEPExecutionError  # type: ignore
        except ImportError:
            from pipelines.dynamics.fep_execution import run_fep_production  # type: ignore
            from pipelines.dynamics.fep_pipeline import FEPExecutionError  # type: ignore
    run_dir = os.path.abspath(os.path.expanduser(args.run_dir))
    try:
        result = run_fep_production(run_dir, resume=not args.no_resume)
    except (FEPExecutionError, FileNotFoundError, ValueError) as exc:
        print_error("FEP production error", exc)
        return 1
    rows = [
        ("Run directory", run_dir),
        ("States completed", f"{result['completed']}/{result['total']}"),
    ]
    if result["failed"]:
        shown = ", ".join(result["failed"][:10]) + (" …" if len(result["failed"]) > 10 else "")
        rows.append(("Failed states", f"[cl.warn.soft]{len(result['failed'])}[/]  [cl.muted]{shown}[/]"))
    kv_block(console, rows, title="FEP production")
    if result["failed"]:
        print_info("Re-run the same command to retry only the pending/failed states.")
    console.print()
    return 0 if result["status"] == "completed" else 1


def _run_dynamic_fep_analyze(args: Namespace) -> int:
    """Estimate binding free energies from a produced ABFE run's dhdl.xvg files."""
    try:
        from ..pipelines.dynamics.fep_analysis import run_fep_analysis
        from ..pipelines.dynamics.fep_pipeline import FEPExecutionError
    except ImportError:
        try:
            from chemlink.pipelines.dynamics.fep_analysis import run_fep_analysis  # type: ignore
            from chemlink.pipelines.dynamics.fep_pipeline import FEPExecutionError  # type: ignore
        except ImportError:
            from pipelines.dynamics.fep_analysis import run_fep_analysis  # type: ignore
            from pipelines.dynamics.fep_pipeline import FEPExecutionError  # type: ignore
    run_dir = os.path.abspath(os.path.expanduser(args.run_dir))
    try:
        result = run_fep_analysis(run_dir)
    except (FEPExecutionError, FileNotFoundError, ValueError) as exc:
        print_error("FEP analysis error", exc)
        return 1
    section(console, "FEP results", "binding free energy")
    for label, entry in result["ligands"].items():
        table = Table(box=rich_box.SIMPLE_HEAD, show_edge=False, pad_edge=False,
                      header_style="cl.muted", border_style="cl.border")
        table.add_column(f"Ligand {label}", style="bold", no_wrap=True)
        table.add_column("ΔG bind (kcal/mol)", justify="right")
        table.add_column("±", justify="right", style="cl.muted")
        for estimator, value in entry["binding_free_energy"].items():
            if value.get("status") == "unavailable":
                table.add_row(estimator, "[cl.warn.soft]unavailable[/]", f"[cl.muted]{value['reason']}[/]")
            else:
                table.add_row(estimator, f"[cl.num]{value['delta_g_kcal_mol']:+.2f}[/]",
                              f"{value['error_kcal_mol']:.2f}")
        console.print(Padding(table, (1, 0, 0, 2)))
        diagnostics = entry["diagnostics"]
        warnings = [item for item in diagnostics["checks"] if item["status"] == "warning"]
        if warnings:
            console.print(status_line("warn", "Reliability — needs review", level_indent=2))
            for item in warnings:
                console.print(f"{indent(4)}[cl.muted]• {item['message']}[/]", soft_wrap=True)
        else:
            console.print(status_line("ok", "Reliability — passed all automatic checks", level_indent=2))

    ranking = result.get("ranking")
    if ranking:
        console.print()
        print_info(f"Ranking (TI): ligand [bold]{ranking['tighter_binder']}[/] binds tighter "
                   f"(ΔΔG = [cl.num]{ranking['delta_delta_g_kcal_mol']:+.2f}[/] kcal/mol)")
    print_info(f"Full report  [cl.path]{run_dir}/fep_analysis.json[/]")
    console.print()
    return 0


def _dynamics_run_dir(explicit: Optional[str]) -> str:
    """RUN_DIR for resume/analyze: given, the cwd when it is a run, or the latest run."""
    return workspace.find_run_dir(explicit, "dynamics", ("run_config.json", "hpc_config.json"))


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
            _dynamics_run_dir(args.run_dir), sim_type_name=args.sim_type, ns_time=args.time,
        )
        stage = controller.detect_stage()
        kv_block(console, [("Run directory", str(controller.work_dir)), ("Detected stage", f"[bold]{stage}[/]")],
                 title="Resume")
        console.print()
        if args.dry_run:
            return 0
        result = controller.run()
        if result == "complete":
            print_ok("The simulation and its analyses are already complete.")
        else:
            print_ok(f"Recovery completed from stage: {result}")
        return 0
    except Exception as exc:
        print_error("Resume error", exc)
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
        from ..pipelines.dynamics.replicas import replica_dirs, write_replicas_summary
    except ImportError:
        try:
            from chemlink.pipelines.dynamics.replicas import (  # type: ignore
                replica_dirs, write_replicas_summary,
            )
        except ImportError:
            from pipelines.dynamics.replicas import (  # type: ignore
                replica_dirs, write_replicas_summary,
            )
    options = dict(
        selection=args.only,
        sim_type_name=args.sim_type,
        ns_time=args.time,
        mmpbsa_cores=args.mmpbsa_cores,
        mmpbsa_method=args.mmpbsa_method,
        mmpbsa_frames=args.mmpbsa_frames,
        mmpbsa_discard=args.mmpbsa_discard,
        mmpbsa_salt=args.mmpbsa_salt,
        mmpbsa_entropy=args.mmpbsa_entropy,
        plot_only=args.plot_only,
    )
    try:
        run_dir = _dynamics_run_dir(args.run_dir)
    except FileNotFoundError as exc:
        print_error("Analysis error", exc)
        return 1
    replicas = replica_dirs(run_dir)
    # A multi-replica root has no trajectory of its own: analyse each replica.
    targets = (
        [str(path) for path in replicas]
        if replicas and not os.path.isfile(os.path.join(run_dir, "md.tpr"))
        else [run_dir]
    )
    try:
        for target in targets:
            controller = DynamicsReanalysisController(target, **options)
            kv_block(console, [("Results directory", str(controller.work_dir)),
                               ("Analyses", f"[bold]{controller.selection}[/]")],
                     title="Re-analysis")
            console.print()
            controller.run()
            print_ok(f"{controller.selection.capitalize()} analysis completed.")
        if targets != [run_dir]:
            summary = write_replicas_summary(run_dir, len(replicas))
            print_ok(f"Replica summary: {summary}")
        return 0
    except Exception as exc:
        print_error("Analysis error", exc)
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

    def _collect() -> list:

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
            ("numpy",    True),
            ("rdkit",    False),
            ("pdbfixer", False),
            ("meeko",    False),
            ("alchemlyb", False),
            ("pymbar",   False),
        ]:
            found = _has_module(pkg)
            ver   = _module_version(pkg) or ("installed" if found else "not found")
            level = ("error" if required else "warning") if not found else "ok"
            checks.append(_chk(
                pkg, "software", level, ver,
                f"pip install {pkg}" if not found else None,
            ))

        from pipelines.dynamics.utils import find_gmx_path  # type: ignore

        gmx_serial = find_gmx_path(serial=True)
        gmx_mpi = find_gmx_path(serial=False)
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

        # -- paths: what a command run from here would use --------------------------

        for role, label in (("receptors", "Receptor inputs"), ("ligands", "Ligand inputs"),
                            ("peptides", "Peptide inputs")):
            try:
                found = workspace.find_inputs(role).describe()
            except FileNotFoundError:
                found = "none in this directory (pass the path, or run from a project directory)"
            checks.append(_chk(label, "paths", "ok", found))

        results = os.path.abspath(workspace.RESULTS_DIR)
        target = results if os.path.isdir(results) else os.path.dirname(results)
        writable = os.access(target, os.W_OK)
        checks.append(_chk(
            "Results directory", "paths",
            "ok" if writable else "error",
            f"{results}  ({'writable' if writable else 'NOT writable'})",
            None if writable else "Run from a directory you can write to, or pass -o DIR.",
        ))
        return checks

    if getattr(args, "json", False):
        checks = _collect()
    else:
        with busy("Inspecting environment"):
            checks = _collect()

    # ── summary counts ─────────────────────────────────────────────────────────

    n_ok     = sum(1 for c in checks if c["level"] == "ok")
    n_warn   = sum(1 for c in checks if c["level"] == "warning")
    n_errors = sum(1 for c in checks if c["level"] == "error")

    # ── JSON output ────────────────────────────────────────────────────────────

    if getattr(args, "json", False):
        payload = {
            "doctor_version":   DOCTOR_VERSION,
            "chemlink_version": VERSION,
            "timestamp":        datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "summary":          {"ok": n_ok, "warnings": n_warn, "errors": n_errors},
            "checks":           checks,
        }
        print(json.dumps(payload, indent=2))
        return 1 if n_errors else 0

    # ── Rich display ───────────────────────────────────────────────────────────

    _CAT_TITLES = {
        "software": "Software & tools",
        "hardware": "Hardware",
        "config":   "Configuration",
        "paths":    "Paths",
    }
    _LEVEL_KEY = {"ok": "ok", "warning": "warn", "error": "err"}

    import platform
    section(console, "Environment check",
            f"{platform.node()} · Python {sys.version.split()[0]} · {platform.system()} {platform.machine()}")

    for cat in ("software", "hardware", "config", "paths"):
        cat_checks = [c for c in checks if c["cat"] == cat]
        if not cat_checks:
            continue
        ready = sum(1 for c in cat_checks if c["level"] == "ok")
        header = Text("\n  ")
        header.append(_CAT_TITLES[cat], style="bold")
        header.append(f"  {ready}/{len(cat_checks)} ready", style="cl.ok.soft" if ready == len(cat_checks) else "cl.muted")
        console.print(header)
        status_block(
            console,
            [(_LEVEL_KEY[c["level"]], c["name"], c["detail"]) for c in cat_checks],
            level=2,
        )

    # Health bar: one segment per check, coloured by state.
    width = max(10, min(40, console.width - 56))
    total = max(1, len(checks))
    seg = {k: round(width * v / total) for k, v in (("ok", n_ok), ("warn", n_warn), ("err", n_errors))}
    seg["ok"] = width - seg["warn"] - seg["err"]
    bar = Text("\n  ")
    bar.append("Health  ", style="bold")
    bar.append("━" * seg["ok"], style="cl.ok.soft")
    bar.append("━" * seg["warn"], style="cl.warn.soft")
    bar.append("━" * seg["err"], style="cl.err.soft")
    bar.append(f"   {n_ok} ok", style="cl.ok.soft")
    bar.append("  ·  ", style="cl.faint")
    bar.append(f"{n_warn} warning{'s' if n_warn != 1 else ''}", style="cl.warn.soft" if n_warn else "cl.muted")
    bar.append("  ·  ", style="cl.faint")
    bar.append(f"{n_errors} error{'s' if n_errors != 1 else ''}", style="cl.err.soft" if n_errors else "cl.muted")
    console.print(bar)

    issues = [c for c in checks if c["level"] in ("error", "warning") and c["suggestion"]]
    issues.sort(key=lambda c: c["level"] != "error")
    if issues:
        section(console, "Suggested fixes", f"{len(issues)} item(s)")
        grid = Table.grid(padding=(0, 1))
        grid.add_column(justify="right", style="cl.faint", no_wrap=True)
        grid.add_column(no_wrap=True)
        grid.add_column(overflow="fold")
        for i, c in enumerate(issues, 1):
            key = _LEVEL_KEY[c["level"]]
            name = Text.assemble((GLYPH[key] + " ", STYLE[key]), (c["name"], "bold"))
            fix = Text.assemble(("→ ", "cl.accent"), (c["suggestion"], "cl.ref"))
            grid.add_row(f"{i}.", name, fix)
        console.print(Padding(grid, (0, 0, 0, 2)))

    console.print()
    if n_errors == 0 and n_warn == 0:
        print_ok(f"All {len(checks)} checks passed — ChemLink is ready.")
    elif n_errors == 0:
        print_warn(f"Usable with {n_warn} warning(s) — affected features will be skipped or run on CPU.")
    else:
        print_err(f"{n_errors} blocking error(s) — fix these before running pipelines.")
    console.print(f"{indent(1)}[cl.faint]doctor {DOCTOR_VERSION} · chemlink {VERSION} · "
                  f"[cl.ref]chemlink doctor --json[/] for machine-readable output[/]\n")
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
    receptor_src, ligand_src, output_base = _docking_inputs(args)
    exts = (".sdf", ".mol2", ".pdb", ".mol", ".pdbqt")
    if ligand_src.origin == "cwd":
        ligand_files = list(ligand_src.files)
    else:
        ligand_files = [
            name for name in os.listdir(ligand_src.directory) if name.lower().endswith(exts)
        ]
    n_ligands = len(ligand_files)

    if n_ligands == 0 and not args.dry_run:
        print_error(
            "No ligands found",
            f"No ligand files found in {ligand_src.directory}",
            hint_text=f"Supported formats: {', '.join(exts)}",
        )
        return 1

    manual_center = getattr(args, "manual_center", None)
    manual_npts   = getattr(args, "manual_npts",   None)
    if manual_npts is not None and manual_center is None:
        print_error("Invalid grid box", "--manual-npts requires --manual-center.")
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

    mode = "container" if args.container else "native"
    repo = _repo_root()
    script_path = os.path.join(repo, "hpc", "slurm", mode, "run_multinode_pipeline.sh")
    run_id = args.run_name or f"run_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = os.path.join(output_base, run_id)
    if args.run_name and os.path.exists(run_dir):
        print_error("Run directory already exists", run_dir,
                    hint_text="Choose another [cl.ref]--run-name[/].")
        return 1

    # Loose files from the working directory are copied next to the run
    # (not inside it: the pipeline script refuses to reuse a named run dir).
    staging = os.path.join(output_base, ".inputs", run_id)
    if args.dry_run:
        receptor_dir, ligand_dir = receptor_src.staged_path(staging), ligand_src.staged_path(staging)
    else:
        receptor_dir = receptor_src.materialize(staging, copy=True)
        ligand_dir = ligand_src.materialize(staging, copy=True)

    env_overrides: dict = {
        "REPO_DIR":                 repo,
        "RUN_ID":                   run_id,
        "RUN_DIR":                  run_dir,
        "INPUT_LIGANDS_DIR":        ligand_dir,
        "INPUT_RECEPTORS_DIR":      receptor_dir,
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
        env_overrides["RUN_NAME_CUSTOM"] = "1"
    if args.partition:
        env_overrides["SLURM_PARTITION"] = args.partition
    if args.gres:
        env_overrides["DOCKING_GRES"] = args.gres
    if args.nodes:
        env_overrides["SLURM_NODELIST"] = args.nodes
    if args.container:
        env_overrides["CONTAINER_IMAGE"] = args.container
    if manual_center:
        env_overrides["MANUAL_CENTER"] = " ".join(f"{v:g}" for v in manual_center)
        if manual_npts:
            env_overrides["MANUAL_NPTS"] = " ".join(str(v) for v in manual_npts)
    if box_config_abs:
        env_overrides["BOX_CONFIG_PATH"] = box_config_abs

    cfg_rows = [
        ("Mode",                mode),
        ("Receptors",           receptor_src.describe()),
        ("Ligands",             ligand_src.describe()),
        ("Run directory",       run_dir),
        ("Ligands detected",    str(n_ligands) if n_ligands else "0  (dry-run)"),
        ("Batch size",          str(batch_size)),
        ("Total batches",       str(total_batches)),
        ("Prep array",          array_range),
        ("Max GPU concurrency", str(args.max_gpu_concurrency)),
        ("Receptor workers",    str(args.receptor_workers)),
        ("Ligand workers",      str(args.ligand_workers)),
        ("Active-site workers", str(args.active_site_workers)),
        ("Docking workers",     str(args.docking_workers)),
        ("Flexible radius",     f"{args.flex_radius:g} A" if args.flex_radius else "disabled"),
    ]
    if manual_center:
        cfg_rows.append(("Grid center (manual)", " ".join(f"{v:g}" for v in manual_center)))
        cfg_rows.append((
            "Grid npts",
            " ".join(str(v) for v in manual_npts) if manual_npts else "auto from ligands",
        ))
    if box_config_abs:
        cfg_rows.append(("Box config", box_config_abs))
    if args.partition:
        cfg_rows.append(("SLURM partition", args.partition))
    if args.gres:
        cfg_rows.append(("GRES", args.gres))
    if args.nodes:
        cfg_rows.append(("Nodes", args.nodes))
    if args.container:
        cfg_rows.append(("Container image", args.container))
    kv_block(console, cfg_rows, title="HPC Docking Configuration")
    console.print()

    if args.dry_run:
        section(console, "Dry run", "nothing was submitted")
        env_lines = Text()
        for k, v in env_overrides.items():
            env_lines.append(f"  {k}", style="cl.accent")
            env_lines.append("=", style="cl.faint")
            env_lines.append(f"{v} \\\n", style="")
        env_lines.append(f"  bash {script_path}", style="cl.ref")
        console.print(Padding(env_lines, (0, 0, 0, 2)))
        console.print()
        return 0

    if not os.path.isfile(script_path):
        print_error("Pipeline script not found", script_path)
        return 1

    env = {**os.environ, **env_overrides}
    result = subprocess.run(["bash", script_path], env=env)
    return result.returncode


# ── HPC Dynamics command handler ──────────────────────────────────────────────

def _run_hpc_dynamics(args: Namespace) -> int:
    try:
        from ..pipelines.dynamics.utils import setup_work_directory
    except ImportError:
        try:
            from chemlink.pipelines.dynamics.utils import setup_work_directory  # type: ignore
        except ImportError:
            from pipelines.dynamics.utils import setup_work_directory  # type: ignore

    dyn_type = args.dyn_type
    sim_id, label, min_files = _DYN_SPEC[dyn_type]

    replicas = getattr(args, "replicas", 1)
    try:
        sim_input_sets, input_interface = _resolve_dynamics_input_sets(
            dyn_type,
            receptor_inputs=getattr(args, "input_receptors", None),
            ligand_inputs=getattr(args, "input_ligands", None),
            allow_multiple_ligands=True,
        )
    except ValueError as exc:
        print_error("Input error", exc)
        return 1

    missing_files = [
        path for files in sim_input_sets for path in files if not os.path.isfile(path)
    ]
    if missing_files:
        print_error(
            "Input file not found",
            f"{len(set(missing_files))} input file(s) do not exist:",
            details="\n".join(sorted(set(missing_files))),
        )
        return 1

    n_sims = len(sim_input_sets)

    # --- Build per-simulation config dicts and write individual JSON files ---
    timestamp = f"{datetime.now():%Y%m%dT%H%M%S}"
    configs: list[dict] = []
    output_base = workspace.output_base("dynamics", getattr(args, "output", None))
    requested_run_name = getattr(args, "run_name", None)
    custom_run_names = [
        indexed_run_name(requested_run_name, index, n_sims)
        for index in range(n_sims)
    ]
    if requested_run_name:
        existing = [
            os.path.join(output_base, name)
            for name in custom_run_names
            if name and os.path.exists(os.path.join(output_base, name))
        ]
        if existing:
            print_error(
                "Run directory already exists",
                "These run directories already exist:",
                details="\n".join(existing),
                hint_text="Choose another [cl.ref]--run-name[/].",
            )
            return 1

    for idx, files in enumerate(sim_input_sets):
        custom_name = custom_run_names[idx]
        if custom_name:
            work_dir = setup_work_directory(output_base, dyn_type, run_name=custom_name)
        elif n_sims > 1:
            # One job per ligand: index the directories so jobs created within
            # the same second do not collide.
            work_dir = os.path.join(output_base, f"run_{dyn_type}_{timestamp}_{idx + 1}")
            os.makedirs(work_dir, exist_ok=True)
        else:
            work_dir = setup_work_directory(output_base, dyn_type)
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
            "mmpbsa_entropy": getattr(args, "mmpbsa_entropy", "ie"),
            "mmpbsa_temperature": 300.0,
            "mmpbsa_ie_segment": 25,
        }

        if dyn_type == "oprotein":
            config["pdb_input"] = files[0]
        elif dyn_type == "oligand":
            try:
                files[0] = _validate_dynamics_ligand(files[0], work_dir)
                charge = _resolve_ligand_charge_for_run(
                    files[0], work_dir,
                    explicit_charge=args.charge,
                    mode=args.charge_mode,
                    ph=args.ligand_ph,
                    interactive=False,
                )
            except Exception as exc:
                print_error("Ligand charge error", exc)
                return 1
            config.update(
                ligand_pdb=files[0],
                ligand_charge=charge.charge,
                ligand_prepared_file=charge.prepared_file,
                ligand_charge_method=charge.method,
                ligand_charge_confidence=charge.confidence,
                ligand_charge_source=charge.source_file,
                ligand_ph=charge.ph,
            )
        elif dyn_type == "pligand":
            try:
                files[1] = _validate_dynamics_ligand(files[1], work_dir)
                charge = _resolve_ligand_charge_for_run(
                    files[1], work_dir,
                    explicit_charge=args.charge,
                    mode=args.charge_mode,
                    ph=args.ligand_ph,
                    interactive=False,
                )
            except Exception as exc:
                print_error("Ligand charge error", exc)
                return 1
            config.update(
                pdb_input=files[0], ligand_pdb=files[1],
                ligand_charge=charge.charge,
                ligand_prepared_file=charge.prepared_file,
                ligand_charge_method=charge.method,
                ligand_charge_confidence=charge.confidence,
                ligand_charge_source=charge.source_file,
                ligand_ph=charge.ph,
                replicas=replicas,
            )
        elif dyn_type in ("ppeptide", "pacid", "pprotein"):
            config.update(pdb_protein=files[0], pdb_partner=files[1],
                          pdb_input=os.path.join(work_dir, "complex.pdb"))
        elif dyn_type == "ppligand":
            try:
                files[2] = _validate_dynamics_ligand(files[2], work_dir)
                charge = _resolve_ligand_charge_for_run(
                    files[2], work_dir,
                    explicit_charge=args.charge,
                    mode=args.charge_mode,
                    ph=args.ligand_ph,
                    interactive=False,
                )
            except Exception as exc:
                print_error("Ligand charge error", exc)
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
    dyn_rows = [
        ("Simulation type", label),
        ("Duration", f"{args.time} ns"),
        ("Simulations", str(n_sims)),
    ]
    if dyn_type == "pligand":
        dyn_rows.append(("Replicas per job", str(replicas)))
    dyn_rows += [
        ("CPUs per job", str(args.cpus)),
        ("Memory per job", args.mem),
        ("Wall time", args.time_limit),
    ]
    if args.partition:
        dyn_rows.append(("SLURM partition", args.partition))
    if args.nodes:
        dyn_rows.append(("Nodes", args.nodes))
    dyn_rows.append(("Config list", master_json))
    kv_block(console, dyn_rows, title="HPC Dynamics Configuration")
    console.print()
    if replicas > 1:
        print_warn(
            f"Each job runs {replicas} replicas back to back ({replicas} × {args.time} ns); "
            f"make sure --time-limit {args.time_limit} covers all of them."
        )

    if args.dry_run:
        record_block(
            console, ["#", "Type", "Work dir"],
            [[str(i + 1), label, cfg["work_dir"]] for i, cfg in enumerate(configs)],
            title="Dry run — jobs that would be submitted",
        )
        console.print()
        return 0

    repo = _repo_root()
    script_path = os.path.join(repo, "hpc", "slurm", "native", "run_dynamics_pipeline.sh")
    if not os.path.isfile(script_path):
        print_error("Pipeline script not found", script_path)
        return 1

    charge = getattr(args, "charge", None)
    env_overrides: dict = {
        "REPO_DIR":         repo,
        "DYN_TYPE":         dyn_type,
        "DYN_NS_TIME":      str(args.time),
        "DYN_CHARGE":       "" if charge is None else str(charge),
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

    # --- Results ---
    headers = ["#", "Type", "Work dir"] + (["Node"] if args.nodes else [])
    node_list = args.nodes.split(",") if args.nodes else []
    rows = []
    for i, cfg in enumerate(configs):
        row = [str(i + 1), label, cfg["work_dir"]]
        if args.nodes:
            row.append(node_list[i % len(node_list)])
        rows.append(row)
    record_block(console, headers, rows, title="Submitted Jobs")
    console.print()
    return result.returncode


# ── Shell completion ───────────────────────────────────────────────────────────
# Generated from the argument parser, so completions always match the options
# each command really accepts.

_PATH_METAVARS = {"DIR", "FILE", "PDB", "PDB_DIR", "RUN_DIR", "IMAGE"}


def _completion_tree(parser: ArgumentParser) -> dict:
    """{"chemlink docking full": (words, options, path_options)}, where words
    are (name, help) pairs for subcommands and positional choices, options
    are (flag, help) pairs, and path_options the flags that take a path."""
    from argparse import _SubParsersAction

    tree: dict = {}

    def clean(text) -> str:
        text = re.sub(r"\x1b\[[0-9;]*m", "", str(text or ""))
        text = " ".join(text.split())
        return text if len(text) <= 72 else text[:71] + "…"

    def walk(p: ArgumentParser, key: str) -> None:
        words, options, paths = [], [], []
        for action in p._actions:
            if isinstance(action, _SubParsersAction):
                helps = {a.dest: a.help for a in action._choices_actions}
                for name, sub in action.choices.items():
                    words.append((name, clean(helps.get(name))))
                    walk(sub, f"{key} {name}")
            elif action.option_strings:
                if action.help is SUPPRESS:
                    continue
                for flag in action.option_strings:
                    options.append((flag, clean(action.help)))
                    if str(action.metavar) in _PATH_METAVARS:
                        paths.append(flag)
            elif action.choices:
                words.extend((str(choice), "") for choice in action.choices)
        tree[key] = (words, options, paths)

    walk(parser, "chemlink")
    return tree


def _bash_completion(tree: dict) -> str:
    def assoc(name: str, pairs) -> str:
        body = "\n".join(f'    ["{key}"]="{value}"' for key, value in pairs)
        return f"declare -gA {name}=(\n{body}\n)"

    words = assoc("_chemlink_words", (
        (key, " ".join([w for w, _ in node[0]] + [o for o, _ in node[1]]))
        for key, node in tree.items()
    ))
    paths = assoc("_chemlink_paths", (
        (f"{key} {flag}", 1) for key, node in tree.items() for flag in node[2]
    ))
    return f"""\
# ChemLink bash completion (generated by `chemlink completion bash`)
# Add to ~/.bashrc:  source <(chemlink completion bash)

{words}
{paths}

_chemlink() {{
    local cur="${{COMP_WORDS[COMP_CWORD]}}" prev="${{COMP_WORDS[COMP_CWORD-1]}}"
    local key=chemlink i
    for (( i = 1; i < COMP_CWORD; i++ )); do
        [[ -n "${{_chemlink_words["$key ${{COMP_WORDS[i]}}"]+x}}" ]] && key="$key ${{COMP_WORDS[i]}}"
    done
    if [[ -n "${{_chemlink_paths["$key $prev"]+x}}" ]]; then
        COMPREPLY=( $(compgen -f -- "$cur") )
        return
    fi
    COMPREPLY=( $(compgen -W "${{_chemlink_words[$key]}}" -- "$cur") )
}}
complete -o default -F _chemlink chemlink
"""


def _zsh_completion(tree: dict) -> str:
    def zstr(entries) -> str:
        text = "\n".join(entries).replace("\\", "\\\\").replace("'", "\\'")
        return "$'" + text.replace("\n", "\\n") + "'"

    def assoc(name: str, pairs) -> str:
        body = "\n".join(f"    '{key}' {value}" for key, value in pairs)
        return f"typeset -gA {name}\n{name}=(\n{body}\n)"

    cmds = assoc("_chemlink_cmds", (
        (key, zstr(f"{w}:{h}" if h else w for w, h in node[0])) for key, node in tree.items()
    ))
    opts = assoc("_chemlink_opts", (
        (key, zstr(f"{o}:{h}" if h else o for o, h in node[1])) for key, node in tree.items()
    ))
    paths = assoc("_chemlink_paths", (
        (f"{key} {flag}", "1") for key, node in tree.items() for flag in node[2]
    ))
    return f"""\
#compdef chemlink
# ChemLink zsh completion (generated by `chemlink completion zsh`)
# source <(chemlink completion zsh)

zstyle ':completion:*:chemlink:*:descriptions' format '%F{{cyan}}%B%d%b%f'
zstyle ':completion:*:chemlink:*' group-name ''

{cmds}
{opts}
{paths}

_chemlink() {{
    local key=chemlink i
    for (( i = 2; i < CURRENT; i++ )); do
        (( ${{+_chemlink_opts[$key $words[i]]}} )) && key="$key $words[i]"
    done
    if (( ${{+_chemlink_paths[$key $words[CURRENT-1]]}} )); then
        _files
        return
    fi
    local -a items
    if [[ $PREFIX == -* ]]; then
        items=( ${{(f)_chemlink_opts[$key]}} )
        _describe -t options option items
        return
    fi
    items=( ${{(f)_chemlink_cmds[$key]}} )
    if (( $#items )); then
        _describe -t commands command items
    else
        _files
    fi
}}

compdef _chemlink chemlink
"""


def _run_demo(args: Namespace) -> int:
    try:
        from .demo import run_demo
    except ImportError:
        from demo import run_demo  # type: ignore
    return run_demo(args.scenario, speed=args.speed, fail=args.fail)


def _run_completion(args: Namespace) -> int:
    shell = args.shell
    if shell in ("bash", "zsh"):
        tree = _completion_tree(build_parser())
        print(_bash_completion(tree) if shell == "bash" else _zsh_completion(tree))
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
        print_ok(f"Done. Restart your shell or run  [cl.ref]source {rc}[/]")
    except OSError as exc:
        print_err(f"Could not write to {rc}: {exc}")


# ── Update command handler ─────────────────────────────────────────────────────

def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _git(repo: str, *git_args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", repo, *git_args], capture_output=True, text=True)


def _git_out(repo: str, *git_args: str) -> str:
    return _git(repo, *git_args).stdout.strip()


def _run_update(args: Namespace) -> int:
    repo = _repo_root()

    if _git(repo, "rev-parse", "--is-inside-work-tree").returncode != 0:
        print_err(f"{repo} is not a git checkout — cannot check for updates.")
        return 1

    branch = _git_out(repo, "rev-parse", "--abbrev-ref", "HEAD")
    if branch == "HEAD":
        print_err("The checkout is on a detached HEAD — check out a branch to update.")
        print_info(f"For example: [cl.ref]git -C {repo} checkout main[/]")
        return 1

    with busy(f"Fetching {branch}"):
        fetch = _git(repo, "fetch", "--quiet", "--tags", "origin", branch)
    if fetch.returncode != 0:
        print_err(f"git fetch failed: {fetch.stderr.strip()}")
        return 1

    local_head  = _git_out(repo, "rev-parse", "HEAD")
    remote_head = _git_out(repo, "rev-parse", f"origin/{branch}")
    behind = int(_git_out(repo, "rev-list", "--count", f"{local_head}..{remote_head}") or 0)
    ahead  = int(_git_out(repo, "rev-list", "--count", f"{remote_head}..{local_head}") or 0)
    local_tag  = _git_out(repo, "describe", "--tags", "--abbrev=0", local_head)  or "untagged"
    remote_tag = _git_out(repo, "describe", "--tags", "--abbrev=0", remote_head) or "untagged"

    if behind == 0:
        print_ok(f"Already up to date  ({local_tag}, {local_head[:8]}).")
        if ahead:
            print_info(f"{ahead} local commit(s) ahead of origin/{branch}.")
        return 0

    tag_suffix = f"  ({local_tag} → {remote_tag})" if local_tag != remote_tag else ""
    console.print()
    section(console, "Update available", f"{behind} new commit(s) on origin/{branch}{tag_suffix}")
    log_lines = _git_out(repo, "log", "--oneline", f"{local_head}..{remote_head}").splitlines()
    for line in log_lines[:20]:
        sha, _, subject = line.partition(" ")
        console.print(f"{indent(2)}[cl.accent]{sha}[/]  {escape(subject)}", soft_wrap=True)
    if len(log_lines) > 20:
        console.print(f"{indent(2)}[cl.muted]… and {len(log_lines) - 20} more[/]")
    console.print()

    if args.check:
        print_info("Run [cl.ref]chemlink update[/] (without --check) to apply.")
        return 0

    if ahead:
        print_err(f"{ahead} local commit(s) not on origin/{branch} — refusing to pull. "
                   "Resolve manually (push, rebase, or reset) and re-run.")
        return 1

    if _git_out(repo, "status", "--porcelain"):
        print_err("Working tree has uncommitted changes — commit or stash them before updating.")
        return 1

    if not args.yes and sys.stdin.isatty():
        answer = console.input(
            f"  [cl.brand]?[/] Update [cl.path]{repo}[/] to [bold]{escape(remote_tag)}[/] ({behind} commit(s))? [cl.muted]\\[y/N][/] "
        ).strip().lower()
        if answer not in ("y", "yes"):
            print_info("Update cancelled.")
            return 0

    print_info(f"Pulling origin/{branch}…")
    merge = _git(repo, "merge", "--ff-only", remote_head)
    if merge.returncode != 0:
        print_err(f"git merge --ff-only failed: {merge.stderr.strip()}")
        return 1

    new_head = _git_out(repo, "rev-parse", "--short", "HEAD")
    print_ok(f"Updated to {remote_tag}  ({new_head}).")

    if _git_out(repo, "diff", "--name-only", local_head, remote_head, "--", "requirements.txt"):
        print_warn("requirements.txt changed — run "
                    "[cl.ref]pip install -r requirements.txt[/] in the chemlink environment.")
    return 0


# ── Argument parser ────────────────────────────────────────────────────────────
# Every path option is optional: omitted paths are resolved from the directory
# the command runs in (see cli/workspace.py for the convention).

def _output_arg(p: ArgumentParser, pipeline: str) -> None:
    p.add_argument("-o", "--output", dest="output", metavar="DIR",
                   help=f"Base directory for runs  [./results/{pipeline}]")


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


def _docking_input_args(p: ArgumentParser, *, receptors: bool = True, ligands: bool = True,
                        prepared: bool = False) -> None:
    if receptors:
        p.add_argument("-ir", "--input-receptors", dest="receptors", metavar="DIR",
                       help=("Prepared receptor directory  [<output>/prepared_receptors_pdbqt]" if prepared
                             else "Receptor PDB directory  [./receptors, else *.pdb here]"))
    if ligands:
        p.add_argument("-il", "--input-ligands", dest="ligands", metavar="DIR",
                       help=("Prepared ligand directory  [<output>/prepared_ligands_pdbqt]" if prepared
                             else "Ligand directory  [./ligands, else *.sdf/*.mol2/*.mol/*.pdbqt here]"))
    _output_arg(p, "docking")


def _box_args(p: ArgumentParser, *, hpc: bool = False, redocking_dir: bool = False) -> None:
    p.add_argument("--manual-center", nargs=3, type=float, metavar=("X", "Y", "Z"),
                   help="Grid center in Å for every receptor without a --box-config entry "
                        "(skips fpocket); without --manual-npts the size comes from the ligands")
    p.add_argument("--manual-npts", nargs=3, type=int, metavar=("NX", "NY", "NZ"),
                   help="AutoDock grid dimensions  (requires --manual-center)")
    p.add_argument("--box-config", dest="box_config", metavar="FILE",
                   help="JSON mapping receptor name -> {center, npts}; overrides the manual "
                        "box and fpocket for the receptors it lists")
    if not hpc:
        p.add_argument("--ligand-box-padding", type=float, default=10.0, metavar="ANGSTROM",
                       help="Total padding per axis when the box is sized from the ligands  [10]")
        if redocking_dir:
            p.add_argument("--redocking-center", nargs="?", const="auto", metavar="PDB_DIR",
                           help="Center the box on the co-crystallized ligand of each original "
                                "receptor PDB  [PDB_DIR: ./receptors, else *.pdb here]")
        else:
            p.add_argument("--redocking-center", action="store_true",
                           help="Center the box on the co-crystallized ligand of each input "
                                "receptor PDB; the size still comes from the docking ligands")
    p.add_argument("--flex-radius", type=float, default=0.0, metavar="ANGSTROM",
                   help="Flex side chains within this distance of the box center  [0 = rigid]")


def _engine_args(p: ArgumentParser) -> None:
    p.add_argument("--engine", choices=("gpu", "cpu", "both"), default="both", metavar="ENGINE",
                   help="gpu (AutoDock-GPU), cpu (AutoDock Vina) or both, run concurrently "
                        "and compared in docking_comparison.csv  [both]")
    p.add_argument("--gpu-nrun", dest="gpu_nrun", type=int, default=50, metavar="N",
                   help="AutoDock-GPU independent GA runs per ligand  (gpu)  [50]")
    p.add_argument("--gpu-nev", dest="gpu_nev", type=int, default=2_500_000, metavar="N",
                   help="AutoDock-GPU max energy evaluations per GA run  (gpu)  [2500000]")
    p.add_argument("--gpu-timeout", dest="gpu_timeout", type=int, default=600, metavar="SECONDS",
                   help="Per-job AutoDock-GPU timeout; slower jobs are marked failed  (gpu)  [600]")
    p.add_argument("--exhaustiveness", type=int, default=8, metavar="N",
                   help="Vina search exhaustiveness  (cpu)  [8]")
    p.add_argument("--num-modes", dest="num_modes", type=int, default=9, metavar="N",
                   help="Max poses Vina reports per ligand  (cpu)  [9]")
    p.add_argument("--vina-cpu", dest="vina_cpu", type=int, default=None, metavar="N",
                   help="CPU threads per Vina job  (cpu)  [cores / docking workers]")
    p.add_argument("--seed", type=int, default=None, metavar="N",
                   help="Vina random seed for reproducible runs  (cpu)")


def _docking_pipeline_args(p: ArgumentParser, *, execution: bool) -> None:
    _docking_input_args(p)
    _run_name_arg(p)
    p.add_argument("--keep-cofactors", action="store_true",
                   help="Keep heme groups and catalytic metal ions (Mg, Zn, Mn, Ca, Fe) "
                        "in the receptor instead of stripping them")
    _box_args(p)
    p.add_argument("-r", "--receptor-workers", type=int, metavar="N",
                   help="Parallel workers for receptor preparation")
    p.add_argument("-l", "--ligand-workers", type=int, metavar="N",
                   help="Parallel workers for ligand preparation")
    p.add_argument("-a", "--active-site-workers", type=int, metavar="N",
                   help="Parallel workers for active-site detection")
    if execution:
        p.add_argument("-d", "--docking-workers", type=int, default=1, metavar="N",
                       help="Concurrent docking jobs  [1]")
        _engine_args(p)
        p.add_argument("--pdb-export-limit", type=int, default=10, metavar="N",
                       help="Top candidates to export as PDB  [10]")


def _peptide_input_args(p: ArgumentParser) -> None:
    p.add_argument("-ip", "--input-peptides", dest="peptides", metavar="DIR",
                   help="Directory of peptide SDF files  [./peptides, else *.sdf here]")
    p.add_argument("-ir", "--input-receptor", dest="receptor", metavar="PDB",
                   help="Receptor PDB (single chain, protonated/clean)  "
                        "[the only *.pdb in ./receptors or here]")
    _output_arg(p, "peptide-docking")


def _lightdock_args(p: ArgumentParser) -> None:
    p.add_argument("--standardize", action="store_true",
                   help="Mutate non-canonical residues to ALA/GLY instead of excluding the "
                        "peptide (the original is kept under original_peptides_pdb/ for MD)")
    p.add_argument("--dedup", action="store_true",
                   help="Dock one representative per real structure (isotope-labeled "
                        "variants of the same compound score near-identically)")
    p.add_argument("--prep-workers", type=int, default=None, metavar="N",
                   help="Parallel workers for preparation  [all CPUs]")
    p.add_argument("--docking-workers", type=int, default=1, metavar="N",
                   help="Peptides docked concurrently  [1]")
    p.add_argument("--cores-per-job", type=int, default=None, metavar="N",
                   help="Cores per LightDock run  [cores / docking workers]")
    p.add_argument("--gso-steps", type=int, default=10, metavar="N",
                   help="GSO simulation steps  [10]")
    p.add_argument("--glowworms", type=int, default=20, metavar="N",
                   help="Glowworms per swarm  [20]")
    p.add_argument("--anm-lig", type=int, default=4, metavar="N",
                   help="ANM normal modes for ligand flexibility  [4]")
    p.add_argument("--swarms", type=int, default=None, metavar="N",
                   help="Fixed swarm count  [auto, from the restraint-site surface area]")


def _robust_gate_args(p: ArgumentParser) -> None:
    p.add_argument("--restraints", dest="restraints_file", metavar="FILE",
                   help="LightDock restraints file, one 'R <chain>.<resname>.<resnum>' line "
                        "per binding-site residue  [the only restraints* file here]")
    p.add_argument("--reference", dest="reference_cid", required=True, metavar="CID",
                   help="Peptide filename (without .sdf) in the batch to gate every other "
                        "candidate against, e.g. an approved drug; only candidates that beat "
                        "it on both score and site distance continue")
    p.add_argument("--max-candidates", type=int, default=None, metavar="N",
                   help="Cap the gate to the top N candidates by score  [no cap]")


def _robust_run_dir_arg(p: ArgumentParser) -> None:
    p.add_argument("run_dir", nargs="?", metavar="RUN_DIR",
                   help="Run created by 'robust lightdock'  [this directory if it is one, "
                        "else the latest under ./results/peptide-docking]")


# ── dynamics: per-type options ──────────────────────────────────────────────────
# Binding-energy (gmx_MMPBSA) options each simulation type actually uses; the
# others either have no binding partner or fix the setting internally.
_DYN_BINDING_OPTIONS = {
    "pligand":  ("method", "frames", "discard", "cores", "salt", "entropy"),
    "ppeptide": ("frames", "discard", "cores", "salt", "entropy"),
    "pprotein": ("frames", "discard", "cores", "salt", "entropy"),
    "ppligand": ("cores", "entropy"),
}
_DYN_LIGAND_TYPES = ("pligand", "ppligand", "oligand")
_DYN_INPUT_HELP = {
    "oprotein": ("Protein PDB", None),
    "pligand":  ("Protein PDB", "Ligand (SDF/MOL/MOL2/PDB/PDBQT)"),
    "ppeptide": ("Protein PDB", "Peptide PDB"),
    "pacid":    ("Protein PDB", "Nucleic-acid PDB"),
    "pprotein": ("Protein PDB", "Partner protein PDB"),
    "ppligand": ("Protein PDB (repeat for both proteins)", "Ligand (SDF/MOL/MOL2/PDB/PDBQT)"),
    "oligand":  (None, "Ligand (SDF/MOL/MOL2/PDB/PDBQT)"),
}


def _mmpbsa_args(p: ArgumentParser, options, *, overrides: bool = False) -> None:
    """gmx_MMPBSA options; with ``overrides`` every default is None so only
    what the user passes replaces the settings saved in the run."""
    def default(value):
        return None if overrides else value

    saved = "  [saved in the run]" if overrides else ""
    if "method" in options:
        p.add_argument("--mmpbsa-method", choices=("gb", "pb", "both"), default=default("both"),
                       help=f"Implicit-solvent method{saved or '  [both]'}")
    if "frames" in options:
        p.add_argument("--mmpbsa-frames", type=_positive_int, default=default(1000), metavar="N",
                       help=f"Max uniformly sampled trajectory frames{saved or '  [1000]'}")
    if "discard" in options:
        p.add_argument("--mmpbsa-discard", type=float, default=default(20.0), metavar="PCT",
                       help=f"Initial trajectory percentage to discard{saved or '  [20]'}")
    if "cores" in options:
        p.add_argument("--mmpbsa-cores", type=_positive_int, default=default(6), metavar="N",
                       help=f"MPI processes for gmx_MMPBSA{saved or '  [6]'}")
    if "salt" in options:
        p.add_argument("--mmpbsa-salt", type=float, default=default(0.150), metavar="M",
                       help=f"Salt concentration in mol/L{saved or '  [0.150]'}")
    if "entropy" in options:
        # Re-analysis turns IE on even when the run saved "none" (docs/Entropia.md).
        p.add_argument("--mmpbsa-entropy", choices=("none", "ie", "c2"), default="ie",
                       help="Entropy correction: interaction entropy, C2 or none  [ie]")


def _dyn_type_args(p: ArgumentParser, name: str, *, hpc: bool) -> None:
    receptor_help, ligand_help = _DYN_INPUT_HELP[name]
    auto = "  [auto-detected here]"
    if receptor_help:
        p.add_argument("-ir", "--input-receptors", action="append", dest="input_receptors",
                       metavar="FILE", help=receptor_help + auto)
    if ligand_help:
        repeat = "; repeat to submit one job per ligand" if hpc and name == "pligand" else ""
        p.add_argument("-il", "--input-ligands", action="append", dest="input_ligands",
                       metavar="FILE", help=ligand_help + repeat + auto)
    if name == "pligand" and not hpc:
        p.add_argument("-ic", "--input-cofactor", action="append", dest="input_cofactor",
                       metavar="FILE",
                       help="Optional cofactor: a monoatomic ion or a small inorganic molecule")
    _output_arg(p, "dynamics")
    p.add_argument("-t", "--time", type=float, default=100.0 if hpc else 0.1, metavar="NS",
                   help=f"Simulation time in nanoseconds  [{100.0 if hpc else 0.1}]")
    if name in _DYN_LIGAND_TYPES:
        p.add_argument("-c", "--charge", type=int, metavar="Q",
                       help="Ligand net charge (overrides automatic detection)")
        modes = ("auto", "strict") if hpc else ("auto", "strict", "prompt")
        p.add_argument("--charge-mode", choices=modes, default="auto",
                       help="Charge resolution: auto, strict (fail when uncertain)"
                            + ("" if hpc else " or prompt") + "  [auto]")
        p.add_argument("--ligand-ph", type=float, default=7.4, metavar="PH",
                       help="pH used to protonate PDB/PDBQT ligands  [7.4]")
    if name == "pligand":
        if not hpc:
            p.add_argument("--cofactor-charge", type=int, metavar="Q",
                           help="Net charge of a polyatomic cofactor (overrides detection)")
            p.add_argument("--cofactor-charge-method", choices=("bcc", "gas"), default="bcc",
                           help="Cofactor partial charges: bcc (AM1-BCC, accurate, slow) or "
                                "gas (Gasteiger, near-instant)  [bcc]")
            p.add_argument("--cofactor-charge-timeout", type=int, metavar="SECONDS",
                           help="ACPYPE timeout for a polyatomic cofactor  [3600]")
        p.add_argument("--replicas", type=_positive_int, default=DEFAULT_MD_REPLICAS, metavar="N",
                       help="Independent MD replicas from the same minimised system  "
                            f"[{DEFAULT_MD_REPLICAS}]")
    if name in _DYN_BINDING_OPTIONS:
        p.add_argument("--no-binding-energy", dest="binding_energy", action="store_false",
                       help="Skip the automatic MM-GBSA/MM-PBSA analysis")
        _mmpbsa_args(p, _DYN_BINDING_OPTIONS[name])
    if name in {"oprotein", "ppeptide", "pprotein"} and not hpc:
        p.add_argument("--force-field", default="default", metavar="NAME",
                       help="GROMACS force field without .ff, e.g. amber99sb-ildn, "
                            "charmm36-jul2022  [default]")
        p.add_argument("--force-field-dir", metavar="DIR",
                       help="Directory containing NAME.ff when it is not a GROMACS built-in")
        p.add_argument("--water-model", default="tip3p", metavar="NAME",
                       help="Water model for the selected force field  [tip3p]")
    if hpc:
        p.add_argument("--nodes", metavar="NODELIST",
                       help="SLURM nodes (comma-separated); jobs are spread across them")
        p.add_argument("--partition", metavar="PARTITION", help="SLURM partition")
        p.add_argument("--time-limit", dest="time_limit", default="72:00:00", metavar="HH:MM:SS",
                       help="Wall time per job  [72:00:00]")
        p.add_argument("--mem", default="0", metavar="MEM",
                       help="Memory per job; 0 = all memory on the node  [0]")
        p.add_argument("--cpus", type=int, default=8, metavar="N", help="CPUs per job  [8]")
        p.add_argument("--dry-run", action="store_true",
                       help="Show the jobs without submitting them")
    _run_name_arg(p)


_DYN_TYPES = [
    (
        "oprotein",
        "Protein-only simulation",
        (
            f"  {bold_cyan('oprotein')}  — Protein-only GROMACS simulation.\n\n"
            f"  Runs the full simulation and analysis flow for a single protein structure.\n"
            f"  Input: 1 protein PDB."
        ),
    ),
    (
        "pligand",
        "Protein + small-molecule ligand",
        (
            f"  {bold_cyan('pligand')}  — Protein + small-molecule ligand simulation.\n\n"
            f"  Adds ligand topology and MM-GBSA/PBSA binding energy with interaction entropy.\n"
            f"  Input: protein PDB + ligand (SDF/MOL/MOL2/PDB/PDBQT); the ligand charge is\n"
            f"  detected automatically, {cyan('-c')} overrides it.\n\n"
            f"  {cyan('-ic')} adds a cofactor: a monoatomic ion (Zn2+, Mg2+, Ca2+…) or a small\n"
            f"  inorganic molecule parametrised with GAFF2/ACPYPE. AM1-BCC charges can take\n"
            f"  minutes on a few dozen atoms; {cyan('--cofactor-charge-method gas')} is near-instant."
        ),
    ),
    (
        "ppeptide",
        "Protein + peptide complex",
        (
            f"  {bold_cyan('ppeptide')}  — Protein + peptide complex simulation.\n\n"
            f"  The two structures are merged into a complex before the simulation starts;\n"
            f"  analysis includes the interaction energy and MM-GBSA binding energy.\n"
            f"  Input: 2 PDB files (protein, peptide); auto-detection takes the smaller as the peptide."
        ),
    ),
    (
        "pacid",
        "Protein + nucleic acid complex",
        (
            f"  {bold_cyan('pacid')}  — Protein + nucleic acid complex simulation.\n\n"
            f"  The two structures are merged into a complex before the simulation starts;\n"
            f"  analysis shows RMSD variation and DNA conformational fitting.\n"
            f"  Input: 2 PDB files (protein, nucleic acid)."
        ),
    ),
    (
        "pprotein",
        "Protein + protein complex",
        (
            f"  {bold_cyan('pprotein')}  — Protein + protein complex simulation.\n\n"
            f"  The two structures are merged into a complex before the simulation starts;\n"
            f"  analysis includes the interaction energy and MM-GBSA binding energy.\n"
            f"  Input: 2 protein PDB files."
        ),
    ),
    (
        "ppligand",
        "Protein + protein + ligand",
        (
            f"  {bold_cyan('ppligand')}  — Protein + protein + ligand simulation.\n\n"
            f"  Combines {cyan('pligand')} and {cyan('pprotein')}: the proteins are merged into a\n"
            f"  complex, the ligand is parametrised and the combined analysis runs.\n"
            f"  Input: 2 protein PDBs + ligand (SDF/MOL/MOL2/PDB/PDBQT)."
        ),
    ),
    (
        "oligand",
        "Ligand only",
        (
            f"  {bold_cyan('oligand')}  — Ligand-only GROMACS simulation in explicit water.\n\n"
            f"  GAFF2/ACPYPE parameters, solvation, neutralisation, minimisation, equilibration\n"
            f"  and production; analysis covers RMSD, Rg, SASA, RMSF and ligand-water contacts.\n"
            f"  Input: 1 ligand (SDF/MOL/MOL2/PDB/PDBQT)."
        ),
    ),
]


def build_parser() -> _Parser:
    epilog = (
        f"  {bold_cyan('EXAMPLES')}\n"
        f"    {cyan('chemlink')}                                   {dim('# interactive launcher')}\n"
        f"    {cyan('chemlink demo')}                              {dim('# preview the interface, no tools/GPU needed')}\n"
        f"    {cyan('chemlink doctor')}                            {dim('# check tools, GPU and paths')}\n"
        f"    {cyan('chemlink docking full')} --engine cpu         {dim('# inputs from ./receptors and ./ligands')}\n"
        f"    {cyan('chemlink dynamic pligand')} -t 10             {dim('# protein + ligand found in this directory')}\n"
        f"    {cyan('chemlink hpc docking')} --dry-run\n"
        f"\n  {bold_cyan('PATHS')}  {dim('(every path option is optional)')}\n"
        f"    Inputs come from ./receptors, ./ligands and ./peptides when they exist, else from\n"
        f"    the matching files in the current directory. Results go to ./results/<pipeline>.\n"
        f"\n  {bold_cyan('GLOBAL FLAGS')}  {dim('(accepted anywhere on the command line)')}\n"
        f"    {cyan('--plain')}      log-style progress lines instead of live bars (SLURM, pipes)\n"
        f"    {cyan('--no-color')}   disable colour  {dim('(or NO_COLOR=1)')}\n"
        f"    {cyan('-v')}, {cyan('-q')}       more / less console logging\n"
        f"    {cyan('--debug')}      full tracebacks on errors"
    )

    parser = _Parser(
        prog="chemlink",
        formatter_class=_Fmt,
        epilog=epilog,
        description=f"  {TAGLINE} — docking, molecular dynamics, free energy and HPC orchestration.",
    )
    parser.add_argument("--version", "-V", action="version", version=f"chemlink {VERSION}")
    parser.add_argument("--plain", action="store_true", help=SUPPRESS)
    parser.add_argument("--no-color", action="store_true", help=SUPPRESS)
    parser.add_argument("-v", "--verbose", action="store_true", help=SUPPRESS)
    parser.add_argument("-q", "--quiet", action="store_true", help=SUPPRESS)
    parser.add_argument("--debug", action="store_true", help=SUPPRESS)

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # ── docking ────────────────────────────────────────────────────────────────
    dock_desc = (
        f"  {bold_cyan('docking')}  — Run molecular docking pipelines.\n\n"
        f"  Pipeline steps (in order):\n"
        f"    {cyan('receptor')}  →  {cyan('ligand')}  →  {cyan('active-site')}  "
        f"→  {cyan('execution')}  →  {cyan('analysis')}\n\n"
        f"  Inputs default to ./receptors and ./ligands (else the matching files in the\n"
        f"  current directory); runs are written to ./results/docking."
    )
    dock = sub.add_parser("docking", help="Molecular docking workflows",
                          description=dock_desc, formatter_class=_Fmt)
    dock_sub = dock.add_subparsers(dest="docking_command", metavar="SUBCOMMAND")

    prep = dock_sub.add_parser("prepare", formatter_class=_Fmt,
                               help="Preparation only  (receptor + ligand + active-site)")
    _docking_pipeline_args(prep, execution=False)
    prep.set_defaults(handler=lambda a: _run_docking_pipeline(a, full=False))

    full_p = dock_sub.add_parser("full", formatter_class=_Fmt,
                                 help="Full pipeline  (preparation + execution + analysis)")
    _docking_pipeline_args(full_p, execution=True)
    full_p.set_defaults(handler=lambda a: _run_docking_pipeline(a, full=True))

    flow_p = dock_sub.add_parser("flow", formatter_class=_Fmt,
                                 help="Run any contiguous subset of pipeline steps")
    _docking_pipeline_args(flow_p, execution=True)
    steps = " | ".join(PIPELINE_STEPS)
    flow_p.add_argument("--from-step", choices=PIPELINE_STEPS, default="receptor", metavar="STEP",
                        help=f"First pipeline step  [receptor]  ({steps})")
    flow_p.add_argument("--to-step", choices=PIPELINE_STEPS, default="analysis", metavar="STEP",
                        help=f"Last pipeline step  [analysis]  ({steps})")
    flow_p.set_defaults(handler=_run_docking_flow)

    rec_p = dock_sub.add_parser("receptor", formatter_class=_Fmt,
                                help="Prepare receptor PDB files to PDBQT")
    _docking_input_args(rec_p, ligands=False)
    rec_p.add_argument("-w", "--workers", type=int, default=1, metavar="N",
                       help="Parallel workers  [1]")
    rec_p.add_argument("--keep-cofactors", action="store_true",
                       help="Keep heme groups and catalytic metal ions in the receptor")
    rec_p.set_defaults(handler=_run_receptor_preparation)

    lig_p = dock_sub.add_parser("ligand", formatter_class=_Fmt,
                                help="Prepare ligand files to PDBQT")
    _docking_input_args(lig_p, receptors=False)
    lig_p.add_argument("-w", "--workers", type=int, default=1, metavar="N",
                       help="Parallel workers  [1]")
    lig_p.set_defaults(handler=_run_ligand_preparation)

    as_p = dock_sub.add_parser("active-site", formatter_class=_Fmt,
                               help="Detect binding sites for prepared receptors")
    _docking_input_args(as_p, prepared=True)
    as_p.add_argument("-w", "--workers", type=int, default=1, metavar="N",
                      help="Parallel workers  [1]")
    _box_args(as_p, redocking_dir=True)
    as_p.set_defaults(handler=_run_active_site)

    run_p = dock_sub.add_parser("run", formatter_class=_Fmt,
                                help="Docking execution on prepared receptors and ligands")
    _docking_input_args(run_p, prepared=True)
    run_p.add_argument("-w", "--workers", type=int, default=1, metavar="N",
                       help="Concurrent docking jobs  [1]")
    _engine_args(run_p)
    run_p.add_argument("--box-config", dest="box_config", metavar="FILE",
                       help="JSON receptor -> {center, npts}  (cpu; else the GPF files are used)")
    run_p.set_defaults(handler=_run_docking_execution)

    ana_p = dock_sub.add_parser("analyze", formatter_class=_Fmt,
                                help="Analyze DLG results and generate reports")
    ana_p.add_argument("run_dir", nargs="?", metavar="RUN_DIR",
                       help="Docking run to analyze  [this directory if it is one, "
                            "else the latest under ./results/docking]")
    ana_p.add_argument("--pdb-export-limit", type=int, default=10, metavar="N",
                       help="Top candidates to export as PDB  [10]")
    ana_p.set_defaults(handler=_run_docking_analysis)

    # ── peptide-docking ───────────────────────────────────────────────────────
    pep_desc = (
        f"  {bold_cyan('peptide-docking')}  — Peptide-protein docking via LightDock.\n\n"
        f"  Pipeline: dedup (optional) -> preparation (SDF -> PDB, standardize, "
        f"per-peptide QC) -> LightDock docking -> ranking.csv (best score per peptide).\n\n"
        f"  {cyan('robust')} extends this into a cross-validation pipeline: gate candidates "
        f"against a known reference, then PRODIGY + HADDOCK3 + PyRosetta/FlexPepDock, "
        f"merged into one combined ranking. Run it stage by stage, or {cyan('all')} at once.\n\n"
        f"  Inputs default to ./peptides (else *.sdf here), the only receptor PDB in\n"
        f"  ./receptors or here, and a restraints* file here; runs go to ./results/peptide-docking."
    )
    pep = sub.add_parser("peptide-docking", help="Peptide-protein docking workflows (LightDock)",
                         description=pep_desc, formatter_class=_Fmt)
    pep_sub = pep.add_subparsers(dest="peptide_docking_command", metavar="SUBCOMMAND")

    pep_full = pep_sub.add_parser("full", formatter_class=_Fmt,
                                  help="Full pipeline  (preparation + docking + ranking)")
    _peptide_input_args(pep_full)
    _run_name_arg(pep_full)
    pep_full.add_argument("--restraints", dest="restraints_file", metavar="FILE",
                          help="LightDock restraints file, one 'R <chain>.<resname>.<resnum>' "
                               "line per binding-site residue  [the only restraints* file here, if any]")
    pep_full.add_argument("--reference", dest="reference_cids", action="append", metavar="CID",
                          help="Peptide filename (without .sdf) to flag as a known reference in "
                               "ranking.csv; repeat for several")
    _lightdock_args(pep_full)
    pep_full.set_defaults(handler=_run_peptide_docking)

    pep_robust = pep_sub.add_parser("robust", formatter_class=_Fmt,
                                    help="Robustness pipeline (triage + gate + PRODIGY + "
                                         "HADDOCK3 + PyRosetta/FlexPepDock + combined ranking)")
    robust_sub = pep_robust.add_subparsers(dest="robust_stage", metavar="STAGE", required=True)

    r_lightdock = robust_sub.add_parser(
        "lightdock", formatter_class=_Fmt,
        help="Stage 1+2: LightDock triage + gate vs. reference -- creates the run",
    )
    _peptide_input_args(r_lightdock)
    _run_name_arg(r_lightdock)
    _robust_gate_args(r_lightdock)
    _lightdock_args(r_lightdock)
    r_lightdock.add_argument("--resume", action="store_true",
                             help="Continue the --run-name run, skipping peptides already docked")
    r_lightdock.set_defaults(handler=_run_peptide_docking_robust_lightdock)

    r_audit = robust_sub.add_parser(
        "audit", formatter_class=_Fmt,
        help="Stage 3: non-canonical/modified-chemistry audit of the gate candidates",
    )
    _robust_run_dir_arg(r_audit)
    r_audit.add_argument("--audit-workers", type=int, default=None, metavar="N",
                         help="Parallel workers  [all CPUs]")
    r_audit.set_defaults(handler=_run_peptide_docking_robust_audit)

    r_prodigy = robust_sub.add_parser(
        "prodigy", formatter_class=_Fmt,
        help="Stage 4: PRODIGY binding-affinity cross-check",
    )
    _robust_run_dir_arg(r_prodigy)
    r_prodigy.add_argument("--prodigy-workers", type=int, default=None, metavar="N",
                           help="Parallel workers  [min(CPUs, candidates)]")
    r_prodigy.set_defaults(handler=_run_peptide_docking_robust_prodigy)

    r_haddock3 = robust_sub.add_parser(
        "haddock3", formatter_class=_Fmt,
        help="Stage 5: HADDOCK3 rigidbody/flexref/emref refinement + convergence",
    )
    _robust_run_dir_arg(r_haddock3)
    r_haddock3.add_argument("--haddock3-workers", type=int, default=None, metavar="N",
                            help="Candidates refined concurrently  [4]")
    r_haddock3.add_argument("--haddock3-sampling", type=int, default=500, metavar="N",
                            help="Rigidbody sampling size  [500]")
    r_haddock3.add_argument("--haddock3-ncores", type=int, default=None, metavar="N",
                            help="CNS processes per candidate  [cores / haddock3 workers]")
    r_haddock3.set_defaults(handler=_run_peptide_docking_robust_haddock3)

    r_rosetta = robust_sub.add_parser(
        "rosetta", formatter_class=_Fmt,
        help="Stage 6: PyRosetta/FlexPepDock refinement",
    )
    _robust_run_dir_arg(r_rosetta)
    r_rosetta.add_argument("--rosetta-workers", type=int, default=None, metavar="N",
                           help="Candidates refined concurrently  [4]")
    r_rosetta.add_argument("--flexpepdock-trials", type=int, default=10, metavar="N",
                           help="Independent-seed FlexPepDock trials per candidate  [10]")
    r_rosetta.set_defaults(handler=_run_peptide_docking_robust_rosetta)

    r_combine = robust_sub.add_parser(
        "combine", formatter_class=_Fmt,
        help="Stage 7: merge every gathered signal into one rank-sum ranking",
    )
    _robust_run_dir_arg(r_combine)
    r_combine.set_defaults(handler=_run_peptide_docking_robust_combine)

    r_all = robust_sub.add_parser(
        "all", formatter_class=_Fmt,
        help="Every stage chained in one call (triage -> gate -> audit -> prodigy -> "
             "haddock3 -> rosetta -> combine)",
    )
    _peptide_input_args(r_all)
    _run_name_arg(r_all)
    _robust_gate_args(r_all)
    _lightdock_args(r_all)
    r_all.add_argument("--audit-workers", type=int, default=None, metavar="N",
                       help="Parallel workers for the standardization audit  [all CPUs]")
    r_all.add_argument("--prodigy-workers", type=int, default=None, metavar="N",
                       help="Parallel workers for PRODIGY  [min(CPUs, candidates)]")
    r_all.add_argument("--haddock3-workers", type=int, default=None, metavar="N",
                       help="Candidates refined through HADDOCK3 concurrently  [4]")
    r_all.add_argument("--haddock3-sampling", type=int, default=500, metavar="N",
                       help="HADDOCK3 rigidbody sampling size  [500]")
    r_all.add_argument("--haddock3-ncores", type=int, default=None, metavar="N",
                       help="CNS processes per HADDOCK3 candidate  [cores / haddock3 workers]")
    r_all.add_argument("--rosetta-workers", type=int, default=None, metavar="N",
                       help="Candidates refined through PyRosetta concurrently  [4]")
    r_all.add_argument("--flexpepdock-trials", type=int, default=10, metavar="N",
                       help="Independent-seed FlexPepDock trials per candidate  [10]")
    r_all.add_argument("--skip-haddock3", action="store_true", help="Skip the HADDOCK3 stage")
    r_all.add_argument("--skip-rosetta", action="store_true",
                       help="Skip the PyRosetta/FlexPepDock stage")
    r_all.add_argument("--resume", action="store_true",
                       help="Continue the --run-name run; every stage skips work already done")
    r_all.set_defaults(handler=_run_peptide_docking_robust_all)

    # ── dynamic ────────────────────────────────────────────────────────────────
    dyn_desc = (
        f"  {bold_cyan('dynamic')}  — Run GROMACS-based molecular dynamics simulations.\n\n"
        f"  Simulation types:\n\n"
        f"    {cyan('oprotein')}   Protein only\n"
        f"    {cyan('pligand')}    Protein + small-molecule ligand  (optional cofactor)\n"
        f"    {cyan('ppeptide')}   Protein + peptide\n"
        f"    {cyan('pacid')}      Protein + nucleic acid\n"
        f"    {cyan('pprotein')}   Protein + protein\n"
        f"    {cyan('ppligand')}   Protein + protein + ligand\n"
        f"    {cyan('oligand')}    Ligand only\n"
        f"    {cyan('fep')}        Binding free energy (ABFE/RBFE)\n\n"
        f"  Input files are detected in the current directory (and ./receptors, ./ligands)\n"
        f"  when {cyan('-ir')}/{cyan('-il')} are omitted; runs are written to ./results/dynamics."
    )
    dyn = sub.add_parser("dynamic", help="Molecular dynamics simulations",
                         description=dyn_desc, formatter_class=_Fmt)
    dyn_sub = dyn.add_subparsers(dest="dyn_type", metavar="TYPE")

    for name, help_text, desc in _DYN_TYPES:
        dp = dyn_sub.add_parser(name, help=help_text, description=desc, formatter_class=_Fmt)
        _dyn_type_args(dp, name, hpc=False)
        dp.set_defaults(handler=_run_dynamic, dyn_type=name)

    fep_p = dyn_sub.add_parser(
        "fep", help="Binding free-energy simulation (ABFE/RBFE)",
        description=(
            f"  {bold_cyan('fep')} — Prepare and run RBFE or ABFE simulations.\n\n"
            "  PDB ligands are converted internally to validated SDF structures.\n"
            "  For ABFE this single command builds the four systems, validates the Boresch\n"
            "  restraints, then runs the pilot, production and analysis. Use --prepare-only or\n"
            "  --build-only to stop earlier, and --resume to continue an existing run.\n\n"
            "  Without -p/-a/-b the protein PDB and the two ligands are detected in the\n"
            "  current directory (ligands A and B in file-name order)."
        ),
        formatter_class=_Fmt,
    )
    fep_p.add_argument("-p", "--protein", metavar="PDB", help="Protein PDB  [auto-detected here]")
    fep_p.add_argument("-a", "--ligand-a", metavar="FILE", help="Ligand A  [auto-detected here]")
    fep_p.add_argument("-b", "--ligand-b", metavar="FILE", help="Ligand B  [auto-detected here]")
    _output_arg(fep_p, "fep")
    fep_p.add_argument("--charge-a", type=int, metavar="Q", help="Net charge of ligand A")
    fep_p.add_argument("--charge-b", type=int, metavar="Q", help="Net charge of ligand B")
    fep_p.add_argument("--charge-mode", choices=("auto", "strict", "prompt"), default="auto",
                       help="Charge resolution: auto, strict or prompt  [auto]")
    fep_p.add_argument("--ligand-ph", type=float, default=7.4, metavar="PH",
                       help="pH used to protonate PDB/PDBQT ligands  [7.4]")
    fep_p.add_argument("--smiles-a", metavar="SMILES",
                       help="Reference chemistry for ligand A bond orders (PDB input)")
    fep_p.add_argument("--smiles-b", metavar="SMILES",
                       help="Reference chemistry for ligand B bond orders (PDB input)")
    fep_p.add_argument("--fep-type", choices=("auto", "rbfe", "abfe"), default="auto",
                       help="Protocol; auto selects ABFE for dissimilar ligands  [auto]")
    fep_p.add_argument("--replicas", type=_positive_int, default=3, metavar="N",
                       help="Independent replicas  [3]")
    fep_p.add_argument("--lambda-windows", type=_positive_int, default=None, metavar="N",
                       help="Lambda windows per replica and leg  [12 RBFE, 20 ABFE]")
    fep_p.add_argument("--equilibration-time", type=float, default=None, metavar="NS",
                       help="Equilibration per phase (NVT/NPT) in ns  [0.5]")
    fep_p.add_argument("--production-time", type=float, default=5.0, metavar="NS",
                       help="Production per lambda window in ns  [5.0]")
    fep_p.add_argument("--temperature", type=float, default=300.0, metavar="K",
                       help="Simulation temperature  [300]")
    fep_p.add_argument("--force-field", choices=("amber99sb-ildn", "amber03"),
                       default="amber99sb-ildn", metavar="NAME",
                       help="amber99sb-ildn or amber03  [amber99sb-ildn]")
    fep_p.add_argument("--water-model", default="tip3p", metavar="NAME", help="Water model  [tip3p]")
    fep_p.add_argument("--prepare-only", action="store_true",
                       help="Validate inputs and create the run layout without executing")
    fep_p.add_argument("--build-only", action="store_true",
                       help="ABFE: build the systems and validate the Boresch restraints, then stop")
    fep_p.add_argument("--resume", nargs="?", const="auto", metavar="RUN_DIR",
                       help="Continue an ABFE run (pilot, production, analysis)  "
                            "[RUN_DIR: the latest under ./results/fep]")
    _run_name_arg(fep_p)
    fep_p.set_defaults(handler=_run_dynamic_fep, dyn_type="fep")

    md_run_dir_help = ("Dynamics run  [this directory if it is one, else the latest "
                       "under ./results/dynamics]")
    resume_p = dyn_sub.add_parser(
        "resume", formatter_class=_Fmt,
        help="Resume an interrupted dynamics run from its latest checkpoint",
        description=(
            f"  {bold_cyan('resume')} — Continue an interrupted ChemLink dynamics run.\n\n"
            "  Detects NVT, NPT, production, post-processing, or analysis state;\n"
            "  validates checkpoint/TPR pairs and skips completed stages."
        ),
    )
    resume_p.add_argument("run_dir", nargs="?", metavar="RUN_DIR", help=md_run_dir_help)
    resume_p.add_argument("--type", dest="sim_type", choices=list(_DYN_SPEC.keys()),
                          help="Simulation type, only for runs without a saved configuration")
    resume_p.add_argument("--time", type=float, metavar="NS",
                          help="Target duration, only for runs without a saved configuration")
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
    analyze_p.add_argument("run_dir", nargs="?", metavar="RUN_DIR", help=md_run_dir_help)
    analyze_p.add_argument("--only", choices=("general", "binding", "all"), default="all",
                           help="Analysis to run  [all]")
    analyze_p.add_argument("--plot-only", action="store_true",
                           help="Regenerate general-analysis plots from existing data")
    analyze_p.add_argument("--type", dest="sim_type", choices=list(_DYN_SPEC.keys()),
                           help="Simulation type, only for runs without a saved configuration")
    analyze_p.add_argument("--time", type=float, metavar="NS",
                           help="Duration, only for runs without a saved configuration")
    _mmpbsa_args(analyze_p, _DYN_BINDING_OPTIONS["pligand"], overrides=True)
    analyze_p.set_defaults(handler=_run_dynamic_analyze)

    # ── demo ───────────────────────────────────────────────────────────────────
    demo_desc = (
        f"  {bold_cyan('demo')}  — Preview ChemLink's interface with a simulated run.\n\n"
        f"  Nothing is computed and no external tools, GPU or input files are needed:\n"
        f"  the demo drives the real progress bars, logs, cards and error panels with\n"
        f"  synthetic timings, so you can check how a run looks on any machine.\n\n"
        f"    {cyan('chemlink demo')}                    {dim('# docking + dynamics')}\n"
        f"    {cyan('chemlink demo dynamics --fail')}    {dim('# include a failure to see error reporting')}\n"
        f"    {cyan('chemlink demo --plain')}            {dim('# how it looks in a SLURM log file')}"
    )
    demo = sub.add_parser("demo", help="Preview the interface with a simulated run (no tools or GPU needed)",
                          description=demo_desc, formatter_class=_Fmt)
    demo.add_argument("scenario", nargs="?", choices=("all", "docking", "dynamics", "logs"), default="all",
                      help="What to simulate  [all]")
    demo.add_argument("--speed", type=float, default=1.0, metavar="X",
                      help="Playback speed multiplier  [1.0; 3 = three times faster]")
    demo.add_argument("--fail", action="store_true",
                      help="Simulate a failure part-way through to preview error reporting")
    demo.set_defaults(handler=_run_demo)

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
    comp.add_argument("shell", choices=["bash", "zsh", "install"], metavar="SHELL",
                      help="Target shell: bash · zsh · install (auto-detect)")
    comp.set_defaults(handler=_run_completion, ui_frame=False)

    # ── update ─────────────────────────────────────────────────────────────────
    upd_desc = (
        f"  {bold_cyan('update')}  — Fetch and apply the latest ChemLink release.\n\n"
        f"  Fetches the current branch from {cyan('origin')}, reports the commits/tags behind,\n"
        f"  and fast-forwards the checkout — refuses to touch a dirty tree or diverged branch."
    )
    upd = sub.add_parser("update", help="Fetch and apply the latest ChemLink release",
                         description=upd_desc, formatter_class=_Fmt)
    upd.add_argument("--check", action="store_true",
                     help="Only check for updates; do not pull")
    upd.add_argument("-y", "--yes", action="store_true",
                     help="Skip the confirmation prompt")
    upd.set_defaults(handler=_run_update)

    # ── hpc ────────────────────────────────────────────────────────────────────
    hpc_desc = (
        f"  {bold_cyan('hpc')}  — Submit pipelines to a SLURM cluster.\n\n"
        f"  Inputs are resolved from the current directory like the local commands, so\n"
        f"  run it from a directory on storage the compute nodes can reach."
    )
    hpc = sub.add_parser("hpc", help="Submit pipelines to SLURM cluster",
                         description=hpc_desc, formatter_class=_Fmt)
    hpc_sub = hpc.add_subparsers(dest="hpc_command", metavar="SUBCOMMAND")

    hpc_dock = hpc_sub.add_parser(
        "docking", formatter_class=_Fmt,
        help="Submit the full docking pipeline to SLURM",
        description=(
            f"  {bold_cyan('hpc docking')}  — Submit molecular docking to SLURM.\n\n"
            f"  Sizes the ligand batches from the ligand count, shows the configuration,\n"
            f"  then submits receptor prep → ligand prep → active-site → batches →\n"
            f"  docking array → merge → analysis."
        ),
    )
    _docking_input_args(hpc_dock)
    _run_name_arg(hpc_dock)
    _box_args(hpc_dock, hpc=True)
    hpc_dock.add_argument("--nodes", metavar="NODELIST",
                          help="SLURM nodelist to restrict submission (e.g. node[01-04])")
    hpc_dock.add_argument("--partition", metavar="PARTITION", help="SLURM partition")
    hpc_dock.add_argument("--gres", metavar="GRES", help="SLURM generic resource (e.g. gpu:1)")
    hpc_dock.add_argument("--batch-size", type=int, metavar="N",
                          help="Ligands per docking batch  [auto: ≤100→all, ≤500→100, ≤2000→200, >2000→500]")
    hpc_dock.add_argument("--prep-tasks", type=int, default=6, metavar="N",
                          help="Array tasks for receptor/ligand preparation  [6]")
    hpc_dock.add_argument("--receptor-workers", type=int, default=4, metavar="N",
                          help="Parallel workers per receptor-prep task  [4]")
    hpc_dock.add_argument("--ligand-workers", type=int, default=8, metavar="N",
                          help="Parallel workers per ligand-prep task  [8]")
    hpc_dock.add_argument("--active-site-workers", type=int, default=4, metavar="N",
                          help="Parallel workers for active-site detection  [4]")
    hpc_dock.add_argument("--docking-workers", type=int, default=1, metavar="N",
                          help="Parallel workers per docking task  [1]")
    hpc_dock.add_argument("--max-gpu-concurrency", type=int, default=6, metavar="N",
                          help="Max concurrent docking array tasks  [6]")
    hpc_dock.add_argument("--container", metavar="IMAGE",
                          help="Run every step inside this container image instead of the "
                               "native environment")
    hpc_dock.add_argument("--dry-run", action="store_true",
                          help="Show the configuration and command without submitting")
    hpc_dock.set_defaults(handler=_run_hpc_docking)

    hpc_dyn = hpc_sub.add_parser(
        "dynamics", formatter_class=_Fmt,
        help="Submit molecular dynamics simulations to SLURM",
        description=(
            f"  {bold_cyan('hpc dynamics')}  — Submit GROMACS simulations to SLURM.\n\n"
            f"  Same simulation types and input detection as {cyan('chemlink dynamic')};\n"
            f"  pligand with several ligands submits one job per ligand."
        ),
    )
    hpc_dyn_sub = hpc_dyn.add_subparsers(dest="dyn_type", metavar="TYPE")
    for name, help_text, desc in _DYN_TYPES:
        hp = hpc_dyn_sub.add_parser(name, help=help_text, description=desc, formatter_class=_Fmt)
        _dyn_type_args(hp, name, hpc=True)
        hp.set_defaults(handler=_run_hpc_dynamics, dyn_type=name)

    return parser


# ── Entry point ────────────────────────────────────────────────────────────────

def _print_banner() -> None:
    try:
        from .banner import SUBTAGLINE, pick_brand, render_brand
    except ImportError:
        from banner import SUBTAGLINE, pick_brand, render_brand  # type: ignore
    console.print()
    name = pick_brand(console.width, 12)
    if name and name != "MARK_LARGE":
        console.print(render_brand(name))
    else:
        console.print(gradient_text("ChemLink"), justify="center")
    console.print()
    console.print(Text(f"{TAGLINE} · {SUBTAGLINE} · v{VERSION}", style="cl.muted"), justify="center")


# Flags that may appear anywhere on the command line. They are stripped before
# argparse sees the argv, so ``chemlink docking full --plain`` works as well as
# ``chemlink --plain docking full`` (argparse would otherwise reject the former,
# and subparser defaults would silently reset the latter).
_GLOBAL_LONG = {"--no-color", "--plain", "--debug", "--verbose", "--quiet"}
_GLOBAL_SHORT = {"-v": "--verbose", "-q": "--quiet"}


def _extract_global_flags(argv: list) -> Tuple[set, list]:
    flags: set = set()
    rest: list = []
    seen_command = False
    for token in argv:
        if token == "--":
            seen_command = True
        if token in _GLOBAL_LONG:
            flags.add(token)
            continue
        if not seen_command and token in _GLOBAL_SHORT:
            flags.add(_GLOBAL_SHORT[token])
            continue
        if not token.startswith("-"):
            seen_command = True
        rest.append(token)
    return flags, rest


def _apply_global_flags(flags: set) -> None:
    import logging
    try:
        from ..utils.logger import capture_warnings_to_console, set_console_level
    except ImportError:
        from utils.logger import capture_warnings_to_console, set_console_level  # type: ignore
    if "--no-color" in flags:
        set_color_enabled(False)
    if "--plain" in flags:
        os.environ["CHEMLINK_PROGRESS"] = "plain"
    if "--debug" in flags:
        os.environ["CHEMLINK_DEBUG"] = "1"
    if "--verbose" in flags:
        set_console_level(logging.DEBUG)
    elif "--quiet" in flags:
        set_console_level(logging.WARNING)
    capture_warnings_to_console()


def _command_words(argv: list) -> list:
    """Leading sub-command words of an argv (``docking full`` from
    ``docking full -ir x``), used for headers and ``--help`` fallbacks."""
    words = []
    for token in argv:
        if token.startswith("-"):
            break
        words.append(token)
    return words[:3]


def _print_run_header(command_line: str) -> None:
    head = Table.grid(expand=True)
    head.add_column(ratio=1, overflow="ellipsis", no_wrap=True)
    head.add_column(justify="right", no_wrap=True)
    left = Text("  ")
    left.append(f"{GLYPH['brand']} ", style="cl.brand")
    first, _, rest = command_line.partition(" ")
    left.append_text(gradient_text(first))
    if rest:
        left.append(" " + rest, style="bold")
    right = Text(f"v{VERSION} · {datetime.now():%d %b %H:%M}  ", style="cl.faint")
    head.add_row(left, right)
    console.print()
    console.print(head)


def _print_run_footer(code: int, elapsed: float) -> None:
    console.print()
    if code == 0:
        console.print(status_line("ok", f"[cl.ok]Done[/] [cl.muted]in {format_duration(elapsed)}[/]"))
    elif code == 130:
        console.print(status_line("warn", f"[cl.warn]Interrupted[/] [cl.muted]after {format_duration(elapsed)}[/]"))
    else:
        console.print(status_line("err", f"[cl.err]Failed[/] [cl.muted]after {format_duration(elapsed)} · exit code {code}[/]"))
    console.print()


_EXC_TITLES = {
    "FileNotFoundError": "File not found",
    "FileExistsError": "Already exists",
    "PermissionError": "Permission denied",
    "TimeoutExpired": "Timed out",
    "CalledProcessError": "External tool failed",
    "SubprocessError": "External tool failed",
    "MemoryError": "Out of memory",
    "ModuleNotFoundError": "Missing Python package",
    "ImportError": "Missing dependency",
}


def report_exception(exc: BaseException) -> None:
    """Print ``exc`` as an error panel; with --debug, also the full traceback.
    Must be called from inside the ``except`` block that caught ``exc``."""
    name = type(exc).__name__
    title = _EXC_TITLES.get(name) or re.sub(r"(?<!^)(?=[A-Z])", " ", name).replace(" Error", " error")
    details = None
    stderr = getattr(exc, "stderr", None)
    if isinstance(stderr, str) and stderr.strip() and stderr.strip() not in str(exc):
        details = "\n".join(stderr.strip().splitlines()[-12:])
    hint_text = None
    if isinstance(exc, ModuleNotFoundError) and getattr(exc, "name", None):
        hint_text = "Install it in the ChemLink environment, then run [cl.ref]chemlink doctor[/]."
    if os.environ.get("CHEMLINK_DEBUG"):
        print_error(title, exc, details=details)
        err_console.print_exception(max_frames=30)
    else:
        print_error(title, exc, details=details, hint_text=hint_text or
                    "Re-run with [cl.ref]--debug[/] for the full traceback.")


def run_command(args: Namespace, command_line: str) -> int:
    """Run a parsed command inside ChemLink's header/footer frame — shared by
    the plain CLI and the TUI so both look identical."""
    framed = getattr(args, "ui_frame", True) and not getattr(args, "json", False)
    if framed:
        _print_run_header(command_line)
    start = time.monotonic()
    try:
        code = args.handler(args) or 0
    except KeyboardInterrupt:
        code = 130
    except SystemExit as exc:  # a handler calling sys.exit(n)
        code = exc.code if isinstance(exc.code, int) else 1
    except Exception as exc:
        report_exception(exc)
        code = 1
    if framed:
        _print_run_footer(code, time.monotonic() - start)
    elif code == 130:
        print_warn("Interrupted.")
    return code


def main() -> int:
    flags, argv = _extract_global_flags(sys.argv[1:])
    _apply_global_flags(flags)

    try:
        import argcomplete  # type: ignore
        _has_argcomplete = True
    except ImportError:
        _has_argcomplete = False

    parser = build_parser()

    if _has_argcomplete:
        argcomplete.autocomplete(parser)

    if not argv:
        if sys.stdin.isatty() and sys.stdout.isatty() and "--plain" not in flags:
            try:
                from .tui.app import run as run_tui
            except ImportError:
                from tui.app import run as run_tui  # type: ignore
            return run_tui(parser, console, err_console, version=VERSION, runner=run_command)
        _print_banner()
        parser.print_help()
        return 0

    if argv[0] in ("-h", "--help"):
        _print_banner()

    _Parser.current_argv = list(argv)
    args = parser.parse_args(argv)

    if not hasattr(args, "handler"):
        parser.parse_args(_command_words(argv) + ["--help"])
        return 0

    import shlex
    return run_command(args, shlex.join(["chemlink", *argv]))


if __name__ == "__main__":
    raise SystemExit(main())
