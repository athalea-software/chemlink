"""Shared Rich progress styling for ChemLink pipelines.

Visual language:
  ◈  pipeline-level tracker   (blue)
  ◉  docking step bar         (per-step colour)
  ◆  dynamics mdrun bar       (magenta / cyan)

Solid bar: 8 sub-character levels per cell (Unicode eighth-blocks) — a
continuous left-to-right fill instead of a dotted/textured look. The unfilled
track renders as a dim solid shade rather than blank space, so the bar reads
as one continuous shape at every percentage, not just once it's mostly full.
"""
from __future__ import annotations

import random
import time
from typing import Generator, Iterable, Optional, TypeVar

from rich.console import Console, Group
from rich.live import Live
from rich.progress import (
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    Task,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.text import Text

T = TypeVar("T")

console = Console(highlight=False)

CTRL_C_HINT = "  Press [bold]Ctrl+C[/bold] at any time to abort.\n"


# ── Solid bar ─────────────────────────────────────────────────────────────────

class SolidBar(ProgressColumn):
    """Horizontal block-fill bar — 8 sub-character levels per cell, filling
    continuously left-to-right.

    Partial-cell characters (1/8 through 7/8 filled, left-aligned):
      ▏ ▎ ▍ ▌ ▋ ▊ ▉ █
    """

    _FULL    = "█"
    _PARTIAL = "▏▎▍▌▋▊▉"  # 1..7 eighths filled (index 0..6 for levels 1-7)
    _TRACK   = "░"

    def __init__(
        self,
        width: int = 36,
        complete_style: str = "cyan",
        finished_style: str = "bright_green",
    ) -> None:
        super().__init__()
        self.width          = width
        self.complete_style = complete_style
        self.finished_style = finished_style

    def render(self, task: Task) -> Text:
        total = float(task.total or 1)
        frac  = min(task.completed / total, 1.0)

        n_units     = self.width * 8      # 8 sub-levels per cell
        filled      = int(frac * n_units)
        full        = filled // 8
        partial     = filled % 8

        full_chars   = self._FULL * full
        partial_char = self._PARTIAL[partial - 1] if partial and full < self.width else ""
        track_chars  = self._TRACK * (self.width - full - len(partial_char))

        bar_style = self.finished_style if frac >= 1.0 else self.complete_style

        t = Text(no_wrap=True)
        t.append(full_chars + partial_char, style=bar_style)
        t.append(track_chars, style="dim")
        return t


# ── Dynamics-specific columns ─────────────────────────────────────────────────

class NsDayColumn(ProgressColumn):
    """Renders live ns/day stored in task.fields['ns_day']."""

    def render(self, task: Task) -> Text:
        v = task.fields.get("ns_day", 0.0)
        if not v or v <= 0:
            return Text("    --.- ns/day", style="dim")
        return Text(f"{v:7.1f} ns/day", style="bold bright_cyan")


class EtaColumn(ProgressColumn):
    """Renders live ETA stored in task.fields['eta_s']."""

    def render(self, task: Task) -> Text:
        v = task.fields.get("eta_s", 0.0)
        if not v or v <= 0:
            return Text("", style="dim")
        m, s = divmod(int(v), 60)
        h, m = divmod(m, 60)
        if h:
            txt = f"ETA {h}h {m:02d}m"
        elif m:
            txt = f"ETA {m}m {s:02d}s"
        else:
            txt = f"ETA {s}s"
        return Text(txt, style="dim")


# ── Internal helpers ──────────────────────────────────────────────────────────

def _sep() -> TextColumn:
    return TextColumn("[dim]│[/dim]")


def _desc(symbol: str, text: str, colour: str) -> str:
    return f"[bold {colour}]{symbol} {text}[/bold {colour}]"


def _make_progress(*columns, transient: bool = True) -> Progress:
    return Progress(*columns, console=console, transient=transient,
                    refresh_per_second=10)


# ── Progress factory functions ────────────────────────────────────────────────

def make_dynamics_progress(bar_width: int = 36) -> Progress:
    """Progress bar for GROMACS mdrun — shows step, ns/day, ETA."""
    return _make_progress(
        TextColumn("{task.description}", justify="left"),
        SolidBar(width=bar_width, complete_style="cyan", finished_style="bright_green"),
        TextColumn("[bold]{task.percentage:3.0f}%[/bold]"),
        _sep(),
        MofNCompleteColumn(),
        TextColumn("[dim]step[/dim]"),
        _sep(),
        NsDayColumn(),
        _sep(),
        EtaColumn(),
        transient=True,
    )


def make_analysis_progress(bar_width: int = 32) -> Progress:
    """Progress bar for frame-based scientific analyses."""
    return _make_progress(
        TextColumn("{task.description}", justify="left"),
        SolidBar(
            width=bar_width,
            complete_style="magenta",
            finished_style="bright_green",
        ),
        TextColumn("[bold]{task.percentage:3.0f}%[/bold]"),
        _sep(),
        MofNCompleteColumn(),
        TextColumn("[dim]frame[/dim]"),
        _sep(),
        TimeElapsedColumn(),
        TextColumn("[dim]<[/dim]"),
        TimeRemainingColumn(),
        transient=True,
    )


def _step_progress(unit: str = "item", bar_width: int = 32) -> Progress:
    return _make_progress(
        TextColumn("{task.description}", justify="left"),
        SolidBar(width=bar_width),
        TextColumn("[bold]{task.percentage:3.0f}%[/bold]"),
        _sep(),
        MofNCompleteColumn(),
        TextColumn(f"[dim]{unit}[/dim]"),
        _sep(),
        TimeElapsedColumn(),
        transient=True,
    )


def _job_progress(bar_width: int = 28) -> Progress:
    return _make_progress(
        TextColumn("{task.description}", justify="left"),
        SolidBar(width=bar_width, complete_style="bright_green",
                 finished_style="green"),
        TextColumn("[bold]{task.percentage:3.0f}%[/bold]"),
        _sep(),
        MofNCompleteColumn(),
        _sep(),
        TimeElapsedColumn(),
        TextColumn("[dim]<[/dim]"),
        TimeRemainingColumn(),
        TextColumn("[dim]{task.fields[postfix]}[/dim]"),
        transient=False,
    )


def _pipeline_progress(bar_width: int = 26) -> Progress:
    return _make_progress(
        TextColumn("{task.description}", justify="left"),
        SolidBar(width=bar_width, complete_style="blue",
                 finished_style="bright_blue"),
        TextColumn("[bold]{task.percentage:3.0f}%[/bold]"),
        _sep(),
        MofNCompleteColumn(),
        TextColumn("[dim]step[/dim]"),
        _sep(),
        TimeElapsedColumn(),
        transient=False,
    )


# ── step_bar_iter ─────────────────────────────────────────────────────────────

def step_bar_iter(
    iterable: Iterable[T],
    desc: str,
    *,
    total: Optional[int] = None,
    unit: str = "item",
    colour: str = "cyan",
) -> Generator[T, None, None]:
    """Wrap any iterable with a solid progress bar.

    Supports generators (pass total= explicitly) or plain sequences.

    Usage::

        for item in step_bar_iter(files, "Ligand Preparation",
                                  unit="ligand", colour="magenta"):
            process(item)

        # With generator / as_completed:
        for future in step_bar_iter(as_completed(futs), "Receptors",
                                    total=len(futs), unit="receptor"):
            ...
    """
    if total is None:
        items = list(iterable)
        total = len(items)
        actual: Iterable[T] = items
    else:
        actual = iterable

    label = _desc("◉", desc, colour)
    with _step_progress(unit=unit) as progress:
        task = progress.add_task(label, total=total)
        for item in actual:
            yield item
            progress.advance(task)


# ── _JobBar — context manager with tqdm-compatible API ───────────────────────

class _JobBar:
    def __init__(self, desc: str, total: int, *, colour: str = "bright_green") -> None:
        self._progress = _job_progress()
        self._label    = _desc("◉", desc.strip(), colour)
        self._total    = total
        self._task: Optional[TaskID] = None

    def __enter__(self) -> "_JobBar":
        self._progress.__enter__()
        self._task = self._progress.add_task(
            self._label, total=self._total, postfix=""
        )
        return self

    def __exit__(self, *exc) -> None:
        self._progress.__exit__(*exc)

    def update(self, n: int = 1) -> None:
        if self._task is not None:
            self._progress.advance(self._task, n)

    def set_postfix_str(self, s: str) -> None:
        if self._task is not None:
            self._progress.update(self._task, postfix=f"  {s}")


def job_bar(
    desc: str,
    total: int,
    *,
    unit: str = "job",
    colour: str = "bright_green",
) -> "_JobBar":
    return _JobBar(desc, total, colour=colour)


# ── _StepBar — tqdm-compatible drop-in for the manual "with ... as pbar: pbar.update()"
#    sub-step pattern used throughout the dynamics pipeline (transient: clears once done,
#    same solid bar as step_bar_iter, instead of tqdm's plain ASCII "###|" bar) ──────────

class _StepBar:
    def __init__(self, desc: str, total: int, *, unit: str = "item", colour: str = "cyan") -> None:
        self._progress = _step_progress(unit=unit)
        self._label    = _desc("◉", desc.strip(), colour)
        self._total    = total
        self._task: Optional[TaskID] = None

    def __enter__(self) -> "_StepBar":
        self._progress.__enter__()
        self._task = self._progress.add_task(self._label, total=self._total)
        return self

    def __exit__(self, *exc) -> None:
        self._progress.__exit__(*exc)

    def update(self, n: int = 1) -> None:
        if self._task is not None:
            self._progress.advance(self._task, n)


def step_bar(desc: str, total: int, *, unit: str = "item", colour: str = "cyan") -> "_StepBar":
    return _StepBar(desc, total, unit=unit, colour=colour)


# ── _PipelineBar — no context manager required (matches dynamics_pipeline API) ─

class _PipelineBar:
    def __init__(self, total: int, desc: str) -> None:
        self._progress = _pipeline_progress()
        self._label    = _desc("◈", desc.strip(), "blue")
        self._total    = total
        self._task: Optional[TaskID] = None
        self._active = True
        self._progress.start()
        self._task = self._progress.add_task(self._label, total=total)

    def update(self, n: int = 1) -> None:
        if self._task is not None:
            self._progress.advance(self._task, n)

    def close(self) -> None:
        if not self._active:
            self._progress.start()
            self._active = True
        self._progress.stop()
        self._active = False

    def pause(self) -> None:
        """Temporarily stop Rich rendering while nested terminal bars run."""
        if self._active:
            self._progress.stop()
            self._active = False

    def resume(self) -> None:
        if not self._active:
            self._progress.start()
            self._active = True

    def __enter__(self) -> "_PipelineBar":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def pipeline_bar(total: int, *, desc: str = "MD Pipeline") -> _PipelineBar:
    return _PipelineBar(total, desc)


# ── BusyIndicator — animated status line for indeterminate waits ──────────────

_BUSY_VERBS = (
    "Coagitating", "Equilibrating", "Solvating", "Percolating", "Titrating",
    "Diffusing", "Annealing", "Nucleating", "Crystallizing", "Calibrating",
)
_SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_SPINNER_FPS = 10


def spinner_frame(elapsed: float) -> str:
    """One glyph of the shared spinner animation for a given elapsed time —
    exposed so anything hand-rolling its own ``Live`` loop (the TUI's launch
    transition, which needs custom text rather than a themed verb) still
    animates with the exact same glyph set as ``BusyIndicator``."""
    return _SPINNER_FRAMES[int(elapsed * _SPINNER_FPS) % len(_SPINNER_FRAMES)]


class _BusyRenderable:
    """Rich renderable recomputed on every Live auto-refresh tick, so both the
    spinner frame and the elapsed-time readout animate on their own — no
    manual ticking thread needed, the same way a ``ProgressColumn`` like
    ``SolidBar`` is re-rendered by ``Progress``'s own live loop."""

    def __init__(self, verb: str, style: str) -> None:
        self.verb  = verb
        self.style = style
        self._start = time.monotonic()

    def __rich__(self) -> Text:
        elapsed = time.monotonic() - self._start
        t = Text(no_wrap=True)
        t.append(f"{spinner_frame(elapsed)} ", style=f"bold {self.style}")
        t.append(f"{self.verb}…", style=f"bold {self.style}")
        t.append(f"  {elapsed:0.0f}s", style="dim")
        return t


class BusyIndicator:
    """Animated status line for a single blocking call with no discrete steps
    to report — a spinner glyph, a cycling themed verb, and elapsed time, in
    the spirit of a coding assistant's own "Thinking…" status line.

    For anything with real sub-steps (a loop, a known step count), use
    ``step_bar``/``job_bar`` instead so progress is honestly represented —
    this is only for the moments in between, where the app is waiting on one
    opaque call (a subprocess, a network fetch) and has nothing truthful to
    report except "still working".

    Usage::

        with busy("Fetching"):
            subprocess.run([...])

        with busy():   # verb chosen at random from a themed set
            long_call()
    """

    def __init__(self, verb: Optional[str] = None, *, style: str = "cyan") -> None:
        self._renderable = _BusyRenderable(verb or random.choice(_BUSY_VERBS), style)
        self._live: Optional[Live] = None

    def __enter__(self) -> "BusyIndicator":
        self._live = Live(
            self._renderable, console=console, transient=True,
            refresh_per_second=_SPINNER_FPS,
        )
        self._live.start()
        return self

    def __exit__(self, *exc) -> None:
        if self._live is not None:
            self._live.stop()


def busy(verb: Optional[str] = None, *, style: str = "cyan") -> "BusyIndicator":
    return BusyIndicator(verb, style=style)


# ── format_duration ─────────────────────────────────────────────────────────

def format_duration(seconds: float) -> str:
    """``"4m 12s"`` / ``"1m 48s"`` / ``"52s"`` — same h/m/s shape as EtaColumn's
    formatting, without the "ETA" prefix."""
    total = max(0, int(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


# ── Checklist — a stack of named phases that stay visible once done ───────────

class ChecklistStep:
    """One item in a Checklist — a context manager returned by
    ``Checklist.step()``. Animates while the ``with`` block runs; on exit,
    auto-finishes as "done" unless ``.finish()`` was already called with a
    custom message (e.g. a final tally). On an exception inside the block, it
    finishes as failed with the exception text, then re-raises — the error is
    never swallowed."""

    def __init__(self, owner: "Checklist", title: str, subtitle: str) -> None:
        self._owner    = owner
        self.title     = title
        self.subtitle  = subtitle
        self.detail    = ""
        self.state     = "running"  # "running" | "ok" | "err"
        self._start    = time.monotonic()
        self._finished = False

    def update(self, detail: str) -> None:
        self.detail = detail
        self._owner._refresh()

    def finish(self, detail: Optional[str] = None, *, ok: bool = True) -> None:
        if self._finished:
            return
        self._finished = True
        self.state  = "ok" if ok else "err"
        self.detail = detail if detail is not None else ("done" if ok else "failed")
        self._owner._refresh()

    def __enter__(self) -> "ChecklistStep":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc is not None:
            self.finish(str(exc), ok=False)
            return False
        self.finish()

    def _render(self) -> Text:
        t = Text(no_wrap=True)
        if self.state == "running":
            t.append(f"{spinner_frame(time.monotonic() - self._start)} ", style="bold cyan")
            t.append(self.title, style="bold cyan")
        else:
            style = "bold green" if self.state == "ok" else "bold red"
            glyph = "✓" if self.state == "ok" else "✗"
            t.append(f"{glyph} ", style=style)
            t.append(self.title, style=style)
        if self.subtitle:
            t.append(f"   ({self.subtitle})", style="dim")
        if self.detail:
            t.append("  ·  ", style="dim")
            t.append(self.detail, style="dim" if self.state != "running" else "")
        return t


class Checklist:
    """A vertical stack of named phases — each animates while running and
    stays visible once done, unlike step_bar/job_bar/pipeline_bar, which each
    render exactly one live line that gets replaced rather than several that
    persist. Owns one non-transient Live for its whole lifetime, so completed
    items remain on screen as a permanent log.

    Usage::

        with checklist() as cl:
            with cl.step("Detecting active site", "fpocket"):
                ...
            with cl.step("Running GPU search", "AutoDock-GPU") as item:
                for done in range(total):
                    ...
                    item.update(f"{done:,} / {total:,} ligands docked")
                item.finish(f"{total:,} / {total:,} ligands docked  ·  100%")
            cl.complete(f"Campaign complete in {format_duration(elapsed)}")
    """

    def __init__(self) -> None:
        self._items: list = []
        self._live: Optional[Live] = None

    def _refresh(self) -> None:
        if self._live is not None:
            self._live.update(Group(*(item._render() for item in self._items)), refresh=True)

    def step(self, title: str, subtitle: str = "") -> ChecklistStep:
        item = ChecklistStep(self, title, subtitle)
        self._items.append(item)
        self._refresh()
        return item

    def complete(self, message: str) -> None:
        """Stop the Live and print a permanent bold-green "✓ {message}" line."""
        if self._live is not None:
            self._live.stop()
            self._live = None
        console.print(f"[bold green]✓[/]  {message}")

    def __enter__(self) -> "Checklist":
        self._live = Live(Group(), console=console, transient=False, refresh_per_second=_SPINNER_FPS)
        self._live.start()
        return self

    def __exit__(self, *exc) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None


def checklist() -> Checklist:
    return Checklist()
