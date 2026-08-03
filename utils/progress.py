"""Shared Rich progress styling for ChemLink pipelines.

Visual language:
  ◈  pipeline-level tracker   (blue)
  ◉  docking step bar         (per-step colour)
  ◆  dynamics mdrun bar       (magenta / cyan)

Braille bar: 8 sub-character levels per cell → very smooth rendering.
"""
from __future__ import annotations

from typing import Generator, Iterable, Optional, TypeVar

from rich.console import Console
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


# ── Braille bar ───────────────────────────────────────────────────────────────

class BrailleBar(ProgressColumn):
    """Horizontal braille bar — each partial cell shows only complete dot rows.

    Sub-cell levels (bottom-to-top, always both left+right dots per row):
      ⠀  ⣀  ⣤  ⣶  ⣿
      0   1   2   3   4 rows filled
    """

    _FILL    = "⣿"
    _EMPTY   = "⠀"
    _PARTIAL = "⣀⣤⣶"  # 1, 2, 3 complete rows (indices 0–2 for levels 1–3)

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

        n_units     = self.width * 4      # 4 sub-levels per cell
        filled      = int(frac * n_units)
        full        = filled // 4
        partial     = filled % 4

        full_chars   = self._FILL * full
        partial_char = self._PARTIAL[partial - 1] if partial and full < self.width else ""
        empty_chars  = self._EMPTY * (self.width - full - len(partial_char))

        bar_style = self.finished_style if frac >= 1.0 else self.complete_style

        t = Text(no_wrap=True)
        t.append("[", style="dim")
        t.append(full_chars + partial_char, style=bar_style)
        t.append(empty_chars, style="dim")
        t.append("]", style="dim")
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
        BrailleBar(width=bar_width, complete_style="cyan", finished_style="bright_green"),
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
        BrailleBar(
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
        BrailleBar(width=bar_width),
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
        BrailleBar(width=bar_width, complete_style="bright_green",
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
        BrailleBar(width=bar_width, complete_style="blue",
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
    """Wrap any iterable with a braille progress bar.

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
