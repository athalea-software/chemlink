"""Shared building blocks for the TUI screens: the header bar, the keycap footer,
panels, and small text helpers. Screens compose these into a full-height
``rich.layout.Layout`` so the header and footer stay pinned while the body
scrolls."""
from __future__ import annotations

import argparse
import os
import re
import textwrap
import time
from typing import Iterable, List, Optional, Sequence, Tuple

from rich import box
from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

try:
    from ...utils.theme import GLYPH, gradient_text
except ImportError:  # pragma: no cover
    from utils.theme import GLYPH, gradient_text  # type: ignore

from .sysinfo import SystemInfo

# Command glyphs — single-cell symbols that render in common terminal fonts.
ICONS = {
    "docking": "◎",
    "dynamic": "≋",
    "hpc": "▦",
    "doctor": "✚",
    "demo": "▶",
    "update": "↻",
    "completion": "⇥",
}

GROUPS: Sequence[Tuple[str, Sequence[str]]] = (
    ("Pipelines", ("docking", "dynamic")),
    ("Cluster", ("hpc",)),
    ("Tools", ("doctor", "demo", "update", "completion")),
)

KEY_STYLE = "bold #e6edf3 on #2b3240"


_BRACKETS = re.compile(r"\[[^\[\]]*\]")


def ansi(text: str) -> Text:
    """Argparse help may carry ANSI colour codes from ``cli.main``; trailing
    ``[default]`` notes are muted the same way ``--help`` shows them."""
    out = Text.from_ansi(text) if "\x1b[" in (text or "") else Text(text or "")
    for m in _BRACKETS.finditer(out.plain):
        out.stylize("cl.muted", m.start(), m.end())
    return out


def description_text(parser: argparse.ArgumentParser) -> Text:
    desc = textwrap.dedent((parser.description or "").strip("\n"))
    return ansi(desc)


def header(breadcrumb: Sequence[str], version: str, sysinfo: Optional[SystemInfo]) -> RenderableType:
    grid = Table.grid(expand=True, padding=(0, 1))
    grid.add_column(no_wrap=True, ratio=1, overflow="ellipsis")
    grid.add_column(no_wrap=True, justify="right")
    left = Text(" ")
    left.append(f"{GLYPH['brand']} ", style="cl.brand")
    left.append_text(gradient_text("ChemLink"))
    if version:
        left.append(f" v{version}", style="cl.faint")
    if breadcrumb:
        left.append("   ")
        for i, part in enumerate(breadcrumb):
            if i:
                left.append(" › ", style="cl.faint")
            left.append(part, style="bold" if i == len(breadcrumb) - 1 else "cl.muted")
    right = sysinfo.chips() if sysinfo is not None else Text("")
    right.append(" ")
    grid.add_row(left, right)
    return Group(grid, Text("─" * 400, style="cl.border", no_wrap=True, overflow="crop"))


def keycaps(pairs: Iterable[Tuple[str, str]]) -> Text:
    out = Text(" ", no_wrap=True, overflow="ellipsis")
    for i, (key, label) in enumerate(pairs):
        if i:
            out.append("   ")
        out.append(f" {key} ", style=KEY_STYLE)
        out.append(f" {label}", style="cl.muted")
    return out


def footer(pairs: Iterable[Tuple[str, str]], status: Optional[Text] = None) -> RenderableType:
    rule = Text("─" * 400, style="cl.border", no_wrap=True, overflow="crop")
    if status is None:
        return Group(rule, keycaps(pairs))
    grid = Table.grid(expand=True)
    grid.add_column(ratio=1, no_wrap=True, overflow="ellipsis")
    grid.add_column(justify="right", no_wrap=True)
    grid.add_row(keycaps(pairs), status)
    return Group(rule, grid)


def panel(body: RenderableType, title: str = "", *, focused: bool = False, subtitle: str = "") -> Panel:
    heading = Text(title, style="cl.brand" if focused else "bold") if title else None
    return Panel(
        body,
        title=heading,
        title_align="left",
        subtitle=Text(subtitle, style="cl.faint") if subtitle else None,
        subtitle_align="right",
        border_style="cl.accent" if focused else "cl.border",
        box=box.ROUNDED,
        padding=(0, 1),
    )


def scroll_window(cursor: int, count: int, rows: int, scroll: int) -> int:
    """New scroll offset keeping ``cursor`` visible in a ``rows``-tall window."""
    rows = max(1, rows)
    if cursor < scroll:
        scroll = cursor
    elif cursor >= scroll + rows:
        scroll = cursor - rows + 1
    return max(0, min(scroll, max(0, count - rows)))


# ── Recent runs ───────────────────────────────────────────────────────────────

# Where commands write runs by default (cli/workspace.py), relative to the cwd.
_RUN_ROOTS = ("results/docking", "results/peptide-docking", "results/dynamics")


def recent_runs(limit: int = 5) -> List[Tuple[str, float]]:
    """Most recently modified run directories under the conventional outputs."""
    found = {}
    for root in _RUN_ROOTS:
        try:
            with os.scandir(root) as it:
                for entry in it:
                    if not entry.is_dir() or entry.name.startswith("."):
                        continue
                    if entry.name.startswith(("prepared_", "docking_results", "inputs", "clean_receptors")):
                        continue
                    try:
                        found[entry.path] = entry.stat().st_mtime
                    except OSError:
                        pass
        except OSError:
            continue
    return sorted(found.items(), key=lambda kv: kv[1], reverse=True)[:limit]


def ago(ts: float) -> str:
    delta = max(0.0, time.time() - ts)
    if delta < 90:
        return "just now"
    if delta < 3600:
        return f"{int(delta // 60)} min ago"
    if delta < 86400:
        return f"{int(delta // 3600)} h ago"
    days = int(delta // 86400)
    return f"{days} day{'s' if days != 1 else ''} ago"


def runs_table(limit: int = 5) -> Optional[RenderableType]:
    runs = recent_runs(limit)
    if not runs:
        return None
    grid = Table.grid(padding=(0, 2))
    grid.add_column(no_wrap=True, overflow="ellipsis", max_width=48)
    grid.add_column(no_wrap=True, style="cl.faint", justify="right")
    for path, mtime in runs:
        rel = os.path.relpath(path)
        grid.add_row(Text(rel, style="cl.path"), ago(mtime))
    return grid
