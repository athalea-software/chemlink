"""Shared flat/minimal terminal styling — the single source of ChemLink's CLI look.

Cyan is the sole brand accent, split by *weight* rather than by color — the same
hue means something different depending on whether it's bold:

  - ``bold cyan``   — structural: section headers/titles (ACCENT).
  - plain ``cyan``  — a reference embedded in a sentence: a command, flag, path,
                       or other literal token the user could type or click (REF).
  - ``dim``         — secondary/meta text (timestamps, byte counts, raw detail).
  - ``dim italic``  — a hint/aside: keybindings, field help text (HINT).
  - plain (no style) — primary body content/values.

Green/yellow/red are reserved for state (ok/warn/error), never for anything else.
No boxed tables or panels — flat, left-aligned, whitespace-driven layout.

Indentation is one ``INDENT`` (2 spaces) per nesting level — use ``indent(n)``
rather than hand-typing a run of spaces, so nested levels stay a clean multiple.
"""
from __future__ import annotations

import os
from typing import Iterable, Optional, Tuple

from rich.console import Console
from rich.text import Text

ACCENT = "cyan"
REF = "cyan"
HINT = "dim italic"

GLYPH = {"ok": "✓", "warn": "▲", "err": "✗", "info": "·"}
STYLE = {"ok": "bold green", "warn": "bold yellow", "err": "bold red", "info": "cyan"}

INDENT = "  "


def indent(level: int = 1) -> str:
    """``level`` repetitions of the 2-space indent unit."""
    return INDENT * level


def ref(text: str) -> str:
    """Mark ``text`` as a reference — a command, flag, or path — inline in a
    sentence: plain (non-bold) cyan, distinct from a bold-cyan section header."""
    return f"[{REF}]{text}[/{REF}]"


def hint(text: str) -> str:
    """Mark ``text`` as a hint/aside — keybindings, field help: dim italic."""
    return f"[{HINT}]{text}[/{HINT}]"


def path_ref(text: str) -> str:
    """``ref()`` if ``text`` looks like a filesystem path, else plain — for
    status/detail columns that mix real paths with prose ("not set", "3.10.20")."""
    if text and (os.sep in text or text.startswith(("/", "~", "./"))):
        return ref(text)
    return text


def _detail_markup(text: str) -> str:
    """A status-row detail: the reference accent for a path, dim for everything
    else (a version number, "not set", free-text notes)."""
    styled = path_ref(text)
    return styled if styled != text else f"[dim]{text}[/dim]"

# Log-level aliases seen across pipeline modules, mapped onto the three states above.
_LOG_LEVEL_MAP = {
    "INFO": "info",
    "NOTE": "warn",
    "WARNING": "warn",
    "WARN": "warn",
    "FATAL": "err",
    "ERROR": "err",
}


def status_block(
    console: Console,
    rows: Iterable[Tuple[str, str, str]],
    *,
    title: Optional[str] = None,
    level: int = 1,
) -> None:
    """Flat aligned status rows: ``  ✓ label   detail``.

    ``rows`` is an iterable of ``(level, label, detail)`` where ``level`` is one
    of ``"ok" | "warn" | "err" | "info"``.
    """
    rows = list(rows)
    if not rows:
        return
    if title:
        console.print(f"\n[bold {ACCENT}]{title}[/bold {ACCENT}]")
    width = max(len(label) for _, label, _ in rows)
    pad = indent(level)
    for row_level, label, detail in rows:
        style = STYLE[row_level]
        glyph = GLYPH[row_level]
        detail_part = f"  {_detail_markup(detail)}" if detail else ""
        console.print(f"{pad}[{style}]{glyph}[/{style}] {label:<{width}}{detail_part}", soft_wrap=True)


def kv_block(
    console: Console,
    rows: Iterable[Tuple[str, str]],
    *,
    title: Optional[str] = None,
    level: int = 1,
) -> None:
    """Flat aligned key/value rows: ``  Label   value`` (no glyph, cyan bold label)."""
    rows = list(rows)
    if not rows:
        return
    if title:
        console.print(f"\n[bold {ACCENT}]{title}[/bold {ACCENT}]")
    width = max(len(label) for label, _ in rows)
    pad = indent(level)
    for label, value in rows:
        console.print(f"{pad}[bold {ACCENT}]{label:<{width}}[/bold {ACCENT}]  {value}", soft_wrap=True)


def record_block(
    console: Console,
    headers: Iterable[str],
    rows: Iterable[Iterable[object]],
    *,
    title: Optional[str] = None,
    level: int = 1,
) -> None:
    """Flat aligned multi-column rows (e.g. a list of submitted jobs).

    Unlike ``kv_block``/``status_block`` this keeps a header row, since the
    columns aren't self-descriptive labels — just plain text, dim, no box.
    Path-like cells (e.g. a "Work dir" column) get the reference accent —
    padding is applied to the plain value first, then the markup wraps around
    it, so the invisible tag characters never throw off column alignment.
    """
    headers = list(headers)
    str_rows = [[str(v) for v in r] for r in rows]
    if not str_rows:
        return
    if title:
        console.print(f"\n[bold {ACCENT}]{title}[/bold {ACCENT}]")
    widths = [
        max(len(headers[i]), *(len(r[i]) for r in str_rows))
        for i in range(len(headers))
    ]
    pad = indent(level)
    console.print(
        pad + "[dim]" + "  ".join(h.ljust(w) for h, w in zip(headers, widths)) + "[/dim]",
        soft_wrap=True,
    )
    for r in str_rows:
        cells = [path_ref(v.ljust(w)) for v, w in zip(r, widths)]
        console.print(pad + "  ".join(cells), soft_wrap=True)


def rule(console: Console, label: Optional[str] = None) -> None:
    """Thin, muted section divider — replaces heavy colored `console.rule(...)` calls."""
    console.rule(f"[dim]{label}[/dim]" if label else "", style="dim")


def log_line(level: str, message: str, phase: str = "") -> Text:
    """Inline tag for a line printed beside a live progress bar.

    Renders e.g. ``✓ message`` (info), ``▲ phase  message`` (note/warning), or
    ``✗ phase  message`` (fatal) — one implementation shared by every pipeline
    step that echoes log lines next to a Rich progress bar.
    """
    key = _LOG_LEVEL_MAP.get(level.upper(), "info")
    style = STYLE[key]
    prefix = GLYPH[key] + (f" {phase}" if phase else "")
    return Text.assemble((prefix + "  ", style), message)
