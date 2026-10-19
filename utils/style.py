"""Building blocks for ChemLink's static terminal output — section headers,
key/value cards, status lists, tables, result blocks, error panels and the
one-line log format shared with the logger and progress display.

Palette and glyphs live in ``utils.theme``; this module only arranges them.

Layout rules
------------
- Sections open with a left-aligned rule: ``◆ Title ─────────``.
- Content is indented one ``INDENT`` (2 spaces) per nesting level.
- Labels are muted, values are bright; anything the user could type or
  open (a command, flag or path) is ``cl.ref``.
- Green/amber/red appear only as state glyphs (``✓ ▲ ✗``) and state words.
"""
from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Any, Iterable, List, Mapping, Optional, Tuple, Union

from rich import box
from rich.console import Console, Group, RenderableType
from rich.markup import escape
from rich.padding import Padding
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

try:
    from .theme import (
        GLYPH, STATE_STYLE, format_count, get_console, iter_lines, register_aliases,
    )
except ImportError:  # loaded as a top-level module
    from theme import (  # type: ignore
        GLYPH, STATE_STYLE, format_count, get_console, iter_lines, register_aliases,
    )

register_aliases(__name__)

# ── Back-compat names (older call sites import these) ─────────────────────────

ACCENT = "cl.brand"
REF = "cl.ref"
HINT = "cl.hint"
STYLE = {"ok": "cl.ok", "warn": "cl.warn", "err": "cl.err", "info": "cl.info"}

INDENT = "  "


def indent(level: int = 1) -> str:
    """``level`` repetitions of the 2-space indent unit."""
    return INDENT * level


def ref(text: str) -> str:
    """Markup for a reference — a command, flag, or path — inside a sentence."""
    return f"[cl.ref]{text}[/cl.ref]"


def hint(text: str) -> str:
    """Markup for an aside — keybindings, field help."""
    return f"[cl.hint]{text}[/cl.hint]"


def looks_like_path(text: str) -> bool:
    return bool(text) and " " not in text.strip() and (
        os.sep in text or text.startswith(("/", "~", "./", "../"))
    )


def path_ref(text: str) -> str:
    """``ref()`` if ``text`` looks like a filesystem path, else unchanged."""
    if text and (os.sep in text or text.startswith(("/", "~", "./"))):
        return ref(text)
    return text


def _detail_markup(text: str) -> str:
    styled = path_ref(text)
    return styled if styled != text else f"[cl.muted]{text}[/cl.muted]"


# ── Sections ──────────────────────────────────────────────────────────────────

def section(console: Console, title: str, subtitle: str = "", *, glyph: str = GLYPH["brand"]) -> None:
    """``◆ Title  subtitle ───────────`` — opens a block of output."""
    label = Text.assemble((f"{glyph} ", "cl.brand"), (title, "cl.brand"))
    if subtitle:
        label.append(f"  {subtitle}", style="cl.muted")
    console.print()
    console.print(Padding(Rule(label, align="left", style="cl.border", characters="─"), (0, 0, 0, 2)))


def rule(console: Console, label: Optional[str] = None) -> None:
    """Thin, muted divider; with a label it becomes a section header."""
    if label:
        section(console, label)
    else:
        console.print(Rule(style="cl.border"))


# ── Key/value, status and record blocks ───────────────────────────────────────

def _value_text(value: Any) -> Text:
    """Render a single value: paths as references, numbers emphasised,
    booleans as state glyphs, ``None`` as an em dash."""
    if value is None:
        return Text("—", style="cl.muted")
    if isinstance(value, bool):
        return Text(GLYPH["ok"] if value else GLYPH["err"], style="cl.ok" if value else "cl.err")
    if isinstance(value, (int, float)):
        return Text(format_count(value), style="cl.num")
    if isinstance(value, Text):
        return value
    text = str(value)
    if looks_like_path(text):
        return Text(text, style="cl.path")
    # Inline Rich markup is allowed in values (e.g. a coloured mode badge).
    if "[" in text and "[/" in text:
        try:
            return Text.from_markup(text)
        except Exception:
            pass
    return Text(text)


def _label(key: Any) -> str:
    text = str(key)
    if "_" in text and " " not in text:
        text = text.replace("_", " ")
        text = text[:1].upper() + text[1:]
    return text


def _inline_mapping(data: Mapping) -> Text:
    """``total 3 · ok 3 · failed 0`` — a flat dict on one line. Non-zero
    ``failed``/``error`` counts are highlighted so problems stand out."""
    out = Text()
    for i, (key, value) in enumerate(data.items()):
        if i:
            out.append("  ·  ", style="cl.faint")
        label = _label(key).lower()
        out.append(f"{label} ", style="cl.muted")
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value and any(
            word in label for word in ("fail", "error", "skip", "warn", "invalid")
        ):
            out.append(format_count(value), style="cl.warn")
        else:
            out.append_text(_value_text(value))
    return out


def _is_flat(data: Mapping) -> bool:
    return all(not isinstance(v, (Mapping, list, tuple)) for v in data.values())


def stats_table(rows: Mapping[str, Any], *, level: int = 1) -> Table:
    """A grid of ``label  value`` rows where values may themselves be dicts
    (rendered inline when flat and short, else as an indented sub-grid)."""
    grid = Table.grid(padding=(0, 3))
    grid.add_column(style="cl.muted", no_wrap=True)
    grid.add_column(overflow="fold")
    for key, value in rows.items():
        if isinstance(value, Mapping) and value:
            if _is_flat(value) and len(value) <= 6:
                grid.add_row(_label(key), _inline_mapping(value))
            else:
                grid.add_row(_label(key), "")
                grid.add_row("", stats_table(value, level=0))
        elif isinstance(value, (list, tuple)):
            grid.add_row(_label(key), Text(", ".join(str(v) for v in value) or "—"))
        else:
            grid.add_row(_label(key), _value_text(value))
    return Padding(grid, (0, 0, 0, len(indent(level))))  # type: ignore[return-value]


def kv_block(
    console: Console,
    rows: Iterable[Tuple[str, Any]],
    *,
    title: Optional[str] = None,
    level: int = 1,
) -> None:
    """Aligned ``Label   value`` rows under an optional section header."""
    rows = list(rows)
    if not rows:
        return
    if title:
        section(console, title)
    console.print(stats_table(dict(rows), level=level))


def stats_block(console: Console, stats: Mapping[str, Any], title: str = "Results") -> None:
    """Result summary for a finished step — nested dicts welcome."""
    if not stats:
        return
    section(console, title)
    console.print(stats_table(stats))
    console.print()


def card(
    rows: Iterable[Tuple[str, Any]],
    *,
    title: str = "",
    subtitle: str = "",
    style: str = "cl.border",
) -> Panel:
    """A rounded panel of key/value rows — for configuration summaries that
    should read as one unit (simulation parameters, compute resources)."""
    grid = Table.grid(padding=(0, 3))
    grid.add_column(style="cl.muted", no_wrap=True)
    grid.add_column(overflow="fold")
    for key, value in rows:
        grid.add_row(_label(key), _value_text(value))
    heading = Text.assemble((f"{GLYPH['brand']} ", "cl.brand"), (title, "cl.brand")) if title else None
    return Panel(
        grid,
        title=heading,
        title_align="left",
        subtitle=Text(subtitle, style="cl.muted") if subtitle else None,
        subtitle_align="right",
        border_style=style,
        box=box.ROUNDED,
        padding=(0, 2),
        expand=False,
    )


def print_card(console: Console, rows: Iterable[Tuple[str, Any]], **kwargs: Any) -> None:
    console.print()
    console.print(Padding(card(rows, **kwargs), (0, 0, 0, 2)))


def status_line(level: str, message: Union[str, Text], *, level_indent: int = 1) -> Text:
    """``  ✓  message`` — one state line (``level`` in ok/warn/err/info/skip)."""
    out = Text(indent(level_indent))
    out.append(GLYPH.get(level, GLYPH["info"]), style=STATE_STYLE.get(level, "cl.info"))
    out.append(" ")
    out.append_text(message if isinstance(message, Text) else Text.from_markup(message))
    return out


def status_block(
    console: Console,
    rows: Iterable[Tuple[str, str, str]],
    *,
    title: Optional[str] = None,
    level: int = 1,
) -> None:
    """Aligned status rows: ``  ✓ label   detail`` (``level`` ∈ ok/warn/err/info)."""
    rows = list(rows)
    if not rows:
        return
    if title:
        section(console, title)
    grid = Table.grid(padding=(0, 2))
    grid.add_column(no_wrap=True)
    grid.add_column(no_wrap=True)
    grid.add_column(overflow="fold")
    for row_level, label, detail in rows:
        key = {"warning": "warn", "error": "err"}.get(row_level, row_level)
        glyph = Text(GLYPH.get(key, GLYPH["info"]), style=STATE_STYLE.get(key, "cl.info"))
        grid.add_row(glyph, Text(label), Text.from_markup(_detail_markup(detail)) if detail else Text(""))
    console.print(Padding(grid, (0, 0, 0, len(indent(level)))))


def record_block(
    console: Console,
    headers: Iterable[str],
    rows: Iterable[Iterable[object]],
    *,
    title: Optional[str] = None,
    level: int = 1,
) -> None:
    """A light table (header rule, no outer box) for multi-column records."""
    headers = list(headers)
    str_rows = [[str(v) for v in r] for r in rows]
    if not str_rows:
        return
    if title:
        section(console, title)
    table = Table(
        box=box.SIMPLE_HEAD,
        show_edge=False,
        pad_edge=False,
        header_style="cl.muted",
        border_style="cl.border",
    )
    for h in headers:
        table.add_column(h, overflow="fold")
    for r in str_rows:
        table.add_row(*[_value_text(v) for v in r])
    console.print(Padding(table, (0, 0, 0, len(indent(level)))))


# ── Messages & errors ─────────────────────────────────────────────────────────

def error_panel(
    title: str,
    message: Union[str, BaseException],
    *,
    hint_text: Optional[str] = None,
    details: Optional[str] = None,
    max_detail_lines: int = 12,
) -> Panel:
    """A red-bordered panel: what went wrong, the detail, and what to do next."""
    body: List[RenderableType] = []
    text = str(message).strip()
    first, _, rest = text.partition("\n")
    body.append(Text(first or title, style="bold"))
    extra = "\n".join(filter(None, [rest.strip(), (details or "").strip()]))
    if extra:
        lines = list(iter_lines(extra))
        shown = lines[:max_detail_lines]
        detail = Text()
        for line in shown:
            detail.append("│ ", style="cl.faint")
            detail.append(line + "\n", style="cl.muted")
        if len(lines) > len(shown):
            detail.append(f"│ … {len(lines) - len(shown)} more line(s)\n", style="cl.faint")
        detail.rstrip()
        body.append(detail)
    if hint_text:
        for line in hint_text.splitlines():
            body.append(Text.assemble((f"{GLYPH['arrow']} ", "cl.accent"), Text.from_markup(line)))
    return Panel(
        Group(*body),
        title=Text.assemble((f"{GLYPH['err']} ", "cl.err"), (title, "cl.err")),
        title_align="left",
        border_style="cl.err.soft",
        box=box.ROUNDED,
        padding=(0, 2),
        expand=False,
    )


# ── Log-line format (shared with utils.logger and pipeline echo lines) ────────

LEVEL_META = {
    # canonical  (glyph, badge, style)
    "debug": ("·", "DEBUG", "cl.debug"),
    "info": ("•", "INFO", "cl.info"),
    "ok": ("✓", "OK", "cl.ok"),
    "note": ("▲", "NOTE", "cl.warn"),
    "warn": ("▲", "WARN", "cl.warn"),
    "err": ("✗", "ERROR", "cl.err"),
    "fatal": ("✗", "FATAL", "cl.err"),
}

_LEVEL_ALIASES = {
    "DEBUG": "debug",
    "INFO": "info",
    "OK": "ok",
    "SUCCESS": "ok",
    "NOTE": "note",
    "WARNING": "warn",
    "WARN": "warn",
    "ERROR": "err",
    "CRITICAL": "fatal",
    "FATAL": "fatal",
}

# Pipelines prefix many messages with a bracketed phase, e.g. "[NVT] mdrun …",
# and GROMACS lines arrive as "NOTE: …" / "WARNING: …" — the badge already says so.
_PHASE_PREFIX = re.compile(r"^\[([^\]]{1,24})\]\s+")
_LEVEL_PREFIX = re.compile(r"^\s*(?:NOTE|WARNING|WARN|ERROR|FATAL ERROR|Fatal error)\s*:\s*", re.I)
MAX_CONSOLE_DETAIL_LINES = 14


def canonical_level(level: str) -> str:
    return _LEVEL_ALIASES.get(str(level).upper(), "info")


def format_log_line(
    level: str,
    message: str,
    *,
    component: str = "",
    when: Optional[datetime] = None,
    show_time: bool = True,
    max_lines: int = MAX_CONSOLE_DETAIL_LINES,
    more_hint: str = "",
) -> Text:
    """``11:42:03  ▲ WARN   ions        Neutralization added 3 Na+``

    Continuation lines (stderr dumps, tracebacks) hang under the message
    behind a faint gutter and are capped at ``max_lines`` on screen — the log
    file keeps every line.
    """
    key = canonical_level(level)
    glyph, badge, style = LEVEL_META[key]
    message = str(message)
    match = _PHASE_PREFIX.match(message)
    if match and not component:
        component, message = match.group(1), message[match.end():]
    if key in ("note", "warn", "err", "fatal"):
        message = _LEVEL_PREFIX.sub("", message, count=1)

    lines = message.splitlines() or [""]
    out = Text(no_wrap=False, overflow="fold")
    prefix_width = 0
    if show_time:
        stamp = (when or datetime.now()).strftime("%H:%M:%S")
        out.append(stamp + "  ", style="cl.faint")
        prefix_width += len(stamp) + 2
    out.append(f"{glyph} ", style=style)
    out.append(f"{badge:<5}", style=style)
    out.append("  ")
    prefix_width += 2 + 5 + 2
    if component:
        comp = component.replace(" equilibration", "").replace(" ", "-").lower()
        comp = comp if len(comp) <= 14 else comp[:13] + "…"
        out.append(f"{comp:<14}", style="cl.muted")
        out.append(" ")
        prefix_width += 15
    body_style = "cl.err.soft" if key in ("err", "fatal") else ""
    out.append(lines[0], style=body_style)

    rest = [ln for ln in lines[1:]]
    while rest and not rest[-1].strip():
        rest.pop()
    if rest:
        shown = rest[:max_lines]
        pad = " " * prefix_width
        for line in shown:
            out.append("\n" + pad)
            out.append("│ ", style="cl.faint")
            out.append(line, style="cl.muted")
        hidden = len(rest) - len(shown)
        if hidden > 0:
            out.append("\n" + pad)
            out.append(f"│ … {hidden} more line(s)" + (f" — {more_hint}" if more_hint else ""), style="cl.faint")
    return out


def log_line(level: str, message: str, phase: str = "") -> Text:
    """A pipeline echo line printed beside a live progress bar — the same
    format the logger uses, so echoed and logged lines look identical."""
    return format_log_line(level, message, component=phase)


__all__ = [
    "ACCENT", "REF", "HINT", "STYLE", "GLYPH", "INDENT",
    "indent", "ref", "hint", "path_ref", "looks_like_path",
    "section", "rule", "kv_block", "stats_block", "stats_table", "card", "print_card",
    "status_line", "status_block", "record_block", "error_panel",
    "format_log_line", "log_line", "canonical_level", "get_console", "escape",
]
