"""Command list screen — the Home menu and every command's subcommand menu, since
both are the same shape: a set of named argparse choices with help text.
Descriptions are read straight from the subparsers action (the same source
``--help`` uses), so there is no separate command table to drift.

Layout::

    ◆ ChemLink v1.0.0   docking                        ● no GPU · CPU mode · 8 threads
    ──────────────────────────────────────────────────────────────────────────────
    ╭ Commands ─────────────────────╮ ╭ ◎ docking ───────────────────────────────╮
    │ PIPELINES                     │ │ Molecular docking workflows              │
    │▌◎ docking        Molecular …  │ │                                          │
    │ ≋ dynamic        GROMACS …    │ │ Pipeline steps (in order): …             │
    │ ...                           │ │ SUBCOMMANDS                              │
    ╰───────────────────────────────╯ ╰──────────────────────────────────────────╯
    ──────────────────────────────────────────────────────────────────────────────
     ↑↓  move    ⏎  open    /  filter    esc  back    q  quit
"""
from __future__ import annotations

import argparse
from typing import List, Optional, Tuple

from rich.console import Group, RenderableType
from rich.layout import Layout
from rich.table import Table
from rich.text import Text

from ..anim import Ease, Reveal
from ..widgets import (
    GROUPS, ICONS, description_text, footer, header, panel, runs_table, scroll_window,
)

try:
    from ...banner import SUBTAGLINE, TAGLINE, brand_size, render_brand, render_byline
except ImportError:  # pragma: no cover
    from banner import SUBTAGLINE, TAGLINE, brand_size, render_brand, render_byline  # type: ignore

try:
    from ....utils.theme import gradient_text, truncate
except ImportError:  # pragma: no cover
    from utils.theme import gradient_text, truncate  # type: ignore

Entry = Tuple[str, argparse.ArgumentParser, str]  # (name, parser, help)


def find_sub_action(parser: argparse.ArgumentParser) -> Optional[argparse._SubParsersAction]:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return action
    return None


def _entries(sub_action: argparse._SubParsersAction) -> List[Entry]:
    """Dedupe alias choices (argparse repeats the same parser object under each
    alias, e.g. 'prep' -> the 'prepare' parser) and pair each with its help text."""
    help_by_name = {pa.dest: (pa.help or "") for pa in sub_action._choices_actions}
    seen_ids = set()
    out: List[Entry] = []
    for name, sub_parser in sub_action.choices.items():
        if id(sub_parser) in seen_ids:
            continue
        seen_ids.add(id(sub_parser))
        out.append((name, sub_parser, help_by_name.get(name, "")))
    return out


def _option_summary(parser: argparse.ArgumentParser) -> RenderableType:
    """A leaf command's arguments at a glance: required inputs, then a count."""
    actions = [
        a for a in parser._actions
        if not isinstance(a, (argparse._HelpAction, argparse._SubParsersAction))
        and a.help != argparse.SUPPRESS
    ]
    required = [a for a in actions if a.required or (not a.option_strings and a.nargs not in ("?", "*") and a.default is None)]
    grid = Table.grid(padding=(0, 2))
    grid.add_column(no_wrap=True, style="cl.ref")
    grid.add_column(overflow="fold", style="cl.muted")
    for a in required[:8]:
        name = a.option_strings[-1] if a.option_strings else str(a.metavar or a.dest).upper()
        grid.add_row(name, a.help or "")
    parts: List[RenderableType] = []
    if required:
        parts.append(Text("REQUIRED", style="cl.faint"))
        parts.append(grid)
        parts.append(Text(""))
    optional = len(actions) - len(required)
    line = Text()
    line.append(f"{optional} option{'s' if optional != 1 else ''}", style="bold")
    line.append(" with sensible defaults — press ", style="cl.muted")
    line.append(" ⏎ ", style="bold #e6edf3 on #2b3240")
    line.append(" to configure and run.", style="cl.muted")
    parts.append(line)
    return Group(*parts)


class CommandListScreen:
    def __init__(
        self,
        title: str,
        sub_action: argparse._SubParsersAction,
        breadcrumb: str = "",
        show_logo: bool = False,
        subtitle: str = "",
        ctx=None,
    ):
        self.title = title
        self.breadcrumb = breadcrumb
        self.show_logo = show_logo
        self.subtitle = subtitle
        self.ctx = ctx
        self.entries = _entries(sub_action)
        if show_logo:
            order = [n for _, names in GROUPS for n in names]
            self.entries.sort(key=lambda e: order.index(e[0]) if e[0] in order else len(order))
        self.cursor = 0
        self.scroll = 0
        self.filter = ""
        self.filter_mode = False
        self._reveal = Reveal(0.28)
        self._marker = Ease(0.0, speed=22.0)

    # ── state ──────────────────────────────────────────────────────────────
    def tick(self) -> bool:
        still_revealing = self._reveal.tick()
        still_gliding = self._marker.tick()
        return still_revealing or still_gliding or self.filter_mode

    def _visible(self) -> List[Entry]:
        if not self.filter:
            return self.entries
        needle = self.filter.lower()
        return [e for e in self.entries if needle in e[0].lower() or needle in e[2].lower()]

    def _crumbs(self) -> List[str]:
        return [p for p in self.breadcrumb.split(" › ") if p] if self.breadcrumb else []

    # ── rendering ──────────────────────────────────────────────────────────
    def _rows(self, visible: List[Entry]) -> List[Tuple[str, Optional[int], str]]:
        """Display rows: ("group", None, label) headers and ("entry", index, "")."""
        rows: List[Tuple[str, Optional[int], str]] = []
        if self.show_logo and not self.filter:
            index = {e[0]: i for i, e in enumerate(visible)}
            placed = set()
            for label, names in GROUPS:
                members = [index[n] for n in names if n in index]
                if not members:
                    continue
                if rows:
                    rows.append(("gap", None, ""))
                rows.append(("group", None, label))
                for i in members:
                    rows.append(("entry", i, ""))
                    placed.add(i)
            rest = [i for i in range(len(visible)) if i not in placed]
            if rest:
                rows.append(("gap", None, ""))
                rows.append(("group", None, "Other"))
                rows.extend(("entry", i, "") for i in rest)
        else:
            rows = [("entry", i, "") for i in range(len(visible))]
        return rows

    def _list_panel(self, visible: List[Entry], height: int, width: int) -> RenderableType:
        rows = self._rows(visible)
        inner_h = max(3, height - 2)
        cursor_row = next((r for r, (k, i, _) in enumerate(rows) if k == "entry" and i == self.cursor), 0)
        self.scroll = scroll_window(cursor_row, len(rows), inner_h, self.scroll)
        window = rows[self.scroll:self.scroll + inner_h]
        name_w = min(18, max((len(n) for n, _, _ in visible), default=8))
        help_w = max(0, width - name_w - 10)
        marker = round(self._marker.value)
        revealed = self._reveal.visible(len(window)) if not self._reveal.done else len(window)

        lines: List[Text] = []
        for r, (kind, idx, label) in enumerate(window):
            if r >= revealed:
                lines.append(Text(""))
                continue
            if kind == "gap":
                lines.append(Text(""))
                continue
            if kind == "group":
                lines.append(Text(f"  {label.upper()}", style="cl.faint"))
                continue
            name, _parser, help_text = visible[idx]
            focused = idx == self.cursor
            line = Text(no_wrap=True, overflow="ellipsis")
            line.append("▌" if idx == marker else " ", style="cl.brand")
            line.append(f"{ICONS.get(name, '•')} ", style="cl.brand" if focused else "cl.muted")
            line.append(f"{name:<{name_w}}", style="bold" if focused else "")
            if help_w > 8:
                line.append("  " + truncate(help_text, help_w), style="cl.muted" if not focused else "")
            if focused:
                line.stylize("on #1e2433")
            lines.append(line)
        if not visible:
            lines.append(Text("  no matches", style="cl.muted"))
        above = self.scroll > 0
        below = self.scroll + inner_h < len(rows)
        sub = "↑ more" if above and not below else ("↓ more" if below and not above else ("↕ more" if above else ""))
        if self.filter_mode or self.filter:
            sub = f"/{self.filter}▏" if self.filter_mode else f"filter: {self.filter}"
        return panel(Group(*lines), "Commands" if self.show_logo else self.title, focused=True, subtitle=sub)

    def _detail_panel(self, visible: List[Entry]) -> RenderableType:
        if not visible:
            return panel(Text("Nothing matches the filter.", style="cl.muted"), "Details")
        name, sub_parser, help_text = visible[self.cursor]
        parts: List[RenderableType] = []
        if help_text:
            parts.append(Text(help_text, style="bold"))
        desc = description_text(sub_parser)
        if desc.plain.strip():
            # The first description line repeats the command name; skip it.
            lines = desc.split("\n")
            if lines and name in lines[0].plain and "—" in lines[0].plain:
                lines = lines[1:]
            while lines and not lines[0].plain.strip():
                lines = lines[1:]
            body = Text("\n").join(lines)
            body.rstrip()
            if body.plain.strip():
                parts.append(Text(""))
                parts.append(body)
        nested = find_sub_action(sub_parser)
        parts.append(Text(""))
        if nested is not None:
            parts.append(Text("SUBCOMMANDS", style="cl.faint"))
            grid = Table.grid(padding=(0, 2))
            grid.add_column(no_wrap=True, style="cl.ref")
            grid.add_column(overflow="ellipsis", no_wrap=True, style="cl.muted")
            for sub_name, _p, sub_help in _entries(nested)[:12]:
                grid.add_row(sub_name, sub_help)
            parts.append(grid)
        else:
            parts.append(_option_summary(sub_parser))
        if self.show_logo:
            runs = runs_table()
            if runs is not None:
                parts.append(Text(""))
                parts.append(Text("RECENT RUNS", style="cl.faint"))
                parts.append(runs)
        title = f"{ICONS.get(name, '•')} {name}"
        return panel(Group(*parts), title)

    def _hero(self, width: int, height: int) -> Tuple[Optional[RenderableType], int]:
        """The logo above the menu, signed "by Athalea" — the biggest size that
        still leaves the command list (≈17 rows) and header/footer room."""
        if not self.show_logo:
            return None, 0
        tag = Text(justify="center")
        tag.append(f"{TAGLINE} · {SUBTAGLINE}", style="cl.muted")
        if self.subtitle:
            tag.append(f"  ·  {self.subtitle}", style="cl.faint")
        room = height - 4 - 17
        for name in ("LOGO_LARGE", "LOGO_MEDIUM"):
            rows, cols = brand_size(name)
            if rows + 4 <= room and cols + 4 <= width:
                return Group(Text(""), render_brand(name), render_byline(), Text(""), tag), rows + 4
        word = Text(justify="center")
        word.append_text(gradient_text("ChemLink"))
        return Group(Text(""), word, render_byline(), tag), 4

    def render(self, width: int = 100, height: int = 30) -> RenderableType:
        visible = self._visible()
        if visible:
            self.cursor = min(self.cursor, len(visible) - 1)
        root = Layout()
        hero, hero_h = self._hero(width, height)
        body_h = max(6, height - 2 - 2 - hero_h)
        sections = [Layout(header(self._crumbs(), getattr(self.ctx, "version", ""), getattr(self.ctx, "sysinfo", None)), size=2)]
        if hero is not None:
            sections.append(Layout(hero, size=hero_h))
        main = Layout(size=body_h)
        two_col = width >= 92
        list_w = max(38, min(56, width * 2 // 5)) if two_col else width
        list_panel = self._list_panel(visible, body_h, list_w - 4)
        if two_col:
            main.split_row(Layout(list_panel, size=list_w), Layout(self._detail_panel(visible)))
        else:
            main.update(list_panel)
        sections.append(main)
        if self.filter_mode:
            keys = [("type", "filter"), ("⏎", "keep"), ("esc", "clear")]
        else:
            keys = [("↑↓", "move"), ("⏎", "open"), ("/", "filter"), ("esc", "back"), ("q", "quit")]
            if self.show_logo:
                del keys[3]
        sections.append(Layout(footer(keys), size=2))
        root.split_column(*sections)
        return root

    # ── input ──────────────────────────────────────────────────────────────
    def _move(self, delta: int, count: int, wrap: bool = False) -> None:
        if not count:
            return
        target = self.cursor + delta
        if wrap and not 0 <= target < count:
            # Circular step past either end; snap so the marker doesn't glide the whole list.
            self.cursor = target % count
            self._marker.snap(float(self.cursor))
            return
        self.cursor = max(0, min(count - 1, target))
        self._marker.set_target(float(self.cursor))

    def handle_key(self, key: str):
        visible = self._visible()

        if self.filter_mode:
            if key in ("enter", "esc"):
                if key == "esc":
                    self.filter = ""
                self.filter_mode = False
            elif key == "backspace":
                self.filter = self.filter[:-1]
            elif key in ("up", "down"):
                self._move(-1 if key == "up" else 1, len(visible), wrap=True)
                return None
            elif len(key) == 1 and key.isprintable():
                self.filter += key
            self.cursor = 0
            self._marker.snap(0.0)  # filtering reshuffles rows; don't glide across it
            return None

        if key in ("up", "k"):
            self._move(-1, len(visible), wrap=True)
        elif key in ("down", "j"):
            self._move(1, len(visible), wrap=True)
        elif key == "pageup":
            self._move(-8, len(visible))
        elif key == "pagedown":
            self._move(8, len(visible))
        elif key == "home":
            self._move(-len(visible), len(visible))
        elif key == "end":
            self._move(len(visible), len(visible))
        elif key == "/":
            self.filter_mode = True
        elif key in ("esc", "left", "h"):
            if self.filter:
                self.filter = ""
                return None
            return ("pop",)
        elif key in ("ctrl-c", "q"):
            return ("quit",)
        elif key in ("enter", "right", "l") and visible:
            name, sub_parser, _ = visible[self.cursor]
            crumb = f"{self.breadcrumb} › {name}" if self.breadcrumb else name
            nested = find_sub_action(sub_parser)
            if nested is not None:
                return ("push", CommandListScreen(name, nested, breadcrumb=crumb, ctx=self.ctx))
            from .form import FormScreen
            return ("push", FormScreen(sub_parser, breadcrumb=crumb, ctx=self.ctx))
        return None
