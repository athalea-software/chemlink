"""Generic command list screen — used for both the Home menu and any command's
subcommand menu, since both are the same shape: a set of named argparse choices with
help text. Descriptions are read straight from the subparsers action, the same source
``--help`` already uses, so there is no separate command/description table to drift.
"""
from __future__ import annotations

import argparse
import textwrap
from typing import List, Optional, Tuple

from rich.console import Group
from rich.text import Text

from ..anim import Ease, Reveal

try:
    from ...banner import LOGO
except ImportError:
    from banner import LOGO  # type: ignore

Entry = Tuple[str, argparse.ArgumentParser, str]  # (name, parser, help)

_LOGO_LINES = textwrap.dedent(LOGO).strip("\n").splitlines()
# Home also needs room for the title/subtitle/menu/hint below the art — this is a
# generous fixed margin rather than an exact per-screen line count.
_LOGO_MIN_HEIGHT = len(_LOGO_LINES) + 14


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


class CommandListScreen:
    def __init__(
        self,
        title: str,
        sub_action: argparse._SubParsersAction,
        breadcrumb: str = "",
        show_logo: bool = False,
        subtitle: str = "",
    ):
        self.title = title
        self.breadcrumb = breadcrumb
        self.show_logo = show_logo
        self.subtitle = subtitle
        self.entries = _entries(sub_action)
        self.cursor = 0
        self.filter = ""
        self.filter_mode = False
        self._reveal = Reveal()
        self._marker = Ease(0.0)

    def tick(self) -> bool:
        still_revealing = self._reveal.tick()
        still_gliding = self._marker.tick()
        return still_revealing or still_gliding

    def _visible(self) -> List[Entry]:
        if not self.filter:
            return self.entries
        needle = self.filter.lower()
        return [e for e in self.entries if needle in e[0].lower() or needle in e[2].lower()]

    def render(self, height: Optional[int] = None) -> Group:
        visible = self._visible()
        lines: List[Text] = []

        if self.show_logo and (height is None or height >= _LOGO_MIN_HEIGHT):
            lines.append(Text("\n".join(_LOGO_LINES), style="cyan", justify="center"))
            lines.append(Text(""))
            header = Text(self.title, style="bold cyan", justify="center")
            if self.subtitle:
                header.append("  " + self.subtitle, style="dim")
            lines.append(header)
        else:
            lines.append(Text(f"◆ {self.breadcrumb or self.title}", style="bold cyan"))
        lines.append(Text(""))

        marker_row = round(self._marker.value)
        name_w = max((len(name) for name, _, _ in visible), default=0)
        for i, (name, _parser, help_text) in enumerate(visible):
            focused = i == self.cursor
            glyph = "▸ " if i == marker_row else "  "
            line = Text(glyph, style="bold cyan" if i == marker_row else "")
            line.append(f"{name:<{name_w}}", style="bold" if focused else "")
            if help_text:
                line.append("   " + help_text, style="dim italic")
            lines.append(line)
        if not visible:
            lines.append(Text("  (no matches)", style="dim"))

        lines.append(Text(""))
        if self.filter_mode:
            lines.append(Text(f"/{self.filter}", style="bold yellow", justify="center"))
        else:
            hint_text = "↑/↓ move   enter open   / filter   esc back   ctrl-c quit"
            lines.append(Text(hint_text, style="dim italic", justify="center"))

        if not self._reveal.done:
            n = self._reveal.visible(len(lines))
            lines = [lines[i] if i < n else Text("") for i in range(len(lines))]
        return Group(*lines)

    def handle_key(self, key: str):
        visible = self._visible()

        if self.filter_mode:
            if key in ("enter", "esc"):
                if key == "esc":
                    self.filter = ""
                self.filter_mode = False
            elif key == "backspace":
                self.filter = self.filter[:-1]
            elif len(key) == 1 and key.isprintable():
                self.filter += key
            self.cursor = 0
            self._marker.snap(0.0)  # filtering can reshuffle rows a lot; don't glide across it
            return None

        if key == "up":
            self.cursor = max(0, self.cursor - 1)
            self._marker.set_target(float(self.cursor))
        elif key == "down":
            self.cursor = min(len(visible) - 1, self.cursor + 1) if visible else 0
            self._marker.set_target(float(self.cursor))
        elif key == "/":
            self.filter_mode = True
        elif key == "esc":
            return ("pop",)
        elif key == "ctrl-c":
            return ("quit",)
        elif key == "enter" and visible:
            name, sub_parser, _ = visible[self.cursor]
            crumb = f"{self.breadcrumb} › {name}" if self.breadcrumb else name
            nested = find_sub_action(sub_parser)
            if nested is not None:
                return ("push", CommandListScreen(name, nested, breadcrumb=crumb))
            from .form import FormScreen
            return ("push", FormScreen(sub_parser, breadcrumb=crumb))
        return None
