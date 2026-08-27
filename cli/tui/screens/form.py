"""A form built by reflecting over an argparse leaf parser's own actions — one editable
field per argument, no separately-maintained field list. Submitting never hand-builds a
Namespace: it serializes the fields back into an argv list and calls the parser's own
``parse_args``, so every existing validator (type=, choices=, custom checks) runs
completely unmodified and can never drift from what the plain CLI enforces.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import re
import shlex
from dataclasses import dataclass
from typing import List, Optional

from rich.console import Group
from rich.text import Text

from ..anim import Blink, Ease, Reveal

_VIEWPORT = 18
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


@dataclass
class Field:
    action: argparse.Action
    kind: str  # "text" | "append" | "toggle" | "choice"
    value: object


def _label(action: argparse.Action) -> str:
    if action.option_strings:
        long = next((o for o in action.option_strings if o.startswith("--")), action.option_strings[0])
        return long
    return f"<{action.dest}>"


def _build_fields(parser: argparse.ArgumentParser) -> List[Field]:
    fields: List[Field] = []
    for action in parser._actions:
        if isinstance(action, (argparse._HelpAction, argparse._SubParsersAction)):
            continue
        if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction, argparse._StoreConstAction)):
            fields.append(Field(action, "toggle", False))
        elif action.choices:
            default = str(action.default) if action.default is not None else str(action.choices[0])
            fields.append(Field(action, "choice", default))
        elif isinstance(action, argparse._AppendAction):
            fields.append(Field(action, "append", ""))
        else:
            fields.append(Field(action, "text", "" if action.default is None else str(action.default)))
    return fields


def _cycle(choices, current, step: int):
    choices = list(choices)
    try:
        i = choices.index(current)
    except ValueError:
        i = 0
    return str(choices[(i + step) % len(choices)])


def _build_argv(fields: List[Field]) -> List[str]:
    positionals: List[str] = []
    optionals: List[str] = []
    for f in fields:
        action = f.action
        is_positional = not action.option_strings
        flag = action.option_strings[0] if action.option_strings else None

        if f.kind == "toggle":
            if f.value:
                optionals.append(flag)
            continue
        if f.kind == "choice":
            optionals.append(flag)
            optionals.append(str(f.value))
            continue

        text = str(f.value or "").strip()
        if not text:
            continue
        tokens = shlex.split(text)
        if is_positional:
            positionals.extend(tokens)
        elif f.kind == "append":
            # argparse's append action expects the flag once per value, unlike a
            # multi-token nargs=3 field (e.g. --manual-center X Y Z) below.
            for tok in tokens:
                optionals.append(flag)
                optionals.append(tok)
        else:
            optionals.append(flag)
            optionals.extend(tokens)
    return positionals + optionals


class FormScreen:
    def __init__(self, parser: argparse.ArgumentParser, breadcrumb: str = ""):
        self.parser = parser
        self.breadcrumb = breadcrumb
        self.fields = _build_fields(parser)
        self.cursor = 0
        self.scroll = 0
        self.editing = False
        self.draft = ""
        self.error: Optional[str] = None
        self._reveal = Reveal()
        self._marker = Ease(0.0)
        self._blink = Blink()

    def tick(self) -> bool:
        still_revealing = self._reveal.tick()
        still_gliding = self._marker.tick()
        still_blinking = self._blink.tick() if self.editing else False
        return still_revealing or still_gliding or still_blinking

    def _scroll_to_cursor(self) -> None:
        if self.cursor < self.scroll:
            self.scroll = self.cursor
        elif self.cursor >= self.scroll + _VIEWPORT:
            self.scroll = self.cursor - _VIEWPORT + 1

    def render(self, height: Optional[int] = None) -> Group:
        lines: List[Text] = [Text(f"◆ {self.breadcrumb}", style="bold cyan", justify="center"), Text("")]

        fields = self.fields
        marker_row = round(self._marker.value)
        if not fields:
            lines.append(Text("  (no options — enter/ctrl-r to run)", style="dim"))
        else:
            window = fields[self.scroll:self.scroll + _VIEWPORT]
            label_w = max(len(_label(f.action)) for f in fields)
            if self.scroll > 0:
                lines.append(Text("  ↑ more above", style="dim"))
            for offset, f in enumerate(window):
                i = self.scroll + offset
                focused = i == self.cursor
                is_marker = i == marker_row
                line = Text("▸ " if is_marker else "  ", style="bold cyan" if is_marker else "")
                line.append(f"{_label(f.action):<{label_w}}", style="bold" if focused else "")
                line.append("  ")
                if f.kind == "toggle":
                    on = bool(f.value)
                    line.append("[x]" if on else "[ ]", style="bold green" if on else "dim")
                elif f.kind == "choice":
                    line.append(f"‹ {f.value} ›", style="cyan")
                else:
                    shown = self.draft if (focused and self.editing) else str(f.value)
                    style = "" if shown else "dim"
                    line.append(shown if shown else "(empty)", style=style)
                    if focused and self.editing and self._blink.on:
                        line.append("▏", style="bold cyan")
                if f.action.help:
                    line.append("   " + f.action.help, style="dim italic")
                lines.append(line)
            if self.scroll + _VIEWPORT < len(fields):
                lines.append(Text("  ↓ more below", style="dim"))

        lines.append(Text(""))
        if self.error:
            lines.append(Text(f"✗ {self.error}", style="bold red"))
            lines.append(Text(""))
        if self.editing:
            hint_text = "type to edit   enter commit   esc cancel"
        else:
            hint_text = "↑/↓ field   ‹/› cycle   space toggle   enter edit/run   ctrl-r run   esc back"
        lines.append(Text(hint_text, style="dim italic", justify="center"))

        if not self._reveal.done:
            n = self._reveal.visible(len(lines))
            lines = [lines[i] if i < n else Text("") for i in range(len(lines))]
        return Group(*lines)

    def _submit(self):
        argv = _build_argv(self.fields)
        buf = io.StringIO()
        try:
            with contextlib.redirect_stderr(buf):
                args = self.parser.parse_args(argv)
        except SystemExit:
            self.error = _ANSI_RE.sub("", buf.getvalue()).strip() or "Invalid input."
            return None
        if not hasattr(args, "handler"):
            self.error = "This needs a subcommand — go back and pick one."
            return None
        self.error = None
        return ("run", args, self.breadcrumb, argv)

    def handle_key(self, key: str):
        fields = self.fields

        if self.editing:
            f = fields[self.cursor]
            if key == "enter":
                f.value = self.draft
                self.editing = False
            elif key == "esc":
                self.editing = False
            elif key == "backspace":
                self.draft = self.draft[:-1]
            elif len(key) == 1 and key.isprintable():
                self.draft += key
            return None

        if key == "esc":
            return ("pop",)
        if key == "ctrl-c":
            return ("quit",)
        if key == "ctrl-r":
            return self._submit()

        if key == "up":
            self.cursor = max(0, self.cursor - 1)
            self._marker.set_target(float(self.cursor))
            self._scroll_to_cursor()
        elif key == "down":
            self.cursor = min(len(fields) - 1, self.cursor + 1) if fields else 0
            self._marker.set_target(float(self.cursor))
            self._scroll_to_cursor()
        elif fields:
            f = fields[self.cursor]
            if f.kind in ("text", "append"):
                if key == "enter":
                    self.draft = str(f.value)
                    self.editing = True
                    self._blink.restart()
            elif f.kind == "toggle":
                if key in ("enter", " "):
                    f.value = not f.value
            elif f.kind == "choice":
                if key == "left":
                    f.value = _cycle(f.action.choices, f.value, -1)
                elif key == "right":
                    f.value = _cycle(f.action.choices, f.value, 1)
                elif key == "enter":
                    return self._submit()
        elif key == "enter":
            return self._submit()
        return None
