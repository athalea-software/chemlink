"""A form built by reflecting over an argparse leaf parser's own actions — one editable
field per argument, no separately-maintained field list. Submitting never hand-builds a
Namespace: it serializes the fields back into an argv list and calls the parser's own
``parse_args``, so every existing validator (type=, choices=, custom checks) runs
completely unmodified and can never drift from what the plain CLI enforces.

Empty fields mean "use the default" and are left off the command line, so the
live command preview shows exactly the flags you changed.
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import io
import os
import re
import shlex
from dataclasses import dataclass, field
from typing import List, Optional

from rich.console import Group, RenderableType
from rich.layout import Layout
from rich.table import Table
from rich.text import Text

from ..anim import Blink, Ease, Reveal
from ..widgets import ansi, footer, header, panel, scroll_window

try:
    from ....utils.theme import truncate
except ImportError:  # pragma: no cover
    from utils.theme import truncate  # type: ignore

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_PATHISH = re.compile(r"(dir|path|file|input|output|receptor|ligand|protein|peptide|pdb|config|exe|bin|image)", re.I)
RUN_ROW = -1


@dataclass
class Field:
    action: argparse.Action
    kind: str  # "text" | "append" | "toggle" | "choice"
    value: object
    initial: object = None
    error: str = ""
    completions: List[str] = field(default_factory=list)

    @property
    def modified(self) -> bool:
        return self.value != self.initial

    @property
    def required(self) -> bool:
        a = self.action
        return bool(a.required) or (not a.option_strings and a.nargs not in ("?", "*") and a.default is None)

    @property
    def pathish(self) -> bool:
        metavar = self.action.metavar if isinstance(self.action.metavar, str) else ""
        return bool(_PATHISH.search(f"{self.action.dest} {metavar}"))


def _label(action: argparse.Action) -> str:
    if action.option_strings:
        return next((o for o in action.option_strings if o.startswith("--")), action.option_strings[0])
    return str(action.metavar or action.dest).upper() if not isinstance(action.metavar, tuple) else action.dest.upper()


def _build_fields(parser: argparse.ArgumentParser) -> List[Field]:
    fields: List[Field] = []
    for action in parser._actions:
        if isinstance(action, (argparse._HelpAction, argparse._SubParsersAction, argparse._VersionAction)):
            continue
        if action.help == argparse.SUPPRESS:
            continue
        if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction, argparse._StoreConstAction)):
            fields.append(Field(action, "toggle", False, False))
        elif action.choices:
            default = str(action.default) if action.default is not None else str(list(action.choices)[0])
            fields.append(Field(action, "choice", default, default))
        elif isinstance(action, argparse._AppendAction):
            fields.append(Field(action, "append", "", ""))
        else:
            fields.append(Field(action, "text", "", ""))
    # Positionals first, then options — the order --help lists them in —
    # except "legacy" positionals superseded by options, which go last.
    fields.sort(key=lambda f: ("legacy" in (f.action.help or "").lower(), bool(f.action.option_strings)))
    return fields


def _cycle(choices, current, step: int):
    choices = [str(c) for c in choices]
    try:
        i = choices.index(str(current))
    except ValueError:
        i = 0
    return choices[(i + step) % len(choices)]


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
            if is_positional:
                if f.modified or action.nargs not in ("?", "*"):
                    positionals.append(str(f.value))
            elif f.modified:
                optionals.extend([flag, str(f.value)])
            continue

        text = str(f.value or "").strip()
        if not text:
            continue
        try:
            tokens = shlex.split(text)
        except ValueError:
            tokens = text.split()
        if is_positional:
            positionals.extend(tokens)
        elif f.kind == "append":
            # argparse's append action expects the flag once per value, unlike a
            # multi-token nargs=3 field (e.g. --manual-center X Y Z) below.
            for tok in tokens:
                optionals.extend([flag, tok])
        else:
            optionals.append(flag)
            optionals.extend(tokens)
    return positionals + optionals


def _validate(f: Field) -> str:
    """Cheap inline checks; the real validation is the parser's own on submit."""
    a = f.action
    if f.kind in ("toggle", "choice"):
        return ""
    text = str(f.value or "").strip()
    if not text:
        return "required" if f.required else ""
    try:
        tokens = shlex.split(text)
    except ValueError:
        return "unbalanced quotes"
    if isinstance(a.nargs, int) and f.kind != "append" and len(tokens) != a.nargs:
        return f"expects {a.nargs} values"
    if a.type in (int, float):
        for tok in tokens:
            try:
                a.type(tok)
            except ValueError:
                return f"'{tok}' is not {'an integer' if a.type is int else 'a number'}"
    return ""


def _default_text(a: argparse.Action) -> str:
    if a.default in (None, argparse.SUPPRESS, False, [], ()):
        return ""
    if isinstance(a.default, (list, tuple)):
        return " ".join(map(str, a.default))
    return str(a.default)


class FormScreen:
    def __init__(self, parser: argparse.ArgumentParser, breadcrumb: str = "", ctx=None):
        self.parser = parser
        self.breadcrumb = breadcrumb
        self.ctx = ctx
        self.fields = _build_fields(parser)
        self.cursor = 0 if self.fields else RUN_ROW
        self.scroll = 0
        self.editing = False
        self.draft = ""
        self.caret = 0
        self.error: Optional[str] = None
        self._reveal = Reveal(0.25)
        self._marker = Ease(0.0, speed=22.0)
        self._blink = Blink(0.55)

    # ── state ──────────────────────────────────────────────────────────────
    def tick(self) -> bool:
        still_revealing = self._reveal.tick()
        still_gliding = self._marker.tick()
        still_blinking = self._blink.tick() if self.editing else False
        return still_revealing or still_gliding or still_blinking

    @property
    def _rows(self) -> int:
        return len(self.fields) + 1  # + the Run button

    def _row_of_cursor(self) -> int:
        return len(self.fields) if self.cursor == RUN_ROW else self.cursor

    def _set_row(self, row: int, wrap: bool = False) -> None:
        wrapped = wrap and not 0 <= row < self._rows
        row = row % self._rows if wrapped else max(0, min(self._rows - 1, row))
        self.cursor = RUN_ROW if row == len(self.fields) else row
        if wrapped:  # circular step past either end; don't glide the whole form
            self._marker.snap(float(row))
        else:
            self._marker.set_target(float(row))

    def _focused(self) -> Optional[Field]:
        return None if self.cursor == RUN_ROW or not self.fields else self.fields[self.cursor]

    def command_words(self) -> List[str]:
        return ["chemlink", *[p for p in self.breadcrumb.split(" › ") if p]]

    def command_line(self) -> str:
        return shlex.join(self.command_words() + _build_argv(self.fields))

    # ── rendering ──────────────────────────────────────────────────────────
    def _value_text(self, f: Field, focused: bool) -> Text:
        a = f.action
        if f.kind == "toggle":
            return Text("● on", style="cl.ok.soft") if f.value else Text("○ off", style="cl.muted")
        if f.kind == "choice":
            out = Text()
            out.append("‹ ", style="cl.faint")
            out.append(str(f.value), style="cl.accent" if f.modified else "")
            out.append(" ›", style="cl.faint")
            choices = [str(c) for c in a.choices]
            if focused and str(f.value) in choices:
                out.append(f"  {choices.index(str(f.value)) + 1}/{len(choices)}", style="cl.faint")
            return out
        if focused and self.editing:
            out = Text(no_wrap=True, overflow="ellipsis")
            draft = self.draft
            out.append(draft[:self.caret], style="bold")
            under = draft[self.caret:self.caret + 1] or " "
            out.append(under, style="reverse" if self._blink.on else "bold underline")
            out.append(draft[self.caret + 1:], style="bold")
            return out
        text = str(f.value or "")
        if text:
            return Text(text, style="cl.path" if f.pathish else "")
        if f.required:
            return Text("required", style="cl.warn.soft")
        default = _default_text(a)
        return Text(f"default {default}" if default else "—", style="italic cl.faint")

    def _fields_panel(self, height: int, width: int) -> RenderableType:
        inner_h = max(3, height - 2)
        row = self._row_of_cursor()
        self.scroll = scroll_window(row, self._rows, inner_h, self.scroll)
        label_w = min(26, max((len(_label(f.action)) for f in self.fields), default=8) + 1)
        value_w = max(8, width - label_w - 8)
        marker = round(self._marker.value)
        lines: List[Text] = []
        rows = list(range(self._rows))[self.scroll:self.scroll + inner_h]
        revealed = self._reveal.visible(len(rows)) if not self._reveal.done else len(rows)
        for n, r in enumerate(rows):
            if n >= revealed:
                lines.append(Text(""))
                continue
            line = Text(no_wrap=True, overflow="ellipsis")
            line.append("▌" if r == marker else " ", style="cl.brand")
            if r == len(self.fields):
                focused = self.cursor == RUN_ROW
                line.append(" " * 1)
                line.append(" ▶ Run ", style="bold #0d1117 on #1fa8db" if focused else "bold cl.accent on #1e2433")
                line.append("   " + truncate(self.command_line(), max(0, width - 14)), style="cl.faint")
                lines.append(line)
                continue
            f = self.fields[r]
            focused = r == self.cursor
            mark = "●" if f.modified else (" " if not f.required else "*")
            line.append(mark, style="cl.accent" if f.modified else "cl.warn.soft")
            label = _label(f.action)
            line.append(f"{label:<{label_w}}", style="bold" if focused else ("" if f.modified else "cl.muted"))
            value = self._value_text(f, focused)
            value.truncate(value_w, overflow="ellipsis")
            line.append_text(value)
            err = _validate(f) if (f.value or f.required) else ""
            if err and err != "required" and not (focused and self.editing):
                line.append(f"  ✗ {err}", style="cl.err.soft")
            if focused:
                line.stylize("on #1e2433")
            lines.append(line)
        sub = f"{sum(1 for f in self.fields if f.modified)} changed" if any(f.modified for f in self.fields) else ""
        if self.scroll + inner_h < self._rows:
            sub = (sub + "  " if sub else "") + "↓ more"
        return panel(Group(*lines), "Options", focused=True, subtitle=sub)

    def _help_panel(self) -> RenderableType:
        f = self._focused()
        if f is None:
            body = Group(
                Text("Run this command", style="bold"),
                Text(""),
                Text("Press ⏎ or Ctrl+R. The terminal is handed over to the command; "
                     "press any key afterwards to come back here.", style="cl.muted"),
            )
            return panel(body, "Run")
        a = f.action
        parts: List[RenderableType] = []
        names = Text()
        for i, opt in enumerate(a.option_strings or [_label(a)]):
            if i:
                names.append(", ", style="cl.faint")
            names.append(opt, style="cl.ref")
        if a.metavar and a.nargs != 0 and a.option_strings:
            mv = " ".join(a.metavar) if isinstance(a.metavar, tuple) else str(a.metavar)
            names.append(f"  {mv}", style="cl.muted")
        parts.append(names)
        if a.help:
            parts.append(Text(""))
            parts.append(ansi(a.help))
        meta = Table.grid(padding=(0, 2))
        meta.add_column(style="cl.faint", no_wrap=True)
        meta.add_column(overflow="fold")
        kind = {"toggle": "flag", "choice": "choice", "append": "repeatable"}.get(f.kind, "")
        if not kind:
            kind = {int: "integer", float: "number"}.get(a.type, "path" if f.pathish else "text")
            if isinstance(a.nargs, int) and a.nargs > 1:
                kind += f" × {a.nargs}"
        meta.add_row("type", kind + ("  · required" if f.required else ""))
        if f.kind == "choice":
            meta.add_row("choices", " | ".join(str(c) for c in a.choices))
        default = _default_text(a)
        if default and f.kind != "toggle":
            meta.add_row("default", default)
        parts.append(Text(""))
        parts.append(meta)
        err = _validate(f)
        if err and (f.value or self.editing):
            parts.append(Text(""))
            parts.append(Text(f"✗ {err}", style="cl.err.soft"))
        if f.completions and self.editing:
            parts.append(Text(""))
            parts.append(Text("MATCHES", style="cl.faint"))
            for c in f.completions[:8]:
                parts.append(Text("  " + c, style="cl.path"))
            if len(f.completions) > 8:
                parts.append(Text(f"  … {len(f.completions) - 8} more", style="cl.faint"))
        return panel(Group(*parts), "Help")

    def _preview_panel(self) -> RenderableType:
        line = Text()
        line.append("$ ", style="cl.faint")
        for i, tok in enumerate(shlex.split(self.command_line())):
            if i:
                line.append(" ")
            quoted = shlex.quote(tok)
            style = "bold" if i < len(self.command_words()) else ("cl.ref" if tok.startswith("-") else "")
            line.append(quoted, style=style)
        body: List[RenderableType] = [line]
        if self.error:
            body.append(Text(""))
            body.append(Text(f"✗ {self.error}", style="cl.err.soft"))
        return panel(Group(*body), "Command")

    def render(self, width: int = 100, height: int = 30) -> RenderableType:
        root = Layout()
        crumbs = [p for p in self.breadcrumb.split(" › ") if p]
        body_h = max(6, height - 4)
        preview_h = 3 + (2 if self.error else 0) + (len(self.command_line()) // max(20, width // 2))
        body = Layout(size=body_h)
        if width >= 100:
            left_w = max(56, min(84, width * 11 // 20))
            right = Layout()
            right.split_column(Layout(self._help_panel()), Layout(self._preview_panel(), size=preview_h + 1))
            body.split_row(Layout(self._fields_panel(body_h, left_w - 4), size=left_w), right)
        else:
            body.split_column(
                Layout(self._fields_panel(body_h - preview_h - 1, width - 4)),
                Layout(self._preview_panel(), size=preview_h + 1),
            )
        if self.editing:
            keys = [("⏎", "save"), ("esc", "cancel"), ("tab", "complete path"), ("←→", "cursor"), ("^U", "clear")]
        else:
            keys = [("↑↓", "field"), ("⏎", "edit"), ("←→", "choice"), ("space", "toggle"),
                    ("^R", "run"), ("⌫", "reset"), ("esc", "back")]
        status = None
        changed = sum(1 for f in self.fields if f.modified)
        if changed:
            status = Text(f"{changed} changed ", style="cl.accent")
        root.split_column(
            Layout(header(crumbs, getattr(self.ctx, "version", ""), getattr(self.ctx, "sysinfo", None)), size=2),
            body,
            Layout(footer(keys, status), size=2),
        )
        return root

    # ── submit ─────────────────────────────────────────────────────────────
    def _submit(self):
        problems = [(f, _validate(f)) for f in self.fields]
        problems = [(f, e) for f, e in problems if e]
        if problems:
            f, e = problems[0]
            self.error = f"{_label(f.action)}: {e}"
            self._set_row(self.fields.index(f))
            return None
        argv = _build_argv(self.fields)
        buf = io.StringIO()
        parser_cls = type(self.parser)
        previous = getattr(parser_cls, "raise_errors", None)
        try:
            if previous is not None:
                parser_cls.raise_errors = True
            with contextlib.redirect_stderr(buf), contextlib.redirect_stdout(io.StringIO()):
                args = self.parser.parse_args(argv)
        except SystemExit:
            self.error = _ANSI_RE.sub("", buf.getvalue()).strip().splitlines()[-1:] or ["Invalid input."]
            self.error = self.error[0]
            return None
        except Exception as exc:  # CLIUsageError from cli.main
            self.error = str(exc)
            return None
        finally:
            if previous is not None:
                parser_cls.raise_errors = previous
        if not hasattr(args, "handler"):
            self.error = "This needs a subcommand — go back and pick one."
            return None
        self.error = None
        return ("run", args, self.breadcrumb, argv)

    # ── editing ────────────────────────────────────────────────────────────
    def _begin_edit(self, f: Field, initial: Optional[str] = None) -> None:
        self.draft = str(f.value) if initial is None else initial
        self.caret = len(self.draft)
        self.editing = True
        f.completions = []
        self._blink.restart()

    def _complete_path(self, f: Field) -> None:
        # Complete the last whitespace-separated token (multi-value fields).
        head, _, token = self.draft[:self.caret].rpartition(" ")
        prefix = head + (" " if head else "")
        pattern = os.path.expanduser(token) + "*"
        matches = sorted(glob.glob(pattern))[:200]
        f.completions = [m + ("/" if os.path.isdir(m) else "") for m in matches]
        if not matches:
            return
        if len(matches) == 1:
            new = f.completions[0]
        else:
            new = os.path.commonprefix(matches)
            if len(new) <= len(os.path.expanduser(token)):
                return
        if token.startswith("~"):
            new = new.replace(os.path.expanduser("~"), "~", 1)
        tail = self.draft[self.caret:]
        self.draft = prefix + new + tail
        self.caret = len(prefix + new)

    def _edit_key(self, f: Field, key: str) -> None:
        d, c = self.draft, self.caret
        if key == "enter":
            f.value = d.strip()
            f.completions = []
            self.editing = False
            self.error = None
        elif key == "esc":
            f.completions = []
            self.editing = False
        elif key == "backspace":
            if c > 0:
                self.draft, self.caret = d[:c - 1] + d[c:], c - 1
        elif key == "delete":
            self.draft = d[:c] + d[c + 1:]
        elif key == "left":
            self.caret = max(0, c - 1)
        elif key == "right":
            self.caret = min(len(d), c + 1)
        elif key == "home":
            self.caret = 0
        elif key == "end":
            self.caret = len(d)
        elif key == "word-left":
            self.caret = len(d[:c].rstrip().rsplit(" ", 1)[0]) if " " in d[:c].rstrip() else 0
        elif key == "word-right":
            nxt = d.find(" ", c + 1)
            self.caret = len(d) if nxt < 0 else nxt
        elif key == "ctrl-u":
            self.draft, self.caret = d[c:], 0
        elif key == "ctrl-w":
            left = d[:c].rstrip()
            cut = left.rsplit(" ", 1)[0] + " " if " " in left else ""
            self.draft, self.caret = cut + d[c:], len(cut)
        elif key == "tab":
            self._complete_path(f)
        elif len(key) == 1 and key.isprintable():
            self.draft, self.caret = d[:c] + key + d[c:], c + 1
        self._blink.restart()

    def handle_key(self, key: str):
        if self.editing:
            f = self._focused()
            if f is not None:
                self._edit_key(f, key)
            else:
                self.editing = False
            return None

        if key == "esc":
            return ("pop",)
        if key == "ctrl-c":
            return ("quit",)
        if key == "ctrl-r":
            return self._submit()

        row = self._row_of_cursor()
        if key in ("up", "shift-tab"):
            self._set_row(row - 1, wrap=True)
            return None
        if key in ("down", "tab"):
            self._set_row(row + 1, wrap=True)
            return None
        if key == "pageup":
            self._set_row(row - 10)
            return None
        if key == "pagedown":
            self._set_row(row + 10)
            return None
        if key == "home":
            self._set_row(0)
            return None
        if key == "end":
            self._set_row(self._rows - 1)
            return None

        f = self._focused()
        if f is None:  # the Run button
            if key in ("enter", " "):
                return self._submit()
            return None

        if key in ("backspace", "delete"):
            f.value = f.initial
            f.completions = []
            return None
        if f.kind in ("text", "append"):
            if key == "enter":
                self._begin_edit(f)
            elif len(key) == 1 and key.isprintable() and key != " ":
                self._begin_edit(f, str(f.value or "") + key)
        elif f.kind == "toggle":
            if key in ("enter", " ", "left", "right"):
                f.value = not f.value
        elif f.kind == "choice":
            if key == "left":
                f.value = _cycle(f.action.choices, f.value, -1)
            elif key in ("right", "enter", " "):
                f.value = _cycle(f.action.choices, f.value, 1)
        self.error = None
        return None
