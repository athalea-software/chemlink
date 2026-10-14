"""Raw-keypress reading for the TUI, built on prompt_toolkit's cross-platform input
backend (handles termios raw mode on Unix, the console API on Windows) so the rest of
the TUI never has to touch terminal mode switching directly.

Raw mode is entered/exited explicitly (not via a single long-lived ``with`` block)
because a handed-off command needs the terminal back in normal *cooked* mode — Ctrl+C
must raise KeyboardInterrupt there, and any interactive ``input()`` prompt (e.g. the
ligand charge confirmation) needs real line editing and echo, neither of which happen
in raw mode.

Keys come back as short names (``"up"``, ``"enter"``, ``"ctrl-r"`` …) or as the
literal character typed. A bracketed paste is expanded into its characters, so
pasting a path into a form field behaves exactly like typing it.
"""
from __future__ import annotations

import select
from typing import List, Optional

from prompt_toolkit.input import create_input
from prompt_toolkit.keys import Keys

_NAMED = {
    Keys.Up: "up",
    Keys.Down: "down",
    Keys.Left: "left",
    Keys.Right: "right",
    Keys.Home: "home",
    Keys.End: "end",
    Keys.PageUp: "pageup",
    Keys.PageDown: "pagedown",
    Keys.ControlM: "enter",
    Keys.ControlJ: "enter",
    Keys.Escape: "esc",
    Keys.ControlI: "tab",
    Keys.BackTab: "shift-tab",
    Keys.ControlC: "ctrl-c",
    Keys.ControlR: "ctrl-r",
    Keys.ControlU: "ctrl-u",
    Keys.ControlW: "ctrl-w",
    Keys.ControlA: "home",
    Keys.ControlE: "end",
    Keys.ControlD: "ctrl-d",
    Keys.ControlL: "ctrl-l",
    Keys.ControlH: "backspace",
    Keys.Backspace: "backspace",
    Keys.Delete: "delete",
    Keys.ControlLeft: "word-left",
    Keys.ControlRight: "word-right",
}


class KeyReader:
    def __init__(self) -> None:
        self._input = create_input()
        self._queue: List[str] = []

    def raw(self):
        """Context manager: put the terminal in raw mode for the TUI's own screens."""
        return self._input.raw_mode()

    def _decode(self, presses) -> List[str]:
        keys: List[str] = []
        for p in presses:
            if p.key == Keys.BracketedPaste:
                keys.extend(ch for ch in (p.data or "").replace("\r", "") if ch.isprintable())
                continue
            if p.key == "\x7f":
                keys.append("backspace")
                continue
            named = _NAMED.get(p.key)
            if named:
                keys.append(named)
            elif isinstance(p.key, str) and len(p.key) == 1:
                keys.append(p.key)
        return keys

    def wait_key(self, timeout: Optional[float] = None) -> Optional[str]:
        """Wait for one keypress and return its normalized name/char.

        With ``timeout`` set and no key arriving in time, returns ``None`` (an
        animation/refresh tick) instead of blocking. A single terminal read can
        return several keys at once (fast typing, a paste); extras are queued.
        """
        if self._queue:
            return self._queue.pop(0)
        ready, _, _ = select.select([self._input.fileno()], [], [], timeout)
        if not ready:
            return None
        keys = self._decode(self._input.read_keys())
        if not keys:
            return None
        self._queue.extend(keys[1:])
        return keys[0]

    def drain(self) -> None:
        """Discard anything typed while a command was running."""
        self._queue.clear()
        while select.select([self._input.fileno()], [], [], 0)[0]:
            if not self._input.read_keys():
                break
