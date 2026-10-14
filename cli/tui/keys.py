"""Raw-keypress reading for the TUI, built on prompt_toolkit's cross-platform input
backend (handles termios raw mode on Unix, the console API on Windows) so the rest of
the TUI never has to touch terminal mode switching directly.

Raw mode is entered/exited explicitly (not via a single long-lived ``with`` block)
because a handed-off command needs the terminal back in normal *cooked* mode — Ctrl+C
must raise KeyboardInterrupt there, and any interactive ``input()`` prompt (e.g. the
ligand charge confirmation) needs real line editing and echo, neither of which happen
in raw mode.
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
    Keys.ControlM: "enter",
    Keys.ControlJ: "enter",
    Keys.Escape: "esc",
    Keys.ControlI: "tab",
    Keys.ControlC: "ctrl-c",
    Keys.ControlR: "ctrl-r",
    Keys.ControlH: "backspace",
    Keys.Delete: "backspace",
}


class KeyReader:
    def __init__(self) -> None:
        self._input = create_input()
        self._queue: List[str] = []

    def raw(self):
        """Context manager: put the terminal in raw mode for the TUI's own screens."""
        return self._input.raw_mode()

    def wait_key(self, timeout: Optional[float] = None) -> Optional[str]:
        """Wait for one physical keypress and return its normalized name/char.

        With ``timeout`` set and no key arriving in time, returns ``None`` (a
        animation "tick") instead of blocking — the TUI's main loop uses this to
        keep redrawing in-flight animations without a keypress. A single terminal
        read can return several keys at once (fast typing, a pasted burst); the
        extras are queued instead of being silently dropped.
        """
        if self._queue:
            return self._queue.pop(0)
        ready, _, _ = select.select([self._input.fileno()], [], [], timeout)
        if not ready:
            return None
        presses = self._input.read_keys()
        keys = [_NAMED.get(p.key, p.key if isinstance(p.key, str) else "") for p in presses]
        if not keys:
            return None
        self._queue.extend(keys[1:])
        return keys[0]
