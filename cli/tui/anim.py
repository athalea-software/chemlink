"""Small stateful helpers behind the TUI's animations. Not a general animation
framework — just the three effects the TUI actually uses: a one-shot top-down reveal
(screen/logo intro), a value that eases toward a target (the sliding selection
marker), and a fixed-interval blink toggle (the text-edit cursor).
"""
from __future__ import annotations

import time
from typing import Optional


class Reveal:
    """One-shot top-down line reveal, driven by wall-clock time so it takes the same
    real duration regardless of how many lines there are."""

    def __init__(self, duration: float = 0.22):
        self.duration = duration
        self.done = False
        self._start: Optional[float] = None

    def visible(self, total: int) -> int:
        if self.done or total <= 0:
            return total
        now = time.monotonic()
        if self._start is None:
            self._start = now
        frac = (now - self._start) / self.duration
        if frac >= 1.0:
            self.done = True
            return total
        return max(1, int(total * frac))

    def tick(self) -> bool:
        return not self.done


class Ease:
    """A float that eases toward a target value each tick — drives the selection
    marker's row position so it visibly glides between rows instead of jumping."""

    def __init__(self, value: float = 0.0, speed: float = 16.0):
        self.value = value
        self.target = value
        self.speed = speed
        self._last: Optional[float] = None

    def set_target(self, target: float) -> None:
        self.target = target

    def snap(self, value: float) -> None:
        self.value = self.target = value
        self._last = None

    def tick(self) -> bool:
        now = time.monotonic()
        if self._last is None:
            self._last = now
        dt = now - self._last
        self._last = now
        diff = self.target - self.value
        if abs(diff) < 0.03:
            self.value = self.target
            return False
        self.value += diff * min(1.0, dt * self.speed)
        return True


class Blink:
    """Toggles .on every `interval` seconds while ticked."""

    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self.on = True
        self._last: Optional[float] = None

    def restart(self) -> None:
        """Show solid again and restart the interval — call when editing begins."""
        self.on = True
        self._last = None

    def tick(self) -> bool:
        now = time.monotonic()
        if self._last is None:
            self._last = now
        if now - self._last >= self.interval:
            self.on = not self.on
            self._last = now
        return True
