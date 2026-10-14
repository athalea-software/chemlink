"""ChemLink progress display — one live region, many trackers.

Every progress indicator in ChemLink (pipeline tracker, step bars, job bars,
GROMACS mdrun bars, checklist items, spinners) is a ``Tracker`` registered
with a single process-wide ``Display``. The display owns the only Rich
``Live`` on the shared console and stacks all active trackers into one
region, so they can nest freely (a step bar inside the pipeline tracker, a
GPU and a CPU docking bar side by side) — Rich itself allows just one live
display per console, which is why the old one-``Live``-per-bar design
crashed with ``LiveError`` the moment two bars overlapped.

Live mode (an interactive terminal)::

    ⠹ Ligand preparation     ━━━━━━━━━━━━━╸━━━━━━━━━━━━━   48%   58/120 ligands   3.4/s   0:17 · ETA 0:18
    ⠹ NVT equilibration      ━━━━━━━━━━━━━━━━━━━━╸━━━━━━   76%   0.08/0.10 ns   231 ns/day   0:41 · ETA 0:12

    ◈ MD pipeline › nvt      ━━━━━━━━━╸━━━━━━━━━━━━━━━━━   36%   5/14 steps   12:03 · ETA 21:30

When a tracker finishes it leaves a permanent one-line record above the live
region, so the scrollback reads as a log of what ran::

    ✓ Ligand preparation     120 ligands · 34.2s · 3.5/s
    ✗ Docking (Vina)         failed at 58/120 jobs · 17.0s — RuntimeError: …

Plain mode (SLURM output files, pipes, CI — see ``theme.progress_mode``)
prints timestamped lines instead of redrawing: a start line, a progress line
every 10 % (at most every 10 s, plus a heartbeat every 10 min for very long
phases), and the same final record.

Public helpers keep their historical signatures: ``step_bar_iter``,
``step_bar``, ``job_bar``, ``pipeline_bar``, ``busy``, ``checklist``,
``make_dynamics_progress`` / ``make_analysis_progress`` (a ``rich.Progress``
look-alike), plus ``track`` for new code.
"""
from __future__ import annotations

import random
import re
import threading
import time
from collections import deque
from datetime import datetime
from typing import (
    Any, Deque, Dict, Generator, Iterable, List, Optional, Tuple, TypeVar,
)

from rich.console import Console, ConsoleOptions, RenderResult
from rich.live import Live
from rich.style import Style
from rich.text import Text

try:
    from .theme import (
        ERR, GLYPH, GRADIENT, OK, TRACK, WARN, format_clock, format_count,
        format_duration, get_console, gradient_colors, plural, progress_mode,
        register_aliases, shared, spinner_frame, truncate,
    )
except ImportError:  # pragma: no cover — loaded as a top-level module
    from theme import (  # type: ignore
        ERR, GLYPH, GRADIENT, OK, TRACK, WARN, format_clock, format_count,
        format_duration, get_console, gradient_colors, plural, progress_mode,
        register_aliases, shared, spinner_frame, truncate,
    )

register_aliases(__name__)

T = TypeVar("T")

console = get_console()

CTRL_C_HINT = "  [cl.muted]Press[/] [cl.key]Ctrl+C[/] [cl.muted]at any time to abort.[/]\n"

__all__ = [
    "console", "CTRL_C_HINT", "Tracker", "track", "step_bar_iter", "step_bar", "job_bar",
    "pipeline_bar", "busy", "BusyIndicator", "Checklist", "ChecklistStep", "checklist",
    "make_dynamics_progress", "make_analysis_progress", "TaskProgress",
    "spinner_frame", "format_duration", "render_bar", "get_display",
]

_REFRESH_HZ = 12
_SPEED_WINDOW_S = 30.0
_ETA_MIN_ELAPSED_S = 2.0
_PLAIN_MIN_GAP_S = 10.0
_PLAIN_HEARTBEAT_S = 600.0
_DESC_MIN, _DESC_MAX = 14, 30
_FINAL_DESC_MIN = 22
_BAR_MIN, _BAR_MAX = 10, 40

# Historical ``colour=`` arguments map onto a small set of bar gradients so
# callers can still hint "this is the docking bar" without picking raw colours.
_GRADIENTS: Dict[str, Tuple[str, ...]] = {
    "default": GRADIENT,
    "green": ("#2ea043", OK, "#7ee787"),
    "magenta": ("#a979ff", "#d2a8ff", "#f778ba"),
    "blue": ("#5cc8ff", "#1fa8db", "#2f86b8"),
}
_COLOUR_ALIASES = {
    "cyan": "default", "bright_cyan": "default", "blue": "blue", "bright_blue": "blue",
    "green": "green", "bright_green": "green", "magenta": "magenta",
    "bright_magenta": "magenta", "purple": "magenta",
}


def _gradient_for(colour: Optional[str]) -> Tuple[str, ...]:
    return _GRADIENTS[_COLOUR_ALIASES.get((colour or "").lower(), "default")]


# ── Bar renderer ──────────────────────────────────────────────────────────────

_FILL, _HEAD, _TRACK_CH = "━", "╸", "━"
_PLAIN_FILL, _PLAIN_HEAD, _PLAIN_TRACK = "█", "▌", "░"


def render_bar(
    width: int,
    fraction: Optional[float],
    *,
    state: str = "running",
    elapsed: float = 0.0,
    gradient: Tuple[str, ...] = GRADIENT,
    color: bool = True,
) -> Text:
    """A ``width``-cell progress bar.

    ``fraction=None`` draws an indeterminate pulse that sweeps back and forth.
    Half-cell resolution (``╸``) keeps the head moving smoothly; the filled
    part is a gradient anchored to the full bar width, so colour encodes
    position rather than shifting as the bar grows. Without colour the bar
    switches to block glyphs so fill and track stay distinguishable.
    """
    width = max(1, width)
    out = Text(no_wrap=True, overflow="crop")
    fill_ch, head_ch, track_ch = (_FILL, _HEAD, _TRACK_CH) if color else (_PLAIN_FILL, _PLAIN_HEAD, _PLAIN_TRACK)
    track_style = Style(color=TRACK) if color else Style(dim=True)

    if fraction is None and state == "running":
        seg = max(3, width // 5)
        span = max(1, width - seg)
        pos = int((elapsed * 18) % (2 * span))
        start = pos if pos <= span else 2 * span - pos
        colors = gradient_colors(seg, gradient) if color else []
        for i in range(width):
            if start <= i < start + seg:
                out.append(fill_ch, Style(color=colors[i - start]) if color else Style(bold=True))
            else:
                out.append(track_ch, track_style)
        return out

    frac = 1.0 if fraction is None else min(max(fraction, 0.0), 1.0)
    halves = int(frac * width * 2)
    full, half = divmod(halves, 2)
    if state == "ok":
        fill_styles = [Style(color=OK)] * width
    elif state == "err":
        fill_styles = [Style(color=ERR)] * width
    elif state == "warn":
        fill_styles = [Style(color=WARN)] * width
    elif color:
        fill_styles = [Style(color=c) for c in gradient_colors(width, gradient)]
    else:
        fill_styles = [Style(bold=True)] * width
    for i in range(full):
        out.append(fill_ch, fill_styles[i])
    used = full
    if half and full < width:
        out.append(head_ch, fill_styles[full])
        used += 1
    if used < width:
        out.append(track_ch * (width - used), track_style)
    return out


# ── Tracker ───────────────────────────────────────────────────────────────────

_COUNT_RE = re.compile(r"^\s*([\d,]+)\s*/\s*([\d,]+)")


class Tracker:
    """One progress indicator. Thread-safe to update from worker threads.

    ``kind`` changes only presentation:
      ``step``/``job``  — count + throughput (``58/120 ligands · 3.4/s``)
      ``pipeline``      — the outer tracker; pinned to the bottom of the region
      ``md``            — GROMACS: simulated ns and ns/day instead of steps/s
      ``analysis``      — like step, for frame-based analyses
      ``check``         — checklist item: title, muted tool name, live detail
      ``busy``          — spinner + verb + elapsed, no bar
    """

    def __init__(
        self,
        description: str,
        total: Optional[float] = None,
        *,
        unit: str = "item",
        kind: str = "step",
        colour: Optional[str] = None,
        transient: bool = False,
        subtitle: str = "",
        display: Optional["Display"] = None,
    ) -> None:
        self.description = _clean_label(description)
        self.subtitle = subtitle
        self.total = total if (total is None or total > 0) else None
        self._declared_total = total
        self.completed: float = 0
        self.unit = unit
        self.kind = kind
        self.gradient = _gradient_for(colour)
        self.transient = transient
        self.postfix = ""
        self.stage = ""
        self.fields: Dict[str, Any] = {}
        self.state = "pending"  # pending | running | ok | err | warn
        self.note = ""
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self._samples: Deque[Tuple[float, float]] = deque()
        self._lock = threading.Lock()
        self._display = display or get_display()
        self._plain_last_emit = 0.0
        self._plain_last_bucket = -1

    # ── lifecycle ──────────────────────────────────────────────────────────
    def start(self) -> "Tracker":
        if self.state != "pending":
            return self
        now = time.monotonic()
        self.start_time = now
        self._samples.append((now, 0.0))
        self.state = "running"
        self._display.add(self)
        return self

    def finish(self, state: str = "ok", note: Optional[str] = None) -> None:
        with self._lock:
            if self.state not in ("running", "pending"):
                return
            if self.state == "pending":
                self.start_time = time.monotonic()
            self.state = state
            if note is not None:
                self.note = note
            self.end_time = time.monotonic()
            if state == "ok" and self.total and self.kind != "pipeline":
                self.completed = max(self.completed, self.total)
        self._display.remove(self)

    def fail(self, reason: Any = "") -> None:
        self.finish("err", _short_reason(reason))

    def cancel(self, note: str = "cancelled") -> None:
        self.finish("warn", note)

    def __enter__(self) -> "Tracker":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is None:
            self.finish("ok")
        elif issubclass(exc_type, KeyboardInterrupt):
            self.cancel("interrupted")
        else:
            self.fail(exc)
        return False

    # ── updates ────────────────────────────────────────────────────────────
    def advance(self, n: float = 1) -> None:
        Tracker.update(self, advance=n)

    def update(
        self,
        completed: Optional[float] = None,
        *,
        total: Optional[float] = None,
        advance: Optional[float] = None,
        description: Optional[str] = None,
        postfix: Optional[str] = None,
        **fields: Any,
    ) -> None:
        now = time.monotonic()
        with self._lock:
            if total is not None:
                self.total = total if total > 0 else None
            if completed is not None:
                self.completed = completed
            if advance:
                self.completed += advance
            if description is not None:
                self.description = _clean_label(description)
            if postfix is not None:
                self.postfix = postfix
            if fields:
                self.fields.update(fields)
            if advance or completed is not None:
                self._samples.append((now, float(self.completed)))
                while len(self._samples) > 2 and now - self._samples[0][0] > _SPEED_WINDOW_S:
                    self._samples.popleft()
        if self._display.plain and self.state == "running":
            self._maybe_emit_plain(now)

    def reset(self, total: Optional[float] = None) -> None:
        now = time.monotonic()
        with self._lock:
            self.completed = 0
            if total is not None:
                self.total = total if total > 0 else None
            self.start_time = now
            self._samples = deque([(now, 0.0)])

    # ── metrics ────────────────────────────────────────────────────────────
    @property
    def elapsed(self) -> float:
        if self.start_time is None:
            return 0.0
        return (self.end_time or time.monotonic()) - self.start_time

    @property
    def fraction(self) -> Optional[float]:
        if not self.total:
            return None
        return min(1.0, self.completed / self.total)

    @property
    def speed(self) -> Optional[float]:
        """Items per second over the recent window (falls back to the run
        average when the window is too short to be meaningful)."""
        samples = list(self._samples)
        if len(samples) >= 2:
            (t0, c0), (t1, c1) = samples[0], samples[-1]
            if t1 - t0 >= 0.5 and c1 > c0:
                return (c1 - c0) / (t1 - t0)
        if self.completed > 0 and self.elapsed > 0.5:
            return self.completed / self.elapsed
        return None

    @property
    def eta(self) -> Optional[float]:
        if self.state != "running":
            return None
        explicit = self.fields.get("eta_s")
        if explicit and explicit > 0:
            return float(explicit)
        if not self.total or self.completed <= 0 or self.elapsed < _ETA_MIN_ELAPSED_S:
            return None
        speed = self.speed
        if not speed:
            return None
        return max(0.0, (self.total - self.completed) / speed)

    # ── text cells ─────────────────────────────────────────────────────────
    def label(self) -> str:
        if self.kind == "pipeline" and self.stage:
            return f"{self.description} › {self.stage}"
        return self.description

    def count_text(self) -> str:
        if self.kind == "md" and self.fields.get("dt_ps") and self.total:
            dt = float(self.fields["dt_ps"])
            done_ns = self.completed * dt / 1000.0
            total_ns = self.total * dt / 1000.0
            prec = 3 if total_ns < 1 else 2
            return f"{done_ns:.{prec}f}/{total_ns:.{prec}f} ns"
        if self.kind in ("busy",):
            return ""
        if self.kind == "check" and not self.total:
            return ""
        unit = plural(self.unit, self.total or self.completed) if self.unit else ""
        if self.total:
            return f"{format_count(self.completed)}/{format_count(self.total)} {unit}".rstrip()
        if self.completed:
            return f"{format_count(self.completed)} {unit}".rstrip()
        return ""

    def rate_text(self) -> str:
        if self.kind == "md":
            ns_day = self.fields.get("ns_day") or 0.0
            return f"{ns_day:,.1f} ns/day" if ns_day > 0 else ""
        if self.kind in ("pipeline", "busy", "check"):
            return ""
        speed = self.speed
        if not speed:
            return ""
        if speed >= 1:
            return f"{speed:,.1f}/s"
        return f"{1 / speed:,.1f}s/{self.unit or 'it'}" if 1 / speed < 600 else f"{1 / speed / 60:,.0f}m/{self.unit or 'it'}"

    def time_text(self) -> str:
        text = format_clock(self.elapsed)
        eta = self.eta
        if eta is not None:
            text += f" · ETA {format_clock(eta)}"
        return text

    def pct_text(self) -> str:
        frac = self.fraction
        return f"{frac * 100:3.0f}%" if frac is not None else ""

    # ── rendering ──────────────────────────────────────────────────────────
    def glyph(self) -> Text:
        if self.state == "running":
            style = "cl.accent" if self.kind != "pipeline" else "cl.brand"
            glyph = GLYPH["pipeline"] if self.kind == "pipeline" else spinner_frame(self.elapsed)
            return Text(glyph, style=style)
        key = {"ok": "ok", "err": "err", "warn": "warn"}.get(self.state, "pending")
        return Text(GLYPH[key], style={"ok": "cl.ok", "err": "cl.err", "warn": "cl.warn"}.get(key, "cl.muted"))

    def show_bar(self) -> bool:
        return self.kind not in ("busy",) and not (self.kind == "check" and not self.total)

    def final_summary(self) -> str:
        """The muted tail of the permanent line left behind once finished."""
        parts: List[str] = []
        count = self.count_text()
        if self.state == "ok":
            if self.kind == "check":
                if self.postfix:
                    parts.append(self.postfix)
            elif self.kind == "pipeline" and self.total:
                parts.append(f"{format_count(self.completed)}/{format_count(self.total)} {plural(self.unit, self.total)}")
            elif self.kind == "md":
                if self.total:
                    parts.append(f"{format_count(self.total)} steps")
                    if self.fields.get("dt_ps"):
                        parts.append(f"{self.total * float(self.fields['dt_ps']) / 1000:.3g} ns")
            elif count:
                total = self.total or self.completed
                parts.append(f"{format_count(total)} {plural(self.unit, total)}" if self.unit else format_count(total))
        else:
            verb = "failed" if self.state == "err" else (self.note or "stopped")
            if count and self.kind != "busy":
                parts.append(f"{verb} at {count}")
            else:
                parts.append(verb)
        parts.append(format_duration(self.elapsed))
        if self.state == "ok":
            rate = self.rate_text()
            if self.kind == "md":
                ns_day = self.fields.get("ns_day") or 0.0
                rate = f"{ns_day:,.1f} ns/day" if ns_day > 0 else ""
            elif self.speed and self.elapsed >= 1.0 and self.kind in ("step", "job", "analysis"):
                speed = self.completed / self.elapsed if self.elapsed else 0
                rate = f"{speed:,.1f}/s" if speed >= 1 else (f"{self.elapsed / self.completed:,.1f}s/{self.unit}" if self.completed else "")
            if rate:
                parts.append(rate)
        text = " · ".join(p for p in parts if p)
        if self.state == "err" and self.note:
            text += f" — {self.note}"
        elif self.state == "ok" and self.note:
            text += f" · {self.note}"
        return text

    def final_line(self, desc_width: int) -> Text:
        line = Text("  ", no_wrap=False, overflow="fold")
        line.append_text(self.glyph())
        line.append(" ")
        title_style = "bold" if self.kind == "pipeline" else ""
        line.append(self.description, style=title_style)
        used = len(self.description)
        if self.subtitle:
            line.append(f"  {self.subtitle}", style="cl.faint")
            used += len(self.subtitle) + 2
        line.append(" " * max(0, desc_width - used) + "  ")
        style = {"err": "cl.err.soft", "warn": "cl.warn.soft"}.get(self.state, "cl.muted")
        line.append(self.final_summary(), style=style)
        return line

    # ── plain mode ─────────────────────────────────────────────────────────
    def plain_start_line(self) -> Text:
        head = Text(f"{datetime.now():%H:%M:%S}  ", style="cl.faint")
        head.append(f"{GLYPH['run']} ", style="cl.brand")
        head.append(self.description, style="bold")
        extras = [self.subtitle] if self.subtitle else []
        if self.total:
            extras.append(f"{format_count(self.total)} {plural(self.unit, self.total)}" if self.kind != "md" else f"{format_count(self.total)} steps")
        if extras:
            head.append("  " + " · ".join(extras), style="cl.muted")
        return head

    def plain_progress_line(self) -> Text:
        line = Text(f"{datetime.now():%H:%M:%S}  ", style="cl.faint")
        line.append("… ", style="cl.accent")
        line.append(self.label())
        bits = [b for b in (self.count_text(), f"({self.pct_text().strip()})" if self.fraction is not None else "", self.rate_text(), self.time_text(), self.postfix) if b]
        line.append("  " + " · ".join(bits), style="cl.muted")
        return line

    def plain_final_line(self) -> Text:
        line = Text(f"{datetime.now():%H:%M:%S}  ", style="cl.faint")
        line.append_text(self.glyph())
        line.append(" ")
        line.append(self.description, style="bold" if self.kind == "pipeline" else "")
        if self.subtitle:
            line.append(f"  {self.subtitle}", style="cl.faint")
        style = {"err": "cl.err.soft", "warn": "cl.warn.soft"}.get(self.state, "cl.muted")
        line.append("  " + self.final_summary(), style=style)
        return line

    def _maybe_emit_plain(self, now: float) -> None:
        if self.kind in ("busy",) or self.transient:
            return
        frac = self.fraction
        bucket = int(frac * 10) if frac is not None else -1
        since = now - (self._plain_last_emit or (self.start_time or now))
        due = (bucket > self._plain_last_bucket and since >= _PLAIN_MIN_GAP_S and frac is not None and frac < 1.0)
        if due or since >= _PLAIN_HEARTBEAT_S:
            self._plain_last_bucket = max(bucket, self._plain_last_bucket)
            self._plain_last_emit = now
            self._display.console.print(self.plain_progress_line(), soft_wrap=True)


def _clean_label(description: str) -> str:
    """Strip Rich markup and the decorative glyphs older callers baked into
    their descriptions (``"[bold magenta]◆ NVT[/]"`` → ``"NVT"``)."""
    text = str(description or "")
    if "[" in text:
        try:
            text = Text.from_markup(text).plain
        except Exception:
            pass
    return text.strip().lstrip("◆◉◈▸•·").strip()


def _short_reason(reason: Any, limit: int = 120) -> str:
    if isinstance(reason, BaseException):
        msg = str(reason).strip().splitlines()[0] if str(reason).strip() else ""
        text = f"{type(reason).__name__}: {msg}" if msg else type(reason).__name__
    else:
        text = str(reason or "").strip().splitlines()[0] if str(reason or "").strip() else ""
    return truncate(text, limit)


# ── Display ───────────────────────────────────────────────────────────────────

class _Region:
    """The renderable handed to ``Live``: lays out all active trackers."""

    def __init__(self, display: "Display") -> None:
        self.display = display

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        try:
            trackers = self.display.snapshot()
            width = options.max_width
            inner = [t for t in trackers if t.kind != "pipeline"]
            outer = [t for t in trackers if t.kind == "pipeline"]
            color = not (console.no_color or console.color_system is None)
            for line in _layout_lines(inner, width, color):
                yield line
            if inner and outer:
                yield Text("")
            for line in _layout_lines(outer, width, color):
                yield line
        except Exception as exc:  # never let a render error kill the live thread
            yield Text(f"  progress display error: {exc}", style="cl.err.soft")


def _layout_lines(trackers: List[Tracker], width: int, color: bool) -> List[Text]:
    if not trackers:
        return []
    desc_w = min(_DESC_MAX, max(_DESC_MIN, *(len(t.label()) + (len(t.subtitle) + 2 if t.subtitle else 0) for t in trackers)))
    cells = [
        {"pct": t.pct_text(), "count": t.count_text(), "rate": t.rate_text(), "time": t.time_text(), "post": t.postfix if t.kind != "check" else ""}
        for t in trackers
    ]
    widths = {k: max(len(c[k]) for c in cells) for k in ("pct", "count", "rate", "time")}
    show = {"pct": True, "count": True, "rate": True, "time": True}

    def fixed() -> int:
        total = 2 + 2 + desc_w + 2
        for key in ("pct", "count", "rate", "time"):
            if show[key] and widths[key]:
                total += widths[key] + 3
        return total

    # Responsive: drop the least important columns first on narrow terminals.
    for key in ("rate", "count", "time", "pct"):
        if width - fixed() >= _BAR_MIN:
            break
        show[key] = False
    while width - fixed() < _BAR_MIN and desc_w > 8:
        desc_w -= 1
    bar_w = max(_BAR_MIN, min(_BAR_MAX, width - fixed()))
    any_bar = any(t.show_bar() for t in trackers)

    lines = []
    for t, c in zip(trackers, cells):
        line = Text("  ", no_wrap=True, overflow="ellipsis")
        line.append_text(t.glyph())
        line.append(" ")
        label = t.label()
        title_style = "bold" if t.kind == "pipeline" else ""
        if t.subtitle:
            room = max(0, desc_w - len(label) - 2)
            sub = truncate(t.subtitle, room)
            label = truncate(label, desc_w)
            line.append(label, style=title_style)
            if sub:
                line.append("  " + sub, style="cl.faint")
            line.append(" " * max(0, desc_w - len(label) - (len(sub) + 2 if sub else 0)))
        else:
            line.append(f"{truncate(label, desc_w):<{desc_w}}", style=title_style)
        line.append("  ")
        if t.kind == "check" and not t.total:
            # A checklist item without a count: its detail sits where the bar would.
            line.append(t.postfix, style="cl.muted")
            line.append(f"   {c['time']}", style="cl.faint")
            lines.append(line)
            continue
        if t.show_bar():
            line.append_text(render_bar(bar_w, t.fraction, elapsed=t.elapsed, gradient=t.gradient, color=color))
        elif any_bar:
            line.append(" " * bar_w)
        if show["pct"] and widths["pct"]:
            line.append(f"  {c['pct']:>{widths['pct']}}", style="bold")
        if show["count"] and widths["count"]:
            line.append(f"   {c['count']:<{widths['count']}}", style="")
        if show["rate"] and widths["rate"]:
            line.append(f"   {c['rate']:>{widths['rate']}}", style="cl.accent")
        if show["time"] and widths["time"]:
            line.append(f"   {c['time']:<{widths['time']}}", style="cl.muted")
        if c["post"]:
            line.append(f"  {c['post']}", style="cl.faint")
        lines.append(line)
    return lines


class Display:
    """Owns the single live region on the shared console."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._trackers: List[Tracker] = []
        self._live: Optional[Live] = None
        self._suspended = 0

    @property
    def console(self) -> Console:
        return get_console()

    @property
    def plain(self) -> bool:
        return progress_mode(self.console) == "plain"

    def snapshot(self) -> List[Tracker]:
        with self._lock:
            return list(self._trackers)

    @property
    def active(self) -> bool:
        return self._live is not None

    def _desc_width(self) -> int:
        # Completion lines share one minimum width so consecutive records align.
        widths = [len(t.description) for t in self._trackers]
        return min(_DESC_MAX, max(_FINAL_DESC_MIN, *widths)) if widths else _FINAL_DESC_MIN

    # Lock discipline: never print or start/stop the Live while holding
    # ``_lock`` — the live refresh thread holds Rich's own lock while it
    # renders, and rendering takes ``_lock`` (via ``snapshot``).

    def _new_live(self) -> Live:
        return Live(
            _Region(self),
            console=self.console,
            refresh_per_second=_REFRESH_HZ,
            transient=True,
            redirect_stdout=True,
            redirect_stderr=True,
        )

    def add(self, tracker: Tracker) -> None:
        plain = self.plain
        live_to_start: Optional[Live] = None
        with self._lock:
            self._trackers.append(tracker)
            if not plain and self._live is None and not self._suspended:
                self._live = live_to_start = self._new_live()
        if plain:
            if not tracker.transient:
                self.console.print(tracker.plain_start_line(), soft_wrap=True)
        elif live_to_start is not None:
            live_to_start.start()

    def remove(self, tracker: Tracker) -> None:
        plain = self.plain
        live_to_stop: Optional[Live] = None
        with self._lock:
            desc_w = max(self._desc_width(), min(_DESC_MAX, len(tracker.description) + (len(tracker.subtitle) + 2 if tracker.subtitle else 0)))
            if tracker in self._trackers:
                self._trackers.remove(tracker)
            if not plain and not self._trackers and self._live is not None:
                live_to_stop, self._live = self._live, None
        if not (tracker.transient and tracker.state == "ok"):
            # While the live region is up this lands above it, as a log line.
            line = tracker.plain_final_line() if plain else tracker.final_line(desc_w)
            self.console.print(line, soft_wrap=True)
        if live_to_stop is not None:
            live_to_stop.stop()

    def suspend(self) -> "_Suspend":
        """Temporarily take the live region down (e.g. to hand the terminal to
        an interactive prompt); it comes back on exit if trackers remain."""
        return _Suspend(self)


class _Suspend:
    def __init__(self, display: Display) -> None:
        self.display = display

    def __enter__(self) -> None:
        d = self.display
        with d._lock:
            d._suspended += 1
            live, d._live = d._live, None
        if live is not None:
            live.stop()

    def __exit__(self, *exc: Any) -> None:
        d = self.display
        live: Optional[Live] = None
        with d._lock:
            d._suspended -= 1
            if not d._suspended and d._trackers and not d.plain and d._live is None:
                d._live = live = d._new_live()
        if live is not None:
            live.start()


def get_display() -> Display:
    return shared("display", Display)


# ── Public helpers ────────────────────────────────────────────────────────────

def track(
    description: str,
    total: Optional[float] = None,
    *,
    unit: str = "item",
    kind: str = "step",
    colour: Optional[str] = None,
    transient: bool = False,
    subtitle: str = "",
) -> Tracker:
    """A new tracker; use as a context manager (``with track(...) as t:``)."""
    return Tracker(description, total, unit=unit, kind=kind, colour=colour,
                   transient=transient, subtitle=subtitle)


def step_bar_iter(
    iterable: Iterable[T],
    desc: str,
    *,
    total: Optional[int] = None,
    unit: str = "item",
    colour: str = "cyan",
) -> Generator[T, None, None]:
    """Wrap an iterable with a progress bar.

    Sequences are counted automatically; pass ``total=`` for generators such
    as ``as_completed(futures)``.
    """
    if total is None:
        try:
            total = len(iterable)  # type: ignore[arg-type]
            actual: Iterable[T] = iterable
        except TypeError:
            actual = list(iterable)
            total = len(actual)  # type: ignore[arg-type]
    else:
        actual = iterable

    tracker = Tracker(desc, total, unit=unit, kind="step", colour=colour).start()
    try:
        for item in actual:
            yield item
            tracker.advance(1)
    except GeneratorExit:
        # The consumer stopped iterating — usually an exception in its loop body.
        if tracker.total and tracker.completed < tracker.total:
            tracker.cancel("stopped")
        else:
            tracker.finish("ok")
        raise
    except KeyboardInterrupt:
        tracker.cancel("interrupted")
        raise
    except BaseException as exc:
        tracker.fail(exc)
        raise
    else:
        tracker.finish("ok")


class _JobBar(Tracker):
    """``job_bar`` handle — tqdm-like ``update(n)`` / ``set_postfix_str``."""

    def update(self, n: float = 1, **kwargs: Any) -> None:  # type: ignore[override]
        if kwargs:
            super().update(advance=n if n else None, **kwargs)
        else:
            super().update(advance=n)

    def set_postfix_str(self, s: str) -> None:
        super().update(postfix=s)


def job_bar(desc: str, total: int, *, unit: str = "job", colour: str = "bright_green") -> _JobBar:
    return _JobBar(desc, total, unit=unit, kind="job", colour=colour)


class _StepBar(_JobBar):
    pass


def step_bar(desc: str, total: int, *, unit: str = "item", colour: str = "cyan") -> _StepBar:
    """Context-managed sub-step bar: ``with step_bar("Solvation", 3) as p: p.update()``."""
    return _StepBar(desc, total, unit=unit, kind="step", colour=colour)


class _PipelineBar:
    """The outer pipeline tracker — pinned to the bottom of the live region.

    Starts immediately (no ``with`` needed). ``pause``/``resume`` are kept for
    old callers but are no longer necessary: nested bars compose natively.
    """

    def __init__(self, total: int, desc: str) -> None:
        self._tracker = Tracker(desc, total, unit="step", kind="pipeline", colour="blue").start()

    @property
    def tracker(self) -> Tracker:
        return self._tracker

    def update(self, n: int = 1) -> None:
        self._tracker.advance(n)

    def set_stage(self, name: str) -> None:
        self._tracker.stage = name

    def set_postfix_str(self, s: str) -> None:
        self._tracker.stage = s

    def fail(self, reason: Any = "") -> None:
        self._tracker.fail(reason)

    def close(self) -> None:
        t = self._tracker
        if t.state != "running":
            return
        if t.total and t.completed < t.total:
            t.cancel("stopped")
        else:
            t.finish("ok")

    def pause(self) -> None:
        pass

    def resume(self) -> None:
        pass

    def __enter__(self) -> "_PipelineBar":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None and self._tracker.state == "running":
            if issubclass(exc_type, KeyboardInterrupt):
                self._tracker.cancel("interrupted")
            else:
                self._tracker.fail(exc)
        self.close()


def pipeline_bar(total: int, *, desc: str = "MD pipeline") -> _PipelineBar:
    return _PipelineBar(total, desc)


# ── Busy indicator ────────────────────────────────────────────────────────────

_BUSY_VERBS = (
    "Equilibrating", "Solvating", "Titrating", "Diffusing", "Annealing",
    "Nucleating", "Crystallizing", "Calibrating", "Percolating", "Folding",
)


class BusyIndicator:
    """Spinner + verb + elapsed time for one opaque blocking call.

    Use a real tracker when there are discrete steps to count; this is for
    the in-between waits (a subprocess, a network fetch) where the only
    honest report is "still working". Leaves no trace once done.
    """

    def __init__(self, verb: Optional[str] = None, *, style: str = "cyan") -> None:
        self.verb = verb or random.choice(_BUSY_VERBS)
        self._tracker = Tracker(f"{self.verb}…", None, kind="busy", transient=True)

    def update(self, verb: str) -> None:
        self._tracker.update(description=f"{verb}…")

    def __enter__(self) -> "BusyIndicator":
        self._tracker.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self._tracker.finish("ok")
        elif issubclass(exc_type, KeyboardInterrupt):
            self._tracker.cancel("interrupted")
        else:
            self._tracker.fail(exc)


def busy(verb: Optional[str] = None, *, style: str = "cyan") -> BusyIndicator:
    return BusyIndicator(verb, style=style)


# ── Checklist ─────────────────────────────────────────────────────────────────

class ChecklistStep(Tracker):
    """One named phase. ``update("12 / 40 ligands docked")`` — a leading
    ``N / M`` count turns the row into a progress bar automatically."""

    def __init__(self, owner: "Checklist", title: str, subtitle: str) -> None:
        super().__init__(title, None, kind="check", subtitle=subtitle, unit="")
        self._owner = owner
        self.title = title

    @property
    def detail(self) -> str:
        return self.postfix

    def update(self, detail: Optional[str] = None, **kwargs: Any) -> None:  # type: ignore[override]
        if detail is None:
            super().update(**kwargs)
            return
        match = _COUNT_RE.match(detail)
        if match:
            done = int(match.group(1).replace(",", ""))
            total = int(match.group(2).replace(",", ""))
            rest = detail[match.end():].strip()
            unit = rest.split()[0] if rest else ""
            self.unit = unit.rstrip("s") if unit else ""
            super().update(completed=done, total=total, postfix=detail, **kwargs)
        else:
            super().update(postfix=detail, **kwargs)

    def finish(self, detail: Optional[str] = None, *, ok: bool = True) -> None:  # type: ignore[override]
        if self.state not in ("running", "pending"):
            return
        if detail is not None:
            self.postfix = detail
        if ok:
            super().finish("ok")
        else:
            super().finish("err", _short_reason(detail or "failed"))

    def __enter__(self) -> "ChecklistStep":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:  # type: ignore[override]
        if exc is not None:
            if isinstance(exc, KeyboardInterrupt):
                self.cancel("interrupted")
            else:
                self.finish(str(exc), ok=False)
            return False
        self.finish()
        return False


class Checklist:
    """A sequence of named phases, each animating while it runs and leaving a
    permanent ``✓``/``✗`` line when done.

    Usage::

        with checklist() as cl:
            with cl.step("Detecting active site", "fpocket"):
                ...
            with cl.step("Running GPU search", "AutoDock-GPU") as item:
                for done in range(total):
                    item.update(f"{done:,} / {total:,} ligands docked")
            cl.complete(f"Campaign complete in {format_duration(elapsed)}")
    """

    def __init__(self) -> None:
        self._items: List[ChecklistStep] = []

    def step(self, title: str, subtitle: str = "") -> ChecklistStep:
        item = ChecklistStep(self, title, subtitle)
        self._items.append(item)
        item.start()
        return item

    def complete(self, message: str) -> None:
        self._close_running("ok")
        con = get_console()
        line = Text("  ")
        if get_display().plain:
            line = Text(f"{datetime.now():%H:%M:%S}  ", style="cl.faint")
        line.append(GLYPH["ok"], style="cl.ok")
        line.append(" ")
        line.append(message, style="cl.ok")
        con.print(line)

    def _close_running(self, state: str, note: str = "") -> None:
        for item in self._items:
            if item.state == "running":
                if state == "ok":
                    item.finish()
                else:
                    Tracker.finish(item, state, note)

    def __enter__(self) -> "Checklist":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self._close_running("ok")
        elif issubclass(exc_type, KeyboardInterrupt):
            self._close_running("warn", "interrupted")
        else:
            self._close_running("err", _short_reason(exc))


def checklist() -> Checklist:
    return Checklist()


# ── rich.Progress look-alike (GROMACS mdrun, gmx_MMPBSA frames) ───────────────

class TaskProgress:
    """Drop-in for the subset of ``rich.progress.Progress`` the dynamics code
    uses (``start/stop``, ``add_task``, ``advance``, ``update``, ``reset``,
    ``console``), backed by display trackers so it nests with everything
    else instead of opening a competing ``Live``."""

    def __init__(self, *, kind: str, unit: str, colour: str) -> None:
        self._kind = kind
        self._unit = unit
        self._colour = colour
        self._tasks: Dict[int, Tracker] = {}
        self._started = False
        self._next = 0

    @property
    def console(self) -> Console:
        return get_console()

    def start(self) -> None:
        self._started = True
        for tracker in self._tasks.values():
            tracker.start()

    def stop(self) -> None:
        for tracker in self._tasks.values():
            if tracker.state == "running":
                if tracker.total and tracker.completed < tracker.total and not tracker.fields.get("_done"):
                    tracker.cancel("stopped")
                else:
                    tracker.finish("ok")
        self._started = False

    def __enter__(self) -> "TaskProgress":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            for tracker in self._tasks.values():
                if tracker.state == "running":
                    if issubclass(exc_type, KeyboardInterrupt):
                        tracker.cancel("interrupted")
                    else:
                        tracker.fail(exc)
        self.stop()

    def add_task(self, description: str, total: Optional[float] = None, **fields: Any) -> int:
        task_id = self._next
        self._next += 1
        tracker = Tracker(description, total, unit=self._unit, kind=self._kind, colour=self._colour)
        tracker.fields.update(fields)
        self._tasks[task_id] = tracker
        if self._started:
            tracker.start()
        return task_id

    def tracker(self, task_id: int) -> Tracker:
        return self._tasks[task_id]

    def advance(self, task_id: int, advance: float = 1) -> None:
        self._tasks[task_id].update(advance=advance)

    def update(self, task_id: int, *, total: Optional[float] = None, completed: Optional[float] = None,
               advance: Optional[float] = None, description: Optional[str] = None, **fields: Any) -> None:
        self._tasks[task_id].update(completed=completed, total=total, advance=advance,
                                    description=description, **fields)

    def reset(self, task_id: int, *, total: Optional[float] = None, **_: Any) -> None:
        self._tasks[task_id].reset(total=total)


def make_dynamics_progress(bar_width: int = 36) -> TaskProgress:
    """GROMACS mdrun: simulated ns, ns/day and ETA (fields ``ns_day``, ``eta_s``, ``dt_ps``)."""
    return TaskProgress(kind="md", unit="step", colour="cyan")


def make_analysis_progress(bar_width: int = 32) -> TaskProgress:
    """Frame-based analyses (gmx_MMPBSA): frames, frames/s and ETA."""
    return TaskProgress(kind="analysis", unit="frame", colour="magenta")
