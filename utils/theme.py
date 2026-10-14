"""ChemLink's terminal design system — palette, Rich theme, glyphs, and the one
shared console every part of the CLI, TUI, logger, and progress display prints to.

Palette
-------
The brand follows the logo: cyan ``#0193C4`` and teal ``#026793``
(docs/images/brand/chemlink-logo.png), lifted slightly so text in those
colours stays readable on both dark and light terminals.
Green / amber / red are reserved for *state* — never decoration.

  ===========  =========================================================
  style name   used for
  ===========  =========================================================
  cl.brand     product name, section headers, the selected row
  cl.accent    secondary emphasis (sub-headers, active values)
  cl.ref       something the user could type or open: command, flag, path
  cl.muted     meta text: timestamps, units, separators, byte counts
  cl.hint      asides: keybindings, help text
  cl.ok/warn/err/info   state
  cl.num       numbers inside prose / stats
  ===========  =========================================================

Every module imports its console from here (``get_console()``) so a live
progress region, a log line and a ``print_ok`` never fight over the cursor.
Rich only allows one live display per console, and two consoles writing to
the same terminal tear each other's redraws — a single shared console avoids
both problems by construction.

Environment
-----------
  NO_COLOR=1             disable colour (https://no-color.org)
  CHEMLINK_PROGRESS=plain  force log-style progress lines even on a TTY
  CHEMLINK_PROGRESS=rich   force live progress bars even when not a TTY
"""
from __future__ import annotations

import os
import sys
import types
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, TypeVar

from rich.color import Color
from rich.console import Console
from rich.style import Style
from rich.text import Text
from rich.theme import Theme

T = TypeVar("T")


# ── Module identity ───────────────────────────────────────────────────────────
#
# Pipeline code imports these helpers both as ``utils.x`` (absolute, with the
# repo root on PYTHONPATH) and as ``chemlink.utils.x`` (relative, inside the
# package). Without care that loads two copies of each module — two consoles,
# two live displays, two registries. ``register_aliases`` makes whichever copy
# loads first answer to both names, and ``shared`` keeps process-wide state in
# one place even if a second copy slipped in before the alias existed.

_STATE_MODULE = "_chemlink_ui_state"


def register_aliases(module_name: str) -> None:
    module = sys.modules.get(module_name)
    if module is None:
        return
    short = module_name[len("chemlink."):] if module_name.startswith("chemlink.") else module_name
    for alias in (short, f"chemlink.{short}"):
        sys.modules.setdefault(alias, module)


def shared(key: str, factory: Callable[[], T]) -> T:
    """Process-wide singleton keyed by ``key`` — survives duplicate imports."""
    state = sys.modules.get(_STATE_MODULE)
    if state is None:
        state = types.ModuleType(_STATE_MODULE)
        sys.modules[_STATE_MODULE] = state
    if not hasattr(state, key):
        setattr(state, key, factory())
    return getattr(state, key)


register_aliases(__name__)


# ── Palette ───────────────────────────────────────────────────────────────────

LOGO_CYAN = "#0193c4"    # exact logo colours, used for the logo art itself
LOGO_TEAL = "#026793"
BRAND = "#1fa8db"        # logo cyan, lifted for text contrast
BRAND_2 = "#2f86b8"      # logo teal, lifted
CYAN = "#5cc8ff"
OK = "#3fb950"
WARN = "#d29922"
ERR = "#f85149"
INFO = "#58a6ff"
MUTED = "#8b949e"
FAINT = "#6e7681"
TRACK = "#3a3f4b"
BORDER = "#3d4450"

GRADIENT: Tuple[str, ...] = (CYAN, BRAND, BRAND_2)

THEME = Theme(
    {
        "cl.brand": f"bold {BRAND}",
        "cl.brand2": BRAND_2,
        "cl.accent": BRAND,
        "cl.title": "bold",
        "cl.ref": CYAN,
        "cl.path": CYAN,
        "cl.num": f"bold {CYAN}",
        "cl.muted": MUTED,
        "cl.faint": FAINT,
        "cl.hint": f"italic {MUTED}",
        "cl.ok": f"bold {OK}",
        "cl.warn": f"bold {WARN}",
        "cl.err": f"bold {ERR}",
        "cl.info": INFO,
        "cl.ok.soft": OK,
        "cl.warn.soft": WARN,
        "cl.err.soft": ERR,
        "cl.border": BORDER,
        "cl.track": TRACK,
        "cl.key": f"bold {BRAND}",
        "cl.sel": "bold on #252b3a",
        "cl.debug": FAINT,
        # Rich's own defaults, retuned to the palette (prompts, tracebacks…)
        "repr.number": f"bold {CYAN}",
        "repr.path": CYAN,
        "repr.filename": CYAN,
        "rule.line": BORDER,
        "progress.spinner": BRAND,
    }
)


# ── Glyphs ────────────────────────────────────────────────────────────────────

GLYPH = {
    "ok": "✓",
    "warn": "▲",
    "err": "✗",
    "info": "•",
    "debug": "·",
    "run": "▸",
    "skip": "○",
    "pending": "○",
    "arrow": "→",
    "brand": "◆",
    "pipeline": "◈",
    "dot": "·",
    "bullet": "•",
}

STATE_STYLE = {
    "ok": "cl.ok",
    "warn": "cl.warn",
    "err": "cl.err",
    "info": "cl.info",
    "debug": "cl.debug",
    "skip": "cl.muted",
    "run": "cl.brand",
}

SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
SPINNER_FPS = 12


def spinner_frame(elapsed: float) -> str:
    return SPINNER_FRAMES[int(elapsed * SPINNER_FPS) % len(SPINNER_FRAMES)]


# ── Consoles ──────────────────────────────────────────────────────────────────

def _make_console(stderr: bool = False) -> Console:
    no_color = bool(os.environ.get("NO_COLOR")) or os.environ.get("TERM") == "dumb"
    return Console(
        theme=THEME,
        stderr=stderr,
        highlight=False,
        no_color=no_color,
        soft_wrap=False,
    )


def get_console() -> Console:
    """The shared stdout console (themed). Live displays attach here."""
    return shared("console", _make_console)


def get_err_console() -> Console:
    """The shared stderr console (themed) for errors that must not hit stdout."""
    return shared("err_console", lambda: _make_console(stderr=True))


def set_color_enabled(enabled: bool) -> None:
    """Toggle colour on both shared consoles (``--no-color``)."""
    for console in (get_console(), get_err_console()):
        console.no_color = not enabled
        if not enabled:
            console._color_system = None  # noqa: SLF001 — Rich has no public setter


def progress_mode(console: Optional[Console] = None) -> str:
    """``"live"`` for animated bars, ``"plain"`` for log-style lines.

    Live bars need a real terminal; under SLURM, ``| tee`` or CI they would
    either print nothing until the end or spray escape codes into a log file,
    so those get timestamped plain lines instead.
    """
    forced = os.environ.get("CHEMLINK_PROGRESS", "").strip().lower()
    if forced in ("plain", "log", "off"):
        return "plain"
    if forced in ("rich", "live", "on"):
        return "live"
    console = console or get_console()
    if not console.is_terminal or console.is_dumb_terminal:
        return "plain"
    return "live"


# ── Colour helpers ────────────────────────────────────────────────────────────

def _lerp(a: Tuple[int, int, int], b: Tuple[int, int, int], t: float) -> Tuple[int, int, int]:
    return tuple(int(round(x + (y - x) * t)) for x, y in zip(a, b))  # type: ignore[return-value]


def gradient_colors(n: int, stops: Sequence[str] = GRADIENT) -> List[str]:
    """``n`` hex colours evenly interpolated across ``stops``."""
    if n <= 0:
        return []
    rgb = [Color.parse(s).get_truecolor() for s in stops]
    triples = [(c.red, c.green, c.blue) for c in rgb]
    if n == 1 or len(triples) == 1:
        r, g, b = triples[0]
        return [f"#{r:02x}{g:02x}{b:02x}"] * n
    out = []
    segments = len(triples) - 1
    for i in range(n):
        pos = i / (n - 1) * segments
        seg = min(int(pos), segments - 1)
        r, g, b = _lerp(triples[seg], triples[seg + 1], pos - seg)
        out.append(f"#{r:02x}{g:02x}{b:02x}")
    return out


def gradient_text(text: str, stops: Sequence[str] = GRADIENT, *, bold: bool = True) -> Text:
    """``text`` coloured character-by-character across ``stops`` (spaces skipped)."""
    visible = [ch for ch in text if not ch.isspace()]
    colors = iter(gradient_colors(len(visible), stops))
    out = Text()
    for ch in text:
        if ch.isspace():
            out.append(ch)
        else:
            out.append(ch, Style(color=next(colors), bold=bold))
    return out


# ── Formatting helpers shared by CLI, TUI and progress ────────────────────────

def format_duration(seconds: float) -> str:
    """``"1h 04m"`` / ``"4m 12s"`` / ``"52s"`` / ``"3.4s"`` (sub-10s keeps a decimal)."""
    seconds = max(0.0, float(seconds))
    if seconds < 10:
        return f"{seconds:.1f}s"
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def format_clock(seconds: float) -> str:
    """``"0:42"`` / ``"12:03"`` / ``"1:04:09"`` — compact elapsed/ETA clocks."""
    total = max(0, int(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def format_count(n: float) -> str:
    """``1,234`` for ints, ``12.5`` for floats, ``1.2M`` past a million."""
    if isinstance(n, float) and not n.is_integer():
        return f"{n:,.1f}"
    n = int(n)
    if abs(n) >= 10_000_000:
        return f"{n / 1_000_000:.1f}M"
    return f"{n:,}"


def plural(unit: str, count: float) -> str:
    if not unit or count == 1 or unit.endswith("s"):
        return unit
    if unit.endswith(("sh", "ch", "x")):
        return unit + "es"
    return unit + "s"


def truncate(text: str, width: int) -> str:
    if width <= 0:
        return ""
    if len(text) <= width:
        return text
    return text[: max(0, width - 1)] + "…"


def iter_lines(text: str) -> Iterable[str]:
    return (line.rstrip() for line in str(text).splitlines())
