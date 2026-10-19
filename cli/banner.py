"""ChemLink's logo for the terminal — shared by the plain-CLI banner
(``cli/main.py``) and the TUI's Home screen.

``render_brand`` draws the official logo (docs/images/brand/chemlink-logo.png)
from the block art in ``logo_art.py``; ``LOGO`` is the older braille mark,
kept for anything that still imports it."""

LOGO = """\

    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣠⣴⣾⠷⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
    ⠀⠀⠀⠀⠀⠀⣀⣤⣶⡿⠿⠛⠉⣀⣤⣶⣤⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
    ⠀⠀⣀⣤⣶⡿⠟⠋⢁⣠⣤⣶⣿⠿⠛⠉⠛⠿⢿⣶⣦⣄⡀⠀⠀⠀⠀⠀⠀⠀
    ⠲⡿⠟⠋⢁⣠⣴⣾⡿⠟⠋⢁⣠⣴⣾⣿⣷⣦⣄⡈⠙⠻⢿⣷⣦⣄⡀⠀⠀⠀
    ⠀⠀⢰⣾⡿⠟⠋⢁⣠⣴⣾⣿⣿⠿⠛⠉⠙⠻⢿⣿⣿⣶⣤⣈⠙⠻⠋⠀⢠⣤
    ⠀⠀⢸⣿⡇⠀⣿⣿⡿⠟⢻⣿⣷⠀⠀⠀⠀⠀⠀⠈⠙⠻⢿⣿⣿⡆⠀⠀⢸⣿
    ⠀⠀⢸⣿⡇⠀⣿⣿⡇⠀⢸⣿⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⣿⡇⠀⠀⢸⣿
    ⠀⠀⢸⣿⡇⠀⣿⣿⡇⠀⢸⣿⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⣿⡇⠀⠀⢸⣿
    ⠀⠀⢸⣿⡇⠀⣿⣿⡇⠀⢸⣿⣷⣤⣄⡀⠀⠀⠀⠀⠀⠀⢸⣿⣿⡇⠀⠀⢸⣿
    ⠀⠀⢸⣿⡇⠀⣿⣿⣷⣤⣀⠈⠙⠻⠿⣿⣷⣶⣤⣀⣀⣤⣾⣿⣿⠇⠀⠀⢸⣿
    ⠀⠀⠸⢿⣷⣦⣄⡉⠛⠿⣿⣿⣷⣦⣄⡀⣉⣽⣿⣿⣿⡿⠟⢋⣡⣄⠀⠀⠸⠿
    ⢾⣶⣤⣀⡈⠙⠻⢿⣷⣦⣄⡉⠛⠿⣿⣿⣿⠿⠛⢉⣠⣴⣾⡿⠟⠋⠁⠀⠀⠀
    ⠀⠈⠙⠻⠿⣷⣦⣄⡈⠙⠻⢿⣷⣦⣄⣉⣠⣴⣾⡿⠟⠋⠁⠀⠀⠀⠀⠀⠀⠀
    ⠀⠀⠀⠀⠀⠀⠉⠛⠿⣷⣦⣄⡈⠙⠻⠿⠟⠋⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠛⠿⣷⡦⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
\
"""

TAGLINE = "Molecular Simulation Toolkit"
SUBTAGLINE = "Docking · Molecular Dynamics · Free Energy · HPC"

# The signature under the logo: ChemLink is an Athalea project.
BYLINE = "by Athalea"
ATHALEA_GREEN = "#3cb5a5"


def render_byline():
    """``by Athalea``, centred under the logo: "by" muted, the name in
    Athalea green."""
    from rich.text import Text

    out = Text(justify="center")
    out.append("by ", style="cl.muted")
    out.append("Athalea", style=f"bold {ATHALEA_GREEN}")
    return out


def logo_lines() -> list:
    import textwrap
    return textwrap.dedent(LOGO).strip("\n").splitlines()


def render_logo(width_hint: int = 0):
    """The braille mark coloured top-to-bottom along the brand gradient."""
    from rich.style import Style
    from rich.text import Text

    try:
        from ..utils.theme import gradient_colors
    except ImportError:  # pragma: no cover
        from utils.theme import gradient_colors  # type: ignore

    lines = logo_lines()
    colors = gradient_colors(len(lines))
    out = Text(justify="center")
    for i, (line, color) in enumerate(zip(lines, colors)):
        if i:
            out.append("\n")
        out.append(line, Style(color=color))
    return out


# ── Official logo (block art) ─────────────────────────────────────────────────

def _art(name: str):
    try:
        from . import logo_art
    except ImportError:  # pragma: no cover
        import logo_art  # type: ignore
    return getattr(logo_art, name)


def brand_size(name: str):
    """(rows, cols) of one of LOGO_LARGE / LOGO_MEDIUM / MARK_LARGE / MARK_SMALL."""
    rows = _art(name)
    return len(rows), max(sum(len(t) for t, *_ in r) for r in rows)


def render_brand(name: str = "LOGO_LARGE", *, justify: str = "center"):
    """The logo as a renderable: the mark in the logo's cyan/teal, the wordmark
    in the terminal's own foreground colour (readable on dark and light themes)."""
    from rich.align import Align
    from rich.style import Style
    from rich.text import Text

    try:
        from ..utils.theme import LOGO_CYAN, LOGO_TEAL
    except ImportError:  # pragma: no cover
        from utils.theme import LOGO_CYAN, LOGO_TEAL  # type: ignore

    colors = {"c": LOGO_CYAN, "t": LOGO_TEAL, "w": "default", ".": None}
    rows = _art(name)
    width = max(sum(len(t) for t, *_ in r) for r in rows)
    out = Text(no_wrap=True, overflow="crop")
    for i, row in enumerate(rows):
        if i:
            out.append("\n")
        used = 0
        for text, fg, bg in row:
            out.append(text, Style(color=colors[fg], bgcolor=colors[bg]))
            used += len(text)
        out.append(" " * (width - used))
    # Align the block as a whole — per-line centring would shear the mark.
    return Align(out, align=justify) if justify in ("center", "right") else out


def pick_brand(width: int, height: int):
    """The largest logo that fits ``width`` × ``height`` cells, or ``None``."""
    for name in ("LOGO_LARGE", "LOGO_MEDIUM", "MARK_LARGE"):
        rows, cols = brand_size(name)
        if cols + 4 <= width and rows <= height:
            return name
    return None
