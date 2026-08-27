"""Main loop: navigate the screen stack, and hand the whole terminal off to a command
when it runs.

Rich's Progress/Live (the braille bars, doctor tables, the dynamics resource card —
everything ChemLink already prints) assume they own the whole terminal while running,
and can't be cleanly nested inside this app's own Live frame without a compositor
framework doing that composition. So running a command pauses this app's frame
entirely, calls the handler in-process with full unmodified terminal control — the
exact same call ``main()`` makes for a plain CLI invocation — and resumes the frame
once it returns. Nothing about the handlers, ``console``/``err_console``, or
``utils/progress.py``/``utils/style.py`` needs to change for this to work.
"""
from __future__ import annotations

import os
import time
import traceback
from argparse import ArgumentParser
from typing import Optional

from rich.align import Align
from rich.console import Console
from rich.live import Live
from rich.text import Text

try:
    from ...utils.progress import spinner_frame
except ImportError:
    from utils.progress import spinner_frame  # type: ignore

from .keys import KeyReader
from .screens.list_screen import CommandListScreen, find_sub_action

_TICK = 0.03  # ~33fps while something is animating; idle waits block indefinitely
_LAUNCH_SPINNER_S = 0.45


def _display_command(breadcrumb: str, argv) -> str:
    return ("chemlink " + breadcrumb.replace(" › ", " ") + (" " + " ".join(argv) if argv else "")).strip()


def _play_launch_spinner(live: Live, display: str, height: int, duration: float = _LAUNCH_SPINNER_S) -> None:
    """Brief animated transition right before handoff, so even a very fast command
    doesn't feel like a hard cut from menu to output. Uses the same spinner glyph
    set as utils.progress.busy(), so every animated wait in the app looks alike."""
    start = time.monotonic()
    end = start + duration
    while time.monotonic() < end:
        frame = spinner_frame(time.monotonic() - start)
        text = Text.assemble((f"{frame} ", "bold cyan"), (f"launching {display}", "bold cyan"))
        live.update(Align(text, align="center", vertical="middle", height=height), refresh=True)
        time.sleep(0.06)


def _report_run(args, breadcrumb: str, argv, console: Console, err_console: Console) -> None:
    display = _display_command(breadcrumb, argv)
    console.print(f"\n[bold cyan]▶[/]  {display}\n")
    start = time.monotonic()
    try:
        code = args.handler(args) or 0
    except KeyboardInterrupt:
        err_console.print("\n[yellow]Interrupted.[/]")
        code = 130
    except Exception as exc:  # mirrors main()'s own top-level handling
        err_console.print(f"\n[bold red]Error:[/] {exc}")
        if os.environ.get("CHEMLINK_DEBUG"):
            traceback.print_exc()
        code = 1
    elapsed = time.monotonic() - start
    status = "[bold green]✓ done[/]" if code == 0 else f"[bold red]✗ exit {code}[/]"
    console.print(f"\n{status}  ·  {elapsed:.1f}s  ·  [dim]press any key to return to ChemLink[/]\n")


def run(parser: ArgumentParser, console: Console, err_console: Console, version: str = "") -> int:
    sub_action = find_sub_action(parser)
    if sub_action is None:
        return 1

    home = CommandListScreen(
        "ChemLink",
        sub_action,
        show_logo=True,
        subtitle=f"v{version}  ·  Molecular Simulation Toolkit" if version else "",
    )
    stack = [home]
    reader = KeyReader()
    live = Live(console=console, screen=True, auto_refresh=False)

    raw_cm = reader.raw()
    raw_cm.__enter__()
    live.start()
    try:
        while stack:
            top = stack[-1]
            animating = top.tick()
            height = console.size.height
            live.update(
                Align(top.render(height=height), align="center", vertical="middle", height=height),
                refresh=True,
            )
            key = reader.wait_key(timeout=_TICK if animating else None)
            if key is None:
                continue  # animation tick — no key to handle, just redraw next pass

            action = top.handle_key(key)
            if action is None:
                continue

            kind = action[0]
            if kind == "push":
                stack.append(action[1])
            elif kind == "pop":
                stack.pop()
            elif kind == "quit":
                stack.clear()
            elif kind == "run":
                _, args, breadcrumb, argv = action
                _play_launch_spinner(live, _display_command(breadcrumb, argv), console.size.height)
                live.stop()
                raw_cm.__exit__(None, None, None)
                _report_run(args, breadcrumb, argv, console, err_console)
                raw_cm = reader.raw()
                raw_cm.__enter__()
                reader.wait_key()
                live.start()
        return 0
    finally:
        live.stop()
        raw_cm.__exit__(None, None, None)
