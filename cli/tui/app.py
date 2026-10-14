"""Main loop: navigate the screen stack, and hand the whole terminal off to a command
when it runs.

The TUI draws on the terminal's alternate screen with one Rich ``Live``. Running a
command stops that frame, restores cooked mode, and calls the handler in-process
with full terminal control — through the same ``run_command`` frame the plain CLI
uses, so a run looks identical whether it was launched here or from a shell. The
command's output stays in the normal scrollback after you leave the TUI.
"""
from __future__ import annotations

import shlex
import time
from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Callable, List, Optional

from rich.align import Align
from rich.console import Console
from rich.live import Live
from rich.text import Text

try:
    from ...utils.theme import GLYPH, format_duration, spinner_frame
except ImportError:  # pragma: no cover
    from utils.theme import GLYPH, format_duration, spinner_frame  # type: ignore

from .keys import KeyReader
from .screens.list_screen import CommandListScreen, find_sub_action
from .sysinfo import SystemInfo
from .widgets import keycaps

_TICK = 1 / 30          # while something animates
_IDLE_TICK = 0.25       # otherwise: cheap polling for resizes / background updates
_LAUNCH_SPINNER_S = 0.4


@dataclass
class AppContext:
    version: str
    sysinfo: SystemInfo


def _display_command(breadcrumb: str, argv) -> str:
    words = ["chemlink", *[p for p in breadcrumb.split(" › ") if p], *argv]
    return shlex.join(words)


def _play_launch(live: Live, display: str, height: int) -> None:
    """A short transition so even an instant command doesn't feel like a hard cut."""
    start = time.monotonic()
    while (elapsed := time.monotonic() - start) < _LAUNCH_SPINNER_S:
        text = Text(justify="center")
        text.append(f"{spinner_frame(elapsed)} ", style="cl.brand")
        text.append("launching ", style="cl.muted")
        text.append(display, style="bold")
        live.update(Align(text, align="center", vertical="middle", height=height), refresh=True)
        time.sleep(1 / 30)


def _fallback_runner(console: Console, err_console: Console):
    """Used when the caller didn't pass ``cli.main.run_command`` (tests, old callers)."""
    def run(args, command_line: str) -> int:
        console.print(f"\n[cl.brand]{GLYPH['brand']}[/] [bold]{command_line}[/]\n")
        start = time.monotonic()
        try:
            code = args.handler(args) or 0
        except KeyboardInterrupt:
            code = 130
        except Exception as exc:
            err_console.print(f"\n[cl.err]{GLYPH['err']} {exc}[/]")
            code = 1
        mark = "[cl.ok]✓ done[/]" if code == 0 else f"[cl.err]✗ exit {code}[/]"
        console.print(f"\n{mark}  [cl.muted]{format_duration(time.monotonic() - start)}[/]")
        return code
    return run


def _after_run_prompt(console: Console, code: int) -> None:
    pairs = [("any key", "back to ChemLink"), ("r", "run again"), ("q", "quit")]
    console.print(keycaps(pairs))


def run(
    parser: ArgumentParser,
    console: Console,
    err_console: Console,
    version: str = "",
    runner: Optional[Callable] = None,
) -> int:
    sub_action = find_sub_action(parser)
    if sub_action is None:
        return 1
    runner = runner or _fallback_runner(console, err_console)

    ctx = AppContext(version=version, sysinfo=SystemInfo())
    home = CommandListScreen(
        "ChemLink",
        sub_action,
        show_logo=True,
        subtitle=f"v{version}" if version else "",
        ctx=ctx,
    )
    stack: List = [home]
    reader = KeyReader()
    live = Live(console=console, screen=True, auto_refresh=False, transient=True)

    raw_cm = reader.raw()
    raw_cm.__enter__()
    live.start()
    last_size = None
    last_sys_ready = ctx.sysinfo.ready
    dirty = True
    try:
        while stack:
            top = stack[-1]
            animating = top.tick()
            size = console.size
            if size != last_size or ctx.sysinfo.ready != last_sys_ready:
                dirty = True
                last_size, last_sys_ready = size, ctx.sysinfo.ready
            if dirty or animating:
                live.update(top.render(width=size.width, height=size.height), refresh=True)
                dirty = False

            key = reader.wait_key(timeout=_TICK if animating else _IDLE_TICK)
            if key is None:
                continue
            dirty = True
            if key == "ctrl-l":
                console.clear()
                continue

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
                display = _display_command(breadcrumb, argv)
                while True:
                    _play_launch(live, display, console.size.height)
                    live.stop()
                    raw_cm.__exit__(None, None, None)
                    try:
                        code = runner(args, display)
                    except KeyboardInterrupt:
                        code = 130
                    _after_run_prompt(console, code)
                    raw_cm = reader.raw()
                    raw_cm.__enter__()
                    reader.drain()
                    choice = reader.wait_key()
                    if choice == "r":
                        args = top.parser.parse_args(argv) if hasattr(top, "parser") else args
                        live.start()
                        continue
                    if choice in ("q", "ctrl-c"):
                        stack.clear()
                    else:
                        live.start()
                    break
        return 0
    finally:
        try:
            live.stop()
        finally:
            raw_cm.__exit__(None, None, None)
