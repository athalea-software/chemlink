"""
mdrun_runner.py
Streaming runner for GROMACS mdrun with Rich live progress.

Progress architecture
----------------------
Primary source  — GROMACS .log file (nvt.log / npt.log / md.log).
  GROMACS writes this file with fflush() every nstlog steps regardless
  of pipe buffering, so data is always available in real time.
  A background thread tails it.

Secondary source — stderr via stdbuf.
  When stdbuf is present the command is wrapped with ``stdbuf -oL -eL``
  to force line-buffered stderr, giving GPU-init diagnostics before the
  first nstlog checkpoint.

Display
--------
  ◆ <phase>  [⣿⣿⣿⣿⣿⣿⣿⣿⡀⠀⠀⠀⠀]  67%  │  34000/50000 step  │  230.4 ns/day  │  ETA 1m 23s

GPU / note / perf lines are printed above the live bar via Rich console.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional
import logging

from rich.progress import Progress, TextColumn, MofNCompleteColumn, TimeElapsedColumn

from utils.progress import (
    NsDayColumn, EtaColumn,
    console as _cons,
    make_dynamics_progress,
)
from utils.style import log_line

# ── Patterns: GROMACS .log file ───────────────────────────────────────────────
_RE_LOG_STEP_HDR = re.compile(r"^\s+Step\s+Time\s*$")
_RE_LOG_STEP_VAL = re.compile(r"^\s+(\d+)\s+[\d.]+\s*$")
_RE_LOG_PERF     = re.compile(r"^\s*Performance:\s+([\d.]+)\s+([\d.]+)")
_RE_LOG_GPU      = re.compile(
    r"(?:GPU info|compute cap\.|Offload|offload|stat:\s*compatible"
    r"|CUDA.*version|update.*GPU|bonded.*GPU|PME.*GPU|nb.*GPU"
    r"|GPU-based offload)", re.I
)
_RE_LOG_NOTE     = re.compile(r"^\s*(?:NOTE|WARNING):", re.I)
_RE_LOG_DISABLED = re.compile(r"disabled|not offload|falling back|skipping", re.I)

# ── Patterns: stderr ──────────────────────────────────────────────────────────
_RE_STDERR_FATAL = re.compile(r"Fatal error|FATAL ERROR")
_RE_STDERR_GPU   = re.compile(r"(?:Offload|GPU info|compute cap|stat:|CUDA error)", re.I)
_RE_STDERR_NOTE  = re.compile(r"^\s*(?:NOTE|WARNING):", re.I)
_RE_STDERR_STEP  = re.compile(
    r"step\s+(\d+).*?remaining wall clock time:\s*([\d.]+)\s*s", re.I
)

# dt used in all our MDP files (ps)
_DT_PS = 0.002

# ── Rich tag helpers ──────────────────────────────────────────────────────────
def _tag(label: str, style: str, phase: str) -> str:
    return f"[{style}][ {label} │ {phase} ][/{style}]"

_PERF = lambda p: _tag("PERF", "bold bright_cyan", p)


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class MdrunResult:
    ns_per_day:   float     = 0.0
    wall_time_s:  float     = 0.0
    gpu_offloads: List[str] = field(default_factory=list)
    notes:        List[str] = field(default_factory=list)
    disabled:     List[str] = field(default_factory=list)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_deffnm(cmd: List[str]) -> str:
    try:
        return cmd[cmd.index("-deffnm") + 1]
    except (ValueError, IndexError):
        return "md"


def _maybe_stdbuf(cmd: List[str]) -> List[str]:
    exe = shutil.which("stdbuf")
    return ([exe, "-oL", "-eL"] + cmd) if exe else cmd


def _is_mpich() -> bool:
    """Return True if the system mpirun is MPICH/Hydra (not OpenMPI)."""
    try:
        r = subprocess.run(["mpirun", "--version"], capture_output=True,
                           text=True, timeout=5)
        return "HYDRA" in r.stdout or "HYDRA" in r.stderr or "mpich" in (r.stdout + r.stderr).lower()
    except Exception:
        return False


def _build_mpi_prefix(n_tasks: int, hosts: str) -> List[str]:
    """Build a mpirun prefix compatible with the installed MPI (MPICH or OpenMPI)."""
    prefix = ["mpirun", "-np", str(n_tasks)]
    if hosts:
        if _is_mpich():
            prefix += ["-hosts", hosts, "-iface", "eno1"]
        else:
            prefix += ["-host", hosts,
                       "--mca", "btl_tcp_if_include", "eno1",
                       "--mca", "btl", "^openib"]
    return prefix


def _live_metrics(step: int, total: int, t_start: float):
    elapsed = time.monotonic() - t_start
    if step <= 0 or elapsed < 1.0:
        return 0.0, 0.0
    sim_ns  = step * _DT_PS / 1000.0
    ns_day  = sim_ns / (elapsed / 86400.0)
    eta_s   = elapsed * (total - step) / step
    return ns_day, eta_s


# ── Public API ────────────────────────────────────────────────────────────────

def run_mdrun_streaming(
    cmd: List[str],
    *,
    work_dir: str,
    phase: str,
    total_steps: int,
    logger: logging.Logger,
    timeout: Optional[int] = None,
    mpi_tasks: int = 1,
    mpi_hosts: str = "",
    quiet_failures: bool = False,
    show_progress: bool = True,
) -> MdrunResult:
    """Run one GROMACS mdrun attempt with real-time Rich progress.

    When mpi_tasks > 1, the mdrun command is wrapped with mpirun for
    domain decomposition across multiple nodes.

    `quiet_failures=True` is for an attempt `run_mdrun_with_fallback` will
    silently retry on CPU if it fails: GPU-disabled/NOTE/FATAL lines and the
    final failure dump still go to the log file, just not to the console —
    a doomed GPU attempt (e.g. steepest-descent EM, which GROMACS never
    supports PME-on-GPU for) shouldn't spam the terminal with a crash it's
    about to recover from anyway.

    `show_progress=False` is for a caller looping over many short windows
    (equilibration phases across several systems, pilot/production lambda
    windows) that shows its own aggregate counter instead: this run's live
    per-step bar and its final "PERF" line are skipped on screen — still
    recorded in the log — so the terminal doesn't get one summary line per
    window on top of the counter.

    Raises subprocess.CalledProcessError on non-zero exit.
    """
    result   = MdrunResult()
    deffnm   = _get_deffnm(cmd)
    log_path = os.path.join(work_dir, f"{deffnm}.log")
    t_start  = time.monotonic()
    # Either reason (a doomed attempt about to retry, or an aggregate
    # counter standing in for per-window detail) hides the same lines from
    # the console while still recording them in the log.
    suppress_console = quiet_failures or not show_progress

    if mpi_tasks > 1:
        # GROMACS 2025+ requires -npme when using -pme gpu with multiple ranks.
        if "-pme" in cmd and cmd[cmd.index("-pme") + 1] == "gpu" and "-npme" not in cmd:
            cmd = cmd + ["-npme", "1"]
        mpi_prefix = _build_mpi_prefix(mpi_tasks, mpi_hosts)
        cmd = mpi_prefix + cmd
        logger.info(
            "[%s] MPI mode active — %d tasks, hosts: %s  prefix: %s",
            phase, mpi_tasks, mpi_hosts or "(default)", " ".join(mpi_prefix),
        )

    actual_cmd = _maybe_stdbuf(cmd)
    stdbuf_tag = " [+stdbuf]" if len(actual_cmd) > len(cmd) else ""
    logger.info(
        "[%s] mdrun START%s  cmd: %s", phase, stdbuf_tag, " ".join(cmd),
        extra={"file_only": True},
    )
    logger.info(
        "[%s] tailing log: %s", phase, log_path,
        extra={"file_only": True},
    )

    phase_label = f"[bold magenta]◆ {phase}[/bold magenta]"

    progress  = make_dynamics_progress()
    if show_progress:
        progress.start()
    task_id   = progress.add_task(phase_label, total=total_steps,
                                  ns_day=0.0, eta_s=0.0)

    last_step: int       = 0
    stderr_lines: List[str] = []
    _done = threading.Event()

    # GROMACS reprints the same GPU-capability/NOTE diagnostics at multiple points
    # within a single phase's log; every repeat still goes to the log file/logger
    # below, but the console only echoes each distinct line once per phase.
    _echoed_lines: set = set()

    def _echo_once(level: str, stripped: str, phase: str) -> None:
        if stripped in _echoed_lines:
            return
        _echoed_lines.add(stripped)
        progress.console.print("  ", log_line(level, stripped, phase=phase), sep="")

    # ── Thread A: tail .log file ──────────────────────────────────────────────
    def _tail_log() -> None:
        nonlocal last_step

        deadline = time.monotonic() + 60
        while not os.path.exists(log_path):
            if _done.is_set() or time.monotonic() > deadline:
                logger.warning("[%s] Log file not found after 60 s: %s",
                               phase, log_path)
                return
            time.sleep(0.5)

        after_hdr = False
        buf = ""

        with open(log_path, "r", errors="replace") as fh:
            while not _done.is_set():
                chunk = fh.read(4096)
                if not chunk:
                    time.sleep(0.3)
                    continue

                buf += chunk
                lines = buf.split("\n")
                buf = lines[-1]

                for line in lines[:-1]:
                    # ── GPU offload lines ─────────────────────────────────
                    if _RE_LOG_GPU.search(line):
                        stripped = line.strip()
                        if _RE_LOG_DISABLED.search(line):
                            result.disabled.append(stripped)
                            if not suppress_console:
                                _echo_once("WARN", stripped, phase)
                            logger.warning(
                                "[%s] %s", phase, stripped,
                                extra={"file_only": suppress_console},
                            )
                        else:
                            result.gpu_offloads.append(stripped)
                            logger.info(
                                "[%s] %s", phase, stripped,
                                extra={"file_only": True},
                            )
                        continue

                    # ── Notes / warnings ──────────────────────────────────
                    if _RE_LOG_NOTE.match(line):
                        stripped = line.strip()
                        result.notes.append(stripped)
                        if not suppress_console:
                            _echo_once("NOTE", stripped, phase)
                        logger.warning(
                            "[%s] %s", phase, stripped,
                            extra={"file_only": True},
                        )
                        continue

                    # ── Step/Time header → next line has step number ───────
                    if _RE_LOG_STEP_HDR.match(line):
                        after_hdr = True
                        continue

                    if after_hdr:
                        after_hdr = False
                        m = _RE_LOG_STEP_VAL.match(line)
                        if m:
                            step  = int(m.group(1))
                            delta = max(0, step - last_step)
                            if delta > 0:
                                progress.advance(task_id, delta)
                                last_step = step
                                ns_day, eta_s = _live_metrics(
                                    step, total_steps, t_start
                                )
                                progress.update(task_id,
                                                ns_day=ns_day, eta_s=eta_s)
                        continue

                    # ── Performance line (end of run) ─────────────────────
                    m = _RE_LOG_PERF.match(line)
                    if m:
                        result.ns_per_day = float(m.group(1))
                        h_per_ns          = float(m.group(2))
                        if show_progress:
                            progress.console.print(
                                f"\n  {_PERF(phase)}  "
                                f"[bold]{result.ns_per_day:.3f} ns/day[/bold]"
                                f"  [dim]({h_per_ns:.3f} h/ns)[/dim]"
                            )
                        logger.info(
                            "[%s] Performance: %.3f ns/day  (%.3f h/ns)",
                            phase,
                            result.ns_per_day,
                            h_per_ns,
                            extra={"file_only": True},
                        )
                        progress.update(task_id, ns_day=result.ns_per_day)

    # ── Thread B: stderr (GPU init + fallback steps) ──────────────────────────
    def _read_stderr(pipe) -> None:
        nonlocal last_step
        for raw in pipe:
            line = raw.rstrip()
            if not line:
                continue
            stderr_lines.append(line)

            if _RE_STDERR_FATAL.search(line):
                if not suppress_console:
                    progress.console.print("  ", log_line("FATAL", line, phase=phase), sep="")
                logger.error(
                    "[%s] FATAL: %s", phase, line,
                    extra={"file_only": suppress_console},
                )
                continue
            if _RE_STDERR_GPU.search(line):
                logger.info(
                    "[%s] %s", phase, line,
                    extra={"file_only": True},
                )
                continue
            if _RE_STDERR_NOTE.match(line):
                if not suppress_console:
                    _echo_once("NOTE", line, phase)
                logger.warning(
                    "[%s] %s", phase, line,
                    extra={"file_only": True},
                )
                continue
            m = _RE_STDERR_STEP.search(line)
            if m and last_step == 0:
                step  = int(m.group(1))
                eta_s = float(m.group(2))
                delta = max(0, step - last_step)
                progress.advance(task_id, delta)
                last_step = step
                progress.update(task_id, eta_s=eta_s)

    def _drain(pipe) -> None:
        for _ in pipe:
            pass

    # ── Launch process ────────────────────────────────────────────────────────
    proc = subprocess.Popen(
        actual_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=work_dir,
        bufsize=1,
    )

    t_log    = threading.Thread(target=_tail_log,                daemon=True)
    t_stderr = threading.Thread(target=_read_stderr,
                                args=(proc.stderr,), daemon=True)
    t_stdout = threading.Thread(target=_drain,
                                args=(proc.stdout,), daemon=True)
    t_log.start()
    t_stderr.start()
    t_stdout.start()

    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        _done.set()
        progress.stop()
        raise RuntimeError(f"[{phase}] mdrun timed out after {timeout}s")
    finally:
        _done.set()
        t_stderr.join(timeout=10)
        t_log.join(timeout=8)
        t_stdout.join(timeout=5)

        if proc.returncode == 0:
            remaining = max(0, total_steps - last_step)
            if remaining:
                progress.advance(task_id, remaining)

        progress.stop()

    result.wall_time_s = time.monotonic() - t_start

    if proc.returncode != 0:
        tail = "\n".join(stderr_lines[-40:])
        logger.error(
            "[%s] mdrun FAILED rc=%d  wall=%.1fs\n--- stderr tail ---\n%s\n---",
            phase, proc.returncode, result.wall_time_s, tail,
            extra={"file_only": suppress_console},
        )
        raise subprocess.CalledProcessError(
            proc.returncode, actual_cmd, stderr="\n".join(stderr_lines)
        )

    logger.info(
        "[%s] mdrun DONE  wall=%.1fs | ns/day=%.3f | gpu_lines=%d"
        " | disabled=%d | notes=%d",
        phase, result.wall_time_s, result.ns_per_day,
        len(result.gpu_offloads), len(result.disabled), len(result.notes),
        extra={"file_only": True},
    )
    if result.disabled:
        logger.warning(
            "[%s] Disabled GPU offloads:\n  %s",
            phase, "\n  ".join(result.disabled),
            extra={"file_only": not show_progress},
        )
    return result


def run_mdrun_with_fallback(
    cmd_gpu: List[str],
    cmd_cpu: List[str],
    *,
    work_dir: str,
    phase: str,
    total_steps: int,
    logger: logging.Logger,
    use_gpu: bool,
    timeout: Optional[int] = None,
    mpi_tasks: int = 1,
    mpi_hosts: str = "",
    show_progress: bool = True,
) -> MdrunResult:
    """Run mdrun GPU-first; on CalledProcessError fall back to CPU.

    When mpi_tasks > 1, the primary (GPU) attempt uses MPI domain
    decomposition.  The CPU fallback intentionally runs without MPI
    so a single-node retry is always available.

    `show_progress=False` is for a caller that shows its own aggregate
    counter over many such calls (see `run_mdrun_streaming`).
    """
    first_cmd = cmd_gpu if use_gpu else cmd_cpu
    try:
        return run_mdrun_streaming(
            first_cmd,
            work_dir=work_dir, phase=phase,
            total_steps=total_steps, logger=logger, timeout=timeout,
            mpi_tasks=mpi_tasks, mpi_hosts=mpi_hosts,
            quiet_failures=use_gpu, show_progress=show_progress,
        )
    except (subprocess.CalledProcessError, RuntimeError) as exc:
        if not use_gpu:
            raise
        logger.warning(
            "[%s] GPU run failed (%s) — retrying on CPU (no MPI).", phase, exc,
            extra={"file_only": True},
        )
        # CPU fallback deliberately omits MPI — runs on local node only.
        return run_mdrun_streaming(
            cmd_cpu,
            work_dir=work_dir, phase=phase,
            total_steps=total_steps, logger=logger, timeout=timeout,
            show_progress=show_progress,
        )
