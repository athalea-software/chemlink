import os
from pathlib import Path
import re
import shutil
import subprocess
import multiprocessing
import sys
from datetime import datetime
try:
    from ...storage.run_directory import create_named_run_directory
except ImportError:
    from storage.run_directory import create_named_run_directory


def _echo(level: str, message: str) -> None:
    """One console line in ChemLink's log format (no-op formatting fallback)."""
    try:
        from utils.style import log_line
        from utils.theme import get_console
        get_console().print(log_line(level, message, phase="dynamics"), soft_wrap=True)
    except ImportError:  # pragma: no cover
        print(f"{level}: {message}")

def calculate_nsteps(ns_time, dt=0.002):
    # 1 ns = 1000 ps. nsteps = (ns * 1000) / dt
    ps_time = ns_time * 1000
    nsteps = int(ps_time / dt)
    return nsteps

def setup_work_directory(
    base_output_path: str,
    sim_type_name: str,
    run_name: str | None = None,
) -> str:
    if run_name is not None:
        return create_named_run_directory(base_output_path, run_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"run_{sim_type_name}_{timestamp}"
    
    work_dir = Path(base_output_path) / folder_name
    work_dir.mkdir(parents=True, exist_ok=True)
    
    return str(work_dir.absolute())

def update_md_nsteps(mdp_path, nsteps):

    if not os.path.exists(mdp_path): # revisa la existencia del archivo MDP
        _echo("ERROR", f"MDP file not found: {mdp_path}")
        return False

    with open(mdp_path, 'r') as f:
        lines = f.readlines()

    new_lines = []
    found = False
    pattern = re.compile(r'^(\s*nsteps\s*=\s*)(\d+)(.*)$')

    for line in lines:
        if pattern.match(line):
            new_line = pattern.sub(rf'\1{nsteps}\3', line)
            new_lines.append(new_line)
            found = True
        else:
            new_lines.append(line)

    # Si por alguna razón no existe la línea nsteps, la añadimos al final
    if not found:
        new_lines.append(f"\n; Agregado por ChemLink\nnsteps = {nsteps}\n")

    with open(mdp_path, 'w') as f:
        f.writelines(new_lines)

    _echo("INFO", f"{os.path.basename(mdp_path)} set to {nsteps:,} steps")
    return True

def _executable_path(candidate):
    """Return an absolute executable path for a command name or path."""
    if not candidate:
        return None
    resolved = shutil.which(os.path.expanduser(str(candidate)))
    if resolved and os.path.isfile(resolved) and os.access(resolved, os.X_OK):
        return os.path.abspath(resolved)
    return None


def _gmx_candidates(binary_name, override_var):
    """Build candidates for GROMACS without assuming it is exported in PATH."""
    candidates = [os.environ.get(override_var), binary_name]

    for directory in (
        os.environ.get("GMXBIN"),
        os.path.join(os.environ["CONDA_PREFIX"], "bin") if os.environ.get("CONDA_PREFIX") else None,
        os.path.join(sys.prefix, "bin"),
        "/usr/bin",
        "/usr/local/bin",
        "/usr/local/gromacs/bin",
        "/opt/gromacs/bin",
    ):
        if directory:
            candidates.append(os.path.join(os.path.expanduser(directory), binary_name))

    # An MPI installation is often next to the serial/thread-MPI executable.
    sibling_name = "gmx_mpi" if binary_name == "gmx" else "gmx"
    sibling = _executable_path(os.environ.get("CHEMLINK_GMX_MPI_BIN") or sibling_name)
    if sibling:
        candidates.append(os.path.join(os.path.dirname(sibling), binary_name))

    return candidates


def find_gmx_path(*, serial=True):
    """Find the absolute path to ``gmx`` (or ``gmx_mpi`` when requested)."""
    binary_name = "gmx" if serial else "gmx_mpi"
    override_var = "CHEMLINK_GMX_BIN" if serial else "CHEMLINK_GMX_MPI_BIN"
    for candidate in _gmx_candidates(binary_name, override_var):
        resolved = _executable_path(candidate)
        if resolved:
            return resolved
    return None


def resolve_gmx_binary(*, require_serial=False, prefer_mpi=True):
    """Resolve GROMACS according to the needs of the calling workflow."""
    serial = find_gmx_path(serial=True)
    if require_serial:
        return serial
    mpi = find_gmx_path(serial=False)
    return (mpi or serial) if prefer_mpi else (serial or mpi)


def check_gmx_installation():
    resolved = resolve_gmx_binary(prefer_mpi=True)
    if resolved:
        return resolved

    raise EnvironmentError(
        "(!) Error: No se encontró GROMACS (gmx o gmx_mpi) instalado en el PATH del sistema.\n"
        "Asegúrese de que GROMACS esté instalado y cargado correctamente."
    )

def convert_pdbqt_to_pdb(input_path, output_path):
    obabel_cmd = [
        "obabel", 
        "-ipdbqt", input_path, 
        "-opdb", "-O", output_path, 
        "-h",
        "-p", "7.0", 
        "--gen3d",                
        "--resname", "LIG"    
    ]
    
    try:
        result = subprocess.run(obabel_cmd, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        _echo("ERROR", f"OpenBabel failed:\n{e.stderr}")
        return False

def get_system_threads():
    try:
        # Best for Linux: returns threads allowed for this specific process
        return len(os.sched_getaffinity(0))
    except (AttributeError, NotImplementedError):
        # Fallback for Windows/macOS or systems without affinity support
        return multiprocessing.cpu_count()
