import os
from pathlib import Path
import re
import shutil
import subprocess
import multiprocessing
from datetime import datetime
try:
    from ...storage.run_directory import create_named_run_directory
except ImportError:
    from storage.run_directory import create_named_run_directory

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
        print(f"[Error] No se encontró el archivo MDP en: {mdp_path}")
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

    print(f"[✓] Archivo {os.path.basename(mdp_path)} actualizado a {nsteps} pasos.")
    return True

def resolve_gmx_binary(*, require_serial=False, prefer_mpi=True):
    """Resolve GROMACS according to the needs of the calling workflow."""
    serial = [
        os.environ.get("CHEMLINK_GMX_BIN"),
        shutil.which("gmx"),
        "/usr/local/gromacs/bin/gmx",
        "/opt/gromacs/bin/gmx",
    ]
    mpi = [
        os.environ.get("CHEMLINK_GMX_MPI_BIN"),
        shutil.which("gmx_mpi"),
        "/usr/local/gromacs/bin/gmx_mpi",
        "/opt/gromacs/bin/gmx_mpi",
    ]
    candidates = serial if require_serial else (mpi + serial if prefer_mpi else serial + mpi)
    return next(
        (path for path in candidates if path and os.path.isfile(path) and os.access(path, os.X_OK)),
        None,
    )


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
        print(f"(!) Error en OpenBabel: {e.stderr}")
        return False

def get_system_threads():
    try:
        # Best for Linux: returns threads allowed for this specific process
        return len(os.sched_getaffinity(0))
    except (AttributeError, NotImplementedError):
        # Fallback for Windows/macOS or systems without affinity support
        return multiprocessing.cpu_count()
