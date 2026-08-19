"""
gmx_optimizer.py
Optimización automática de flags mdrun según hardware disponible.

RTX 5000 / Blackwell (sm_120) — estado con GROMACS 2025.4 + CUDA 13
----------------------------------------------------------------------
Con GROMACS 2025.4 y CUDA 13.0 los kernels sm_120 son nativos (sin PTX JIT).
El cuello de botella principal en sistemas de 30-50k átomos es la
**saturación de GPU**: el sistema es demasiado pequeño para usar la GPU al 100%.
Solución: correr múltiples simulaciones concurrentes en la misma GPU.

Barostato
----------
Parrinello-Rahman crea un punto de sincronización CPU-GPU (cálculo de virial)
que reduce el rendimiento ~50%. Los MDPs usan C-rescale (ensemble NPT correcto,
100% compatible con GPU-update).

CPU híbridas (Intel Arrow Lake, Alder Lake, Raptor Lake)
---------------------------------------------------------
Mezclar P-cores y E-cores en un bloque OpenMP contiguo crea load imbalance
porque los E-cores son ~15-25% más lentos. Esta función detecta cuántos P-cores
hay para recomendar el ntomp óptimo.
"""

import os
import re
import subprocess
from typing import Any, Dict, List, Optional, Tuple

# Architectures with stable, production-grade GROMACS GPU offloads.
_MATURE_ARCHS   = {75, 80, 86, 89, 90}     # Turing / Ampere / Ada / Hopper
# Architectures where GROMACS GPU kernel support may be incomplete.
_NEXT_GEN_ARCHS = {100, 102, 120}           # Blackwell B100 / B200 / RTX 5000


def get_optimal_mdrun_flags(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return a dict with GPU and CPU flag lists for mdrun, plus metadata:

      gpu                 : List[str]  — flags for the GPU run attempt
      cpu                 : List[str]  — flags for the CPU fallback
      ntomp               : int        — OpenMP threads per rank (single sim)
      gpu_arch            : int        — sm capability without dot (86, 120…)
      is_mature           : bool       — known-good GPU offload support
      is_next_gen         : bool       — Blackwell or newer
      update_gpu          : bool       — whether -update gpu is applied
      nstlist             : int        — recommended nstlist for MDP
      p_core_count        : int        — detected high-frequency (P) cores
      max_concurrent_sims : int        — max sims that fit in GPU VRAM
      diagnostics         : List[str]  — human-readable notes logged by callers
    """
    gpu_ids    = config.get("gpu_ids", [])
    use_gpu    = bool(gpu_ids)
    total_cpus = os.cpu_count() or 8
    diagnostics: List[str] = []

    p_core_count = _get_p_core_count()

    if not use_gpu:
        ntomp = min(p_core_count, total_cpus)
        return {
            "gpu":                 [],
            "cpu":                 ["-ntomp", str(ntomp), "-pin", "on"],
            "ntomp":               ntomp,
            "gpu_arch":            0,
            "is_mature":           False,
            "is_next_gen":         False,
            "update_gpu":          False,
            "nstlist":             10,
            "p_core_count":        p_core_count,
            "max_concurrent_sims": 1,
            "gpu_name":            None,
            "gpu_memory_total_mb": 0,
            "gpu_memory_free_mb":  0,
            "total_cpu_count":     total_cpus,
            "diagnostics":         ["No GPU detected — running CPU-only."],
        }

    gpu_id     = gpu_ids[0]
    gpu_mem_mb = _get_gpu_memory_mb(gpu_id)
    gpu_arch   = _get_gpu_compute_capability(gpu_id)
    gpu_name   = _get_gpu_name(gpu_id)

    is_mature   = gpu_arch in _MATURE_ARCHS
    is_next_gen = gpu_arch in _NEXT_GEN_ARCHS or gpu_arch >= 100

    # ── ntomp: prefer P-cores only to avoid OpenMP load imbalance on hybrid CPUs
    # Both mature and next-gen use the same formula now that C-rescale removes
    # the CPU-side barostat stall.  P-cores are faster and form a clean NUMA set.
    ntomp = max(4, min(8, p_core_count))

    # ── GPU offload flags ─────────────────────────────────────────────────────
    gpu_flags: List[str] = [
        "-ntomp", str(ntomp),
        "-pin",   "on",
        "-nb",    "gpu",
        "-pme",   "gpu",
    ]
    if gpu_arch >= 70:
        gpu_flags += ["-bonded", "gpu"]

    # -update gpu: enabled for mature archs; for next-gen GROMACS 2024+ enables
    # it automatically when conditions are met (no need to pass the flag).
    use_update_gpu = is_mature and not is_next_gen
    if use_update_gpu:
        gpu_flags += ["-update", "gpu"]

    cpu_flags: List[str] = ["-ntomp", str(ntomp), "-pin", "on"]

    # ── nstlist: starting value for GROMACS auto-tuner ─────────────────────────
    # GROMACS will override this upward (to ~80-100 for GPU runs), so this is
    # just a lower-bound hint.
    nstlist = 40 if is_next_gen else (20 if is_mature else 10)

    # ── Concurrent simulation capacity ────────────────────────────────────────
    gpu_mem_used_mb = _get_gpu_memory_used_mb(gpu_id)
    gpu_mem_free_mb = gpu_mem_mb - gpu_mem_used_mb
    # Conservative estimate: use current process VRAM as per-sim baseline.
    # Floor at 200 MB to avoid divide-by-zero on fresh GPU.
    per_sim_mb = max(200, gpu_mem_used_mb)
    # Reserve 10% headroom
    max_concurrent = max(1, int(gpu_mem_free_mb * 0.9 // per_sim_mb) + 1)

    # CPU-side limit: each sim needs ntomp threads
    cpu_concurrent = max(1, total_cpus // ntomp)
    max_concurrent = min(max_concurrent, cpu_concurrent)

    # ── Diagnostics ───────────────────────────────────────────────────────────
    diagnostics.append(
        f"GPU {gpu_id}: {gpu_name}  sm_{gpu_arch}  "
        f"{gpu_mem_mb} MB total / {gpu_mem_free_mb} MB free"
    )
    diagnostics.append(
        f"CPU: {total_cpus} cores total / {p_core_count} P-cores detected  "
        f"→ ntomp={ntomp}"
    )
    diagnostics.append(
        f"Arch: {'MATURE' if is_mature else 'NEXT-GEN' if is_next_gen else 'UNKNOWN'}  "
        f"| update_gpu={use_update_gpu}  | nstlist hint={nstlist}"
    )
    diagnostics.append(
        f"Concurrent capacity: up to {max_concurrent} sims on this GPU "
        f"({per_sim_mb} MB/sim estimated, {gpu_mem_free_mb} MB free)"
    )

    if max_concurrent > 1:
        diagnostics.append(
            f"THROUGHPUT TIP — GPU is ~50% utilized per simulation. "
            f"Running {min(max_concurrent, 3)} concurrent sims on the same GPU "
            f"can multiply throughput {min(max_concurrent, 3)}x. "
            f"Use -gpu_id 0 -pin on -pinoffset <N*{ntomp}> -ntomp {ntomp} per run. "
            f"Example for 2 sims: "
            f"sim1: -ntomp {ntomp} -pinoffset 0  |  "
            f"sim2: -ntomp {ntomp} -pinoffset {ntomp}"
        )

    if is_next_gen:
        diagnostics.append(
            "BLACKWELL sm_120: GROMACS 2025.4+CUDA 13 → native kernels active. "
            "C-rescale barostat in MDP removes CPU sync stall. "
            "GPU-aware MPI: set GMX_FORCE_GPU_AWARE_MPI=1 if MPI was built with CUDA support."
        )

    return {
        "gpu":                 gpu_flags,
        "cpu":                 cpu_flags,
        "ntomp":               ntomp,
        "gpu_arch":            gpu_arch,
        "is_mature":           is_mature,
        "is_next_gen":         is_next_gen,
        "update_gpu":          use_update_gpu,
        "nstlist":             nstlist,
        "p_core_count":        p_core_count,
        "max_concurrent_sims": max_concurrent,
        "gpu_name":            gpu_name,
        "gpu_memory_total_mb": gpu_mem_mb,
        "gpu_memory_free_mb":  gpu_mem_free_mb,
        "total_cpu_count":     total_cpus,
        "diagnostics":         diagnostics,
    }


def get_cached_optimal_mdrun_flags(config: Dict[str, Any]) -> Dict[str, Any]:
    """Return one hardware decision shared by every phase in a simulation."""
    cached = config.get("_gmx_optimizer")
    if isinstance(cached, dict):
        return cached
    optimized = get_optimal_mdrun_flags(config)
    config["_gmx_optimizer"] = optimized
    return optimized


def get_concurrent_configs(
    config: Dict[str, Any],
    n_sims: int,
) -> List[Dict[str, Any]]:
    """
    Return a list of n_sims flag dicts, each with -pinoffset set so that
    concurrent GROMACS instances use non-overlapping CPU core ranges.

    Usage (bash, same GPU):
        for i in 0 1 2; do
            cd sim_$i && mpirun -np 1 gmx_mpi mdrun -gpu_id 0 \\
                <flags from configs[i]> -deffnm md &
        done
    """
    base = get_optimal_mdrun_flags(config)
    ntomp = base["ntomp"]
    configs = []
    for i in range(n_sims):
        flags = list(base["gpu"])
        # Replace or add -pinoffset
        if "-pinoffset" in flags:
            idx = flags.index("-pinoffset")
            flags[idx + 1] = str(i * ntomp)
        else:
            flags += ["-pinoffset", str(i * ntomp)]
        configs.append({**base, "gpu": flags, "pinoffset": i * ntomp})
    return configs


# ── Hardware query helpers ─────────────────────────────────────────────────────

def _get_p_core_count() -> int:
    """
    Return the number of high-performance cores by reading per-core max
    frequencies from sysfs.  On hybrid CPUs (Arrow Lake, Alder Lake) P-cores
    run at higher max frequencies than E-cores.  Falls back to total CPU count.
    """
    try:
        freqs: List[int] = []
        for i in range(os.cpu_count() or 8):
            path = f"/sys/devices/system/cpu/cpu{i}/cpufreq/cpuinfo_max_freq"
            try:
                with open(path) as f:
                    freqs.append(int(f.read().strip()))
            except OSError:
                pass

        if not freqs:
            return os.cpu_count() or 8

        max_freq = max(freqs)
        # P-cores are those within 20% of the maximum frequency
        threshold = max_freq * 0.80
        p_cores = sum(1 for f in freqs if f >= threshold)
        return max(1, p_cores)
    except Exception:
        return os.cpu_count() or 8


def _get_gpu_name(gpu_id: int) -> str:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name",
             "--format=csv,noheader", f"--id={gpu_id}"],
            capture_output=True, text=True, timeout=5,
        )
        return r.stdout.strip()
    except Exception:
        return "unknown"


def _get_gpu_memory_mb(gpu_id: int) -> int:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total",
             "--format=csv,noheader,nounits", f"--id={gpu_id}"],
            capture_output=True, text=True, timeout=5,
        )
        return int(r.stdout.strip())
    except Exception:
        return 8000


def _get_gpu_memory_used_mb(gpu_id: int) -> int:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits", f"--id={gpu_id}"],
            capture_output=True, text=True, timeout=5,
        )
        return int(r.stdout.strip())
    except Exception:
        return 500


def _get_gpu_compute_capability(gpu_id: int) -> int:
    """Return compute capability as integer without dot, e.g. 8.6 → 86."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap",
             "--format=csv,noheader", f"--id={gpu_id}"],
            capture_output=True, text=True, timeout=5,
        )
        return int(r.stdout.strip().replace(".", ""))
    except Exception:
        return 86
