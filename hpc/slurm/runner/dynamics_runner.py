"""
dynamics_runner.py
Thin entry point called by the SLURM dynamics.slurm script.
Reads a JSON config file and runs DynamicsPipeline.

Usage (from SLURM script):
    python -m hpc.slurm.runner.dynamics_runner /path/to/config.json
"""
from __future__ import annotations

import json
import os
import sys


def main() -> int:
    if len(sys.argv) < 2:
        print("[dynamics_runner] Error: config JSON path required as first argument", flush=True)
        return 1

    config_path = sys.argv[1]
    try:
        with open(config_path) as fh:
            config = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[dynamics_runner] Error reading config {config_path}: {exc}", flush=True)
        return 1

    try:
        from chemlink.pipelines.dynamics.dynamics_pipeline import DynamicsPipeline
    except ImportError:
        try:
            from chemlink.pipelines.dynamics.dynamics_pipeline import DynamicsPipeline  # type: ignore
        except ImportError:
            try:
                from ...pipelines.dynamics.dynamics_pipeline import DynamicsPipeline  # type: ignore
            except ImportError as exc:
                print(f"[dynamics_runner] Cannot import DynamicsPipeline: {exc}", flush=True)
                return 1

    # ── MPI injection ─────────────────────────────────────────────────────────
    mpi_tasks = int(os.environ.get("DYN_MPI_TASKS", "1"))
    mpi_hosts = os.environ.get("DYN_MPI_HOSTS", "")
    if mpi_tasks > 1:
        config["mpi_tasks"] = mpi_tasks
        config["mpi_hosts"] = mpi_hosts
        print(
            f"[dynamics_runner] MPI mode: {mpi_tasks} tasks, hosts: {mpi_hosts or '(default)'}",
            flush=True,
        )

    print(f"[dynamics_runner] Starting simulation: {config.get('sim_type_label', '?')} "
          f"({config.get('ns_time', '?')} ns)", flush=True)
    print(f"[dynamics_runner] Work dir: {config.get('work_dir', '?')}", flush=True)

    DynamicsPipeline(config).execute()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
