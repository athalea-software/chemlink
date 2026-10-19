"""Independent MD replicas that share one prepared (minimised) system.

The parent run directory holds the preparation outputs (topology, solvated and
neutralised system, ``em.gro``). Each replica lives in ``replica_NN/`` with a
copy of those inputs and is a complete, self-contained run directory, so the
existing equilibration, production, post-processing, analysis, ``resume`` and
``analyze`` code works on it unchanged. NVT uses ``gen_seed = -1``, so every
replica starts from independently drawn velocities.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import statistics
from pathlib import Path

DEFAULT_REPLICAS = 2
REPLICA_PREFIX = "replica_"

# Parent-level bookkeeping files that must not be inherited by a replica.
_PARENT_ONLY_FILES = {
    "run_config.json",
    "hpc_config.json",
    "simulation.log",
    "simulation_summary.md",
    "replicas_summary.md",
}

_RMSD_SERIES = (
    ("Backbone RMSD (nm)", "rmsd_backbone.xvg"),
    ("Ligand RMSD, protein fit (nm)", "rmsd_ligand_fit_protein.xvg"),
)


def replica_dir_name(index: int) -> str:
    return f"{REPLICA_PREFIX}{index:02d}"


def replica_dirs(work_dir: Path) -> list[Path]:
    return sorted(
        path for path in Path(work_dir).glob(f"{REPLICA_PREFIX}*") if path.is_dir()
    )


def replica_count(config: dict) -> int:
    return max(1, int(config.get("replicas", 1) or 1))


def prepare_replica(parent_dir: Path, index: int, config: dict) -> dict:
    """Create ``replica_NN`` from the prepared system and return its config.

    Regular files are copied (they are small and GROMACS may rewrite some of
    them); directories such as ``acpype_work`` are linked read-only. Existing
    files in the replica are kept so a partially run replica can be resumed.
    """
    parent_dir = Path(parent_dir)
    if not (parent_dir / "em.gro").is_file():
        raise FileNotFoundError(
            f"Cannot create {replica_dir_name(index)}: em.gro is missing in {parent_dir}"
        )
    replica_dir = parent_dir / replica_dir_name(index)
    replica_dir.mkdir(exist_ok=True)

    for entry in parent_dir.iterdir():
        if entry.name.startswith(REPLICA_PREFIX) or entry.name in _PARENT_ONLY_FILES:
            continue
        target = replica_dir / entry.name
        if target.exists() or target.is_symlink():
            continue
        if entry.is_dir():
            try:
                target.symlink_to(os.path.relpath(entry, replica_dir), target_is_directory=True)
            except OSError:
                shutil.copytree(entry, target)
        elif entry.is_file():
            shutil.copy2(entry, target)

    replica_config = copy.deepcopy(config)
    replica_config.update(
        work_dir=str(replica_dir.resolve()),
        parent_work_dir=str(parent_dir.resolve()),
        replica_index=index,
        replicas=replica_count(config),
    )
    replica_config.pop("_config_path", None)
    (replica_dir / "run_config.json").write_text(
        json.dumps(replica_config, indent=2), encoding="utf-8"
    )
    return replica_config


def _xvg_mean(path: Path) -> float | None:
    if not path.is_file():
        return None
    values = []
    for line in path.read_text(errors="replace").splitlines():
        if not line.strip() or line.startswith(("#", "@")):
            continue
        parts = line.split()
        if len(parts) >= 2:
            try:
                values.append(float(parts[1]))
            except ValueError:
                continue
    return statistics.fmean(values) if values else None


def _binding_energies(replica_dir: Path) -> dict[str, float]:
    """Return {label: kcal/mol} from a replica's gmx_MMPBSA metadata."""
    metadata = replica_dir / "binding_energy_analysis/gmx_MMPBSA/analysis_metadata.json"
    if not metadata.is_file():
        return {}
    try:
        data = json.loads(metadata.read_text())
    except (OSError, ValueError):
        return {}
    energies = {}
    for method, result in (data.get("energies") or {}).items():
        if "total_mean" in result:
            energies[f"MM-{method}SA ΔH (kcal/mol)"] = float(result["total_mean"])
        if "corrected_mean" in result:
            energies[f"MM-{method}SA ΔG (kcal/mol)"] = float(result["corrected_mean"])
    return energies


def _format_stats(values: list[float]) -> str:
    if not values:
        return "N/A"
    mean = statistics.fmean(values)
    if len(values) < 2:
        return f"{mean:.3f} (n=1)"
    return f"{mean:.3f} ± {statistics.stdev(values):.3f} (n={len(values)})"


def write_replicas_summary(parent_dir: Path, replicas: int) -> str:
    """Aggregate per-replica binding energies and RMSD into Markdown."""
    parent_dir = Path(parent_dir)
    rows: dict[str, dict[int, float]] = {}
    status_rows = []
    for index in range(1, replicas + 1):
        replica_dir = parent_dir / replica_dir_name(index)
        done = (replica_dir / "SIMULATION_SUMMARY.txt").is_file()
        status_rows.append(
            f"| {replica_dir_name(index)} | {'Completed' if done else 'Pending or interrupted'} |"
        )
        metrics = _binding_energies(replica_dir)
        for label, filename in _RMSD_SERIES:
            value = _xvg_mean(replica_dir / "rmsd_analysis" / filename)
            if value is not None:
                metrics[label] = value
        for label, value in metrics.items():
            rows.setdefault(label, {})[index] = value

    header = "| Metric | " + " | ".join(
        replica_dir_name(i) for i in range(1, replicas + 1)
    ) + " | Mean ± SD |"
    divider = "|---|" + "---|" * (replicas + 1)
    metric_rows = []
    for label, per_replica in rows.items():
        cells = [
            f"{per_replica[i]:.3f}" if i in per_replica else "N/A"
            for i in range(1, replicas + 1)
        ]
        metric_rows.append(
            f"| {label} | " + " | ".join(cells)
            + f" | {_format_stats(list(per_replica.values()))} |"
        )

    output = parent_dir / "replicas_summary.md"
    output.write_text(
        "# ChemLink replicas summary\n\n"
        f"- **Requested replicas:** {replicas}\n"
        "- **Shared preparation:** topology, solvation, ions and minimization "
        "in the root directory; each replica starts from `em.gro` with independent "
        "initial velocities (`gen_seed = -1`).\n\n"
        "## Status\n\n| Replica | Status |\n|---|---|\n"
        + "\n".join(status_rows)
        + "\n\n## Per-replica metrics\n\n"
        + ("\n".join([header, divider, *metric_rows]) if metric_rows else "No metrics available.")
        + "\n\nThe standard deviation is computed across replicas (n-1).\n",
        encoding="utf-8",
    )
    return str(output)
