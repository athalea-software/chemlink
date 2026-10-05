"""Checkpoint-aware recovery for interrupted ChemLink dynamics runs."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import re
from typing import Optional

from hpc.cluster.resource_detector import get_hardware_profile
from pipelines.dynamics.gmx_optimizer import get_optimal_mdrun_flags
from pipelines.dynamics.mdrun_runner import run_mdrun_with_fallback
from pipelines.dynamics.utils import check_gmx_installation, get_system_threads

logger = logging.getLogger(__name__)

SIM_TYPES = {
    "oprotein": ("1", "Protein only"),
    "pligand": ("2", "Protein + ligand"),
    "ppeptide": ("3", "Protein + peptide"),
    "pacid": ("4", "Protein + nucleic acid"),
    "pprotein": ("5", "Protein + protein"),
    "ppligand": ("6", "Protein + protein + ligand"),
    "oligand": ("7", "Ligand only"),
}


def _mdp_number(path: Path, key: str) -> Optional[float]:
    if not path.is_file():
        return None
    match = re.search(
        rf"^\s*{re.escape(key)}\s*=\s*([-+]?\d+(?:\.\d+)?)",
        path.read_text(errors="replace"), re.MULTILINE | re.IGNORECASE,
    )
    return float(match.group(1)) if match else None


def phase_completed(work_dir: Path, phase: str) -> bool:
    """Require both the final structure and GROMACS completion marker."""
    gro = work_dir / f"{phase}.gro"
    log = work_dir / f"{phase}.log"
    return (
        gro.is_file() and gro.stat().st_size > 0 and log.is_file()
        and "Finished mdrun" in log.read_text(errors="replace")
    )


def analysis_completed(work_dir: Path, config: dict) -> bool:
    if not (work_dir / "SIMULATION_SUMMARY.txt").is_file():
        return False
    if not config.get("binding_energy", True):
        return True
    metadata = work_dir / "binding_energy_analysis/gmx_MMPBSA/analysis_metadata.json"
    if not metadata.is_file():
        return config.get("sim_type") not in {"2", "3", "5", "6"}
    try:
        return json.loads(metadata.read_text()).get("status") in {
            "completed", "completed_with_warnings",
        }
    except (OSError, ValueError):
        return False


def detect_resume_stage(work_dir: Path, config: dict) -> str:
    for phase in ("nvt", "npt", "md"):
        if phase_completed(work_dir, phase):
            continue
        if (work_dir / f"{phase}.cpt").is_file() and (work_dir / f"{phase}.tpr").is_file():
            return phase
        if phase == "npt" and phase_completed(work_dir, "nvt"):
            return "npt-new"
        if phase == "md" and phase_completed(work_dir, "npt"):
            return "md-new"
        raise RuntimeError(
            f"Cannot resume {phase.upper()}: a valid {phase}.cpt/{phase}.tpr pair was not found."
        )
    if any(not (work_dir / name).is_file() for name in ("md_center.xtc", "md.pdb", "index.ndx")):
        return "post-processing"
    if not analysis_completed(work_dir, config):
        return "analysis"
    return "complete"


def load_resume_config(
    work_dir: Path,
    sim_type_name: Optional[str] = None,
    ns_time: Optional[float] = None,
) -> dict:
    config_path = work_dir / "run_config.json"
    if not config_path.is_file():
        config_path = work_dir / "hpc_config.json"
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}

    if sim_type_name:
        config["sim_type"], config["sim_type_label"] = SIM_TYPES[sim_type_name]
    elif "sim_type" not in config:
        if (work_dir / "ligand.itp").is_file() or (work_dir / "ligand_charge.json").is_file():
            config["sim_type"], config["sim_type_label"] = SIM_TYPES["pligand"]
        else:
            raise ValueError("The simulation type is ambiguous; provide --type TYPE.")

    if ns_time is not None:
        config["ns_time"] = ns_time
    elif "ns_time" not in config:
        nsteps = _mdp_number(work_dir / "md.mdp", "nsteps")
        dt = _mdp_number(work_dir / "md.mdp", "dt")
        if nsteps is None or dt is None:
            raise ValueError("Simulation duration is unavailable; provide --time NS.")
        config["ns_time"] = nsteps * dt / 1000.0

    config["work_dir"] = str(work_dir.resolve())
    config.setdefault("threads", get_system_threads())
    config.setdefault("binding_energy", True)
    config.setdefault("mmpbsa_method", "both")
    config.setdefault("mmpbsa_frames", 1000)
    config.setdefault("mmpbsa_discard", 20.0)
    config.setdefault("mmpbsa_cores", config["threads"])
    config.setdefault("mmpbsa_salt", 0.150)
    config.setdefault("mmpbsa_entropy", "none")
    config.setdefault("mmpbsa_temperature", 300.0)
    config.setdefault("mmpbsa_ie_segment", 25.0)
    if config.get("resolved_force_field_dir"):
        os.environ["GMXLIB"] = config["resolved_force_field_dir"]
    config["gpu_ids"] = get_hardware_profile().gpu_indices
    return config


class DynamicsResumeController:
    def __init__(
        self,
        work_dir: str,
        *,
        sim_type_name: Optional[str] = None,
        ns_time: Optional[float] = None,
    ):
        self.work_dir = Path(work_dir).expanduser().resolve()
        if not self.work_dir.is_dir():
            raise FileNotFoundError(f"Run directory was not found: {self.work_dir}")
        self.config = load_resume_config(self.work_dir, sim_type_name, ns_time)
        self.gmx_bin = check_gmx_installation()

    def detect_stage(self) -> str:
        return detect_resume_stage(self.work_dir, self.config)

    def _resume_mdrun(self, phase: str) -> None:
        opt = get_optimal_mdrun_flags(self.config)
        nsteps = int(_mdp_number(self.work_dir / f"{phase}.mdp", "nsteps") or 0)
        if nsteps < 1:
            raise ValueError(f"Could not read nsteps from {phase}.mdp")
        base = [
            self.gmx_bin, "mdrun", "-v", "-deffnm", phase,
            "-cpi", f"{phase}.cpt", "-append",
        ]
        run_mdrun_with_fallback(
            cmd_gpu=base + opt["gpu"], cmd_cpu=base + opt["cpu"],
            work_dir=str(self.work_dir), phase=f"Resume {phase.upper()}",
            total_steps=nsteps, logger=logger,
            use_gpu=bool(self.config.get("gpu_ids")), timeout=None,
            mpi_tasks=int(self.config.get("mpi_tasks", 1)),
            mpi_hosts=self.config.get("mpi_hosts", ""),
        )

    def run(self, requested_stage: str = "auto") -> str:
        # Keep validation/dry-run lightweight and free from plotting imports.
        from pipelines.dynamics.steps.analysis import AnalysisStep
        from pipelines.dynamics.steps.equilibration import EquilibrationStep
        from pipelines.dynamics.steps.post_processing import PostProcessingStep
        from pipelines.dynamics.steps.production import ProductionStep

        stage = self.detect_stage() if requested_stage == "auto" else requested_stage
        if stage == "complete":
            return stage
        if stage == "nvt":
            self._resume_mdrun("nvt")
            EquilibrationStep(self.config, self.gmx_bin).run_npt_only()
            ProductionStep(self.config, self.gmx_bin).run()
        elif stage == "npt-new":
            EquilibrationStep(self.config, self.gmx_bin).run_npt_only()
            ProductionStep(self.config, self.gmx_bin).run()
        elif stage == "npt":
            self._resume_mdrun("npt")
            ProductionStep(self.config, self.gmx_bin).run()
        elif stage == "md-new":
            ProductionStep(self.config, self.gmx_bin).run()
        elif stage == "md":
            self._resume_mdrun("md")

        if stage in {"nvt", "npt-new", "npt", "md-new", "md", "post-processing"}:
            PostProcessingStep(self.config, self.gmx_bin).run()
        AnalysisStep(self.config, self.gmx_bin).run()
        return stage
