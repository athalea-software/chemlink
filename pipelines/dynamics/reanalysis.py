"""Re-run analyses for an existing ChemLink molecular-dynamics run."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pipelines.dynamics.resume import load_resume_config
from pipelines.dynamics.steps.analysis import AnalysisStep
from pipelines.dynamics.utils import check_gmx_installation


ANALYSIS_INPUTS = ("md.tpr", "md_center.xtc", "index.ndx")
BINDING_SIM_TYPES = {"2", "3", "5", "6"}


class DynamicsReanalysisController:
    """Validate a completed run and execute only the requested analysis stage."""

    def __init__(
        self,
        run_dir: str,
        *,
        selection: str = "all",
        sim_type_name: Optional[str] = None,
        ns_time: Optional[float] = None,
        mmpbsa_cores: Optional[int] = None,
        mmpbsa_method: Optional[str] = None,
        mmpbsa_frames: Optional[int] = None,
        mmpbsa_discard: Optional[float] = None,
        mmpbsa_salt: Optional[float] = None,
        plot_only: bool = False,
    ) -> None:
        self.work_dir = Path(run_dir).expanduser().resolve()
        if not self.work_dir.is_dir():
            raise FileNotFoundError(f"Run directory was not found: {self.work_dir}")
        if selection not in {"general", "binding", "all"}:
            raise ValueError(f"Unsupported analysis selection: {selection}")
        self.selection = selection
        self.config = load_resume_config(self.work_dir, sim_type_name, ns_time)
        overrides = {
            "mmpbsa_cores": mmpbsa_cores,
            "mmpbsa_method": mmpbsa_method,
            "mmpbsa_frames": mmpbsa_frames,
            "mmpbsa_discard": mmpbsa_discard,
            "mmpbsa_salt": mmpbsa_salt,
        }
        self.config.update({key: value for key, value in overrides.items() if value is not None})
        self.config["binding_energy"] = selection in {"binding", "all"}
        self.config["plot_only"] = bool(plot_only)
        self.gmx_bin = check_gmx_installation()

    def validate(self) -> None:
        missing = [name for name in ANALYSIS_INPUTS if not (self.work_dir / name).is_file()]
        if missing:
            raise FileNotFoundError("Missing analysis inputs: " + ", ".join(missing))
        if self.selection in {"binding", "all"}:
            sim_type = str(self.config.get("sim_type"))
            if sim_type not in BINDING_SIM_TYPES:
                raise ValueError(
                    f"Binding-energy reanalysis is not available for simulation type {sim_type}"
                )
            if not (self.work_dir / "topol.top").is_file():
                raise FileNotFoundError("Missing binding-energy input: topol.top")

    def run(self) -> str:
        self.validate()
        step = AnalysisStep(self.config, self.gmx_bin)
        if self.selection == "general":
            if not step._run_general_analysis():
                raise RuntimeError("General dynamics analysis failed; inspect simulation.log")
        elif self.selection == "binding":
            if not step._run_binding_energy_analysis():
                raise RuntimeError("Binding-energy analysis failed; inspect analysis_metadata.json")
            step._generate_final_report()
        else:
            if not step._run_general_analysis():
                raise RuntimeError("General dynamics analysis failed; inspect simulation.log")
            if not step._run_binding_energy_analysis():
                raise RuntimeError("Binding-energy analysis failed; inspect analysis_metadata.json")
            step._generate_final_report()
        return self.selection
