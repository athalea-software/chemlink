import time
from datetime import datetime
from pathlib import Path

from hpc.cluster.resource_detector import get_hardware_profile
from rich.padding import Padding
from utils.progress import pipeline_bar as make_pipeline_bar
from utils.style import card, section, status_line
from utils.theme import format_duration, get_console
from pipelines.dynamics.steps.ligand_topology import LigandTopologyStep
from pipelines.dynamics.steps.cofactor_topology import CofactorTopologyStep
from pipelines.dynamics.gmx_optimizer import get_cached_optimal_mdrun_flags
from .replicas import (
    prepare_replica,
    replica_count,
    replica_dir_name,
    write_replicas_summary,
)
from .utils import check_gmx_installation

LIGAND_ONLY_SIM_TYPE = "7"
from .steps import (
    ComplexBuilderStep,
    TopologyStep,
    SolvationStep,
    IonsStep,
    EnergyMinStep,
    EquilibrationStep,
    ProductionStep,
    PostProcessingStep,
    AnalysisStep,
)


class DynamicsPipeline:
    def __init__(self, config: dict) -> None:
        self.config   = config
        self.gmx_bin  = check_gmx_installation()
        # 6 shared preparation steps + 4 per replica (equilibration,
        # production, post-processing, analysis).
        self.total_steps = 6 + 4 * self.replicas

        # Detect hardware and expose GPU indices to all steps via config.
        # Steps read config["gpu_ids"] to decide whether to pass GPU flags
        # to mdrun; an empty list means CPU-only execution.
        hw = get_hardware_profile()
        self.config.setdefault("gpu_ids", hw.gpu_indices)
        self.resources = get_cached_optimal_mdrun_flags(self.config)
        self.console = get_console()
        self._started_at = time.monotonic()
        self._started_wall = datetime.now()
        self._show_resource_card()

    @property
    def replicas(self) -> int:
        return replica_count(self.config)

    def _show_resource_card(self) -> None:
        """Show only the compute resources selected for this simulation."""
        resources = self.resources
        if resources.get("gpu_name"):
            memory_gb = resources["gpu_memory_total_mb"] / 1024
            gpu_line = (
                f"GPU {self.config['gpu_ids'][0]} · {resources['gpu_name']} · "
                f"{memory_gb:.1f} GB · sm_{resources['gpu_arch']}"
            )
        else:
            gpu_line = "[cl.warn.soft]none detected[/] · running on CPU"
        cpu_line = (
            f"{resources['total_cpu_count']} cores · "
            f"{resources['p_core_count']} P-cores · {resources['ntomp']} threads per simulation"
        )
        self.console.print()
        self.console.print(Padding(
            card([("GPU", gpu_line), ("CPU", cpu_line)], title="Compute resources"),
            (0, 0, 0, 2),
        ))

    def _write_final_summary(self, status: str, current_step: str, error: Exception | None) -> str:
        """Write a compact final Markdown report for hardware and phase execution."""
        work_dir = Path(self.config["work_dir"])
        work_dir.mkdir(parents=True, exist_ok=True)
        output = work_dir / "simulation_summary.md"
        resources = self.resources
        elapsed = time.monotonic() - self._started_at
        gpu_flags = " ".join(resources.get("gpu", [])) or "CPU only"
        cpu_flags = " ".join(resources.get("cpu", []))

        phases = [("Minimization", "", "em.log", "em.gro")]
        md_phases = [
            ("NVT equilibration", "nvt.log", "nvt.gro"),
            ("NPT equilibration", "npt.log", "npt.gro"),
            ("Production", "md.log", "md.gro"),
        ]
        if self.replicas > 1:
            for index in range(1, self.replicas + 1):
                prefix = replica_dir_name(index)
                phases += [
                    (f"{label} ({prefix})", prefix, log_name, artifact_name)
                    for label, log_name, artifact_name in md_phases
                ]
        else:
            phases += [(label, "", log_name, artifact_name)
                       for label, log_name, artifact_name in md_phases]
        phase_rows = []
        for label, subdir, log_name, artifact_name in phases:
            log_path = work_dir / subdir / log_name
            artifact = work_dir / subdir / artifact_name
            log_ref = f"{subdir}/{log_name}" if subdir else log_name
            phase_status = "Completed" if artifact.is_file() else "Pending or interrupted"
            log_status = f"`{log_ref}`" if log_path.is_file() else "Not generated"
            phase_rows.append(
                f"| {label} | {phase_status} | {log_status} |"
            )

        gpu_description = "Not available"
        if resources.get("gpu_name"):
            gpu_description = (
                f"GPU {self.config['gpu_ids'][0]} — {resources['gpu_name']} "
                f"(sm_{resources['gpu_arch']}, "
                f"{resources['gpu_memory_total_mb']} MB total, "
                f"{resources['gpu_memory_free_mb']} MB free at start)"
            )

        diagnostics = "\n".join(f"- {item}" for item in resources.get("diagnostics", []))
        error_section = f"\n## Final error\n\n`{error}`\n" if error else ""
        output.write_text(
            "# ChemLink simulation summary\n\n"
            f"- **Status:** {status}\n"
            f"- **System:** {self.config.get('sim_type_label', 'N/A')}\n"
            f"- **Requested time:** {self.config.get('ns_time', 'N/A')} ns\n"
            f"- **Replicas:** {self.replicas}\n"
            f"- **Started:** {self._started_wall.isoformat(timespec='seconds')}\n"
            f"- **Run time:** {elapsed:.1f} s\n"
            f"- **Last stage:** {current_step}\n\n"
            "## Resources used\n\n"
            f"- **GPU:** {gpu_description}\n"
            f"- **CPU:** {resources['total_cpu_count']} cores in total, "
            f"{resources['p_core_count']} P-cores detected\n"
            f"- **OpenMP threads:** {resources['ntomp']}\n"
            f"- **Computed concurrent capacity:** {resources['max_concurrent_sims']}\n\n"
            "## GROMACS configuration\n\n"
            f"- **GPU flags:** `{gpu_flags}`\n"
            f"- **CPU fallback:** `{cpu_flags}`\n"
            f"- **Recommended nstlist:** {resources['nstlist']}\n"
            f"- **GPU update:** {resources['update_gpu']}\n\n"
            "## Main stages\n\n"
            "| Stage | Status | Detailed log |\n"
            "|---|---|---|\n"
            + "\n".join(phase_rows)
            + "\n\n## Resource diagnostics\n\n"
            + diagnostics
            + "\n\n## Traceability\n\n"
            "The full technical detail stays in `simulation.log`; the native GROMACS logs "
            "are kept as `em.log`, `nvt.log`, `npt.log` and `md.log`.\n"
            + error_section,
            encoding="utf-8",
        )
        return str(output)

    def execute(self) -> None:
        label = self.config["sim_type_label"]
        replicas = f"{self.replicas} replicas" if self.replicas > 1 else "1 replica"
        section(self.console, "Molecular dynamics", f"{label} · {self.config.get('ns_time', '?')} ns · {replicas}")
        self.console.print()

        pipeline_bar = make_pipeline_bar(self.total_steps, desc="MD pipeline")
        current_step = "initialisation"
        # Set by _run_md_stages so failures name the exact stage and replica.
        self._current_step: str | None = None
        status = "Running"
        final_error = None
        try:
            # 0. COMPLEX BUILDING (sim types 3–6 only)
            current_step = "complex building"
            pipeline_bar.set_stage(current_step)
            if self.config["sim_type"] in ["3", "4", "5", "6"]:
                ComplexBuilderStep(self.config).run()
            pipeline_bar.update(1)

            # 1. TOPOLOGY
            current_step = "topology"
            pipeline_bar.set_stage(current_step)
            if self.config["sim_type"] != LIGAND_ONLY_SIM_TYPE:
                TopologyStep(self.config, self.gmx_bin).run()
            pipeline_bar.update(1)

            # 1.5 LIGAND TOPOLOGY (sim types 2 and 6 only)
            current_step = "ligand topology"
            pipeline_bar.set_stage(current_step)
            if self.config["sim_type"] == LIGAND_ONLY_SIM_TYPE:
                LigandTopologyStep(self.config, self.gmx_bin, standalone=True).run()
                self.config["current_gro"] = "processed.gro"
            elif self.config["sim_type"] in ["2", "6"]:
                LigandTopologyStep(self.config, self.gmx_bin).run()
                if self.config.get("cofactor_pdb"):
                    current_step = "cofactor topology"
                    CofactorTopologyStep(self.config, self.gmx_bin).run()
                    current_step = "ligand topology"
                self.config["current_gro"] = "complex.gro"
            else:
                self.config["current_gro"] = "processed.gro"
            pipeline_bar.update(1)

            # 2. SOLVATION
            current_step = "solvation"
            pipeline_bar.set_stage(current_step)
            SolvationStep(self.config, self.gmx_bin).run()
            pipeline_bar.update(1)

            # 3. IONS
            current_step = "ions"
            pipeline_bar.set_stage(current_step)
            IonsStep(self.config, self.gmx_bin).run()
            pipeline_bar.update(1)

            # 4. ENERGY MINIMISATION
            current_step = "energy minimisation"
            pipeline_bar.set_stage(current_step)
            EnergyMinStep(self.config, self.gmx_bin).run()
            pipeline_bar.update(1)

            # 5–8. EQUILIBRATION → PRODUCTION → POST-PROCESSING → ANALYSIS
            if self.replicas == 1:
                self._run_md_stages(self.config, pipeline_bar, "")
            else:
                # Replicas run back to back; each starts from the shared em.gro
                # with freshly generated NVT velocities.
                for index in range(1, self.replicas + 1):
                    tag = replica_dir_name(index)
                    self._current_step = f"{tag} setup"
                    section(self.console, f"Replica {index}/{self.replicas}", tag, glyph="▸")
                    replica_config = prepare_replica(
                        Path(self.config["work_dir"]), index, self.config
                    )
                    self._run_md_stages(replica_config, pipeline_bar, tag)

            status = "Completed"

        except BaseException as exc:
            current_step = self._current_step or current_step
            status = "Failed"
            final_error = exc
            if isinstance(exc, KeyboardInterrupt):
                pipeline_bar.tracker.cancel(f"interrupted during {current_step}")
            else:
                pipeline_bar.fail(f"{current_step}: {exc}")
            raise
        finally:
            pipeline_bar.close()
            current_step = self._current_step or current_step
            summary_path = self._write_final_summary(status, current_step, final_error)
            self.console.print()
            if final_error is None:
                self.console.print(status_line(
                    "ok", f"Simulation finished in {format_duration(time.monotonic() - self._started_at)}",
                ))
            else:
                self.console.print(status_line("err", f"Simulation stopped at [bold]{current_step}[/]"))
            self.console.print(status_line("info", f"Summary   [cl.path]{summary_path}[/]"))
            log_path = Path(self.config["work_dir"]) / "simulation.log"
            if log_path.is_file():
                self.console.print(status_line("info", f"Full log  [cl.path]{log_path}[/]"))
            if self.replicas > 1:
                replicas_path = write_replicas_summary(
                    Path(self.config["work_dir"]), self.replicas
                )
                self.console.print(status_line("info", f"Replicas  [cl.path]{replicas_path}[/]"))

    def _run_md_stages(self, config: dict, pipeline_bar, tag: str) -> None:
        """Run equilibration through analysis in ``config["work_dir"]``."""
        suffix = f" ({tag})" if tag else ""

        self._current_step = f"equilibration{suffix}"
        pipeline_bar.set_stage(self._current_step)
        EquilibrationStep(config, self.gmx_bin).run()
        pipeline_bar.update(1)

        self._current_step = f"production{suffix}"
        pipeline_bar.set_stage(self._current_step)
        ProductionStep(config, self.gmx_bin).run()
        pipeline_bar.update(1)

        self._current_step = f"post-processing{suffix}"
        pipeline_bar.set_stage(self._current_step)
        PostProcessingStep(config, self.gmx_bin).run()
        pipeline_bar.update(1)

        self._current_step = f"analysis{suffix}"
        pipeline_bar.set_stage(self._current_step)
        AnalysisStep(config, self.gmx_bin).run()
        pipeline_bar.update(1)
