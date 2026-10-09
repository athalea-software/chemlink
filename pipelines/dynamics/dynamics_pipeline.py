import os
import time
from datetime import datetime
from pathlib import Path

from hpc.cluster.resource_detector import get_hardware_profile
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from utils.progress import pipeline_bar as make_pipeline_bar
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
        self.console = Console(highlight=False)
        self._started_at = time.monotonic()
        self._started_wall = datetime.now()
        self._show_resource_card()

    @property
    def replicas(self) -> int:
        return replica_count(self.config)

    def _show_resource_card(self) -> None:
        """Show only the compute resources selected for this simulation."""
        resources = self.resources
        table = Table.grid(padding=(0, 2))
        table.add_column(style="bold cyan", justify="right")
        table.add_column()
        if resources.get("gpu_name"):
            memory_gb = resources["gpu_memory_total_mb"] / 1024
            table.add_row(
                "GPU",
                f"GPU {self.config['gpu_ids'][0]} · {resources['gpu_name']} · "
                f"{memory_gb:.1f} GB · sm_{resources['gpu_arch']}",
            )
        else:
            table.add_row("GPU", "No disponible · ejecución por CPU")
        table.add_row(
            "CPU",
            f"{resources['total_cpu_count']} núcleos · "
            f"{resources['p_core_count']} P-cores · {resources['ntomp']} hilos por simulación",
        )
        self.console.print(
            Panel(
                table,
                title="[bold cyan]Recursos de cómputo[/bold cyan]",
                border_style="cyan",
                padding=(0, 1),
            )
        )

    def _write_final_summary(self, status: str, current_step: str, error: Exception | None) -> str:
        """Write a compact final Markdown report for hardware and phase execution."""
        work_dir = Path(self.config["work_dir"])
        work_dir.mkdir(parents=True, exist_ok=True)
        output = work_dir / "simulation_summary.md"
        resources = self.resources
        elapsed = time.monotonic() - self._started_at
        gpu_flags = " ".join(resources.get("gpu", [])) or "CPU only"
        cpu_flags = " ".join(resources.get("cpu", []))

        phases = [("Minimización", "", "em.log", "em.gro")]
        md_phases = [
            ("Equilibración NVT", "nvt.log", "nvt.gro"),
            ("Equilibración NPT", "npt.log", "npt.gro"),
            ("Producción", "md.log", "md.gro"),
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
            phase_status = "Completada" if artifact.is_file() else "Pendiente o interrumpida"
            log_status = f"`{log_ref}`" if log_path.is_file() else "No generado"
            phase_rows.append(
                f"| {label} | {phase_status} | {log_status} |"
            )

        gpu_description = "No disponible"
        if resources.get("gpu_name"):
            gpu_description = (
                f"GPU {self.config['gpu_ids'][0]} — {resources['gpu_name']} "
                f"(sm_{resources['gpu_arch']}, "
                f"{resources['gpu_memory_total_mb']} MB total, "
                f"{resources['gpu_memory_free_mb']} MB libres al iniciar)"
            )

        diagnostics = "\n".join(f"- {item}" for item in resources.get("diagnostics", []))
        error_section = f"\n## Error final\n\n`{error}`\n" if error else ""
        output.write_text(
            "# Resumen de simulación ChemLink\n\n"
            f"- **Estado:** {status}\n"
            f"- **Sistema:** {self.config.get('sim_type_label', 'N/D')}\n"
            f"- **Tiempo solicitado:** {self.config.get('ns_time', 'N/D')} ns\n"
            f"- **Réplicas:** {self.replicas}\n"
            f"- **Inicio:** {self._started_wall.isoformat(timespec='seconds')}\n"
            f"- **Duración de ejecución:** {elapsed:.1f} s\n"
            f"- **Última etapa:** {current_step}\n\n"
            "## Recursos utilizados\n\n"
            f"- **GPU:** {gpu_description}\n"
            f"- **CPU:** {resources['total_cpu_count']} núcleos totales, "
            f"{resources['p_core_count']} P-cores detectados\n"
            f"- **Hilos OpenMP:** {resources['ntomp']}\n"
            f"- **Capacidad concurrente calculada:** {resources['max_concurrent_sims']}\n\n"
            "## Configuración de GROMACS\n\n"
            f"- **Flags GPU:** `{gpu_flags}`\n"
            f"- **Fallback CPU:** `{cpu_flags}`\n"
            f"- **nstlist recomendado:** {resources['nstlist']}\n"
            f"- **Actualización en GPU:** {resources['update_gpu']}\n\n"
            "## Etapas principales\n\n"
            "| Etapa | Estado | Log detallado |\n"
            "|---|---|---|\n"
            + "\n".join(phase_rows)
            + "\n\n## Diagnóstico de recursos\n\n"
            + diagnostics
            + "\n\n## Trazabilidad\n\n"
            "El detalle técnico completo permanece en `simulation.log`; los logs nativos de "
            "GROMACS se conservan como `em.log`, `nvt.log`, `npt.log` y `md.log`.\n"
            + error_section,
            encoding="utf-8",
        )
        return str(output)

    def execute(self) -> None:
        label = self.config["sim_type_label"].upper()
        print(f"\n CHEMLINK DYNAMICS ORCHESTRATOR: {label}")
        print("=" * 50)

        pipeline_bar = make_pipeline_bar(self.total_steps)
        current_step = "initialisation"
        # Set by _run_md_stages so failures name the exact stage and replica.
        self._current_step: str | None = None
        status = "En ejecución"
        final_error = None
        try:
            # 0. COMPLEX BUILDING (sim types 3–6 only)
            current_step = "complex building"
            if self.config["sim_type"] in ["3", "4", "5", "6"]:
                ComplexBuilderStep(self.config).run()
            pipeline_bar.update(1)

            # 1. TOPOLOGY
            current_step = "topology"
            if self.config["sim_type"] != LIGAND_ONLY_SIM_TYPE:
                TopologyStep(self.config, self.gmx_bin).run()
            pipeline_bar.update(1)

            # 1.5 LIGAND TOPOLOGY (sim types 2 and 6 only)
            current_step = "ligand topology"
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
            SolvationStep(self.config, self.gmx_bin).run()
            pipeline_bar.update(1)

            # 3. IONS
            current_step = "ions"
            IonsStep(self.config, self.gmx_bin).run()
            pipeline_bar.update(1)

            # 4. ENERGY MINIMISATION
            current_step = "energy minimisation"
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
                    print(f"\n▶ Réplica {index}/{self.replicas} ({tag})")
                    replica_config = prepare_replica(
                        Path(self.config["work_dir"]), index, self.config
                    )
                    self._run_md_stages(replica_config, pipeline_bar, tag)

            status = "Completada"

        except Exception as exc:
            current_step = self._current_step or current_step
            status = "Fallida"
            final_error = exc
            print(f"\n[✗] CRITICAL ERROR in {current_step}: {exc}")
            raise
        finally:
            pipeline_bar.close()
            current_step = self._current_step or current_step
            summary_path = self._write_final_summary(status, current_step, final_error)
            self.console.print(f"[dim]Resumen final:[/dim] [cyan]{summary_path}[/cyan]")
            if self.replicas > 1:
                replicas_path = write_replicas_summary(
                    Path(self.config["work_dir"]), self.replicas
                )
                self.console.print(
                    f"[dim]Resumen de réplicas:[/dim] [cyan]{replicas_path}[/cyan]"
                )

    def _run_md_stages(self, config: dict, pipeline_bar, tag: str) -> None:
        """Run equilibration through analysis in ``config["work_dir"]``."""
        suffix = f" ({tag})" if tag else ""

        self._current_step = f"equilibration{suffix}"
        EquilibrationStep(config, self.gmx_bin).run()
        pipeline_bar.update(1)

        self._current_step = f"production{suffix}"
        ProductionStep(config, self.gmx_bin).run()
        pipeline_bar.update(1)

        self._current_step = f"post-processing{suffix}"
        PostProcessingStep(config, self.gmx_bin).run()
        pipeline_bar.update(1)

        self._current_step = f"analysis{suffix}"
        # Rich and tqdm both reposition the terminal cursor. Pause the
        # persistent pipeline renderer while analysis uses nested tqdm bars.
        pipeline_bar.pause()
        AnalysisStep(config, self.gmx_bin).run()
        pipeline_bar.update(1)
        pipeline_bar.resume()
