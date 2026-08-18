import os

from hpc.cluster.resource_detector import get_hardware_profile
from utils.progress import pipeline_bar as make_pipeline_bar
from pipelines.dynamics.steps.ligand_topology import LigandTopologyStep
from .utils import check_gmx_installation
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
        self.total_steps = 10

        # Detect hardware and expose GPU indices to all steps via config.
        # Steps read config["gpu_ids"] to decide whether to pass GPU flags
        # to mdrun; an empty list means CPU-only execution.
        hw = get_hardware_profile()
        self.config.setdefault("gpu_ids", hw.gpu_indices)
        if hw.has_gpu:
            gpu_names = ", ".join(
                f"GPU {g.index} {g.name} ({g.memory_total_gb} GB)"
                for g in hw.gpus
            )
            print(f" Hardware  : {gpu_names}")
        else:
            print(" Hardware  : No NVIDIA GPU detected — running on CPU")

    def execute(self) -> None:
        label = self.config["sim_type_label"].upper()
        print(f"\n CHEMLINK DYNAMICS ORCHESTRATOR: {label}")
        print("=" * 50)

        pipeline_bar = make_pipeline_bar(self.total_steps)
        current_step = "initialisation"
        try:
            # 0. COMPLEX BUILDING (sim types 3–6 only)
            current_step = "complex building"
            if self.config["sim_type"] in ["3", "4", "5", "6"]:
                ComplexBuilderStep(self.config).run()
            pipeline_bar.update(1)

            # 1. TOPOLOGY
            current_step = "topology"
            TopologyStep(self.config, self.gmx_bin).run()
            pipeline_bar.update(1)

            # 1.5 LIGAND TOPOLOGY (sim types 2 and 6 only)
            current_step = "ligand topology"
            if self.config["sim_type"] in ["2", "6"]:
                LigandTopologyStep(self.config, self.gmx_bin).run()
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

            # 5. EQUILIBRATION
            current_step = "equilibration"
            EquilibrationStep(self.config, self.gmx_bin).run()
            pipeline_bar.update(1)

            # 6. PRODUCTION
            current_step = "production"
            ProductionStep(self.config, self.gmx_bin).run()
            pipeline_bar.update(1)

            # 7. POST-PROCESSING
            current_step = "post-processing"
            PostProcessingStep(self.config, self.gmx_bin).run()
            pipeline_bar.update(1)

            # 8. ANALYSIS
            current_step = "analysis"
            # Rich and tqdm both reposition the terminal cursor. Pause the
            # persistent pipeline renderer while analysis uses nested tqdm bars.
            pipeline_bar.pause()
            AnalysisStep(self.config, self.gmx_bin).run()
            pipeline_bar.update(1)
            pipeline_bar.resume()

        except Exception as exc:
            print(f"\n[✗] CRITICAL ERROR in {current_step}: {exc}")
            raise
        finally:
            pipeline_bar.close()
