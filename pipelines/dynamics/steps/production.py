import os

from ..logger import get_step_logger
from ..mdrun_runner import run_mdrun_with_fallback
from utils.retry import SubprocessError, run_subprocess
from pipelines.dynamics.gmx_optimizer import get_cached_optimal_mdrun_flags

_GROMPP_TIMEOUT = 120


class ProductionStep:
    def __init__(self, config, gmx_bin):
        self.config      = config
        self.gmx_bin     = gmx_bin
        work             = self.config["work_dir"]
        self.npt_gro     = os.path.join(work, "npt.gro")
        self.npt_cpt     = os.path.join(work, "npt.cpt")
        self.topol       = os.path.join(work, "topol.top")
        self.md_mdp      = os.path.join(work, "md.mdp")
        self.md_tpr      = os.path.join(work, "md.tpr")
        self.output_base = "md"
        self.logger      = get_step_logger(
            __name__, os.path.join(work, "simulation.log")
        )

    def _create_production_mdp(self, nstlist: int) -> int:
        """Write md.mdp and return the total nsteps."""
        ns_time = float(self.config["ns_time"])
        nsteps  = int((ns_time * 1000) / 0.002)
        with open(self.md_mdp, "w") as fh:
            fh.write(
                f"integrator           = md\n"
                f"nsteps               = {nsteps}\n"
                f"dt                   = 0.002\n"
                f"nstxout              = 0\n"
                f"nstvout              = 0\n"
                f"nstfout              = 0\n"
                f"nstenergy            = 5000\n"
                f"nstlog               = 5000\n"
                f"nstxout-compressed   = 5000\n"
                f"compressed-x-grps    = System\n"
                f"continuation         = yes\n"
                f"constraint_algorithm = lincs\n"
                f"constraints          = h-bonds\n"
                f"cutoff-scheme        = Verlet\n"
                f"ns_type              = grid\n"
                f"nstlist              = {nstlist}\n"
                f"rcoulomb             = 1.0\n"
                f"rvdw                 = 1.0\n"
                f"coulombtype          = PME\n"
                f"pme_order            = 4\n"
                f"fourierspacing       = 0.16\n"
                f"tcoupl               = V-rescale\n"
                f"tc-grps              = System\n"
                f"tau_t                = 0.1\n"
                f"ref_t                = 300\n"
                f"pcoupl               = C-rescale\n"
                f"pcoupltype           = isotropic\n"
                f"tau_p                = 1.0\n"
                f"ref_p                = 1.0\n"
                f"compressibility      = 4.5e-5\n"
                f"pbc                  = xyz\n"
                f"gen_vel              = no\n"
            )
        return nsteps

    def run(self) -> None:
        opt     = get_cached_optimal_mdrun_flags(self.config)
        use_gpu = bool(self.config.get("gpu_ids"))

        mpi_tasks = int(self.config.get("mpi_tasks", 1))
        mpi_hosts = self.config.get("mpi_hosts", "")

        nsteps = self._create_production_mdp(opt["nstlist"])

        grompp_cmd = [
            self.gmx_bin, "grompp",
            "-f", self.md_mdp, "-c", self.npt_gro, "-t", self.npt_cpt,
            "-p", self.topol, "-o", self.md_tpr, "-maxwarn", "10",
        ]

        mdrun_base = [self.gmx_bin, "mdrun", "-v", "-deffnm", self.output_base]

        try:
            run_subprocess(
                grompp_cmd,
                timeout=_GROMPP_TIMEOUT,
                retries=2,
                logger=self.logger,
            )

            run_mdrun_with_fallback(
                cmd_gpu=mdrun_base + opt["gpu"],
                cmd_cpu=mdrun_base + opt["cpu"],
                work_dir=self.config["work_dir"],
                phase="Production MD",
                total_steps=nsteps,
                logger=self.logger,
                use_gpu=use_gpu,
                timeout=None,
                mpi_tasks=mpi_tasks,
                mpi_hosts=mpi_hosts,
            )

        except SubprocessError as exc:
            self.logger.error("Production grompp failed:\n%s", exc.stderr)
            raise RuntimeError(str(exc)) from exc
        except Exception:
            self.logger.exception("Unexpected error in ProductionStep")
            raise
