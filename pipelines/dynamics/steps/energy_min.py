import os
from typing import List, Optional
from utils.progress import step_bar

from ..logger import get_step_logger
from utils.retry import SubprocessError, run_subprocess

# grompp preprocessing is fast; mdrun minimisation caps at 50 000 steps
_GROMPP_TIMEOUT = 120
_MDRUN_TIMEOUT  = 3600


class EnergyMinStep:
    def __init__(self, config, gmx_bin):
        self.config     = config
        self.gmx_bin    = gmx_bin
        self.input_gro  = os.path.join(self.config["work_dir"], "ionized.gro")
        self.topol      = os.path.join(self.config["work_dir"], "topol.top")
        self.em_mdp     = os.path.join(self.config["work_dir"], "em.mdp")
        self.em_tpr     = os.path.join(self.config["work_dir"], "em.tpr")
        self.em_gro     = os.path.join(self.config["work_dir"], "em.gro")
        self.logger     = get_step_logger(
            __name__, os.path.join(self.config["work_dir"], "simulation.log")
        )

    def _create_em_mdp(self) -> None:
        with open(self.em_mdp, "w") as fh:
            fh.write(
                "integrator    = steep\n"
                "emtol         = 1000.0\n"
                "emstep        = 0.01\n"
                "nsteps        = 50000\n"
                "nstlist       = 1\n"
                "cutoff-scheme = Verlet\n"
                "ns_type       = grid\n"
                "coulombtype   = PME\n"
                "rcoulomb      = 1.0\n"
                "rvdw          = 1.0\n"
                "pbc           = xyz\n"
            )

    def _gpu_fallback(self, mdrun_cpu: List[str]):
        """Return an on_retry callback that switches to CPU on the first retry."""
        def _callback(attempt: int, exc: Exception) -> Optional[List[str]]:
            if attempt == 1:
                self.logger.warning(
                    "GPU energy minimisation failed — retrying on CPU only."
                )
            return mdrun_cpu if attempt == 1 else None
        return _callback

    def run(self) -> None:
        self._create_em_mdp()

        grompp_cmd = [
            self.gmx_bin, "grompp",
            "-f", self.em_mdp,
            "-c", self.input_gro,
            "-p", self.topol,
            "-o", self.em_tpr,
            "-maxwarn", "10",
        ]

        threads    = str(self.config.get("threads", 8))
        use_gpu    = bool(self.config.get("gpu_ids"))
        mdrun_base = [
            self.gmx_bin, "mdrun",
            "-v", "-deffnm", "em",
            "-ntomp", threads,
            "-pin", "on",
        ]
        mdrun_gpu = mdrun_base + ["-nb", "gpu"]
        mdrun_cpu = mdrun_base

        with step_bar("Energy Minimization", 2) as pbar:
            try:
                run_subprocess(
                    grompp_cmd,
                    timeout=_GROMPP_TIMEOUT,
                    retries=2,
                    logger=self.logger,
                )
                pbar.update(1)

                run_subprocess(
                    mdrun_gpu if use_gpu else mdrun_cpu,
                    timeout=_MDRUN_TIMEOUT,
                    retries=2 if use_gpu else 1,
                    cwd=self.config["work_dir"],
                    logger=self.logger,
                    on_retry=self._gpu_fallback(mdrun_cpu) if use_gpu else None,
                )
                pbar.update(1)

                if not os.path.exists(self.em_gro):
                    raise FileNotFoundError(
                        "Minimisation finished but output file was not found."
                    )

            except SubprocessError as exc:
                self.logger.error("Energy Minimization failed:\n%s", exc.stderr)
                raise RuntimeError(str(exc)) from exc
            except Exception as exc:
                self.logger.exception("Unexpected error in EnergyMinStep")
                raise
