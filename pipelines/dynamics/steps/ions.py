import os
from utils.progress import step_bar

from ..logger import get_step_logger
from utils.retry import SubprocessError, run_subprocess

_GROMPP_TIMEOUT = 120
_GENION_TIMEOUT = 120


class IonsStep:
    def __init__(self, config, gmx_bin):
        self.config      = config
        self.gmx_bin     = gmx_bin
        work             = self.config["work_dir"]
        self.solvated_gro = os.path.join(work, "solvated.gro")
        self.topol       = os.path.join(work, "topol.top")
        self.ions_mdp    = os.path.join(work, "ions.mdp")
        self.ions_tpr    = os.path.join(work, "ions.tpr")
        self.ionized_gro = os.path.join(work, "ionized.gro")
        self.logger      = get_step_logger(__name__, os.path.join(work, "simulation.log"))

    def _create_ions_mdp(self) -> None:
        with open(self.ions_mdp, "w") as fh:
            fh.write(
                "integrator    = steep\n"
                "emtol         = 1000.0\n"
                "emstep        = 0.01\n"
                "nsteps        = 50000\n"
                "nstlist       = 1\n"
                "cutoff-scheme = Verlet\n"
                "ns_type       = grid\n"
                "coulombtype   = cutoff\n"
                "rcoulomb      = 1.0\n"
                "rvdw          = 1.0\n"
                "pbc           = xyz\n"
            )

    def run(self) -> None:
        self._create_ions_mdp()

        grompp_cmd = [
            self.gmx_bin, "grompp",
            "-f", self.ions_mdp,
            "-c", self.solvated_gro,
            "-p", self.topol,
            "-o", self.ions_tpr,
            "-maxwarn", "10",
        ]
        genion_cmd = [
            self.gmx_bin, "genion",
            "-s", self.ions_tpr,
            "-o", self.ionized_gro,
            "-p", self.topol,
            "-pname", "NA",
            "-nname", "CL",
            "-neutral",
        ]

        with step_bar("System Neutralization", 3) as pbar:
            try:
                run_subprocess(
                    grompp_cmd,
                    timeout=_GROMPP_TIMEOUT,
                    retries=2,
                    logger=self.logger,
                )
                pbar.update(1)

                run_subprocess(
                    genion_cmd,
                    timeout=_GENION_TIMEOUT,
                    retries=2,
                    input_data="SOL\n",
                    logger=self.logger,
                )
                pbar.update(1)

                if not os.path.exists(self.ionized_gro):
                    raise FileNotFoundError(
                        "GROMACS finished, but ionized file was not found."
                    )
                pbar.update(1)

            except SubprocessError as exc:
                self.logger.error("Neutralization failed:\n%s", exc.stderr)
                raise RuntimeError(str(exc)) from exc
            except Exception:
                self.logger.exception("Unexpected error in IonsStep")
                raise
