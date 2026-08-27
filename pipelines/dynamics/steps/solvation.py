import os
from utils.progress import step_bar

from ..logger import get_step_logger
from utils.retry import SubprocessError, run_subprocess

_EDITCONF_TIMEOUT = 120
_SOLVATE_TIMEOUT  = 300


class SolvationStep:
    def __init__(self, config, gmx_bin):
        self.config      = config
        self.gmx_bin     = gmx_bin
        work             = self.config["work_dir"]
        gro_name         = self.config.get("current_gro", "processed.gro")
        self.input_gro   = os.path.join(work, gro_name)
        self.topol       = os.path.join(work, "topol.top")
        self.boxed_gro   = os.path.join(work, "boxed.gro")
        self.solvated_gro = os.path.join(work, "solvated.gro")
        self.logger      = get_step_logger(__name__, os.path.join(work, "simulation.log"))

    def run(self) -> None:
        editconf_cmd = [
            self.gmx_bin, "editconf",
            "-f", self.input_gro,
            "-o", self.boxed_gro,
            "-bt", "cubic",
            "-d", "1.0",
            "-c",
        ]
        solvate_cmd = [
            self.gmx_bin, "solvate",
            "-cp", self.boxed_gro,
            "-cs", "spc216.gro",
            "-o", self.solvated_gro,
            "-p", self.topol,
        ]

        with step_bar("System Solvation", 2) as pbar:
            try:
                run_subprocess(
                    editconf_cmd,
                    timeout=_EDITCONF_TIMEOUT,
                    retries=2,
                    logger=self.logger,
                )
                pbar.update(1)

                run_subprocess(
                    solvate_cmd,
                    timeout=_SOLVATE_TIMEOUT,
                    retries=2,
                    logger=self.logger,
                )
                pbar.update(1)

                if not os.path.exists(self.solvated_gro):
                    raise FileNotFoundError(
                        "GROMACS execution finished, but solvated file was not found."
                    )

            except SubprocessError as exc:
                self.logger.error("Solvation failed:\n%s", exc.stderr)
                raise RuntimeError(str(exc)) from exc
            except Exception:
                self.logger.exception("Unexpected error in SolvationStep")
                raise
