import os
from typing import List

from ..logger import get_step_logger
from ..mdrun_runner import run_mdrun_with_fallback
from utils.retry import SubprocessError, run_subprocess
from pipelines.dynamics.gmx_optimizer import get_optimal_mdrun_flags

_GROMPP_TIMEOUT = 120
_MDRUN_TIMEOUT  = 7200   # 2 h per phase

_NVT_STEPS = 50_000
_NPT_STEPS = 50_000


class EquilibrationStep:
    def __init__(self, config, gmx_bin):
        self.config   = config
        self.gmx_bin  = gmx_bin
        work          = self.config["work_dir"]
        self.em_gro   = os.path.join(work, "em.gro")
        self.topol    = os.path.join(work, "topol.top")
        self.nvt_mdp  = os.path.join(work, "nvt.mdp")
        self.nvt_tpr  = os.path.join(work, "nvt.tpr")
        self.nvt_gro  = os.path.join(work, "nvt.gro")
        self.nvt_cpt  = os.path.join(work, "nvt.cpt")
        self.npt_mdp  = os.path.join(work, "npt.mdp")
        self.npt_tpr  = os.path.join(work, "npt.tpr")
        self.npt_gro  = os.path.join(work, "npt.gro")
        self.logger   = get_step_logger(
            __name__, os.path.join(work, "simulation.log")
        )

    def _create_nvt_mdp(self, nstlist: int) -> None:
        with open(self.nvt_mdp, "w") as fh:
            fh.write(
                "integrator           = md\n"
                f"nsteps               = {_NVT_STEPS}\n"
                "dt                   = 0.002\n"
                "nstxout              = 5000\n"
                "nstvout              = 5000\n"
                "nstenergy            = 5000\n"
                "nstlog               = 5000\n"
                "continuation         = no\n"
                "constraint_algorithm = lincs\n"
                "constraints          = h-bonds\n"
                "cutoff-scheme        = Verlet\n"
                "ns_type              = grid\n"
                f"nstlist              = {nstlist}\n"
                "rcoulomb             = 1.0\n"
                "rvdw                 = 1.0\n"
                "coulombtype          = PME\n"
                "pme_order            = 4\n"
                "fourierspacing       = 0.16\n"
                "tcoupl               = V-rescale\n"
                "tc-grps              = System\n"
                "tau_t                = 0.1\n"
                "ref_t                = 300\n"
                "pcoupl               = no\n"
                "pbc                  = xyz\n"
                "gen_vel              = yes\n"
                "gen_temp             = 300\n"
                "gen_seed             = -1\n"
            )

    def _create_npt_mdp(self, nstlist: int) -> None:
        with open(self.npt_mdp, "w") as fh:
            fh.write(
                "integrator           = md\n"
                f"nsteps               = {_NPT_STEPS}\n"
                "dt                   = 0.002\n"
                "nstxout              = 5000\n"
                "nstvout              = 5000\n"
                "nstenergy            = 5000\n"
                "nstlog               = 5000\n"
                "continuation         = yes\n"
                "constraint_algorithm = lincs\n"
                "constraints          = h-bonds\n"
                "cutoff-scheme        = Verlet\n"
                "ns_type              = grid\n"
                f"nstlist              = {nstlist}\n"
                "rcoulomb             = 1.0\n"
                "rvdw                 = 1.0\n"
                "coulombtype          = PME\n"
                "pme_order            = 4\n"
                "fourierspacing       = 0.16\n"
                "tcoupl               = V-rescale\n"
                "tc-grps              = System\n"
                "tau_t                = 0.1\n"
                "ref_t                = 300\n"
                "pcoupl               = C-rescale\n"
                "pcoupltype           = isotropic\n"
                "tau_p                = 1.0\n"
                "ref_p                = 1.0\n"
                "compressibility      = 4.5e-5\n"
                "refcoord_scaling     = com\n"
                "pbc                  = xyz\n"
                "gen_vel              = no\n"
            )

    def _grompp(self, flags: List[str]) -> None:
        run_subprocess(
            flags,
            timeout=_GROMPP_TIMEOUT,
            retries=2,
            logger=self.logger,
        )

    def run(self) -> None:
        opt = get_optimal_mdrun_flags(self.config)
        use_gpu = bool(self.config.get("gpu_ids"))

        mpi_tasks = int(self.config.get("mpi_tasks", 1))
        mpi_hosts = self.config.get("mpi_hosts", "")

        for msg in opt["diagnostics"]:
            if "WARNING" in msg or "PERF" in msg:
                self.logger.warning("gmx_optimizer: %s", msg)
            else:
                self.logger.info("gmx_optimizer: %s", msg)

        nstlist = opt["nstlist"]

        try:
            self._run_nvt(opt, use_gpu, mpi_tasks, mpi_hosts)
            self._run_npt(opt, use_gpu, mpi_tasks, mpi_hosts)

            if not os.path.exists(self.npt_gro):
                raise FileNotFoundError(
                    "Equilibration finished but npt.gro was not found."
                )

        except SubprocessError as exc:
            self.logger.error("Equilibration grompp failed:\n%s", exc.stderr)
            raise RuntimeError(str(exc)) from exc
        except Exception:
            self.logger.exception("Unexpected error in EquilibrationStep")
            raise

    def _run_nvt(self, opt, use_gpu: bool, mpi_tasks: int, mpi_hosts: str) -> None:
        self._create_nvt_mdp(opt["nstlist"])
        self._grompp([
            self.gmx_bin, "grompp",
            "-f", self.nvt_mdp, "-c", self.em_gro, "-r", self.em_gro,
            "-p", self.topol, "-o", self.nvt_tpr, "-maxwarn", "10",
        ])
        base = [self.gmx_bin, "mdrun", "-v", "-deffnm", "nvt"]
        run_mdrun_with_fallback(
            cmd_gpu=base + opt["gpu"], cmd_cpu=base + opt["cpu"],
            work_dir=self.config["work_dir"], phase="NVT equilibration",
            total_steps=_NVT_STEPS, logger=self.logger, use_gpu=use_gpu,
            timeout=_MDRUN_TIMEOUT, mpi_tasks=mpi_tasks, mpi_hosts=mpi_hosts,
        )

    def _run_npt(self, opt, use_gpu: bool, mpi_tasks: int, mpi_hosts: str) -> None:
        self._create_npt_mdp(opt["nstlist"])
        self._grompp([
            self.gmx_bin, "grompp",
            "-f", self.npt_mdp, "-c", self.nvt_gro, "-r", self.nvt_gro,
            "-t", self.nvt_cpt, "-p", self.topol,
            "-o", self.npt_tpr, "-maxwarn", "10",
        ])
        base = [self.gmx_bin, "mdrun", "-v", "-deffnm", "npt"]
        run_mdrun_with_fallback(
            cmd_gpu=base + opt["gpu"], cmd_cpu=base + opt["cpu"],
            work_dir=self.config["work_dir"], phase="NPT equilibration",
            total_steps=_NPT_STEPS, logger=self.logger, use_gpu=use_gpu,
            timeout=_MDRUN_TIMEOUT, mpi_tasks=mpi_tasks, mpi_hosts=mpi_hosts,
        )

    def run_npt_only(self) -> None:
        """Start NPT after a resumed or already completed NVT phase."""
        opt = get_optimal_mdrun_flags(self.config)
        self._run_npt(
            opt, bool(self.config.get("gpu_ids")),
            int(self.config.get("mpi_tasks", 1)), self.config.get("mpi_hosts", ""),
        )
        if not os.path.exists(self.npt_gro):
            raise FileNotFoundError("NPT equilibration finished but npt.gro was not found.")
