"""Adapter for LightDock command-line execution.

LightDock ships two CLI entry points instead of one: `lightdock3_setup.py`
(generates starting swarm positions from a receptor/ligand pair) and
`lightdock3.py` (runs the GSO simulation over those swarms). Both write
processed copies of their inputs next to the input files themselves
(`lightdock_<name>.pdb`) and refuse to run again if that companion file
already exists -- so each call must happen in an isolated working
directory. That isolation is the caller's responsibility, not this
adapter's; see pipelines/docking/steps for how jobs get their own dirs.
"""

import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import List, Optional, Sequence


@dataclass(frozen=True)
class LightDockSetupResult:
    """Structured output of a `lightdock3_setup.py` run."""

    setup_file: str
    swarms: int
    log_file: str


@dataclass(frozen=True)
class LightDockRunResult:
    """Structured output of a `lightdock3.py` run."""

    swarm_dirs: List[str]
    log_file: str


class LightDockAdapter:
    """Wrapper around lightdock3_setup.py / lightdock3.py binary invocation."""

    DEFAULT_SETUP_EXECUTABLE = os.environ.get("LIGHTDOCK_SETUP_BIN", "lightdock3_setup.py")
    DEFAULT_RUN_EXECUTABLE = os.environ.get("LIGHTDOCK_RUN_BIN", "lightdock3.py")

    def __init__(self, setup_executable: Optional[str] = None, run_executable: Optional[str] = None):
        self.setup_executable = self._resolve_executable(setup_executable or self.DEFAULT_SETUP_EXECUTABLE)
        self.run_executable = self._resolve_executable(run_executable or self.DEFAULT_RUN_EXECUTABLE)

    @staticmethod
    def _resolve_executable(executable: str) -> str:
        """Resolve executable from an explicit path or PATH lookup."""
        if os.path.isabs(executable) and os.path.isfile(executable):
            return executable

        resolved = shutil.which(executable)
        return resolved or executable

    def validate(self) -> None:
        """Ensure both LightDock entry points exist and are executable."""
        for executable, name in (
            (self.setup_executable, "lightdock3_setup.py"),
            (self.run_executable, "lightdock3.py"),
        ):
            if not os.path.isfile(executable):
                raise FileNotFoundError(f"{name} not found: {executable}")
            if not os.access(executable, os.X_OK):
                raise RuntimeError(f"{name} is not executable: {executable}")

    def setup(
        self,
        receptor_pdb: str,
        ligand_pdb: str,
        cwd: str,
        restraints_file: Optional[str] = None,
        swarms: Optional[int] = None,
        glowworms: int = 20,
        use_anm: bool = True,
        anm_rec: int = 0,
        anm_lig: int = 10,
        noh: bool = True,
        noxt: bool = True,
        fixed_distance: Optional[float] = None,
        timeout_seconds: int = 1800,
    ) -> LightDockSetupResult:
        """Run lightdock3_setup.py and return the generated setup.json path.

        `cwd` must be a directory private to this job (see module docstring).
        `receptor_pdb`/`ligand_pdb` are resolved relative to `cwd` by
        LightDock itself if not absolute, so pass paths accordingly.

        The swarm starting-position calculation (SASA + occlusion filter
        over the whole receptor) is the other big per-call cost -- for a
        ~380-residue receptor it took ~485s on its own, separate from and
        larger than the ANM cost. LightDock reuses it for free if an
        `init/` directory already exists in `cwd` (its own documented
        behavior, see lightdock/prep/simulation.py::calculate_starting_positions).
        That calculation only depends on the receptor + restraints + the
        swarm-to-surface distance, NOT on which ligand is docked -- so
        passing a fixed `fixed_distance` (instead of letting it vary per
        ligand diameter) makes one precomputed `init/` folder valid to
        copy into every peptide's job directory before calling setup()
        again, skipping straight past that cost. See
        pipelines/peptide_docking/steps/lightdock_execution.py.

        anm_rec defaults to 0 (rigid receptor): LightDock recomputes the
        receptor's normal modes from scratch on every single setup() call
        with no caching, and the cost scales with receptor size -- for a
        ~380-residue receptor that dominated every peptide's setup time
        (minutes each) regardless of how small the peptide was. Docking
        here is meant as a fast pre-filter; receptor flexibility is
        handled properly afterwards by the MD stage (pipelines/dynamics),
        so paying that cost per-peptide during docking buys nothing.
        """
        cmd = [self.setup_executable, receptor_pdb, ligand_pdb, "-g", str(glowworms)]
        if noh:
            cmd.append("--noh")
        if noxt:
            cmd.append("--noxt")
        if use_anm:
            cmd += ["-anm", "-ar", str(anm_rec), "-al", str(anm_lig)]
        if restraints_file:
            cmd += ["--rst", restraints_file]
        if swarms:
            cmd += ["-s", str(swarms)]
        if fixed_distance is not None:
            cmd += ["-fd", str(fixed_distance)]

        log_file = os.path.join(cwd, "setup.log")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_seconds, cwd=cwd)
        with open(log_file, "w", encoding="utf-8") as handle:
            handle.write(result.stdout)
            handle.write(result.stderr)

        setup_file = os.path.join(cwd, "setup.json")
        if result.returncode != 0 or not os.path.isfile(setup_file):
            tail = (result.stderr or result.stdout or "").strip()[-800:]
            raise RuntimeError(f"lightdock3_setup.py failed for {ligand_pdb}: {tail or 'no output'}")

        swarm_dirs = [d for d in os.listdir(cwd) if d.startswith("swarm_")]
        return LightDockSetupResult(setup_file=setup_file, swarms=len(swarm_dirs), log_file=log_file)

    def run(
        self,
        setup_file: str,
        steps: int,
        cwd: str,
        cores: int = 1,
        swarm_list: Optional[Sequence[int]] = None,
        scoring_function: Optional[str] = None,
        timeout_seconds: int = 7200,
    ) -> LightDockRunResult:
        """Run lightdock3.py over the swarms generated by setup() and return their dirs."""
        cmd = [self.run_executable, setup_file, str(steps), "-c", str(cores)]
        if scoring_function:
            cmd += ["-s", scoring_function]
        if swarm_list:
            cmd += ["-l", *(str(s) for s in swarm_list)]

        # Cap each of lightdock3.py's own `-c N` worker PROCESSES to one BLAS/
        # OpenMP thread. -c already gives it N-way process parallelism for
        # the swarm computation; without this, each of those N processes can
        # ALSO spawn as many numpy/BLAS threads as it sees cores, so real CPU
        # usage becomes N x N instead of N -- invisible with a single job,
        # but exactly the oversubscription that made peptide preparation's
        # RDKit embedding fail more often under concurrency this session
        # (see peptide_preparation.py). One thread per worker process is the
        # right cap regardless of how many jobs run at once.
        env = dict(os.environ)
        for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[_var] = "1"

        log_file = os.path.join(cwd, "simulation.log")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_seconds, cwd=cwd, env=env)
        with open(log_file, "w", encoding="utf-8") as handle:
            handle.write(result.stdout)
            handle.write(result.stderr)

        combined_output = (result.stdout or "") + (result.stderr or "")
        if result.returncode != 0 or "lightdock] INFO: Finished." not in combined_output:
            tail = combined_output.strip()[-800:]
            raise RuntimeError(f"lightdock3.py failed for {setup_file}: {tail or 'no output'}")

        swarm_dirs = [
            os.path.join(cwd, d) for d in sorted(os.listdir(cwd))
            if d.startswith("swarm_") and os.path.isfile(os.path.join(cwd, d, f"gso_{steps}.out"))
        ]
        return LightDockRunResult(swarm_dirs=swarm_dirs, log_file=log_file)
