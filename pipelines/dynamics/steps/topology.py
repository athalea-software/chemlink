import os
from pathlib import Path
import shutil
import json
from tqdm import tqdm
from pdbfixer import PDBFixer
from openmm.app import PDBFile

from ..logger import get_step_logger
from utils.retry import SubprocessError, run_subprocess

_PDB2GMX_TIMEOUT = 300


def force_field_search_dirs(gmx_bin: str, custom_dir: str | None = None) -> list[Path]:
    """Return ordered GROMACS force-field roots, including ChemFusion layouts."""
    candidates = [
        custom_dir,
        os.environ.get("CHEMLINK_FORCEFIELD_DIR"),
        os.environ.get("GMXLIB"),
        str(Path(__file__).resolve().parents[3] / "forcefields"),
        "/usr/local/gromacs/share/gromacs/top",
        "/opt/gromacs/share/gromacs/top",
        "/usr/share/gromacs/top",
    ]
    resolved_gmx = shutil.which(gmx_bin) or gmx_bin
    if resolved_gmx:
        gmx_path = Path(resolved_gmx).resolve()
        candidates.insert(3, str(gmx_path.parent.parent / "share/gromacs/top"))
    result: list[Path] = []
    for candidate in candidates:
        if candidate:
            path = Path(candidate).expanduser().resolve()
            if path.is_dir() and path not in result:
                result.append(path)
    return result


def resolve_force_field(requested: str, search_dirs: list[Path]) -> tuple[str, Path]:
    """Resolve a field name; ``chemfusion`` means the newest CHARMM36 variant."""
    aliases = {"chemfusion", "charmm36m"}
    if requested.lower() not in aliases:
        for root in search_dirs:
            if (root / f"{requested}.ff").is_dir():
                return requested, root
        raise FileNotFoundError(f"GROMACS force field '{requested}.ff' was not found")

    exact_names = ("charmm36m", "charmm36-jul2022")
    for name in exact_names:
        for root in search_dirs:
            if (root / f"{name}.ff").is_dir():
                return name, root
    matches: list[tuple[str, Path]] = []
    for root in search_dirs:
        for directory in root.glob("charmm36*.ff"):
            if directory.is_dir():
                matches.append((directory.name[:-3], root))
    if matches:
        return sorted(matches, key=lambda item: item[0], reverse=True)[0]
    searched = ", ".join(str(path) for path in search_dirs) or "no valid directories"
    raise FileNotFoundError(
        "ChemFusion's CHARMM36m force field was not found. "
        "Use --force-field-dir DIR or CHEMLINK_FORCEFIELD_DIR. Searched: " + searched
    )


class TopologyStep:
    def __init__(self, config, gmx_bin):
        self.config        = config
        self.gmx_bin       = gmx_bin
        self.pdb_input_abs = os.path.abspath(self.config["pdb_input"])
        self.output_gro    = "processed.gro"
        self.logger        = get_step_logger(
            __name__, os.path.join(self.config["work_dir"], "simulation.log")
        )

    def _repair_pdb(self, input_pdb: str, output_pdb_path: str) -> bool:
        with tqdm(total=4, desc="  └─ Repairing Structure", leave=False) as pbar:
            try:
                fixer = PDBFixer(filename=input_pdb)
                pbar.update(1)
                fixer.findMissingResidues()
                pbar.update(1)
                fixer.findMissingAtoms()
                fixer.addMissingAtoms()
                pbar.update(1)
                fixer.addMissingHydrogens(7.0)
                with open(output_pdb_path, "w") as fh:
                    PDBFile.writeFile(fixer.topology, fixer.positions, fh)
                pbar.update(1)
                return True
            except Exception:
                self.logger.exception("PDBFixer failed")
                return False

    def _build_pdb2gmx_cmd(self, pdb_path: str) -> list:
        sim_type = self.config.get("sim_type")
        requested = self.config.get("force_field", "default")
        water = self.config.get("water_model", "tip3p")
        base = [
            self.gmx_bin, "pdb2gmx",
            "-f", pdb_path,
            "-o", self.output_gro,
            "-water", water,
            "-ignh",
        ]
        if requested != "default":
            resolved, root = resolve_force_field(
                requested,
                force_field_search_dirs(self.gmx_bin, self.config.get("force_field_dir")),
            )
            self.config["resolved_force_field"] = resolved
            self.config["resolved_force_field_dir"] = str(root)
            self.config["water_model"] = water
            os.environ["GMXLIB"] = str(root)
            config_path = Path(self.config["work_dir"]) / "run_config.json"
            if config_path.is_file():
                config_path.write_text(json.dumps(self.config, indent=2) + "\n", encoding="utf-8")
            return base + ["-ff", resolved, "-chainsep", "id"]
        if sim_type in ["3", "4", "5", "6"]:
            return base + ["-ff", "amber99sb-ildn", "-chainsep", "id"]
        return base + ["-ff", "amber03"]

    def run(self) -> None:
        work_dir         = self.config["work_dir"]
        repaired_pdb_abs = os.path.join(os.path.abspath(work_dir), "complex_repaired.pdb")
        cmd              = self._build_pdb2gmx_cmd(self.pdb_input_abs)
        environment      = os.environ.copy()
        if self.config.get("resolved_force_field_dir"):
            environment["GMXLIB"] = self.config["resolved_force_field_dir"]

        try:
            run_subprocess(
                cmd,
                timeout=_PDB2GMX_TIMEOUT,
                retries=1,
                cwd=work_dir,
                env=environment,
                logger=self.logger,
            )
        except SubprocessError as exc:
            # Attempt PDBFixer repair when the error is atom-related.
            if "not found in the input file" in exc.stderr or "atom" in exc.stderr:
                if self._repair_pdb(self.pdb_input_abs, repaired_pdb_abs):
                    repaired_cmd = self._build_pdb2gmx_cmd("complex_repaired.pdb")
                    try:
                        run_subprocess(
                            repaired_cmd,
                            timeout=_PDB2GMX_TIMEOUT,
                            retries=1,
                            cwd=work_dir,
                            env=environment,
                            logger=self.logger,
                        )
                        return
                    except SubprocessError as exc2:
                        self.logger.error("Persistent error after repair:\n%s", exc2.stderr)
                        raise RuntimeError(str(exc2)) from exc2
            self.logger.error("pdb2gmx failed:\n%s", exc.stderr)
            raise RuntimeError(str(exc)) from exc
