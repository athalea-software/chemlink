import os
import shutil
from pathlib import Path
from tqdm import tqdm

from ..dihedral_sanitization import sanitize_ligand_dihedrals
from ..logger import get_step_logger
from utils.cofactor_classifier import (
    CofactorError,
    classify_cofactor,
    find_ion_moleculetype,
    format_gro_atom_line,
)
from utils.retry import SubprocessError, run_subprocess
from .topology import force_field_search_dirs, resolve_force_field

_ACPYPE_TIMEOUT = 600


def _resolve_cofactor_force_field(config: dict, gmx_bin: str) -> tuple[str, Path]:
    """Resolve the force-field name/directory this run's protein topology used.

    Mirrors TopologyStep._build_pdb2gmx_cmd's own default selection so the
    cofactor's ion is matched against the same force field the protein was
    actually built with, whether or not --force-field was passed explicitly.
    """
    if config.get("resolved_force_field_dir"):
        return config["resolved_force_field"], Path(config["resolved_force_field_dir"])
    requested = config.get("force_field", "default")
    if requested == "default":
        requested = "amber99sb-ildn" if config.get("sim_type") in {"3", "4", "5", "6"} else "amber03"
    search_dirs = force_field_search_dirs(gmx_bin, config.get("force_field_dir"))
    return resolve_force_field(requested, search_dirs)


class CofactorTopologyStep:
    """Fuses an optional ion / small inorganic cofactor into the MD system.

    Must run after LigandTopologyStep: it appends onto the protein+ligand
    complex.gro/topol.top that step already produced, the same way
    LigandTopologyStep appends the ligand onto the bare protein.
    """

    def __init__(self, config: dict, gmx_bin: str):
        self.config = config
        self.gmx_bin = gmx_bin
        self.work_dir = config["work_dir"]
        self.cofactor_pdb = os.path.abspath(config["cofactor_pdb"])
        self.complex_gro = os.path.join(self.work_dir, "complex.gro")
        self.topol_file = os.path.join(self.work_dir, "topol.top")
        self.acpype_workdir = os.path.join(self.work_dir, "acpype_cofactor")
        self.logger = get_step_logger(__name__, os.path.join(self.work_dir, "simulation.log"))

    def run(self) -> None:
        with tqdm(total=2, desc="  └─ Cofactor Topology Generation", leave=False) as pbar:
            try:
                classification = classify_cofactor(self.cofactor_pdb)
                pbar.update(1)

                if classification.kind == "ion":
                    resname = self._merge_ion(classification)
                    self.config["cofactor_kind"] = "ion"
                    self.config["cofactor_resname"] = resname
                else:
                    self._merge_gaff_cofactor(classification)
                    self.config["cofactor_kind"] = "molecule"
                    self.config["cofactor_resname"] = "COF"
                pbar.update(1)
            except SubprocessError as exc:
                self.logger.error("ACPYPE failed for cofactor:\n%s", exc.stderr)
                raise RuntimeError(str(exc)) from exc
            except CofactorError:
                self.logger.exception("Cofactor classification/resolution failed")
                raise
            except Exception:
                self.logger.exception("Unexpected error in CofactorTopologyStep")
                raise

    # -- ion branch -----------------------------------------------------

    def _merge_ion(self, classification) -> str:
        ff_name, ff_dir = _resolve_cofactor_force_field(self.config, self.gmx_bin)
        ions_itp = ff_dir / f"{ff_name}.ff" / "ions.itp"
        if not ions_itp.is_file():
            raise CofactorError(
                f"Force field '{ff_name}' does not ship an ions.itp ({ions_itp}); "
                "cannot resolve this ion cofactor against it."
            )
        atom = classification.atoms[0]
        resname, atom_name = find_ion_moleculetype(ions_itp, classification.resname)

        with open(self.complex_gro) as fh:
            lines = fh.readlines()
        header, body, box = lines[0], lines[2:-1], lines[-1]
        natoms = int(lines[1].strip())
        last_resnum = int(body[-1][:5]) if body else 0
        new_atom_line = format_gro_atom_line(
            last_resnum + 1, resname, atom_name, natoms + 1,
            atom.x / 10.0, atom.y / 10.0, atom.z / 10.0,
        )
        with open(self.complex_gro, "w") as fh:
            fh.write(header)
            fh.write(f"{natoms + 1}\n")
            fh.writelines(body)
            fh.write(new_atom_line)
            fh.write(box)

        with open(self.topol_file) as fh:
            top_lines = fh.readlines()
        include_line = f'#include "{ff_name}.ff/ions.itp"\n'
        already_included = any(line.strip() == include_line.strip() for line in top_lines)
        new_top: list[str] = []
        for line in top_lines:
            if "[ molecules ]" in line and not already_included:
                new_top.append("; Cofactor ion parameters\n")
                new_top.append(include_line)
                new_top.append("\n")
                already_included = True
            new_top.append(line)
        new_top.append(f"{resname:<19}1\n")
        with open(self.topol_file, "w") as fh:
            fh.writelines(new_top)

        return resname

    # -- GAFF2 molecule branch -------------------------------------------

    def _merge_gaff_cofactor(self, classification) -> None:
        charge = self.config.get("cofactor_charge")
        if charge is None:
            raise CofactorError(
                "Cofactor net charge was not resolved. Provide --cofactor-charge or "
                "enable charge detection."
            )
        os.makedirs(self.acpype_workdir, exist_ok=True)
        source = os.path.abspath(self.config.get("cofactor_prepared_file", self.cofactor_pdb))
        internal_ligand = os.path.join(self.acpype_workdir, "cofactor.pdb")
        shutil.copy(source, internal_ligand)

        run_subprocess(
            [
                "acpype", "-i", "cofactor.pdb",
                "-b", "COF", "-c", "bcc",
                "-n", str(int(charge)), "-a", "gaff2",
            ],
            timeout=_ACPYPE_TIMEOUT,
            retries=1,
            cwd=self.acpype_workdir,
            logger=self.logger,
        )

        acpype_out = next(
            (
                os.path.join(self.acpype_workdir, entry)
                for entry in os.listdir(self.acpype_workdir)
                if entry.endswith(".acpype")
            ),
            None,
        )
        if not acpype_out:
            raise FileNotFoundError("ACPYPE output directory was not found for cofactor.")

        acpype_gro = os.path.join(acpype_out, "COF_GMX.gro")
        acpype_itp = os.path.join(acpype_out, "COF_GMX.itp")

        dihedral_report = sanitize_ligand_dihedrals(
            acpype_itp, os.path.join(self.work_dir, "cofactor_dihedral_sanitization.json"),
        )
        if dihedral_report["removed_null_count"]:
            self.logger.info(
                "NOTA: Se eliminaron %d diedros nulos del cofactor "
                "(periodicidad 0 y constante de fuerza 0); sin cambio en fuerzas.",
                dihedral_report["removed_null_count"],
                extra={"file_only": True},
            )

        new_atomtypes = self._clean_cofactor_itp(acpype_itp)
        self._merge_gro(acpype_gro)
        self._patch_topology(new_atomtypes)

    def _clean_cofactor_itp(self, itp_path: str) -> list[str]:
        with open(itp_path) as fh:
            lines = fh.readlines()
        atomtype_lines: list[str] = []
        clean, skip = [], False
        for line in lines:
            if "[ atomtypes ]" in line:
                skip = True
                continue
            if skip and line.startswith("["):
                skip = False
            if skip:
                if line.strip() and not line.strip().startswith(";"):
                    atomtype_lines.append(line)
                continue
            clean.append(line)
        with open(os.path.join(self.work_dir, "cofactor.itp"), "w") as fh:
            fh.writelines(clean)
        return atomtype_lines

    def _merge_gro(self, cofactor_gro: str) -> None:
        with open(self.complex_gro) as fh:
            p_lines = fh.readlines()
        with open(cofactor_gro) as fh:
            c_lines = fh.readlines()
        p_atoms = p_lines[2:-1]
        c_atoms = c_lines[2:-1]
        box = p_lines[-1]
        total = len(p_atoms) + len(c_atoms)
        with open(self.complex_gro, "w") as fh:
            fh.write(p_lines[0])
            fh.write(f"{total}\n")
            fh.writelines(p_atoms)
            fh.writelines(c_atoms)
            fh.write(box)

    def _existing_atomtype_names(self) -> set[str]:
        names: set[str] = set()
        in_types = False
        with open(self.topol_file) as fh:
            for line in fh:
                if "[ atomtypes ]" in line:
                    in_types = True
                    continue
                if in_types:
                    stripped = line.strip()
                    # This codebase always writes a blank line right after an
                    # atomtypes block (see LigandTopologyStep._patch_topology
                    # and _patch_topology below), so treat it as the closing
                    # marker too -- the next non-blank line is not
                    # necessarily another "[ ... ]" section header.
                    if not stripped or stripped.startswith("["):
                        in_types = False
                        continue
                    if stripped and not stripped.startswith(";"):
                        names.add(stripped.split()[0])
        return names

    def _patch_topology(self, cofactor_atomtype_lines: list[str]) -> None:
        # The ligand step already inserted GAFF2 atomtypes right after
        # forcefield.itp. Re-declaring a type GROMACS already knows (e.g. a
        # shared generic "c3"/"os"/"n" from the ligand) makes grompp reject
        # the topology as a duplicate, so only genuinely new type names are
        # appended here -- in their own block, right above the include that
        # uses them, which keeps ordering valid without touching the
        # ligand's block at all.
        existing = self._existing_atomtype_names()
        unique_new = [
            line for line in cofactor_atomtype_lines
            if line.strip() and not line.strip().startswith(";") and line.split()[0] not in existing
        ]

        with open(self.topol_file) as fh:
            top_lines = fh.readlines()

        insertion: list[str] = []
        if unique_new:
            insertion.append("; Cofactor-specific atomtypes (GAFF2)\n")
            insertion.append("[ atomtypes ]\n")
            insertion.extend(unique_new)
            insertion.append("\n")
        insertion.append("; Include cofactor topology\n")
        insertion.append('#include "cofactor.itp"\n\n')

        new_top: list[str] = []
        inserted = False
        for line in top_lines:
            if "[ molecules ]" in line and not inserted:
                new_top.extend(insertion)
                inserted = True
            new_top.append(line)
        new_top.append(f"{'COF':<19}1\n")

        with open(self.topol_file, "w") as fh:
            fh.writelines(new_top)
