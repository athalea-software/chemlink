import os
import shutil
from pathlib import Path
from tqdm import tqdm

from ..dihedral_sanitization import sanitize_ligand_dihedrals
from ..logger import get_step_logger
from ..utils import convert_pdbqt_to_pdb
from utils.cofactor_classifier import (
    CofactorError,
    classify_cofactor,
    find_ion_moleculetype,
    format_gro_atom_line,
)
from utils.retry import SubprocessError, run_subprocess
from .topology import force_field_search_dirs, resolve_force_field

# AM1-BCC (the "bcc" charge method) runs a semi-empirical QM optimisation
# (sqm) per fragment; on a cofactor with a few dozen heavy atoms -- larger
# than the small ions/inorganic anions this path was originally sized for --
# that routinely takes well past 10 minutes on a single core. The default
# stays generous rather than tight, and config["cofactor_charge_timeout"]
# (in seconds) can override it.
_ACPYPE_TIMEOUT = 3600


def _directive(line: str) -> str:
    """A .top line's content with any ';' comment stripped, whitespace-trimmed.

    Section detection below must never do a plain substring search for
    "[ atomtypes ]"/"[ molecules ]" -- a comment that merely mentions a
    section name (or a longer directive that happens to contain the text)
    would otherwise be mistaken for the real header.
    """
    return line.split(";", 1)[0].strip()


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
            if _directive(line) == "[ molecules ]" and not already_included:
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
        # The prepared file may be the original cofactor PDB, or -- when it
        # went through pH-dependent charge detection -- an SDF written by
        # prepare_protonated_sdf. ACPYPE picks its parser from the file
        # extension, so the copy must keep the source's real format instead
        # of always landing on "cofactor.pdb" (which made ACPYPE try to read
        # SDF content as PDB and fail with no usable residue).
        source_suffix = os.path.splitext(source)[1].lower()
        internal_suffix = ".pdb" if source_suffix == ".pdbqt" else source_suffix
        if internal_suffix not in {".pdb", ".sdf", ".mol", ".mol2"}:
            raise CofactorError(f"Unsupported cofactor format for ACPYPE: {source_suffix}")
        internal_name = f"cofactor{internal_suffix}"
        internal_ligand = os.path.join(self.acpype_workdir, internal_name)

        if source_suffix == ".pdbqt":
            if not convert_pdbqt_to_pdb(source, internal_ligand):
                raise CofactorError(f"Failed to convert cofactor PDBQT to PDB: {source}")
        else:
            shutil.copy(source, internal_ligand)

        charge_method = self.config.get("cofactor_gaff_charge_method", "bcc")
        timeout = self.config.get("cofactor_charge_timeout", _ACPYPE_TIMEOUT)
        run_subprocess(
            [
                "acpype", "-i", internal_name,
                "-b", "COF", "-c", charge_method,
                "-n", str(int(charge)), "-a", "gaff2",
            ],
            timeout=timeout,
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
            if _directive(line) == "[ atomtypes ]":
                skip = True
                continue
            if skip and _directive(line).startswith("["):
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
                if _directive(line) == "[ atomtypes ]":
                    in_types = True
                    continue
                if in_types:
                    directive = _directive(line)
                    # This codebase always writes a blank line right after an
                    # atomtypes block (see LigandTopologyStep._patch_topology
                    # and _patch_topology below), so treat it as the closing
                    # marker too -- the next non-blank line is not
                    # necessarily another "[ ... ]" section header.
                    if not line.strip() or directive.startswith("["):
                        in_types = False
                        continue
                    if directive:
                        names.add(directive.split()[0])
        return names

    def _patch_topology(self, cofactor_atomtype_lines: list[str]) -> None:
        # GROMACS requires every [ atomtypes ] entry to appear before the
        # FIRST [ moleculetype ] anywhere in the file ("Invalid order for
        # directive atomtypes" otherwise) -- it is not enough for a type to
        # precede the specific #include that uses it. The ligand step's own
        # [ atomtypes ] block already sits early (right after
        # forcefield.itp, before any moleculetype), so new cofactor types
        # are appended INTO that same block instead of opening a second one
        # further down next to "#include cofactor.itp", which would land
        # after the protein's and ligand's moleculetype sections. Types
        # GROMACS already knows (e.g. a shared generic "c3"/"os"/"n" from
        # the ligand) are skipped -- redeclaring them makes grompp reject
        # the topology as a duplicate.
        existing = self._existing_atomtype_names()
        unique_new = [
            line for line in cofactor_atomtype_lines
            if line.strip() and not line.strip().startswith(";") and line.split()[0] not in existing
        ]

        with open(self.topol_file) as fh:
            top_lines = fh.readlines()

        new_top: list[str] = []
        in_types = False
        types_found = False
        inserted_types = not unique_new
        inserted_include = False
        for line in top_lines:
            if _directive(line) == "[ atomtypes ]":
                in_types = True
                types_found = True
                new_top.append(line)
                continue
            if in_types:
                directive = _directive(line)
                if not line.strip() or directive.startswith("["):
                    if not inserted_types:
                        new_top.append("; Cofactor-specific atomtypes (GAFF2)\n")
                        new_top.extend(unique_new)
                        inserted_types = True
                    in_types = False
                else:
                    new_top.append(line)
                    continue
            if _directive(line) == "[ molecules ]" and not inserted_include:
                new_top.append("; Include cofactor topology\n")
                new_top.append('#include "cofactor.itp"\n\n')
                inserted_include = True
            new_top.append(line)

        if in_types and not inserted_types:
            new_top.append("; Cofactor-specific atomtypes (GAFF2)\n")
            new_top.extend(unique_new)
            inserted_types = True
        if not types_found and unique_new:
            raise CofactorError(
                f"{self.topol_file} has no existing [ atomtypes ] section to extend with "
                "the cofactor's GAFF2 atom types (expected one inserted by LigandTopologyStep)."
            )
        if not inserted_include:
            new_top.append("; Include cofactor topology\n")
            new_top.append('#include "cofactor.itp"\n\n')

        new_top.append(f"{'COF':<19}1\n")

        with open(self.topol_file, "w") as fh:
            fh.writelines(new_top)
