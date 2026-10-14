import os
import shutil
from pathlib import Path
from utils.progress import step_bar
from pipelines.dynamics.utils import convert_pdbqt_to_pdb
from pipelines.dynamics.dihedral_sanitization import sanitize_ligand_dihedrals

from ..logger import get_step_logger
from utils.retry import SubprocessError, run_subprocess

_ACPYPE_TIMEOUT = 600


sanitize_zero_periodicity_dihedrals = sanitize_ligand_dihedrals


class LigandTopologyStep:
    def __init__(self, config, gmx_bin, standalone=False):
        self.standalone    = standalone
        self.config        = config
        self.gmx_bin       = gmx_bin
        self.work_dir      = self.config["work_dir"]
        self.ligand_pdb    = os.path.abspath(self.config["ligand_pdb"])
        self.ligand_source = os.path.abspath(
            self.config.get("ligand_prepared_file", self.ligand_pdb)
        )
        self.protein_gro   = os.path.join(self.work_dir, "processed.gro")
        self.topol_file    = os.path.join(self.work_dir, "topol.top")
        if "ligand_charge" not in self.config:
            raise ValueError(
                "Ligand net charge was not resolved. Provide --charge or enable charge detection."
            )
        self.charge        = int(self.config["ligand_charge"])
        self.acpype_workdir = os.path.join(self.work_dir, "acpype_work")
        self.logger        = get_step_logger(
            __name__, os.path.join(self.work_dir, "simulation.log")
        )

    def run(self) -> None:
        with step_bar("Ligand Topology Generation", 5) as pbar:
            try:
                os.makedirs(self.acpype_workdir, exist_ok=True)
                source_suffix = os.path.splitext(self.ligand_source)[1].lower()
                internal_suffix = ".pdb" if source_suffix == ".pdbqt" else source_suffix
                if internal_suffix not in {".pdb", ".sdf", ".mol", ".mol2"}:
                    raise ValueError(f"Unsupported ligand format for ACPYPE: {source_suffix}")
                internal_ligand = os.path.join(
                    self.acpype_workdir, f"ligand{internal_suffix}"
                )

                # Format conversion: PDBQT → PDB or plain copy.
                if source_suffix == ".pdbqt":
                    if not convert_pdbqt_to_pdb(self.ligand_source, internal_ligand):
                        self.logger.error(
                            "Failed to convert %s to PDB using OpenBabel.", self.ligand_source
                        )
                        raise RuntimeError("Failed to convert ligand PDBQT to PDB.")
                else:
                    try:
                        shutil.copy(self.ligand_source, internal_ligand)
                    except FileNotFoundError:
                        self.logger.error("Source ligand file not found: %s", self.ligand_source)
                        raise
                pbar.update(1)

                if not os.path.exists(internal_ligand):
                    raise FileNotFoundError(
                        f"Expected ligand input not found after conversion/copy: {internal_ligand}"
                    )
                if os.path.getsize(internal_ligand) == 0:
                    raise RuntimeError(f"Ligand input is empty: {internal_ligand}")

                # Run ACPYPE for GAFF2 parameters.
                run_subprocess(
                    [
                        "acpype", "-i", os.path.basename(internal_ligand),
                        "-b", "LIG", "-c", "bcc",
                        "-n", str(self.charge), "-a", "gaff2",
                    ],
                    timeout=_ACPYPE_TIMEOUT,
                    retries=1,
                    cwd=self.acpype_workdir,
                    logger=self.logger,
                )
                pbar.update(1)

                acpype_out_folder = next(
                    (
                        os.path.join(self.acpype_workdir, d)
                        for d in os.listdir(self.acpype_workdir)
                        if d.endswith(".acpype")
                    ),
                    None,
                )
                if not acpype_out_folder:
                    raise FileNotFoundError("ACPYPE output directory was not found.")

                acpype_gro = os.path.join(acpype_out_folder, "LIG_GMX.gro")
                acpype_itp = os.path.join(acpype_out_folder, "LIG_GMX.itp")

                dihedral_report = sanitize_zero_periodicity_dihedrals(
                    acpype_itp,
                    os.path.join(self.work_dir, "ligand_dihedral_sanitization.json"),
                )
                if dihedral_report["removed_null_count"]:
                    self.logger.info(
                        "NOTA: Se eliminaron %d diedros nulos del ligando "
                        "(periodicidad 0 y constante de fuerza 0); sin cambio en fuerzas.",
                        dihedral_report["removed_null_count"],
                        extra={"file_only": True},
                    )
                self._clean_ligand_itp(acpype_itp)
                pbar.update(1)
                if self.standalone:
                    self._write_standalone_system(acpype_gro, acpype_itp)
                    pbar.update(2)
                else:
                    self._merge_gro(self.protein_gro, acpype_gro)
                    pbar.update(1)
                    self._patch_topology(acpype_itp)
                    pbar.update(1)

            except SubprocessError as exc:
                self.logger.error("ACPYPE failed:\n%s", exc.stderr)
                raise RuntimeError(str(exc)) from exc
            except Exception:
                self.logger.exception("Unexpected error in LigandTopologyStep")
                raise

    def _clean_ligand_itp(self, itp_path: str) -> None:
        with open(itp_path) as fh:
            lines = fh.readlines()
        clean, skip = [], False
        for line in lines:
            if "[ atomtypes ]" in line:
                skip = True
                continue
            if skip and line.startswith("["):
                skip = False
            if not skip:
                clean.append(line)
        with open(os.path.join(self.work_dir, "ligand.itp"), "w") as fh:
            fh.writelines(clean)

    def _write_standalone_system(self, lig_gro: str, original_itp: str) -> None:
        """Ligand-only system: processed.gro plus a topol.top built from scratch."""
        with open(lig_gro) as fh:
            lines = fh.readlines()
        with open(os.path.join(self.work_dir, "processed.gro"), "w") as fh:
            fh.write("Ligand System\n")
            fh.writelines(lines[1:])

        atomtypes: list = []
        capture = False
        with open(original_itp) as fh:
            for line in fh:
                if "[ atomtypes ]" in line:
                    capture = True
                    atomtypes.append(line)
                    continue
                if capture and line.startswith("["):
                    break
                if capture:
                    atomtypes.append(line)

        with open(self.topol_file, "w") as fh:
            fh.write("; Ligand-only topology generated by ChemLink (GAFF2 / ACPYPE)\n")
            fh.write('#include "amber03.ff/forcefield.itp"\n\n')
            fh.write("; Ligand-specific atomtypes (GAFF2)\n")
            fh.writelines(atomtypes)
            fh.write('\n; Include ligand topology\n#include "ligand.itp"\n\n')
            fh.write('; Water and ions\n#include "amber03.ff/tip3p.itp"\n')
            fh.write('#include "amber03.ff/ions.itp"\n\n')
            fh.write("[ system ]\nLigand in water\n\n[ molecules ]\nLIG                1\n")

    def _merge_gro(self, prot_gro: str, lig_gro: str) -> None:
        with open(prot_gro) as fh:
            p_lines = fh.readlines()
        with open(lig_gro) as fh:
            l_lines = fh.readlines()
        p_atoms = p_lines[2:-1]
        l_atoms = l_lines[2:-1]
        box     = p_lines[-1]
        total   = len(p_atoms) + len(l_atoms)
        with open(os.path.join(self.work_dir, "complex.gro"), "w") as fh:
            fh.write("Complex System\n")
            fh.write(f"{total}\n")
            fh.writelines(p_atoms)
            fh.writelines(l_atoms)
            fh.write(box)

    def _patch_topology(self, original_itp: str) -> None:
        atomtypes: list = []
        with open(original_itp) as fh:
            capture = False
            for line in fh:
                if "[ atomtypes ]" in line:
                    capture = True
                    atomtypes.append(line)
                    continue
                if capture and line.startswith("["):
                    break
                if capture:
                    atomtypes.append(line)

        with open(self.topol_file) as fh:
            top_lines = fh.readlines()

        new_top: list = []
        types_inserted = itp_included = False

        for line in top_lines:
            if "forcefield.itp" in line and not types_inserted:
                new_top.append(line)
                new_top.append("\n; Ligand-specific atomtypes (GAFF2)\n")
                new_top.extend(atomtypes)
                new_top.append("\n")
                types_inserted = True
                continue
            if "[ molecules ]" in line and not itp_included:
                new_top.append("; Include ligand topology\n")
                new_top.append('#include "ligand.itp"\n\n')
                new_top.append(line)
                itp_included = True
                continue
            if line.strip() == "LIG                1":
                continue
            new_top.append(line)

        if not any("LIG" in l for l in top_lines[-5:]):
            new_top.append("LIG                1\n")
        with open(self.topol_file, "w") as fh:
            fh.writelines(new_top)
