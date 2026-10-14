"""Receptor processing utilities for cleanup and PDBQT conversion."""

import math
import os
import subprocess
from datetime import datetime
from typing import Dict, List, Optional, Tuple

try:
    from openbabel import openbabel as ob
except ImportError:
    import openbabel as ob

from meeko import MoleculePreparation, PDBQTWriterLegacy
from pdbfixer import PDBFixer
from openmm import Context, CustomExternalForce, LocalEnergyMinimizer, VerletIntegrator, unit
from openmm.app import ForceField, HBonds, NoCutoff, PDBFile
from rdkit import Chem

from ..storage.file_manager import create_out_file, find_compound_name
from .logger import setup_logger
from .receptor_cofactors import (
    append_cofactors,
    find_cofactor_residues,
    residue_key,
    write_cofactor_pdbqt,
)

logger = setup_logger(__name__)


class ReceptorProcessor:
    """Handle receptor cleanup and conversion to PDBQT."""

    PROTEIN_RESIDUES = {
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
        "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
        "THR", "TRP", "TYR", "VAL",
        "HID", "HIE", "HIP", "HSE", "HSD", "HSP",
        "CYX", "CYM",
        "ACE", "NME",
    }

    def __init__(
        self,
        remove_water: bool = True,
        remove_ligands: bool = True,
        remove_ions: bool = True,
        keep_clean_pdb: bool = True,
        keep_cofactors: bool = False,
    ):
        self.mol = ob.OBMol()
        self.remove_water = remove_water
        self.remove_ligands = remove_ligands
        self.remove_ions = remove_ions
        self.keep_clean_pdb = keep_clean_pdb
        # Keep hemes and catalytic metal ions in the docking receptor (see
        # utils/receptor_cofactors.py). They bypass PDBFixer/Meeko and are
        # appended to the receptor PDBQT afterwards.
        self.keep_cofactors = keep_cofactors
        self._input_path: Optional[str] = None
        self._cofactors: Dict[Tuple[str, str, str, str], str] = {}

    def _read_receptor(self, receptor_file: str) -> None:
        if not os.path.exists(receptor_file):
            abs_path = os.path.abspath(receptor_file)
            cwd = os.getcwd()

            error_msg = f"Receptor file not found: {receptor_file}\n"
            error_msg += f"  Absolute path attempted: {abs_path}\n"
            error_msg += f"  Current working directory: {cwd}\n"

            basename = os.path.basename(receptor_file)
            possible_paths = [
                f"data/input/receptors/{basename}",
                f"./data/input/receptors/{basename}",
                f"../data/input/receptors/{basename}",
            ]

            for possible in possible_paths:
                if os.path.exists(possible):
                    error_msg += f"  File found at: {possible}\n"
                    error_msg += "  Use this path instead!\n"
                    break

            raise RuntimeError(error_msg.strip())

        if not os.access(receptor_file, os.R_OK):
            raise RuntimeError(
                f"Cannot read receptor file (permission denied): {receptor_file}\n"
                f"  Try: chmod +r {receptor_file}"
            )

        self.mol = ob.OBMol()
        self._input_path = os.path.abspath(receptor_file)
        conv = ob.OBConversion()
        conv.SetInFormat("pdb")

        logger.info(f"Reading receptor: {receptor_file}")

        # PDB files do not encode aromatic bond orders.  While reading a
        # protein, Open Babel may therefore print a non-fatal
        # ``Failed to kekulize aromatic bonds`` warning directly from its C++
        # layer.  It is not actionable during receptor cleanup and disrupts
        # the progress display.  Limit Open Babel to real errors for this
        # operation only, then restore the caller's previous verbosity.
        error_log = getattr(ob, "obErrorLog", None)
        previous_output_level = None
        if error_log is not None and hasattr(error_log, "GetOutputLevel"):
            previous_output_level = error_log.GetOutputLevel()
            error_log.SetOutputLevel(ob.obError)

        try:
            read_ok = conv.ReadFile(self.mol, receptor_file)
        finally:
            if previous_output_level is not None:
                error_log.SetOutputLevel(previous_output_level)

        if not read_ok:
            file_size = os.path.getsize(receptor_file)
            error_msg = f"OpenBabel failed to read receptor: {receptor_file}\n"
            error_msg += "  File exists: Yes\n"
            error_msg += f"  File size: {file_size} bytes\n"
            error_msg += f"  File readable: {os.access(receptor_file, os.R_OK)}\n"

            try:
                with open(receptor_file, "r") as handle:
                    first_line = handle.readline().strip()
                    error_msg += f"  First line: {first_line[:50]}...\n"
            except Exception as exc:
                error_msg += f"  Cannot read file content: {exc}\n"

            error_msg += "  This may indicate a corrupted or malformed PDB file"
            raise RuntimeError(error_msg.strip())

        num_atoms = self.mol.NumAtoms()
        if num_atoms == 0:
            raise RuntimeError(
                f"Read 0 atoms from {receptor_file}. File may be empty or corrupted."
            )

        logger.info(f"Successfully read {num_atoms:,} atoms from {receptor_file}")

        self._cofactors = {}
        if self.keep_cofactors:
            with open(receptor_file) as handle:
                self._cofactors = find_cofactor_residues(handle, self.PROTEIN_RESIDUES)

    def _get_residue_summary(self) -> Dict[str, int]:
        residues: Dict[str, int] = {}
        for atom in ob.OBMolAtomIter(self.mol):
            res = atom.GetResidue()
            if res:
                name = res.GetName().strip().upper()
                residues[name] = residues.get(name, 0) + 1
        return residues

    def _is_protein(self) -> bool:
        """Detect if molecule is a standard protein."""
        residues = self._get_residue_summary()
        protein_residue_count = sum(
            count for name, count in residues.items() 
            if name in self.PROTEIN_RESIDUES
        )
        total_atoms = self.mol.NumAtoms()
        
        # Si tiene residuos de proteína, es proteína
        if protein_residue_count > 0:
            return True
        
        # Si tiene solo heteroátomos sin residuos, es molécula sintética
        return False

    def _identify_atoms_to_remove(self):
        atoms_to_remove = []
        removed_info = {"water": 0, "ligand": 0, "ion": 0, "other": 0}

        for atom in ob.OBMolAtomIter(self.mol):
            res = atom.GetResidue()

            if not res:
                atoms_to_remove.append(atom.GetIdx())
                removed_info["other"] += 1
                continue

            name = res.GetName().strip().upper()
            if name in self.PROTEIN_RESIDUES:
                continue

            if self._is_cofactor_residue(res):
                continue

            if name in {"HOH", "WAT", "H2O", "DOD"}:
                if self.remove_water:
                    atoms_to_remove.append(atom.GetIdx())
                    removed_info["water"] += 1
                continue

            if name in {"NA", "CL", "MG", "CA", "ZN", "FE", "K", "MN", "CU"}:
                if self.remove_ions:
                    atoms_to_remove.append(atom.GetIdx())
                    removed_info["ion"] += 1
                continue

            if self.remove_ligands:
                atoms_to_remove.append(atom.GetIdx())
                removed_info["ligand"] += 1

        return atoms_to_remove, removed_info

    def _is_cofactor_residue(self, res) -> bool:
        if not self._cofactors:
            return False
        key = (
            res.GetChain() or " ",
            res.GetNumString().strip(),
            res.GetName().strip().upper(),
        )
        return key in {(chain, num, name) for chain, num, _icode, name in self._cofactors}

    def _remove_non_protein(self) -> Dict[str, int]:
        """Classify non-protein atoms (water/ion/ligand) and log removal counts.

        This no longer mutates self.mol or writes through it — see
        _save_clean_receptor() for why the actual removal happens there,
        directly on the original PDB text.
        """
        _, info = self._identify_atoms_to_remove()

        logger.info(
            f"Removed: {info['water']} water, {info['ligand']} ligand, "
            f"{info['ion']} ion, {info['other']} other atoms"
        )
        return info

    def _save_clean_receptor(self, output_file: str) -> None:
        """Write the cleaned receptor by filtering the original PDB text directly.

        OpenBabel's PDB writer does not preserve the original chain ID —
        OBResidue.GetChain() is ignored at write time, and chain letters
        are re-derived from OpenBabel's own perceived backbone
        connectivity instead. On structures where a metal ion (e.g. Zn,
        Mg) or a bound ligand was coordinating a loop, deleting that
        hetero group can distort the perceived backbone continuity enough
        that OpenBabel silently splits one real chain into two on write
        (verified: atom count is preserved, only the chain ID/boundary is
        wrong). PDBFixer then treats the false chain end as a genuine
        C-terminus and adds a spurious OXT atom there, which RDKit/Meeko
        later reject downstream ("Explicit valence ... is greater than
        permitted"). Re-emitting the original, already-correctly-labeled
        ATOM/HETATM lines (minus the removed residues) sidesteps that
        OpenBabel round-trip entirely.
        """
        if not self._input_path:
            raise RuntimeError("_save_clean_receptor called before _read_receptor")

        water_names = {"HOH", "WAT", "H2O", "DOD"}
        ion_names = {"NA", "CL", "MG", "CA", "ZN", "FE", "K", "MN", "CU"}

        out_lines = []
        cofactor_lines = []
        with open(self._input_path) as handle:
            for line in handle:
                if line.startswith(("ATOM", "HETATM")) and residue_key(line) in self._cofactors:
                    element = line[76:78].strip().upper()
                    if element != "H":
                        cofactor_lines.append(line)
                    continue
                if line.startswith(("ATOM", "HETATM")):
                    # Receptor protonation belongs to PDBFixer/Meeko.  Keeping
                    # hydrogens supplied by the input is unsafe: PDBFixer's
                    # topology retains them even though _write_heavy_atom_pdb
                    # later hides them from the intermediate file.  If the
                    # geometry-validation fallback then calls
                    # addMissingHydrogens(), OpenMM can see a mixture of the
                    # input protonation and its own terminal templates (for
                    # example ARG matching NARG with one hydrogen too many).
                    element = line[76:78].strip().upper()
                    atom_name = line[12:16].strip().upper()
                    if element == "H" or (not element and atom_name.startswith("H")):
                        continue
                    resname = line[17:20].strip().upper()
                    if resname in self.PROTEIN_RESIDUES:
                        out_lines.append(line)
                    elif resname in water_names:
                        if not self.remove_water:
                            out_lines.append(line)
                    elif resname in ion_names:
                        if not self.remove_ions:
                            out_lines.append(line)
                    elif not self.remove_ligands:
                        out_lines.append(line)
                    continue
                if line.startswith(("TER", "END")):
                    out_lines.append(line)

        if not out_lines:
            raise RuntimeError(f"No atoms remained after filtering {self._input_path}")

        with open(output_file, "w") as handle:
            handle.writelines(out_lines)

        kept_atoms = sum(1 for line in out_lines if line.startswith(("ATOM", "HETATM")))
        logger.info(f"Remaining atoms: {kept_atoms}")

        if cofactor_lines:
            cofactor_pdb = self._cofactor_pdb_for(output_file)
            with open(cofactor_pdb, "w") as handle:
                handle.writelines(cofactor_lines)
                handle.write("END\n")
            logger.info(
                "Kept %d cofactor residue(s) for docking: %s",
                len(self._cofactors),
                ", ".join(
                    f"{name} {chain}:{resnum} ({kind})"
                    for (chain, resnum, _icode, name), kind in self._cofactors.items()
                ),
            )

    @staticmethod
    def _cofactor_pdb_for(clean_pdb: str) -> str:
        return os.path.splitext(clean_pdb)[0] + "_cofactors.pdb"

    @staticmethod
    def _insert_chain_break_ters(pdb_path: str) -> List[Tuple[str, str]]:
        """Split chains only at genuine backbone breaks (disordered loops, missing termini).

        Receptors with disordered loops (common in GPCRs, cryo-EM structures)
        have skips in residue numbering where OpenBabel writes a single
        contiguous chain. PDBFixer then renumbers everything sequentially and
        keeps the same chain letter across the gap, so Meeko's polymer
        builder later infers a peptide bond from chain/residue adjacency
        alone and fails ("Expected N paddings ... but got 0") even though a
        bare TER is present. Assigning a new chain letter to the segment
        after each gap (in addition to the TER) makes both tools treat it as
        a genuinely separate, unbonded chain.

        Crucially, a residue-number gap alone is NOT a break: some structures
        (e.g. trypsin's chymotrypsin numbering, 3ptb) skip numbers while the
        backbone stays covalently continuous. Splitting there would sever a
        real peptide bond and make Meeko fail the *opposite* way ("excess
        inter-residue bond(s)"). So a gap is only split when the peptide bond
        C(i)->N(i+1) is too long to be bonded.

        Returns the (chain, resnum) of each residue immediately before an
        inserted gap -- these are synthetic chain ends, not real chain
        termini. PDBFixer still protonates them as if they were: it caps
        each with a terminal OXT, which Meeko's bond-order perception can
        then misassign to CA instead of the backbone C, producing an
        overvalent carbon. Callers must strip those OXT atoms again after
        protonation (see _strip_spurious_oxt).
        """
        with open(pdb_path) as handle:
            lines = handle.readlines()

        # Pass 1: record each residue's backbone N and C coordinates, in order,
        # so a numbering gap can be tested for an actual backbone discontinuity.
        PEPTIDE_BOND_MAX = 2.5  # Å; a real C(i)->N(i+1) peptide bond is ~1.33 Å

        residues: List[dict] = []
        index_by_key: Dict[Tuple[str, int], int] = {}
        for line in lines:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            chain = line[21]
            try:
                resnum = int(line[22:26])
            except ValueError:
                continue
            rkey = (chain, resnum)
            if rkey not in index_by_key:
                index_by_key[rkey] = len(residues)
                residues.append({"chain": chain, "resnum": resnum, "N": None, "C": None})
            atom_name = line[12:16].strip()
            if atom_name in ("N", "C") and residues[index_by_key[rkey]][atom_name] is None:
                try:
                    residues[index_by_key[rkey]][atom_name] = (
                        float(line[30:38]), float(line[38:46]), float(line[46:54])
                    )
                except ValueError:
                    pass

        # A gap is a real break only if the backbone C->N distance exceeds a
        # peptide bond (or a backbone atom is missing, so continuity is unknown).
        break_after = set()  # original (chain, resnum) after which to split
        for i in range(1, len(residues)):
            prev_res, curr_res = residues[i - 1], residues[i]
            if curr_res["chain"] != prev_res["chain"]:
                continue
            if curr_res["resnum"] - prev_res["resnum"] <= 1:
                continue
            c_prev, n_curr = prev_res["C"], curr_res["N"]
            bonded = (
                c_prev is not None
                and n_curr is not None
                and math.dist(c_prev, n_curr) <= PEPTIDE_BOND_MAX
            )
            if not bonded:
                break_after.add((prev_res["chain"], prev_res["resnum"]))

        # Pass 2: rewrite, inserting a TER + new chain letter only at real breaks.
        used_chains = {line[21] for line in lines if line.startswith(("ATOM", "HETATM"))}
        available_chains = [c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if c not in used_chains]

        out_lines = []
        synthetic_termini: List[Tuple[str, str]] = []
        prev_key = None  # (original_chain, resnum) of the last ATOM/HETATM seen
        chain_override: Dict[str, str] = {}
        for line in lines:
            if line.startswith(("ATOM", "HETATM")):
                orig_chain = line[21]
                resnum = int(line[22:26])
                key = (orig_chain, resnum)

                if prev_key is not None and key != prev_key and prev_key in break_after:
                    out_lines.append("TER\n")
                    # prev_key's residue was written under its own original
                    # chain letter (the override below only applies going
                    # forward), so that's the (chain, resnum) PDBFixer will
                    # see as this segment's C-terminus.
                    synthetic_termini.append((prev_key[0], str(prev_key[1])))
                    if available_chains:
                        chain_override[orig_chain] = available_chains.pop(0)
                    else:
                        logger.warning(
                            f"Ran out of spare chain letters splitting chain "
                            f"{orig_chain!r} at residue {resnum}; this segment "
                            "keeps the previous chain ID and may still confuse "
                            "Meeko's polymer builder."
                        )

                current_chain = chain_override.get(orig_chain, orig_chain)
                if current_chain != orig_chain:
                    line = line[:21] + current_chain + line[22:]

                prev_key = key
            out_lines.append(line)

        with open(pdb_path, "w") as handle:
            handle.writelines(out_lines)

        return synthetic_termini

    @staticmethod
    def _strip_spurious_oxt(pdb_path: str, synthetic_termini: List[Tuple[str, str]]) -> None:
        """Remove OXT atoms PDBFixer added at synthetic chain-break points.

        _insert_chain_break_ters() splits disordered-loop gaps into separate
        chains so Meeko's polymer builder treats them as unbonded. PDBFixer,
        seeing what looks like a genuine chain end there, caps it with a
        terminal OXT -- but that residue isn't actually the protein's real
        C-terminus, just a gap boundary, so the cap is spurious and confuses
        RDKit's bond-order perception downstream in Meeko (a 5-valent
        carbon: "Explicit valence ... is greater than permitted").
        """
        if not synthetic_termini:
            return

        targets = set(synthetic_termini)
        with open(pdb_path) as handle:
            lines = handle.readlines()

        out_lines = []
        removed = 0
        for line in lines:
            if line.startswith("ATOM") and line[12:16].strip() == "OXT":
                chain = line[21]
                resnum = line[22:26].strip()
                if (chain, resnum) in targets:
                    removed += 1
                    continue
            out_lines.append(line)

        if removed:
            logger.info(
                f"Stripped {removed} spurious OXT atom(s) at synthetic "
                f"chain-break boundaries: {sorted(targets)}"
            )
            with open(pdb_path, "w") as handle:
                handle.writelines(out_lines)

    @staticmethod
    def _repair_invalid_terminal_oxt(
        pdb_path: str,
        min_nonbonded_distance: float = 1.70,
        warning_log: Optional[str] = None,
    ) -> List[Tuple[str, str]]:
        """Rebuild terminal OXT atoms with impossible PDBFixer geometry."""
        with open(pdb_path) as handle:
            lines = handle.readlines()

        residues = {}
        for index, line in enumerate(lines):
            if not line.startswith("ATOM"):
                continue
            key = (line[21], line[22:26].strip(), line[26])
            name = line[12:16].strip()
            xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            residues.setdefault(key, {})[name] = (index, xyz)

        repaired = []
        for (chain, resnum, _icode), atoms in residues.items():
            if not {"CA", "C", "O", "OXT"}.issubset(atoms):
                continue
            oxt_index, oxt = atoms["OXT"]
            _, carbon = atoms["C"]
            _, oxygen = atoms["O"]
            _, alpha_carbon = atoms["CA"]
            c_oxt = math.dist(carbon, oxt)
            o_oxt = math.dist(oxygen, oxt)
            invalid = not 1.05 <= c_oxt <= 1.55
            invalid = invalid or o_oxt < min_nonbonded_distance
            invalid = invalid or math.dist(alpha_carbon, oxt) < min_nonbonded_distance
            if not invalid:
                continue

            def direction(target):
                vector = tuple(target[i] - carbon[i] for i in range(3))
                length = math.sqrt(sum(value * value for value in vector))
                if length < 1e-6:
                    raise RuntimeError(
                        f"Cannot repair OXT at {chain}:{resnum}: degenerate coordinates"
                    )
                return tuple(value / length for value in vector)

            ca_direction = direction(alpha_carbon)
            o_direction = direction(oxygen)
            opposite_bisector = tuple(
                -(ca_direction[i] + o_direction[i]) for i in range(3)
            )
            norm = math.sqrt(sum(value * value for value in opposite_bisector))
            if norm < 1e-6:
                raise RuntimeError(
                    f"Cannot repair OXT at {chain}:{resnum}: collinear CA/C/O"
                )
            new_oxt = tuple(
                carbon[i] + 1.25 * opposite_bisector[i] / norm for i in range(3)
            )
            line = lines[oxt_index]
            lines[oxt_index] = (
                f"{line[:30]}{new_oxt[0]:8.3f}{new_oxt[1]:8.3f}"
                f"{new_oxt[2]:8.3f}{line[54:]}"
            )
            repaired.append((chain, resnum))
            warning_message = (
                "Rebuilt invalid terminal OXT at %s:%s "
                "(old C-OXT %.2f A, old O-OXT %.2f A)"
                % (chain, resnum, c_oxt, o_oxt)
            )
            if warning_log:
                os.makedirs(os.path.dirname(os.path.abspath(warning_log)), exist_ok=True)
                with open(warning_log, "a", encoding="utf-8") as handle:
                    timestamp = datetime.now().isoformat(timespec="seconds")
                    receptor = os.path.basename(pdb_path)
                    handle.write(
                        f"{timestamp} | WARNING | receptor_preparation | "
                        f"{receptor} | {warning_message}\n"
                    )

        if repaired:
            with open(pdb_path, "w") as handle:
                handle.writelines(lines)
        return repaired

    @staticmethod
    def _validate_rdkit_geometry(pdb_path: str) -> None:
        """Validate perceived bonds and report residue context before Meeko."""
        mol = Chem.MolFromPDBFile(pdb_path, sanitize=False, removeHs=False)
        if mol is None:
            raise RuntimeError(f"RDKit could not parse repaired receptor: {pdb_path}")
        try:
            Chem.SanitizeMol(mol)
        except Exception as exc:
            offenders = []
            for atom in mol.GetAtoms():
                order = sum(bond.GetBondTypeAsDouble() for bond in atom.GetBonds())
                if (atom.GetSymbol() == "O" and order > 2.0) or atom.GetDegree() > 4:
                    info = atom.GetPDBResidueInfo()
                    if info is not None:
                        offenders.append(
                            f"{info.GetName().strip()} {info.GetResidueName()} "
                            f"{info.GetChainId()}:{info.GetResidueNumber()}"
                        )
            detail = f" Suspect atoms: {', '.join(offenders[:8])}." if offenders else ""
            raise RuntimeError(
                "Receptor has impossible bond geometry after automatic repair."
                f"{detail} RDKit: {exc}"
            ) from exc

    @staticmethod
    def _relax_rebuilt_atoms(
        fixer: PDBFixer, original_pdb: str, jitter_seed: Optional[int] = None
    ) -> None:
        """Relax PDBFixer additions while restraining experimental atoms.

        A residue missing its entire side chain in the deposited structure
        (see REMARK 470) forces PDBFixer to place several atoms at once with
        no local reference geometry; the restrained minimization below can
        settle them into a position that is locally valid but still clashes
        sterically with an unrelated neighboring residue. When jitter_seed is
        given, the rebuilt (non-experimental) heavy atoms are nudged by a
        small random offset before minimizing, letting a retry escape that
        particular local minimum instead of reproducing it deterministically.
        """
        original_atoms = set()
        with open(original_pdb) as handle:
            for line in handle:
                if line.startswith(("ATOM", "HETATM")):
                    original_atoms.add(
                        (line[21], line[22:26].strip(), line[26], line[12:16].strip())
                    )

        fixer.addMissingHydrogens(7.0)

        if jitter_seed is not None:
            import random

            rng = random.Random(jitter_seed)
            positions = fixer.positions.value_in_unit(unit.nanometer)
            jittered = []
            for index, atom in enumerate(fixer.topology.atoms()):
                residue = atom.residue
                key = (residue.chain.id, residue.id, residue.insertionCode, atom.name)
                x, y, z = positions[index]
                if key not in original_atoms and atom.element.symbol != "H":
                    x += rng.uniform(-0.15, 0.15)
                    y += rng.uniform(-0.15, 0.15)
                    z += rng.uniform(-0.15, 0.15)
                jittered.append((x, y, z))
            fixer.positions = unit.Quantity(jittered, unit.nanometer)

        forcefield = ForceField("amber14-all.xml")
        system = forcefield.createSystem(
            fixer.topology, nonbondedMethod=NoCutoff, constraints=HBonds
        )
        restraint = CustomExternalForce(
            "0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)"
        )
        restraint.addGlobalParameter(
            "k", 10000.0 * unit.kilojoule_per_mole / unit.nanometer**2
        )
        restraint.addPerParticleParameter("x0")
        restraint.addPerParticleParameter("y0")
        restraint.addPerParticleParameter("z0")
        restrained = 0
        for index, atom in enumerate(fixer.topology.atoms()):
            residue = atom.residue
            key = (residue.chain.id, residue.id, residue.insertionCode, atom.name)
            if key in original_atoms and atom.element.symbol != "H":
                restraint.addParticle(index, fixer.positions[index])
                restrained += 1
        system.addForce(restraint)

        integrator = VerletIntegrator(0.001 * unit.picoseconds)
        context = Context(system, integrator)
        try:
            context.setPositions(fixer.positions)
            LocalEnergyMinimizer.minimize(context, maxIterations=1000)
            fixer.positions = context.getState(getPositions=True).getPositions()
        finally:
            del context
            del integrator
        logger.warning(
            "Restrained minimization repaired PDBFixer-added atoms; "
            "%d experimental heavy atoms were restrained",
            restrained,
        )

    @staticmethod
    def _write_heavy_atom_pdb(fixer: PDBFixer, pdb_path: str) -> None:
        """Write a PDBFixer model while excluding all hydrogens."""
        with open(pdb_path, "w") as handle:
            PDBFile.writeFile(fixer.topology, fixer.positions, handle, keepIds=True)
        with open(pdb_path) as handle:
            pdb_lines = handle.readlines()
        with open(pdb_path, "w") as handle:
            for line in pdb_lines:
                if line.startswith(("ATOM", "HETATM")):
                    element = line[76:78].strip().upper()
                    atom_name = line[12:16].strip().upper()
                    if element == "H" or (not element and atom_name.startswith("H")):
                        continue
                handle.write(line)

    @staticmethod
    def _detect_disulfide_residues(pdb_path: str, max_bond_dist: float = 2.5) -> List[Tuple[str, str]]:
        """Return (chain, resnum) pairs of CYS residues linked by a disulfide bond.

        Meeko perceives bonds by distance and treats a Cys-Cys S-S bridge
        (~2.05 A) as an inter-residue bond; near a chain-break gap this
        confuses its polymer builder ("Expected N paddings ... but got 0").
        Declaring the involved cysteines as CYX via --set_template tells
        Meeko they are deliberately cross-linked, not a chain-connectivity
        error.
        """
        sg_atoms = []
        with open(pdb_path) as handle:
            for line in handle:
                if line.startswith("ATOM") and line[12:16].strip() == "SG" and line[17:20].strip() == "CYS":
                    chain = line[21]
                    resnum = line[22:26].strip()
                    xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
                    sg_atoms.append((chain, resnum, xyz))

        disulfide_residues = set()
        for i in range(len(sg_atoms)):
            for j in range(i + 1, len(sg_atoms)):
                chain_a, resnum_a, xyz_a = sg_atoms[i]
                chain_b, resnum_b, xyz_b = sg_atoms[j]
                dist = math.dist(xyz_a, xyz_b)
                if dist <= max_bond_dist:
                    disulfide_residues.add((chain_a, resnum_a))
                    disulfide_residues.add((chain_b, resnum_b))

        return sorted(disulfide_residues)

    def _run_prepare_receptor(
        self,
        input_pdb: str,
        output_pdbqt: str,
        synthetic_termini: Optional[List[Tuple[str, str]]] = None,
        warning_log: Optional[str] = None,
    ) -> None:
        """Repair heavy atoms with PDBFixer and parameterize/write PDBQT with Meeko.

        Hydrogens are deliberately left to Meeko.  OpenMM/PDBFixer places newly
        added hydrogens stochastically and, for some large experimental structures,
        can place two hydrogens almost on top of one another.  Meeko then perceives
        spurious inter-residue bonds and rejects otherwise valid residues.  Meeko's
        polymer templates add the hydrogens it needs deterministically.
        """
        protonated_pdb = os.path.splitext(input_pdb)[0] + "_h.pdb"

        fixer = PDBFixer(filename=input_pdb)
        fixer.findMissingResidues()
        # Discard missing-residue/loop gaps instead of rebuilding them:
        # PDBFixer models them de novo via OpenMM energy minimization, which
        # reliably diverges to NaN coordinates on large gaps (e.g. disordered
        # GPCR loops or truncated termini). Those regions are typically not
        # part of the binding pocket anyway, so only missing atoms within
        # existing residues (incomplete side chains) and hydrogens are added.
        # findMissingAtoms() still requires missingResidues to exist as an
        # (empty) dict, so it's cleared rather than skipped.
        fixer.missingResidues = {}
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        # keepIds=True preserves chain splits and original residue numbering.
        self._write_heavy_atom_pdb(fixer, protonated_pdb)

        self._strip_spurious_oxt(protonated_pdb, synthetic_termini or [])
        self._repair_invalid_terminal_oxt(protonated_pdb, warning_log=warning_log)
        try:
            self._validate_rdkit_geometry(protonated_pdb)
        except RuntimeError as initial_error:
            logger.warning("Initial repaired geometry is invalid: %s", initial_error)
            # A residue missing its whole side chain (REMARK 470) occasionally
            # settles into a position that's locally valid but clashes with a
            # neighboring residue even after restrained minimization. Retry a
            # few times, nudging only the rebuilt atoms on later attempts, so
            # a bad local minimum doesn't sink the whole receptor.
            max_relax_attempts = 3
            last_error = initial_error
            for attempt in range(max_relax_attempts):
                relax_fixer = PDBFixer(filename=input_pdb)
                relax_fixer.findMissingResidues()
                relax_fixer.missingResidues = {}
                relax_fixer.findMissingAtoms()
                relax_fixer.addMissingAtoms()
                jitter_seed = attempt or None
                self._relax_rebuilt_atoms(relax_fixer, input_pdb, jitter_seed=jitter_seed)
                self._write_heavy_atom_pdb(relax_fixer, protonated_pdb)
                self._strip_spurious_oxt(protonated_pdb, synthetic_termini or [])
                self._repair_invalid_terminal_oxt(protonated_pdb, warning_log=warning_log)
                try:
                    self._validate_rdkit_geometry(protonated_pdb)
                    last_error = None
                    break
                except RuntimeError as retry_error:
                    last_error = retry_error
                    logger.warning(
                        "Relax attempt %d/%d still invalid: %s",
                        attempt + 1, max_relax_attempts, retry_error,
                    )
            if last_error is not None:
                raise last_error

        abs_output_pdbqt = os.path.abspath(output_pdbqt)
        out_prefix = abs_output_pdbqt[:-len(".pdbqt")] if abs_output_pdbqt.endswith(".pdbqt") else abs_output_pdbqt
        cmd = [
            "mk_prepare_receptor.py",
            "--read_pdb", os.path.abspath(protonated_pdb),
            "-o", out_prefix,
            "-p",
        ]

        disulfides = self._detect_disulfide_residues(protonated_pdb)
        if disulfides:
            set_template = ",".join(f"{chain}:{resnum}=CYX" for chain, resnum in disulfides)
            cmd.extend(["--set_template", set_template])
            logger.info(f"Detected {len(disulfides)} disulfide-bonded CYS -> CYX: {disulfides}")

        logger.info(f"Running: {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        except subprocess.TimeoutExpired:
            raise RuntimeError("mk_prepare_receptor.py timed out after 300s")

        if result.returncode != 0:
            error_msg = f"mk_prepare_receptor.py failed with code {result.returncode}\n"
            if result.stdout:
                error_msg += f"STDOUT:\n{result.stdout}\n"
            if result.stderr:
                error_msg += f"STDERR:\n{result.stderr}\n"
            logger.error(error_msg)
            raise RuntimeError(error_msg.strip())

        if not os.path.isfile(abs_output_pdbqt) or os.path.getsize(abs_output_pdbqt) == 0:
            raise RuntimeError(f"Output PDBQT empty or missing: {abs_output_pdbqt}")

        # Unlike clean_pdb (a debug/audit artifact gated by keep_clean_pdb),
        # protonated_pdb is a required input for GPF generation in the
        # active-site-detection step (Meeko needs a PDB, not a .pdbqt) and is
        # always kept regardless of keep_clean_pdb.
        logger.info(f"Generated PDBQT: {abs_output_pdbqt}")

    def _run_prepare_ligand(self, input_pdb: str, output_pdbqt: str) -> None:
        """Prepare a non-protein 'synthetic receptor' with Meeko's ligand-side path.

        Mirrors process_ligand(): unlike proteins, these molecules keep their
        original conformation, so hydrogens are added onto the existing
        coordinates instead of re-embedding a new one.
        """
        if not os.path.isfile(input_pdb):
            raise FileNotFoundError(f"Input PDB file not found: {input_pdb}")

        mol = Chem.MolFromPDBFile(input_pdb, removeHs=False)
        if mol is None:
            raise RuntimeError(f"RDKit failed to read receptor: {input_pdb}")

        mol = Chem.AddHs(mol, addCoords=True)

        mk_prep = MoleculePreparation()
        molsetup = mk_prep(mol)[0]
        pdbqt_string, is_ok, err_msg = PDBQTWriterLegacy.write_string(molsetup)
        if not is_ok:
            raise RuntimeError(f"Meeko failed to write PDBQT: {err_msg}")

        with open(output_pdbqt, "w") as handle:
            handle.write(pdbqt_string)

        logger.info(f"Generated PDBQT (synthetic molecule): {output_pdbqt}")

    def prepare_receptor(self, receptor_file: str, output_path: str) -> str:
        """Run receptor cleanup + Meeko conversion and return output PDBQT path.

        Handles both standard proteins (PDBFixer + Meeko receptor path) and
        synthetic molecules like quitosano (Meeko ligand path).
        """
        base = find_compound_name(receptor_file)
        safe_base = base.replace(" ", "_").replace(",", "_")

        self._read_receptor(receptor_file)
        residues = self._get_residue_summary()
        logger.info(f"Residues found: {residues}")

        is_protein = self._is_protein()
        
        if is_protein:
            logger.info(f"Detected standard protein molecule - using PDBFixer + Meeko")
            
            clean_pdb = create_out_file(
                f"{output_path}/clean_receptors",
                f"{safe_base}_clean.pdb",
            )
            
            self._remove_non_protein()
            self._save_clean_receptor(clean_pdb)
            synthetic_termini = self._insert_chain_break_ters(clean_pdb)
            logger.info(f"Saved cleaned receptor to {clean_pdb}")

            out_pdbqt = create_out_file(
                f"{output_path}/prepared_receptors_pdbqt",
                f"{safe_base}_clean.pdbqt",
            )
            self._run_prepare_receptor(
                clean_pdb,
                out_pdbqt,
                synthetic_termini=synthetic_termini,
                warning_log=os.path.join(output_path, "receptor_preparation.log"),
            )

            cofactor_pdb = self._cofactor_pdb_for(clean_pdb)
            if self._cofactors and os.path.isfile(cofactor_pdb):
                # The sidecars stay next to the clean receptor even when
                # keep_clean_pdb is off: active-site detection re-appends them
                # after Meeko rewrites the receptor PDBQT with the GPF.
                cofactor_pdbqt = os.path.splitext(cofactor_pdb)[0] + ".pdbqt"
                write_cofactor_pdbqt(cofactor_pdb, cofactor_pdbqt)
                append_cofactors(out_pdbqt, cofactor_pdbqt)
            
            if not self.keep_clean_pdb and os.path.exists(clean_pdb):
                os.remove(clean_pdb)
                logger.debug(f"Removed intermediate: {clean_pdb}")
        else:
            logger.info(f"Detected synthetic molecule (non-protein) - using Meeko (ligand path)")
            # Para moléculas sintéticas, usar el PDB original directamente sin limpiar
            
            out_pdbqt = create_out_file(
                f"{output_path}/prepared_receptors_pdbqt",
                f"{safe_base}.pdbqt",
            )
            self._run_prepare_ligand(receptor_file, out_pdbqt)

        return out_pdbqt
