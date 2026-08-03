"""Receptor processing utilities for cleanup and PDBQT conversion."""

import math
import os
import subprocess
from typing import Dict, List, Optional, Tuple

try:
    from openbabel import openbabel as ob
except ImportError:
    import openbabel as ob

from meeko import MoleculePreparation, PDBQTWriterLegacy
from pdbfixer import PDBFixer
from openmm.app import PDBFile
from rdkit import Chem

from ..storage.file_manager import create_out_file, find_compound_name
from .logger import setup_logger

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
    ):
        self.mol = ob.OBMol()
        self.remove_water = remove_water
        self.remove_ligands = remove_ligands
        self.remove_ions = remove_ions
        self.keep_clean_pdb = keep_clean_pdb
        self._input_path: Optional[str] = None

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

        if not conv.ReadFile(self.mol, receptor_file):
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
        with open(self._input_path) as handle:
            for line in handle:
                if line.startswith(("ATOM", "HETATM")):
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

    @staticmethod
    def _insert_chain_break_ters(pdb_path: str) -> List[Tuple[str, str]]:
        """Split chains at residue-number gaps (disordered loops, missing termini).

        Receptors with disordered loops (common in GPCRs, cryo-EM structures)
        have skips in residue numbering where OpenBabel writes a single
        contiguous chain. PDBFixer then renumbers everything sequentially and
        keeps the same chain letter across the gap, so Meeko's polymer
        builder later infers a peptide bond from chain/residue adjacency
        alone and fails ("Expected N paddings ... but got 0") even though a
        bare TER is present. Assigning a new chain letter to the segment
        after each gap (in addition to the TER) makes both tools treat it as
        a genuinely separate, unbonded chain.

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

                if (
                    prev_key is not None
                    and key != prev_key
                    and orig_chain == prev_key[0]
                    and resnum - prev_key[1] > 1
                ):
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
    ) -> None:
        """Protonate with PDBFixer and parameterize/write PDBQT with Meeko."""
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
        fixer.addMissingHydrogens(7.0)
        with open(protonated_pdb, "w") as handle:
            # keepIds=True: the default reassigns chain letters and
            # residue numbers from scratch, which erases the chain splits
            # from _insert_chain_break_ters() and reintroduces the gap
            # ambiguity that breaks Meeko's polymer builder downstream.
            PDBFile.writeFile(fixer.topology, fixer.positions, handle, keepIds=True)

        self._strip_spurious_oxt(protonated_pdb, synthetic_termini or [])

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
            self._run_prepare_receptor(clean_pdb, out_pdbqt, synthetic_termini=synthetic_termini)
            
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
