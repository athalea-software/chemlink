"""Peptide structure processing.

Rebuilds a clean, LightDock-ready PDB directly from the chemical
connectivity of a source SDF, instead of relying on Open Babel's residue
perception (OBChainsParser), which is unreliable for peptides beyond
~5-10 residues regardless of input geometry (verified empirically against
lightdock_test/: it produces "UNK" residues or mismatched atom names,
which LightDock's DFIRE scoring then rejects, or geometrically degenerate
output that hangs lightdock3_setup's swarm calculation).

Method:
  1. RDKit generates a real 3D conformation (ETKDG + MMFF), the same
     technique MoleculeProcessor uses for small-molecule ligands.
  2. The backbone is located with an N-CA-C(=O) pattern and residues are
     chained by their amide bonds, N-terminal to C-terminal.
  3. Each side chain is identified by its own composition (heavy atom
     count, formula, rings) against the 20 standard amino acids, and
     assigned the correct PDB atom names (CB, CG, OG1, ...) by walking
     the graph from CA.
"""

import math
import os
import subprocess
import sys
import traceback
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem

from ..storage.file_manager import find_compound_name, create_out_file
from .logger import setup_logger

logger = setup_logger(__name__)


class NonCanonicalResidueError(RuntimeError):
    """Raised when a residue or chain topology doesn't match any of the
    20 standard amino acids (unusual ring, unrecognized side-chain
    formula, ambiguous/unresolved backbone). Distinguished from a plain
    RuntimeError so callers (process_peptide) can tell "this peptide is
    genuinely non-canonical" apart from an unexpected bug, and exclude
    it with a clear label instead of a raw traceback -- DFIRE couldn't
    score it either way, since it only has templates for the 20
    standard residues.
    """


class PeptideProcessor:
    """Rebuilds standard-residue peptide PDBs from SDF connectivity."""

    BACKBONE_SMARTS = Chem.MolFromSmarts("[NX3][CX4][CX3](=[OX1])")

    # (heavy_atom_count, tuple(sorted(Counter(symbol).items()))) -> resname.
    # LEU/ILE share a formula (4C) and are broken by the degree of the atom
    # bonded to CB instead. Ring-containing side chains (HIS/PHE/TYR/TRP)
    # are resolved separately via GetRingInfo.
    SIGNATURES = {
        (1, (("C", 1),)): "ALA",
        (2, (("C", 1), ("O", 1))): "SER",
        (2, (("C", 1), ("S", 1))): "CYS",
        (3, (("C", 2), ("O", 1))): "THR",
        (3, (("C", 3),)): "VAL",
        (4, (("C", 3), ("S", 1))): "MET",
        (4, (("C", 2), ("O", 2))): "ASP",
        (4, (("C", 2), ("N", 1), ("O", 1))): "ASN",
        (5, (("C", 3), ("O", 2))): "GLU",
        (5, (("C", 3), ("N", 1), ("O", 1))): "GLN",
        (5, (("C", 4), ("N", 1))): "LYS",
        (7, (("C", 4), ("N", 3))): "ARG",
    }

    NAME_LISTS = {
        "ALA": ["CB"],
        "SER": ["CB", "OG"],
        "CYS": ["CB", "SG"],
        "THR": ["CB", "OG1", "CG2"],
        "VAL": ["CB", "CG1", "CG2"],
        "LEU": ["CB", "CG", "CD1", "CD2"],
        "ILE": ["CB", "CG1", "CG2", "CD1"],
        "MET": ["CB", "CG", "SD", "CE"],
        "ASP": ["CB", "CG", "OD1", "OD2"],
        "ASN": ["CB", "CG", "OD1", "ND2"],
        "GLU": ["CB", "CG", "CD", "OE1", "OE2"],
        "GLN": ["CB", "CG", "CD", "OE1", "NE2"],
        "LYS": ["CB", "CG", "CD", "CE", "NZ"],
        "ARG": ["CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"],
    }

    def __init__(self, num_conformers: int = 3, random_seed: int = 42, standardize: bool = False):
        """Initialize processor with conformer generation parameters.

        Args:
            num_conformers: Number of 3D conformers to try before keeping
                the lowest-MMFF-energy one
            random_seed: Random seed for reproducibility
            standardize: If True, instead of excluding non-canonical peptides,
                mutate every residue the classifier can't recognize to alanine
                (backbone N-CA-C-O + CB, the rest of the side chain dropped) and
                linearize crosslinked/lariat topologies. The result is a plain
                standard-residue peptide that DFIRE/LightDock can score -- a
                docking pre-filter approximation that loses the modified side
                chains. Names of the mutated residues are collected in
                ``self.standardized_residues`` after each build_pdb().
        """
        self.num_conformers = num_conformers
        self.random_seed = random_seed
        self.standardize = standardize
        self.standardized_residues: List[Tuple[int, str]] = []

    def read_molecule(self, sdf_path: str) -> Chem.Mol:
        """Read the peptide molecule from an SDF file.

        Args:
            sdf_path: Path to the source SDF file

        Returns:
            RDKit molecule object

        Raises:
            RuntimeError: If the file could not be read
        """
        suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
        mol = suppl[0] if suppl and len(suppl) > 0 else None
        if mol is None:
            raise RuntimeError(f"RDKit failed to read molecule from {sdf_path}")
        return mol

    def embed_3d(self, mol: Chem.Mol) -> Chem.Mol:
        """Generate real 3D conformers and return the lowest-MMFF-energy one.

        Args:
            mol: RDKit molecule (typically 2D, as read from a PubChem SDF)

        Returns:
            Molecule with a single, heavy-atom-only 3D conformer

        Raises:
            RuntimeError: If conformer generation fails
        """
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = self.random_seed
        params.pruneRmsThresh = 0.5
        # The default distance-geometry embedding fails outright on
        # larger/flexible peptides (>~10 residues); random coords as a
        # starting point make it converge reliably instead.
        params.useRandomCoords = True
        conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=self.num_conformers, params=params)
        if not conf_ids:
            raise RuntimeError("RDKit could not generate any 3D conformer")

        results = AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=1)
        converged = [(cid, e) for cid, (status, e) in zip(conf_ids, results) if status == 0]
        best_id = min(converged, key=lambda t: t[1])[0] if converged else conf_ids[0]

        best = Chem.Mol(mol)
        best.RemoveAllConformers()
        best.AddConformer(mol.GetConformer(best_id), assignId=True)
        # RemoveHs (unlike RemoveAllHs) keeps H on formally charged atoms or
        # certain double bonds (e.g. ARG's guanidino =NH); LightDock strips
        # hydrogens at setup anyway (--noh), so there's no reason to keep any.
        return Chem.RemoveAllHs(best)

    def find_residues(self, mol: Chem.Mol) -> List[Dict]:
        """Pair each Calpha with its backbone N and C(=O).

        Args:
            mol: RDKit molecule with a 3D conformer

        Returns:
            List of {"n", "ca", "c", "o"} atom-index dicts, one per residue
        """
        residues = []
        seen_ca = set()
        for n_i, ca_i, c_i, o_i in mol.GetSubstructMatches(self.BACKBONE_SMARTS):
            if ca_i in seen_ca:
                continue
            seen_ca.add(ca_i)
            residues.append({"n": n_i, "ca": ca_i, "c": c_i, "o": o_i})
        return residues

    def order_chain(
        self, mol: Chem.Mol, residues: List[Dict], backbone_atoms: set
    ) -> List[List[Dict]]:
        """Split residues into one or more chains, N-terminal -> C-terminal,
        via the real amide bonds.

        Most peptides are a single line: one N-terminal residue whose N is
        bonded only to its own CA (optionally plus a small dead-end cap
        like an N-terminal acetyl group), unbroken from there to the
        C-terminus. Some are lariat/bicyclic designs where a side-chain
        atom on one residue (e.g. a Lys NZ, reached through a Glu/Asp
        side chain forming a lactam bridge) forms an isopeptide bond into
        another residue's N, starting what looks like a second backbone
        segment. That segment's "N" has an extra heavy neighbor too, but
        unlike a simple cap, walking outward from that neighbor reaches
        another residue's backbone -- so it's returned as its own chain
        instead of being confused for the one true, free N-terminus.

        Args:
            mol: RDKit molecule
            residues: Unordered residues from find_residues()
            backbone_atoms: {n, ca, c, o} atom indices for every residue
                in the molecule, used to tell a branch/crosslink (reaches
                another residue's backbone) apart from a simple terminal
                cap (a small dead-end group, e.g. an acetyl)

        Returns:
            A list of chains (each a residue list, N-to-C); chains[0] is
            always the chain with the true free N-terminus.

        Raises:
            RuntimeError: If there isn't exactly one true free N-terminus,
                or the backbone doesn't fully resolve into linear chains.
        """
        n_to_idx = {r["n"]: i for i, r in enumerate(residues)}

        next_of = {}
        cut_targets = set()
        for i, r in enumerate(residues):
            for nbr in mol.GetAtomWithIdx(r["c"]).GetNeighbors():
                j = nbr.GetIdx()
                if j in n_to_idx and n_to_idx[j] != i:
                    next_of[i] = n_to_idx[j]
                    cut_targets.add(j)

        candidates = [i for i, r in enumerate(residues) if r["n"] not in cut_targets]

        def is_true_terminus(residue_idx: int) -> bool:
            n_atom, ca_atom = residues[residue_idx]["n"], residues[residue_idx]["ca"]
            extra = [nb.GetIdx() for nb in mol.GetAtomWithIdx(n_atom).GetNeighbors() if nb.GetIdx() != ca_atom]
            if not extra:
                return True
            visited = {n_atom, *extra}
            frontier = list(extra)
            while frontier:
                nxt = []
                for node in frontier:
                    if node in backbone_atoms and node != n_atom:
                        return False  # reaches another residue's backbone: a crosslink, not a cap
                    for nb in mol.GetAtomWithIdx(node).GetNeighbors():
                        j = nb.GetIdx()
                        if j not in visited:
                            visited.add(j)
                            nxt.append(j)
                frontier = nxt
            return True

        def walk(start_idx):
            chain = [start_idx]
            seen = {start_idx}
            while chain[-1] in next_of:
                nxt = next_of[chain[-1]]
                if nxt in seen:  # next_of cíclico (péptido cíclico): cortar el bucle
                    break
                chain.append(nxt)
                seen.add(nxt)
            return chain

        true_starts = [i for i in candidates if is_true_terminus(i)]

        if len(true_starts) != 1:
            if not self.standardize:
                raise NonCanonicalResidueError(
                    f"Could not determine a single N-terminus (candidates: {len(true_starts)})"
                )
            # standardize: elegir un inicio y linealizar (ignorar el crosslink).
            # Preferir un terminus real; si no hay, el candidato cuya caminata
            # cubre más residuos.
            pool = true_starts or candidates or list(range(len(residues)))
            true_starts = [max(pool, key=lambda i: len(walk(i)))]

        branch_starts = [i for i in candidates if i not in true_starts]

        chain_indices = [walk(true_starts[0])]
        claimed = set(chain_indices[0])
        for start in branch_starts:
            if start in claimed:
                continue
            branch = walk(start)
            chain_indices.append(branch)
            claimed.update(branch)

        total = sum(len(c) for c in chain_indices)
        if total != len(residues):
            if not self.standardize:
                raise NonCanonicalResidueError(f"Chain walk resolved {total} of {len(residues)} residues")
            # standardize: residuos no reclamados (por el crosslink) como cadenas propias
            for i in range(len(residues)):
                if i not in claimed:
                    chain_indices.append([i])
                    claimed.add(i)

        return [[residues[i] for i in chain] for chain in chain_indices]

    def side_chain_atoms(self, mol: Chem.Mol, residue: Dict, backbone_atoms: set):
        """BFS from CB avoiding this residue's own backbone, and stopping
        at any *other* residue's backbone reached through it (a
        side-chain isopeptide bond into another chain, see order_chain).

        If the walk loops back to the residue's own N, it's proline (ring).

        Args:
            mol: RDKit molecule
            residue: One entry from find_residues()/order_chain()
            backbone_atoms: {n, ca, c, o} atom indices for every residue
                in the molecule (all chains), precomputed once in
                build_pdb() so this stays a cheap set lookup here.

        Returns:
            ((cb_index, {side_chain_atom_indices}), reaches_own_n) or
            (None, False) for glycine (no side chain)
        """
        ca, n_i, c_i = residue["ca"], residue["n"], residue["c"]
        cb_candidates = [
            nb.GetIdx() for nb in mol.GetAtomWithIdx(ca).GetNeighbors()
            if nb.GetIdx() not in (n_i, c_i)
        ]
        if not cb_candidates:
            return None, False  # GLY: no side chain
        cb = cb_candidates[0]

        visited, frontier, reaches_n = {cb}, [cb], False
        while frontier:
            nxt = []
            for node in frontier:
                for nb in mol.GetAtomWithIdx(node).GetNeighbors():
                    j = nb.GetIdx()
                    if j == n_i:
                        reaches_n = True
                    elif j in backbone_atoms:
                        continue  # another residue's backbone via a crosslink; stop here
                    elif j != ca and j not in visited:
                        visited.add(j)
                        nxt.append(j)
            frontier = nxt
        return (cb, visited), reaches_n

    @staticmethod
    def _subtree_size(mol, node, came_from, atom_set):
        size = 1
        for nb in mol.GetAtomWithIdx(node).GetNeighbors():
            j = nb.GetIdx()
            if j != came_from and j in atom_set:
                size += PeptideProcessor._subtree_size(mol, j, node, atom_set)
        return size

    @classmethod
    def _bfs_order(cls, mol, start, exclude, atom_set):
        """BFS from start, breaking ties by descending atomic number and
        descending subtree size. Reproduces standard PDB numbering (e.g.
        OG1 before CG2 in THR, chain-continuing CG1 before branch CG2 in
        ILE) without any residue-specific special-casing.
        """
        order = [start]
        frontier = [(start, exclude)]
        while frontier:
            children = []
            for node, came_from in frontier:
                kids = [
                    nb.GetIdx() for nb in mol.GetAtomWithIdx(node).GetNeighbors()
                    if nb.GetIdx() != came_from and nb.GetIdx() in atom_set
                ]
                kids.sort(key=lambda j: (
                    -mol.GetAtomWithIdx(j).GetAtomicNum(),
                    -cls._subtree_size(mol, j, node, atom_set),
                    j,
                ))
                children.extend((k, node) for k in kids)
            order.extend(k for k, _ in children)
            frontier = children
        return order

    @staticmethod
    def _name_his(mol, cg, side_set):
        ring_nbrs = [i for i in side_set if mol.GetBondBetweenAtoms(cg, i)]
        nd1 = next(i for i in ring_nbrs if mol.GetAtomWithIdx(i).GetSymbol() == "N")
        cd2 = next(i for i in ring_nbrs if mol.GetAtomWithIdx(i).GetSymbol() == "C")
        ce1 = next(nb.GetIdx() for nb in mol.GetAtomWithIdx(nd1).GetNeighbors() if nb.GetIdx() != cg)
        ne2 = next(nb.GetIdx() for nb in mol.GetAtomWithIdx(cd2).GetNeighbors() if nb.GetIdx() != cg)
        return {cg: "CG", nd1: "ND1", cd2: "CD2", ce1: "CE1", ne2: "NE2"}

    @staticmethod
    def _name_phe_tyr(mol, cg, side_set, is_tyr):
        ring_nbrs = [i for i in side_set if mol.GetBondBetweenAtoms(cg, i)]
        cd1, cd2 = ring_nbrs[0], ring_nbrs[1]
        ce1 = next(nb.GetIdx() for nb in mol.GetAtomWithIdx(cd1).GetNeighbors() if nb.GetIdx() != cg)
        ce2 = next(nb.GetIdx() for nb in mol.GetAtomWithIdx(cd2).GetNeighbors() if nb.GetIdx() != cg)
        cz = next(nb.GetIdx() for nb in mol.GetAtomWithIdx(ce1).GetNeighbors() if nb.GetIdx() != cd1)
        names = {cg: "CG", cd1: "CD1", cd2: "CD2", ce1: "CE1", ce2: "CE2", cz: "CZ"}
        if is_tyr:
            oh = next(nb.GetIdx() for nb in mol.GetAtomWithIdx(cz).GetNeighbors() if nb.GetIdx() not in (ce1, ce2))
            names[oh] = "OH"
        return names

    @staticmethod
    def _name_trp(mol, cg, side_set):
        ring_nbrs = [i for i in side_set if mol.GetBondBetweenAtoms(cg, i)]

        def ring_degree(i):
            return sum(1 for nb in mol.GetAtomWithIdx(i).GetNeighbors() if nb.GetIdx() in side_set and nb.GetIdx() != cg)

        cd2 = max(ring_nbrs, key=ring_degree)  # fusion atom (ring degree 2)
        cd1 = next(i for i in ring_nbrs if i != cd2)

        ne1 = next(nb.GetIdx() for nb in mol.GetAtomWithIdx(cd1).GetNeighbors() if nb.GetIdx() != cg)
        cd2_nbrs = [i for i in side_set if mol.GetBondBetweenAtoms(cd2, i) and i != cg]
        ne1_nbrs = {nb.GetIdx() for nb in mol.GetAtomWithIdx(ne1).GetNeighbors()}
        ce2 = next(i for i in cd2_nbrs if i in ne1_nbrs)
        ce3 = next(i for i in cd2_nbrs if i != ce2)

        cz2 = next(nb.GetIdx() for nb in mol.GetAtomWithIdx(ce2).GetNeighbors() if nb.GetIdx() not in (cd2, ne1))
        cz3 = next(nb.GetIdx() for nb in mol.GetAtomWithIdx(ce3).GetNeighbors() if nb.GetIdx() != cd2)
        ch2 = next(nb.GetIdx() for nb in mol.GetAtomWithIdx(cz2).GetNeighbors() if nb.GetIdx() != ce2)

        return {
            cg: "CG", cd1: "CD1", ne1: "NE1", cd2: "CD2",
            ce2: "CE2", ce3: "CE3", cz2: "CZ2", cz3: "CZ3", ch2: "CH2",
        }

    def classify_residue(
        self, mol: Chem.Mol, residue: Dict, ring_info, backbone_atoms: set
    ) -> Tuple[str, Dict[int, str]]:
        """Identify one residue's amino acid type and PDB atom names.

        Args:
            mol: RDKit molecule
            residue: One entry from order_chain()
            ring_info: mol.GetRingInfo(), passed in so it's computed once
            backbone_atoms: see side_chain_atoms()

        Returns:
            (3-letter resname, {atom_idx: pdb_atom_name})
        """
        ca, n_i = residue["ca"], residue["n"]
        result, reaches_n = self.side_chain_atoms(mol, residue, backbone_atoms)
        if result is None:
            return "GLY", {}

        cb, side_set = result
        if reaches_n:
            try:
                cd = next(i for i in side_set if mol.GetBondBetweenAtoms(i, n_i))
                cg = next(i for i in side_set if i not in (cb, cd))
            except StopIteration as exc:
                raise NonCanonicalResidueError(
                    f"Ring closing back to N ({len(side_set)} atoms) doesn't match standard proline topology"
                ) from exc
            return "PRO", {cb: "CB", cg: "CG", cd: "CD"}

        rings = [set(r) for r in ring_info.AtomRings() if set(r) <= side_set]
        if rings:
            cg = next(i for i in side_set if mol.GetBondBetweenAtoms(cb, i))
            ring_atoms = side_set - {cb}  # CB is also a neighbor of CG; not part of the ring
            try:
                if len(rings) == 2:
                    names = self._name_trp(mol, cg, ring_atoms)
                    resname = "TRP"
                elif len(rings[0]) == 5:
                    names = self._name_his(mol, cg, ring_atoms)
                    resname = "HIS"
                else:
                    is_tyr = Counter(mol.GetAtomWithIdx(i).GetSymbol() for i in side_set).get("O", 0) == 1
                    names = self._name_phe_tyr(mol, cg, ring_atoms, is_tyr)
                    resname = "TYR" if is_tyr else "PHE"
            except (StopIteration, IndexError) as exc:
                # The ring-naming helpers assume a canonical His/Trp/Phe/Tyr
                # connectivity; anything else (an unusual heterocycle, a
                # non-canonical aromatic side chain) breaks that assumption
                # partway through instead of failing a clean signature
                # check up front.
                raise NonCanonicalResidueError(
                    f"Ring side chain ({len(rings)} ring(s), {len(ring_atoms)} atoms) "
                    f"doesn't match standard His/Trp/Phe/Tyr topology"
                ) from exc
            names[cb] = "CB"
            return resname, names

        counts = Counter(mol.GetAtomWithIdx(i).GetSymbol() for i in side_set)
        key = (len(side_set), tuple(sorted(counts.items())))
        if key == (4, (("C", 4),)):
            cb_degree = sum(1 for i in side_set if mol.GetBondBetweenAtoms(cb, i))
            resname = "ILE" if cb_degree == 2 else "LEU"
        else:
            resname = self.SIGNATURES.get(key)
            if resname is None:
                raise NonCanonicalResidueError(
                    f"Unrecognized side chain: {len(side_set)} atoms, {dict(counts)}"
                )

        order = self._bfs_order(mol, cb, ca, side_set)
        return resname, dict(zip(order, self.NAME_LISTS[resname]))

    @staticmethod
    def _cap_atom_name(mol, idx, taken_names):
        symbol = mol.GetAtomWithIdx(idx).GetSymbol()
        base = "OXT" if symbol == "O" else ("NT" if symbol == "N" else f"X{symbol.upper()}")
        name = base
        i = 1
        while name in taken_names:
            i += 1
            name = f"{base[:3]}{i}"
        return name

    def build_pdb(self, sdf_path: str, out_pdb_path: str) -> Dict:
        """Rebuild a clean, standard-residue peptide PDB and write it out.

        Args:
            sdf_path: Path to the source SDF file
            out_pdb_path: Path to write the reconstructed PDB to

        Returns:
            {"residues": int, "chains": int, "atoms": int,
            "modified_termini": [(resnum, resname, atom_name), ...]} --
            modified_termini is non-empty when the peptide has a
            chemically modified N- or C-terminus (e.g. C-terminal
            amidation, N-terminal acetylation) that DFIRE cannot score;
            see the comment above the cap-atom loop. chains > 1 means a
            lariat/bicyclic peptide (see order_chain) -- its extra
            chain(s) get PDB chain letters B, C, ... after the main
            chain's A, all still fully standard-residue and DFIRE-scorable.
        """
        self.standardized_residues = []
        mol = self.read_molecule(sdf_path)
        mol = self.embed_3d(mol)
        Chem.GetSymmSSSR(mol)
        ring_info = mol.GetRingInfo()

        all_residues = self.find_residues(mol)
        backbone_atoms = set()
        for r in all_residues:
            backbone_atoms.update((r["n"], r["ca"], r["c"], r["o"]))

        chains = self.order_chain(mol, all_residues, backbone_atoms)

        chain_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        atom_names: Dict[int, Tuple[str, int, str, str]] = {}
        for chain_idx, chain_residues in enumerate(chains):
            chain_letter = chain_letters[chain_idx]
            for resnum, residue in enumerate(chain_residues, start=1):
                try:
                    resname, side_names = self.classify_residue(mol, residue, ring_info, backbone_atoms)
                except NonCanonicalResidueError:
                    if not self.standardize:
                        raise
                    # mutar a ALA (backbone + CB); GLY si no tiene cadena lateral
                    sc, _ = self.side_chain_atoms(mol, residue, backbone_atoms)
                    if sc is None:
                        resname, side_names = "GLY", {}
                    else:
                        cb, _ = sc
                        resname, side_names = "ALA", {cb: "CB"}
                    self.standardized_residues.append((resnum, resname))
                atom_names[residue["n"]] = (chain_letter, resnum, resname, "N")
                atom_names[residue["ca"]] = (chain_letter, resnum, resname, "CA")
                atom_names[residue["c"]] = (chain_letter, resnum, resname, "C")
                atom_names[residue["o"]] = (chain_letter, resnum, resname, "O")
                for idx, name in side_names.items():
                    atom_names[idx] = (chain_letter, resnum, resname, name)

        # Terminal cap atoms hanging off an already-named atom. A single
        # extra oxygen (OXT, the free -COOH's second carboxyl O) is a
        # standard PDB name that DFIRE recognizes on any residue. Anything
        # else here (an amide N from C-terminal amidation, an acetyl's
        # extra carbons from N-terminal acetylation, ...) is a real
        # chemical modification with no standard name -- DFIRE only has
        # templates for the 20 unmodified amino acids, so it rejects
        # whatever we call these atoms. Tracked in modified_termini so
        # callers can flag/exclude the peptide instead of silently
        # producing a PDB that will fail deep inside LightDock's scoring
        # step. Loops in case the cap is more than one atom long.
        modified_termini: List[Tuple[int, str, str]] = []
        changed = True
        while changed:
            changed = False
            for idx, (chain_letter, resnum, resname, _) in list(atom_names.items()):
                for nbr in mol.GetAtomWithIdx(idx).GetNeighbors():
                    j = nbr.GetIdx()
                    if j not in atom_names:
                        # standardize: descartar TODOS los atomos de cap terminal
                        # no-backbone (OXT2, NT, XC, ...). DFIRE no los necesita y
                        # rechaza cualquiera que no sea de los 20 AA estandar.
                        if self.standardize:
                            continue
                        taken = {n for (_, _, _, n) in atom_names.values()}
                        cap_name = self._cap_atom_name(mol, j, taken)
                        atom_names[j] = (chain_letter, resnum, resname, cap_name)
                        if not cap_name.startswith("OXT"):
                            modified_termini.append((resnum, resname, cap_name))
                        changed = True

        if len(atom_names) != mol.GetNumAtoms():
            missing = mol.GetNumAtoms() - len(atom_names)
            if not self.standardize:
                raise NonCanonicalResidueError(f"{missing} heavy atoms left unclassified")
            # standardize: los átomos no nombrados (cadenas laterales descartadas
            # al mutar a ALA, o átomos de crosslink) simplemente no se escriben.

        conf = mol.GetConformer()
        lines = []
        # Ordenar por (cadena, residuo) -- NO por indice de RDKit (orden del SDF
        # original). Herramientas como CNS (via HADDOCK) exigen que los atomos de
        # un mismo residuo formen un bloque contiguo en el PDB; el orden del SDF
        # no respeta eso, y LightDock lo tolera pero CNS aborta con
        # "SEGMNT-ERR: attempt to enter duplicate residue" al ver un residuo
        # reaparecer tras haberse "cerrado".
        ordered_atoms = sorted(atom_names.items(), key=lambda item: (item[1][0], item[1][1]))
        for serial, (idx, (chain_letter, resnum, resname, name)) in enumerate(ordered_atoms, start=1):
            symbol = mol.GetAtomWithIdx(idx).GetSymbol()
            pos = conf.GetAtomPosition(idx)
            atom_field = name if len(name) == 4 else f" {name:<3}"
            lines.append(
                f"ATOM  {serial:>5} {atom_field}{resname:>4} {chain_letter}{resnum:>4}    "
                f"{pos.x:>8.3f}{pos.y:>8.3f}{pos.z:>8.3f}{1.0:>6.2f}{0.0:>6.2f}"
                f"          {symbol:>2}"
            )
        lines.append("TER")
        lines.append("END")

        with open(out_pdb_path, "w") as handle:
            handle.write("\n".join(lines) + "\n")

        return {
            "residues": sum(len(c) for c in chains),
            "chains": len(chains),
            "atoms": mol.GetNumAtoms(),
            "modified_termini": modified_termini,
        }


def _write_original_pdb(sdf_path: str, out_pdb: str) -> bool:
    """Escribe el PDB de la molecula ORIGINAL completa (con sus modificaciones).

    Se usa en modo standardize: el docking corre sobre la version estandarizada
    (aproximada), pero este PDB conserva el farmaco real para que la etapa de
    dinamica (MD/MMPBSA) use la molecula verdadera, no la recortada. Open Babel
    da mejores nombres de residuo que RDKit para el paso posterior de MD.
    """
    try:
        subprocess.run(["obabel", sdf_path, "-O", out_pdb],
                       capture_output=True, text=True, timeout=180, check=False)
        return os.path.isfile(out_pdb) and os.path.getsize(out_pdb) > 0
    except Exception as exc:  # pragma: no cover - best-effort
        logger.warning(f"No se pudo generar el PDB original {out_pdb}: {exc}")
        return False


def process_peptide(
    peptide_file: str, output_path: str, standardize: bool = False
) -> Tuple[str, bool, Optional[dict], List[str]]:
    """Rebuild a single peptide PDB from its source file (SDF).

    Standalone function suitable for multiprocessing, mirroring
    utils.molecule_processor.process_ligand()'s
    (name, success, error_details, warnings) contract.

    Peptides DFIRE can't score are still reported as "successful" (the
    diagnosis itself succeeded) rather than "failed", with the reason in
    `warnings` -- this keeps a batch of many peptides from failing over a
    handful that were never going to be dockable, while still surfacing
    exactly which peptides were skipped and why:
      - Modified N-/C-terminus (see PeptideProcessor.build_pdb): the PDB
        builds fine, so it's written to
        prepared_peptides_pdb/excluded_modified_termini/ instead of the
        main output directory (LightDockExecution never sees it there).
      - Non-canonical residue or unusual topology (NonCanonicalResidueError,
        e.g. an unrecognized side chain, a non-standard aromatic ring, an
        ambiguous branch): build_pdb() raises before writing anything, so
        there's nothing to move -- just the diagnostic reason in `warnings`.

    A real, unexpected failure (a bug, a malformed input file) still comes
    back as `success=False` with the full traceback in `error_details`.

    Args:
        peptide_file: Path to the source peptide file (SDF)
        output_path: Base output directory

    Returns:
        Tuple of (peptide_name, success, error_details, warnings)
    """
    peptide_name = find_compound_name(peptide_file)

    try:
        processor = PeptideProcessor(standardize=standardize)
        out_pdb = create_out_file(f"{output_path}/prepared_peptides_pdb", f"{peptide_name}.pdb")
        info = processor.build_pdb(peptide_file, out_pdb)

        if standardize:
            # El docking usa el PDB estandarizado (arriba). Guardamos TAMBIEN el
            # PDB original completo para que solo la molecula verdadera pase a MD.
            orig_pdb = create_out_file(f"{output_path}/original_peptides_pdb", f"{peptide_name}.pdb")
            _write_original_pdb(peptide_file, orig_pdb)
            std = processor.standardized_residues
            if std:
                logger.info(
                    f"{peptide_name}: estandarizado para docking "
                    f"({len(std)} residuo(s) -> ALA/GLY: {std}); "
                    f"molecula original guardada en original_peptides_pdb/{peptide_name}.pdb"
                )
            # En modo standardize nada se excluye: el peptido es dockeable.
            return (peptide_name, True, None, [])

        modified_termini = info.get("modified_termini") or []
        if modified_termini:
            excluded_pdb = create_out_file(
                f"{output_path}/prepared_peptides_pdb/excluded_modified_termini",
                f"{peptide_name}.pdb",
            )
            os.replace(out_pdb, excluded_pdb)
            details = ", ".join(f"{resname}.{resnum} atom {name}" for resnum, resname, name in modified_termini)
            warning = (
                f"Excluded from LightDock docking: modified terminus ({details}) -- "
                f"DFIRE only supports the 20 unmodified amino acids. PDB saved to "
                f"prepared_peptides_pdb/excluded_modified_termini/{peptide_name}.pdb"
            )
            logger.warning(f"{peptide_name}: {warning}")
            return (peptide_name, True, None, [warning])

        return (peptide_name, True, None, [])
    except NonCanonicalResidueError as e:
        # Same reasoning as the modified-termini branch above: DFIRE only
        # has templates for the 20 standard amino acids, so a residue or
        # topology it can't classify is just as undockable as a capped
        # terminus. Reported as a clean exclusion instead of a failure --
        # build_pdb() raises before writing anything, so there's no
        # partial/corrupt PDB left behind, just the diagnostic reason.
        warning = (
            f"Excluded from LightDock docking: non-canonical residue or unusual "
            f"structure ({e}) -- DFIRE only supports the 20 standard amino acids."
        )
        logger.warning(f"{peptide_name}: {warning}")
        return (peptide_name, True, None, [warning])
    except Exception as e:
        error_details = {
            "peptide": peptide_name,
            "file": peptide_file,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
        logger.error(f"{peptide_name}: Failed to prepare - {str(e)}")
        return (peptide_name, False, error_details, [])


# Reasonable ranges (A) for the three backbone bonds that appear once per
# residue -- N-CA and CA-C within a residue, C(i)-N(i+1) across the peptide
# bond. Loose on purpose (real values cluster near 1.33-1.52 A): this is a
# corruption smoke test (collapsed/exploded coordinates from a bad embed),
# not a geometry validator, so it should not flag normal residues.
_BACKBONE_BOND_RANGE = (0.8, 2.2)


def validate_prepared_pdb(pdb_path: str) -> Dict[str, Any]:
    """QC pass over one prepared peptide PDB before it's trusted for docking.

    Catches the failure modes seen in practice that a bare "did build_pdb
    raise" check misses -- the file parses fine but is silently wrong:
      - residues split across non-contiguous blocks (the real bug found
        this session: build_pdb sorted atoms by RDKit index instead of by
        residue, which CNS/HADDOCK reject outright and other tools may
        mis-handle silently)
      - NaN/Inf coordinates (a failed embed that RDKit didn't error on)
      - a degenerate bounding box (atoms collapsed to ~one point)
      - backbone bonds (N-CA, CA-C, peptide C-N) outside a sane range,
        which flags corrupted 3D coordinates even when every other check
        passes

    Returns {"qc_status": "OK"|"WARN"|"FAIL", "issues": [str, ...]}.
    FAIL means don't trust this file for docking; WARN means usable but
    worth a second look; OK means no issues found.
    """
    issues: List[str] = []
    atoms: List[Tuple[str, int, str, Tuple[float, float, float]]] = []  # chain, resnum, name, xyz

    if not os.path.isfile(pdb_path) or os.path.getsize(pdb_path) == 0:
        return {"qc_status": "FAIL", "issues": ["file missing or empty"]}

    with open(pdb_path, "r", errors="ignore") as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            try:
                chain = line[21]
                resnum = int(line[22:26])
                name = line[12:16].strip()
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            except (ValueError, IndexError):
                issues.append(f"unparseable ATOM line: {line.strip()[:60]}")
                continue
            atoms.append((chain, resnum, name, (x, y, z)))

    if not atoms:
        return {"qc_status": "FAIL", "issues": issues + ["no ATOM records found"]}

    for _, _, name, (x, y, z) in atoms:
        if any(math.isnan(c) or math.isinf(c) for c in (x, y, z)):
            issues.append(f"NaN/Inf coordinate on atom {name}")
            break  # one is enough to flag; no need to enumerate every atom

    xs = [a[3][0] for a in atoms]
    ys = [a[3][1] for a in atoms]
    zs = [a[3][2] for a in atoms]
    bbox = (max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))
    if max(bbox) < 1.0:
        issues.append(f"degenerate bounding box {bbox} (atoms collapsed to ~one point)")

    seen_closed = set()
    current = None
    for chain, resnum, _, _ in atoms:
        key = (chain, resnum)
        if key != current:
            if current is not None:
                seen_closed.add(current)
            current = key
            if key in seen_closed:
                issues.append(f"residue {chain}.{resnum} is not contiguous in the file")
                break

    by_res: Dict[Tuple[str, int], Dict[str, Tuple[float, float, float]]] = defaultdict(dict)
    for chain, resnum, name, xyz in atoms:
        by_res[(chain, resnum)][name] = xyz
    ordered_keys = sorted(by_res)  # good enough for a bond-length smoke test
    bad_bonds = 0
    for chain, resnum in ordered_keys:
        res = by_res[(chain, resnum)]
        if "N" in res and "CA" in res:
            if not (_BACKBONE_BOND_RANGE[0] <= math.dist(res["N"], res["CA"]) <= _BACKBONE_BOND_RANGE[1]):
                bad_bonds += 1
        if "CA" in res and "C" in res:
            if not (_BACKBONE_BOND_RANGE[0] <= math.dist(res["CA"], res["C"]) <= _BACKBONE_BOND_RANGE[1]):
                bad_bonds += 1
    if bad_bonds:
        issues.append(f"{bad_bonds} backbone bond(s) outside {_BACKBONE_BOND_RANGE} A")

    if not issues:
        return {"qc_status": "OK", "issues": []}
    fatal = any("no ATOM" in i or "NaN/Inf" in i or "degenerate bounding box" in i
                or "not contiguous" in i for i in issues)
    return {"qc_status": "FAIL" if fatal else "WARN", "issues": issues}


def deduplicate_peptides(input_dir: str) -> Dict[str, List[str]]:
    """Group SDF files in `input_dir` by real structure (canonical SMILES,
    isotopes stripped).

    Isotope labels (deuterium, 13C, ...) don't change docking behavior --
    scoring functions key on element type, not mass -- so isotopologues of
    the same compound are one docking job, not several. See the memantine
    batch this session: 24 deuterated/stereo variants of one molecule that
    scored near-identically, confirming this in practice.

    Returns {canonical_smiles: [cid, ...]}, one entry per distinct real
    structure. Molecules that fail to parse are returned as their own
    singleton group keyed "UNPARSEABLE:<cid>" rather than dropped silently.
    """
    groups: Dict[str, List[str]] = defaultdict(list)
    cids = sorted(f[:-4] for f in os.listdir(input_dir) if f.endswith(".sdf"))
    for cid in cids:
        path = os.path.join(input_dir, cid + ".sdf")
        try:
            mol = Chem.MolFromMolFile(path, sanitize=True)
            if mol is None:
                groups[f"UNPARSEABLE:{cid}"].append(cid)
                continue
            for atom in mol.GetAtoms():
                atom.SetIsotope(0)
            smiles = Chem.MolToSmiles(mol, canonical=True)
            groups[smiles].append(cid)
        except Exception:
            groups[f"UNPARSEABLE:{cid}"].append(cid)
    return dict(groups)


if __name__ == "__main__":
    # Subprocess entrypoint for one peptide, used by PeptidePreparation's
    # timeout-safe path: `python -m chemlink.utils.peptide_processor <sdf>
    # <output_dir> <standardize:0|1>`. A hung RDKit embed (seen in practice
    # on a handful of pathological topologies -- see the 145949006 /
    # 49770777 cases this session, each stuck >10-20 min on molecules no
    # larger than ones that finished in seconds) can only be killed
    # reliably by terminating its OS process; running one peptide per
    # subprocess under an external `timeout`/subprocess.run(timeout=...)
    # is what makes that possible without an in-process signal handler
    # racing against a C-extension call that won't yield control back.
    if len(sys.argv) != 4:
        print(f"usage: python -m chemlink.utils.peptide_processor <sdf> <output_dir> <standardize:0|1>",
              file=sys.stderr)
        sys.exit(2)
    _sdf, _outdir, _std = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
    _name, _ok, _err, _warnings = process_peptide(_sdf, _outdir, standardize=_std)
    _qc = {"qc_status": "SKIPPED", "issues": []}
    if _ok and not _warnings:
        _pdb_path = os.path.join(_outdir, "prepared_peptides_pdb", f"{_name}.pdb")
        if os.path.isfile(_pdb_path):
            _qc = validate_prepared_pdb(_pdb_path)
    import json as _json
    print(_json.dumps({
        "peptide": _name, "success": _ok, "warnings": _warnings,
        "error": _err, "qc_status": _qc["qc_status"], "qc_issues": _qc["issues"],
    }))
    sys.exit(0 if _ok else 1)
