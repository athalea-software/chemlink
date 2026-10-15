"""Detection and PDBQT typing of receptor cofactors kept during docking.

By default receptor cleanup strips every hetero group. For targets whose
binding site is built around a prosthetic group (cytochrome P450 / CYP51 and
its heme) or a catalytic metal (Mg, Zn ...), redocking can only reproduce the
reference poses if that cofactor stays in the receptor.

Meeko's polymer builder has no residue templates for hemes or bare metal
ions, so cofactors never go through PDBFixer/Meeko. They are split off into a
``<receptor>_cofactors.pdb`` sidecar next to the clean receptor, typed here
for AutoDock (united-atom AD4 types + Gasteiger charges), and appended to the
receptor PDBQT that Meeko produced. The grid parameter file then gets the
extra receptor atom types (e.g. ``Fe``) so AutoGrid maps them.
"""

import math
import os
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import rdPartialCharges

from .logger import setup_logger

logger = setup_logger(__name__)

# Residue names of heme and heme-like porphyrins in the PDB Chemical
# Component Dictionary. Unlisted names are still caught by the geometric
# check in _looks_like_heme() (Fe coordinated by four porphyrin nitrogens).
HEME_RESIDUES = {"HEM", "HEC", "HEA", "HEB", "HAS", "HDD", "HEO", "DHE", "HNI"}

# Single-atom metal ions kept as cofactors, with their formal charge. Only
# elements present in AutoGrid4's built-in AD4.1 parameter set are listed;
# any other metal would make AutoGrid abort on an unknown receptor type.
METAL_ION_CHARGES = {
    "MG": 2.0,
    "ZN": 2.0,
    "MN": 2.0,
    "MN3": 3.0,
    "CA": 2.0,
    "FE": 3.0,
    "FE2": 2.0,
}

# AD4 atom type for each metal element (case as AutoDock expects it).
METAL_AD_TYPES = {"MG": "Mg", "ZN": "Zn", "MN": "Mn", "CA": "Ca", "FE": "Fe"}

COFACTOR_MARKER = "REMARK CHEMLINK COFACTORS"

_METAL_N_COORDINATION = 2.6  # A; Fe-N(porphyrin) is ~2.0 A

# Bond-order perception from heavy-atom geometry (cofactors carry no H).
_SP2_MEAN_ANGLE = 115.0      # degrees; sp3 is ~109.5, sp2 ~120
_SP2_SHORT_BOND = 1.45       # A; a chain sp2 atom has at least one bond this short
_PLANAR_RING_TOL = 0.20      # A; max out-of-plane deviation of an aromatic ring
_TERMINAL_DOUBLE_MAX = {     # A; longest bond still read as a terminal double bond
    (6, 6): 1.45,            # =CH2 (C-C single is ~1.50)
    (6, 7): 1.36,
    (6, 8): 1.36,            # C=O (C-O single is ~1.43)
    (6, 16): 1.75,
}


ResidueKey = Tuple[str, str, str, str]  # (chain, resnum, icode, resname)


def residue_key(line: str) -> ResidueKey:
    return (line[21], line[22:26].strip(), line[26].strip(), line[17:20].strip().upper())


def _element(line: str) -> str:
    element = line[76:78].strip().upper()
    if element:
        return element
    letters = "".join(char for char in line[12:16] if char.isalpha())
    return letters[:1].upper()


def _xyz(line: str) -> Tuple[float, float, float]:
    return float(line[30:38]), float(line[38:46]), float(line[46:54])


def _looks_like_heme(atom_lines: List[str]) -> bool:
    """True for an iron porphyrin: one Fe with four N within bonding distance."""
    irons = [_xyz(line) for line in atom_lines if _element(line) == "FE"]
    nitrogens = [_xyz(line) for line in atom_lines if _element(line) == "N"]
    if len(irons) != 1 or len(nitrogens) < 4:
        return False
    coordinated = sum(
        1 for xyz in nitrogens if math.dist(irons[0], xyz) <= _METAL_N_COORDINATION
    )
    return coordinated >= 4


def find_cofactor_residues(lines: Iterable[str], protein_residues: Set[str]) -> Dict[ResidueKey, str]:
    """Map each cofactor residue in a PDB to its kind (``"heme"`` or ``"ion"``)."""
    groups: Dict[ResidueKey, List[str]] = {}
    for line in lines:
        if not line.startswith(("ATOM", "HETATM")):
            continue
        key = residue_key(line)
        if key[3] in protein_residues:
            continue
        groups.setdefault(key, []).append(line)

    cofactors: Dict[ResidueKey, str] = {}
    for key, atom_lines in groups.items():
        resname = key[3]
        heavy = [line for line in atom_lines if _element(line) != "H"]
        if resname in HEME_RESIDUES or _looks_like_heme(heavy):
            cofactors[key] = "heme"
        elif len(heavy) == 1 and resname in METAL_ION_CHARGES:
            cofactors[key] = "ion"
    return cofactors


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    v1, v2 = a - b, c - b
    cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def _planar_ring_atoms(mol: Chem.Mol, xyz: np.ndarray) -> Set[int]:
    """Atoms of 5- and 6-membered rings that lie in one plane."""
    atoms: Set[int] = set()
    for ring in Chem.GetSymmSSSR(mol):
        ring = list(ring)
        if len(ring) not in (5, 6):
            continue
        centered = xyz[ring] - xyz[ring].mean(axis=0)
        normal = np.linalg.svd(centered)[2][-1]
        if np.abs(centered @ normal).max() <= _PLANAR_RING_TOL:
            atoms.update(ring)
    return atoms


def _sp2_atoms(mol: Chem.Mol, xyz: np.ndarray) -> Set[int]:
    """Trigonal atoms, judged from bond angles (and ring planarity)."""
    planar = _planar_ring_atoms(mol, xyz)
    sp2: Set[int] = set()
    for atom in mol.GetAtoms():
        index = atom.GetIdx()
        neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
        if len(neighbors) not in (2, 3):
            continue
        # Pyrrole angles are ~108 degrees, so ring atoms go by planarity.
        if index in planar:
            sp2.add(index)
            continue
        angles = [
            _angle(xyz[a], xyz[index], xyz[b])
            for k, a in enumerate(neighbors) for b in neighbors[k + 1:]
        ]
        shortest = min(float(np.linalg.norm(xyz[index] - xyz[n])) for n in neighbors)
        if sum(angles) / len(angles) > _SP2_MEAN_ANGLE and (
            len(neighbors) == 3 or shortest <= _SP2_SHORT_BOND
        ):
            sp2.add(index)
    return sp2


def _max_matching(n: int, edges: List[Tuple[int, int]]) -> Set[Tuple[int, int]]:
    """Maximum-cardinality matching of a general graph (Edmonds' blossom).

    Conjugated rings with an odd member (pyrroles) make the graph
    non-bipartite, so a plain augmenting-path search is not enough.
    """
    adj: List[List[int]] = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    match = [-1] * n

    def find_augmenting_path(root: int):
        used = [False] * n
        parent = [-1] * n
        base = list(range(n))
        used[root] = True
        queue = [root]

        def lowest_common_base(a: int, b: int) -> int:
            seen = [False] * n
            while True:
                a = base[a]
                seen[a] = True
                if match[a] == -1:
                    break
                a = parent[match[a]]
            while True:
                b = base[b]
                if seen[b]:
                    return b
                b = parent[match[b]]

        def mark_blossom(v: int, b: int, child: int, blossom: List[bool]) -> None:
            while base[v] != b:
                blossom[base[v]] = blossom[base[match[v]]] = True
                parent[v] = child
                child = match[v]
                v = parent[match[v]]

        head = 0
        while head < len(queue):
            v = queue[head]
            head += 1
            for to in adj[v]:
                if base[v] == base[to] or match[v] == to:
                    continue
                if to == root or (match[to] != -1 and parent[match[to]] != -1):
                    shared = lowest_common_base(v, to)
                    blossom = [False] * n
                    mark_blossom(v, shared, to, blossom)
                    mark_blossom(to, shared, v, blossom)
                    for i in range(n):
                        if blossom[base[i]]:
                            base[i] = shared
                            if not used[i]:
                                used[i] = True
                                queue.append(i)
                elif parent[to] == -1:
                    parent[to] = v
                    if match[to] == -1:
                        return to, parent
                    used[match[to]] = True
                    queue.append(match[to])
        return -1, parent

    for root in range(n):
        if match[root] != -1:
            continue
        end, parent = find_augmenting_path(root)
        while end != -1:
            prev = parent[end]
            following = match[prev]
            match[end], match[prev] = prev, end
            end = following
    return {(u, v) for u, v in enumerate(match) if v > u}


def _double_bonds(mol: Chem.Mol, xyz: np.ndarray, sp2: Set[int]) -> Set[Tuple[int, int]]:
    """Place double bonds on a heavy-atom graph that only has connectivity."""

    def can_double(atom) -> bool:
        element, degree, index = atom.GetAtomicNum(), atom.GetDegree(), atom.GetIdx()
        if element == 6:
            return degree <= 3 and (index in sp2 or degree == 1)
        if element == 7:
            return degree <= 2 and (index in sp2 or degree == 1)
        if element in (8, 16):
            return degree == 1
        return False

    terminal_pairs = []
    edges = []
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtom(), bond.GetEndAtom()
        if not (can_double(a) and can_double(b)):
            continue
        terminal = [atom for atom in (a, b) if atom.GetDegree() == 1]
        if not terminal:
            edges.append((a.GetIdx(), b.GetIdx()))
            continue
        other = b if terminal[0] is a else a
        length = float(np.linalg.norm(xyz[a.GetIdx()] - xyz[b.GetIdx()]))
        limit = _TERMINAL_DOUBLE_MAX.get(tuple(sorted((a.GetAtomicNum(), b.GetAtomicNum()))))
        if limit is not None and length <= limit and other.GetIdx() in sp2:
            terminal_pairs.append((length, a.GetIdx(), b.GetIdx()))
        elif (terminal[0].GetAtomicNum() == 6 and other.GetAtomicNum() == 6
              and other.GetDegree() == 2 and other.GetIdx() in sp2):
            # A =CH2 on a two-neighbour sp2 carbon is a vinyl even when the
            # deposited geometry is poor; a methyl sits on a ring carbon.
            edges.append((a.GetIdx(), b.GetIdx()))

    # Terminal C=O and =CH2 first, shortest bond wins: geometry alone cannot
    # tell a carboxyl C=O from the C=C it could form with its CH2 neighbour.
    doubles: Set[Tuple[int, int]] = set()
    used: Set[int] = set()
    for _length, u, v in sorted(terminal_pairs):
        if u not in used and v not in used:
            doubles.add((u, v))
            used.update((u, v))
    remaining = [(u, v) for u, v in edges if u not in used and v not in used]
    return doubles | _max_matching(mol.GetNumAtoms(), remaining)


def _perceive_framework(organic_lines: List[str]) -> Tuple[Chem.Mol, Set[int]]:
    """RDKit molecule (bond orders, implicit H) of a cofactor's heavy atoms."""
    with rdBase.BlockLogs():
        mol = Chem.MolFromPDBBlock(
            "".join(organic_lines) + "END\n",
            sanitize=False, removeHs=False, proximityBonding=True,
        )
    if mol is None or mol.GetNumAtoms() != len(organic_lines):
        read = 0 if mol is None else mol.GetNumAtoms()
        raise RuntimeError(
            f"RDKit read {read} atoms from a {len(organic_lines)}-atom cofactor framework"
        )
    xyz = mol.GetConformer().GetPositions()
    sp2 = _sp2_atoms(mol, xyz)

    editable = Chem.RWMol(mol)
    for u, v in _double_bonds(mol, xyz, sp2):
        editable.GetBondBetweenAtoms(u, v).SetBondType(Chem.BondType.DOUBLE)
    for atom in editable.GetAtoms():
        atom.SetNoImplicit(False)
        atom.SetNumExplicitHs(0)
        atom.SetFormalCharge(0)
    perceived = editable.GetMol()
    try:
        with rdBase.BlockLogs():
            Chem.SanitizeMol(perceived)
    except Exception as exc:
        # Keep typing the cofactor on its single-bond graph rather than
        # failing the whole receptor; the charges are rougher.
        logger.warning("Cofactor bond orders could not be perceived (%s); using single bonds", exc)
        perceived = Chem.RWMol(mol)
        for atom in perceived.GetAtoms():
            atom.SetNoImplicit(False)
        perceived = perceived.GetMol()
        perceived.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(
            perceived,
            Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,
            catchErrors=True,
        )
    return perceived, sp2


def _ad_type(atom, coordinated: bool, sp2: Set[int]) -> str:
    """United-atom AutoDock type for a heavy atom of an organic cofactor."""
    element = atom.GetAtomicNum()
    if element == 6:
        # Ring carbons of the conjugated macrocycle (pyrrole rings) are
        # aromatic for AutoDock; methyl, vinyl and propionate carbons are not.
        in_small_ring = any(atom.IsInRingSize(size) for size in (5, 6))
        return "A" if in_small_ring and atom.GetIdx() in sp2 else "C"
    if element == 7:
        # Porphyrin nitrogens donate their lone pair to the metal and are no
        # longer hydrogen-bond acceptors.
        if coordinated:
            return "N"
        return "NA" if atom.GetTotalNumHs() == 0 and atom.GetDegree() < 3 else "N"
    if element == 8:
        return "OA"
    if element == 16:
        return "SA"
    return Chem.GetPeriodicTable().GetElementSymbol(element)


def _united_atom_charge(atom) -> float:
    """Gasteiger charge with the implicit hydrogens' share folded in."""
    charge = atom.GetDoubleProp("_GasteigerCharge")
    if atom.HasProp("_GasteigerHCharge"):
        charge += atom.GetDoubleProp("_GasteigerHCharge")
    return charge if math.isfinite(charge) else 0.0


def _pdbqt_line(line: str, charge: float, ad_type: str) -> str:
    record = line[:54].ljust(54)
    return f"{record}  1.00  0.00    {charge:+.3f} {ad_type:<2}\n"


def _type_organic_cofactor(atom_lines: List[str]) -> List[str]:
    """Return PDBQT lines for one polyatomic cofactor (heme)."""
    # Gasteiger has no metal parameters, and with Fe bonded to the
    # nitrogens it zeroes the whole molecule. Type and charge the organic
    # framework alone; the metal gets 0.0, as AutoDockTools does for heme
    # iron, and its coordinating atoms are typed as non-acceptors.
    metal_lines = [line for line in atom_lines if _element(line) in METAL_AD_TYPES]
    organic_lines = [line for line in atom_lines if _element(line) not in METAL_AD_TYPES]

    mol, sp2 = _perceive_framework(organic_lines)
    rdPartialCharges.ComputeGasteigerCharges(mol)

    metal_xyz = [_xyz(line) for line in metal_lines]
    typed = {}
    # RDKit keeps input order, so atom i is organic_lines[i].
    for atom, line in zip(mol.GetAtoms(), organic_lines):
        coordinated = any(
            math.dist(_xyz(line), xyz) <= _METAL_N_COORDINATION for xyz in metal_xyz
        )
        typed[id(line)] = _pdbqt_line(line, _united_atom_charge(atom), _ad_type(atom, coordinated, sp2))
    for line in metal_lines:
        typed[id(line)] = _pdbqt_line(line, 0.0, METAL_AD_TYPES[_element(line)])
    return [typed[id(line)] for line in atom_lines]


def write_cofactor_pdbqt(cofactor_pdb: str, output_pdbqt: str) -> Set[str]:
    """Type every cofactor in ``cofactor_pdb``; return the AD types written."""
    groups: Dict[ResidueKey, List[str]] = {}
    with open(cofactor_pdb) as handle:
        for line in handle:
            if line.startswith(("ATOM", "HETATM")):
                groups.setdefault(residue_key(line), []).append(line)

    out_lines: List[str] = []
    for key, atom_lines in groups.items():
        if len(atom_lines) == 1:
            line = atom_lines[0]
            element = _element(line)
            ad_type = METAL_AD_TYPES.get(element)
            if ad_type is None:
                raise RuntimeError(f"Unsupported cofactor ion {key[3]} ({element})")
            out_lines.append(_pdbqt_line(line, METAL_ION_CHARGES.get(key[3], 2.0), ad_type))
        else:
            out_lines.extend(_type_organic_cofactor(atom_lines))

    with open(output_pdbqt, "w") as handle:
        handle.writelines(out_lines)

    types = {line[77:79].strip() for line in out_lines}
    logger.info(
        "Typed %d cofactor residue(s) for docking: %s (AD types: %s)",
        len(groups),
        ", ".join(f"{key[3]} {key[0]}:{key[1]}" for key in groups),
        " ".join(sorted(types)),
    )
    return types


def cofactor_sidecar(clean_receptors_dir: str, receptor_stem: str, ext: str) -> str:
    """Path of the cofactor sidecar for a ``<name>_clean`` receptor stem."""
    return os.path.join(clean_receptors_dir, f"{receptor_stem}_cofactors{ext}")


def append_cofactors(receptor_pdbqt: str, cofactor_pdbqt: str) -> None:
    """Append the typed cofactor atoms to a rigid receptor PDBQT (idempotent)."""
    with open(receptor_pdbqt) as handle:
        lines = handle.readlines()
    if any(line.startswith(COFACTOR_MARKER) for line in lines):
        return
    with open(cofactor_pdbqt) as handle:
        cofactor_lines = [line for line in handle if line.startswith(("ATOM", "HETATM"))]
    if not cofactor_lines:
        return

    body = [line for line in lines if not line.startswith("END")]
    serial = 0
    for line in body:
        if line.startswith(("ATOM", "HETATM")):
            try:
                serial = max(serial, int(line[6:11]))
            except ValueError:
                pass
    renumbered = []
    for line in cofactor_lines:
        serial += 1
        renumbered.append(f"{line[:6]}{serial % 100000:>5}{line[11:]}")

    with open(receptor_pdbqt, "w") as handle:
        handle.writelines(body)
        handle.write(f"{COFACTOR_MARKER}\n")
        handle.writelines(renumbered)


def add_gpf_receptor_types(gpf_path: str, extra_types: Iterable[str]) -> None:
    """Add receptor atom types (e.g. Fe) to a grid parameter file."""
    with open(gpf_path) as handle:
        lines = handle.readlines()
    for index, line in enumerate(lines):
        if not line.startswith("receptor_types"):
            continue
        content, _, comment = line.partition("#")
        fields = content.split()
        types = fields[1:]
        for ad_type in sorted(extra_types):
            if ad_type not in types:
                types.append(ad_type)
        new_line = "receptor_types " + " ".join(types)
        if comment:
            new_line += "    #" + comment.rstrip("\n")
        lines[index] = new_line + "\n"
        break
    else:
        raise RuntimeError(f"No receptor_types line in {gpf_path}")
    with open(gpf_path, "w") as handle:
        handle.writelines(lines)


def read_pdbqt_types(pdbqt_path: str) -> Set[str]:
    with open(pdbqt_path) as handle:
        return {
            line[77:79].strip() for line in handle
            if line.startswith(("ATOM", "HETATM")) and line[77:79].strip()
        }


def attach_cofactors_to_grid(
    clean_receptors_dir: str,
    receptor_stem: str,
    receptor_pdbqt: str,
    gpf_path: Optional[str],
) -> bool:
    """Append a receptor's cofactor sidecar to its PDBQT and GPF, if it has one."""
    sidecar = cofactor_sidecar(clean_receptors_dir, receptor_stem, ".pdbqt")
    if not os.path.isfile(sidecar):
        return False
    append_cofactors(receptor_pdbqt, sidecar)
    if gpf_path:
        add_gpf_receptor_types(gpf_path, read_pdbqt_types(sidecar))
    return True
