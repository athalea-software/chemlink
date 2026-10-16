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

try:
    from openbabel import openbabel as ob
except ImportError:
    import openbabel as ob

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


def _ad_type(atom, coordinated: bool) -> str:
    """United-atom AutoDock type for a heavy atom of an organic cofactor."""
    element = atom.GetAtomicNum()
    if element == 6:
        # Ring carbons of the conjugated macrocycle (pyrrole rings) are
        # aromatic for AutoDock; methyl, vinyl and propionate carbons are not.
        in_small_ring = any(atom.IsInRingSize(size) for size in (5, 6))
        return "A" if in_small_ring and atom.GetHyb() == 2 else "C"
    if element == 7:
        # Porphyrin nitrogens donate their lone pair to the metal and are no
        # longer hydrogen-bond acceptors.
        if coordinated:
            return "N"
        return "NA" if atom.IsHbondAcceptor() else "N"
    if element == 8:
        return "OA"
    if element == 16:
        return "SA"
    return ob.GetSymbol(element)


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

    mol = ob.OBMol()
    conv = ob.OBConversion()
    conv.SetInFormat("pdb")
    conv.ReadString(mol, "".join(organic_lines) + "END\n")
    if mol.NumAtoms() != len(organic_lines):
        raise RuntimeError(
            f"Open Babel read {mol.NumAtoms()} atoms from a "
            f"{len(organic_lines)}-atom cofactor framework"
        )
    ob.OBChargeModel.FindType("gasteiger").ComputeCharges(mol)

    metal_xyz = [_xyz(line) for line in metal_lines]
    typed = {}
    # Open Babel keeps input order, so atom index i is organic_lines[i - 1].
    for atom, line in zip(ob.OBMolAtomIter(mol), organic_lines):
        coordinated = any(
            math.dist(_xyz(line), xyz) <= _METAL_N_COORDINATION for xyz in metal_xyz
        )
        typed[id(line)] = _pdbqt_line(line, atom.GetPartialCharge(), _ad_type(atom, coordinated))
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
