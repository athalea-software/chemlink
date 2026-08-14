"""Selection and audit helpers for limited receptor side-chain flexibility."""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Sequence, Tuple


BACKBONE_ATOMS = {"N", "CA", "C", "O", "OXT"}
NON_ROTATABLE_SIDECHAINS = {"ALA", "GLY"}
PROTEIN_RESIDUES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
    "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
    "THR", "TRP", "TYR", "VAL", "HID", "HIE", "HIP", "HSD",
    "HSE", "HSP", "CYM", "CYX",
}


@dataclass(frozen=True)
class FlexibleResidue:
    chain: str
    number: int
    insertion_code: str
    name: str
    distance: float

    @property
    def meeko_id(self) -> str:
        # Meeko identifies residues as CHAIN:NUMBER. Insertion codes cannot be
        # represented unambiguously by its CLI, so those residues are rejected.
        return f"{self.chain}:{self.number}" if self.chain else f":{self.number}"


def _atom_records(pdb_path: str) -> Iterable[Dict[str, object]]:
    with open(pdb_path, encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            try:
                yield {
                    "atom": line[12:16].strip().upper(),
                    "altloc": line[16:17].strip(),
                    "resname": line[17:20].strip().upper(),
                    "chain": line[21:22].strip(),
                    "number": int(line[22:26].strip()),
                    "icode": line[26:27].strip(),
                    "xyz": (
                        float(line[30:38]),
                        float(line[38:46]),
                        float(line[46:54]),
                    ),
                    "element": (line[76:78].strip() or line[12:14].strip()).upper(),
                }
            except (TypeError, ValueError):
                continue


def detect_disulfide_residues(pdb_path: str) -> List[Tuple[str, int]]:
    """Return CYS residue identifiers whose SG atoms form a disulfide bond."""
    sg_atoms = []
    for atom in _atom_records(pdb_path):
        if atom["resname"] == "CYS" and atom["atom"] == "SG":
            sg_atoms.append(((str(atom["chain"]), int(atom["number"])), atom["xyz"]))
    bonded = set()
    for index, (left_id, left_xyz) in enumerate(sg_atoms):
        for right_id, right_xyz in sg_atoms[index + 1:]:
            if math.dist(left_xyz, right_xyz) <= 2.5:
                bonded.update((left_id, right_id))
    return sorted(bonded)
def select_flexible_residues(
    pdb_path: str,
    center: Sequence[float],
    radius: float = 10.0,
) -> Tuple[List[FlexibleResidue], List[Dict[str, object]]]:
    """Select residues having a heavy side-chain atom within ``radius`` Å."""
    if radius < 0:
        raise ValueError("Flexible-residue radius must be >= 0")

    grouped: Dict[Tuple[str, int, str, str], List[Dict[str, object]]] = {}
    for atom in _atom_records(pdb_path):
        key = (atom["chain"], atom["number"], atom["icode"], atom["resname"])
        grouped.setdefault(key, []).append(atom)

    disulfide_keys = set()
    cysteine_sg = []
    for key, atoms in grouped.items():
        if key[3] not in {"CYS", "CYX"}:
            continue
        for atom in atoms:
            if atom["atom"] == "SG":
                cysteine_sg.append((key, atom["xyz"]))
    for index, (left_key, left_xyz) in enumerate(cysteine_sg):
        for right_key, right_xyz in cysteine_sg[index + 1:]:
            if math.dist(left_xyz, right_xyz) <= 2.5:
                disulfide_keys.update((left_key, right_key))

    selected: List[FlexibleResidue] = []
    excluded: List[Dict[str, object]] = []
    cx, cy, cz = (float(value) for value in center)
    for (chain, number, icode, resname), atoms in sorted(grouped.items()):
        if resname not in PROTEIN_RESIDUES:
            continue
        if resname in NON_ROTATABLE_SIDECHAINS:
            excluded.append({"chain": chain, "number": number, "name": resname,
                             "reason": "side chain has no useful rotatable bond"})
            continue
        if (chain, number, icode, resname) in disulfide_keys or resname == "CYX":
            excluded.append({"chain": chain, "number": number, "name": resname,
                             "reason": "disulfide-bonded cysteine"})
            continue
        if icode:
            excluded.append({"chain": chain, "number": number, "name": resname,
                             "insertion_code": icode,
                             "reason": "Meeko CLI cannot identify insertion codes safely"})
            continue

        distances = []
        for atom in atoms:
            atom_name = str(atom["atom"])
            element = str(atom["element"])
            if atom_name in BACKBONE_ATOMS or element in {"H", "D"} or atom_name.startswith("H"):
                continue
            x, y, z = atom["xyz"]
            distances.append(math.dist((x, y, z), (cx, cy, cz)))
        if not distances:
            continue
        distance = min(distances)
        if distance <= radius:
            selected.append(FlexibleResidue(chain, number, icode, resname, distance))

    return selected, excluded


def write_flexibility_report(
    path: str,
    receptor: str,
    center: Sequence[float],
    radius: float,
    selected: Sequence[FlexibleResidue],
    excluded: Sequence[Dict[str, object]],
    rigid_file: str,
    flexible_file: str | None,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "receptor": receptor,
        "center": [float(value) for value in center],
        "radius_angstrom": float(radius),
        "selection_rule": "minimum heavy side-chain atom distance to box center",
        "selected_count": len(selected),
        "selected": [asdict(residue) | {"meeko_id": residue.meeko_id} for residue in selected],
        "excluded": list(excluded),
        "rigid_pdbqt": rigid_file,
        "flexible_pdbqt": flexible_file,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
