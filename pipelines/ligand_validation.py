"""Chemical-integrity validation shared by docking and dynamics pipelines."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from rdkit import Chem
from rdkit.Chem import rdMolTransforms


SUPPORTED_FORMATS = {".sdf", ".mol", ".mol2", ".pdb", ".pdbqt"}


def perceive_pdb_bond_orders(path: Path, *, sanitize: bool = False) -> Optional[Chem.Mol]:
    """Recover real bond orders/aromaticity/formal charges from a plain PDB.

    PDB CONECT records carry connectivity only, never bond multiplicity, so a
    direct RDKit parse collapses every bond to single and fills valence with
    implicit hydrogens. That silently turns a nitrile carbon into a strained
    sp3 centre and a pyrylium O+ into a neutral, over-valent oxygen. Open
    Babel infers proper bond orders from the 3D geometry alone; round-tripping
    through SDF (which encodes bond order explicitly) recovers them.

    Open Babel infers bond order from bond lengths alone, which can misjudge
    conjugated/mesoionic systems — e.g. assigning a C=O double bond to an
    oxygen whose formal charge the PDB already declares as O-, an explicit
    valence contradiction. The perceived structure is trusted only when it
    actually sanitizes; otherwise this returns None so the caller falls back
    to the conservative single-bond parse, which preserves declared formal
    charges from the PDB element-field annotation (e.g. "O1-", "N1+") and is
    valence-consistent by construction (all-single bonds plus those charges).

    Returns None — callers fall back to the raw PDB parse — if Open Babel is
    unavailable, the conversion fails, or the perceived structure is invalid.
    """
    executable = shutil.which("obabel")
    if not executable:
        return None
    try:
        with tempfile.TemporaryDirectory(prefix="chemlink-pdb2sdf-") as temp_dir:
            sdf_path = os.path.join(temp_dir, "converted.sdf")
            result = subprocess.run(
                [executable, str(path), "-O", sdf_path],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode != 0 or not os.path.isfile(sdf_path):
                return None
            supplier = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=False)
            mol = supplier[0] if supplier and len(supplier) else None
            if mol is None:
                return None
            sanitized_copy = Chem.Mol(mol)
            try:
                Chem.SanitizeMol(sanitized_copy)
            except Exception:
                return None
            return sanitized_copy if sanitize else mol
    except Exception:
        return None


@dataclass
class LigandValidationResult:
    path: str
    valid: bool
    atoms: int
    bonds: int
    fragments: int
    errors: list[str]
    warnings: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


class LigandValidationError(ValueError):
    def __init__(self, result: LigandValidationResult):
        self.result = result
        super().__init__(f"Ligand validation failed for {result.path}: " + "; ".join(result.errors))


def _looks_like_pdbqt(source_lines: list[str]) -> bool:
    """Docking tools often write PDBQT-typed atom records under a .pdb name."""
    return any(
        line.startswith(("ROOT", "ENDROOT", "BRANCH", "ENDBRANCH", "TORSDOF"))
        for line in source_lines
    )


def _is_pdbqt_ligand(path: Path) -> bool:
    if path.suffix.lower() == ".pdbqt":
        return True
    if path.suffix.lower() != ".pdb":
        return False
    return _looks_like_pdbqt(path.read_text(errors="replace").splitlines())


def _read_ligand(path: Path) -> Optional[Chem.Mol]:
    suffix = path.suffix.lower()
    if suffix == ".sdf":
        supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=False)
        return supplier[0] if supplier and len(supplier) else None
    if suffix == ".mol":
        return Chem.MolFromMolFile(str(path), removeHs=False, sanitize=False)
    if suffix == ".mol2":
        return Chem.MolFromMol2File(str(path), removeHs=False, sanitize=False)
    if suffix in {".pdb", ".pdbqt"}:
        source_lines = path.read_text(errors="replace").splitlines()
        if suffix == ".pdbqt" or _looks_like_pdbqt(source_lines):
            lines = []
            for line in source_lines:
                if line.startswith(("ATOM", "HETATM")):
                    autodock_type = line.split()[-1] if line.split() else ""
                    element_map = {
                        "A": "C", "C": "C", "N": "N", "NA": "N", "NS": "N",
                        "O": "O", "OA": "O", "OS": "O", "S": "S", "SA": "S",
                        "P": "P", "H": "H", "HD": "H", "HS": "H", "F": "F",
                        "Cl": "Cl", "CL": "Cl", "Br": "Br", "BR": "Br", "I": "I",
                        "Mg": "Mg", "MG": "Mg", "Mn": "Mn", "MN": "Mn",
                        "Zn": "Zn", "ZN": "Zn", "Ca": "Ca", "CA": "Ca",
                        "Fe": "Fe", "FE": "Fe",
                    }
                    element = element_map.get(autodock_type)
                    if element is None:
                        atom_name = line[12:16].strip()
                        letters = "".join(char for char in atom_name if char.isalpha())
                        element = (letters[:2].title() if letters[:2].title() in element_map.values() else letters[:1].upper())
                    lines.append(line[:66].ljust(76) + element.rjust(2))
                elif line.startswith(("CONECT", "TER", "END")):
                    lines.append(line)
            block = "\n".join(lines) + "\n"
            return Chem.MolFromPDBBlock(
                block, removeHs=False, sanitize=False, proximityBonding=True,
            )
        perceived = perceive_pdb_bond_orders(path)
        if perceived is not None:
            return perceived
        block = "\n".join(source_lines) + "\n"
        return Chem.MolFromPDBBlock(
            block, removeHs=False, sanitize=False, proximityBonding=True,
        )
    return None


def _distance(conf: Chem.Conformer, first: int, second: int) -> float:
    a, b = conf.GetAtomPosition(first), conf.GetAtomPosition(second)
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def validate_ligand(path: str, *, report_path: Optional[str] = None) -> LigandValidationResult:
    """Validate graph and 3D geometry; raise no exception and return a report."""
    ligand = Path(path).expanduser().resolve()
    errors: list[str] = []
    warnings: list[str] = []
    atoms = bonds = fragments = 0
    if not ligand.is_file():
        errors.append("file does not exist")
        mol = None
    elif ligand.suffix.lower() not in SUPPORTED_FORMATS:
        errors.append(f"unsupported ligand format {ligand.suffix or '(none)'}")
        mol = None
    else:
        try:
            mol = _read_ligand(ligand)
        except Exception as exc:
            mol = None
            errors.append(f"structure could not be read: {exc}")
        if mol is None and not errors:
            errors.append("structure could not be read or contains no molecule")

    if mol is not None:
        atoms, bonds = mol.GetNumAtoms(), mol.GetNumBonds()
        if atoms < 1:
            errors.append("molecule contains no atoms")
        try:
            Chem.SanitizeMol(mol)
        except Exception as exc:
            errors.append(f"invalid valence, aromaticity, or bond order: {exc}")
        fragments = len(Chem.GetMolFrags(mol))
        if fragments != 1:
            errors.append(f"molecule is disconnected ({fragments} fragments); all atoms must be connected")
        if atoms > 1 and bonds < atoms - 1:
            errors.append(f"insufficient connectivity ({atoms} atoms, {bonds} bonds)")

        if mol.GetNumConformers() == 0:
            errors.append("molecule has no 3D coordinates")
        else:
            conf = mol.GetConformer()
            validate_geometry = conf.Is3D()
            if not validate_geometry:
                warnings.append("coordinates are marked as 2D; downstream preparation will generate a 3D conformer")
            table = Chem.GetPeriodicTable()
            bonded_pairs: set[tuple[int, int]] = set()
            for bond in mol.GetBonds():
                i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                bonded_pairs.add(tuple(sorted((i, j))))
                length = _distance(conf, i, j)
                expected = table.GetRcovalent(mol.GetAtomWithIdx(i).GetAtomicNum()) + table.GetRcovalent(mol.GetAtomWithIdx(j).GetAtomicNum())
                lower, upper = max(0.45, expected * 0.62), expected * 1.35
                if validate_geometry and (not math.isfinite(length) or length < lower or length > upper):
                    errors.append(f"implausible bond length atoms {i + 1}-{j + 1}: {length:.2f} Å (expected {lower:.2f}-{upper:.2f} Å)")

            # Detect severe non-bonded overlaps, excluding 1-3 neighbours.
            one_three = set(bonded_pairs)
            for atom in mol.GetAtoms():
                neighbours = [n.GetIdx() for n in atom.GetNeighbors()]
                for i in neighbours:
                    for j in neighbours:
                        if i < j:
                            one_three.add((i, j))
            for i in range(atoms):
                for j in range(i + 1, atoms):
                    if (i, j) in one_three:
                        continue
                    clash_limit = 0.55 * (
                        table.GetRcovalent(mol.GetAtomWithIdx(i).GetAtomicNum())
                        + table.GetRcovalent(mol.GetAtomWithIdx(j).GetAtomicNum())
                    )
                    distance = _distance(conf, i, j)
                    if validate_geometry and distance < clash_limit:
                        errors.append(f"non-bonded atomic clash atoms {i + 1}-{j + 1}: {distance:.2f} Å")

            ideals = {
                Chem.HybridizationType.SP: (180.0, 30.0),
                Chem.HybridizationType.SP2: (120.0, 38.0),
                Chem.HybridizationType.SP3: (109.5, 42.0),
            }
            ring_info = mol.GetRingInfo()
            # PDBQT does not preserve bond orders, so RDKit cannot infer
            # trustworthy hybridisation/ideal angles from that format alone
            # — whether named .pdbqt or a docking tool's .pdb with PDBQT typing.
            angle_atoms = (
                mol.GetAtoms()
                if validate_geometry and not _is_pdbqt_ligand(ligand)
                else []
            )
            for center in angle_atoms:
                neighbours = [n.GetIdx() for n in center.GetNeighbors()]
                ideal = ideals.get(center.GetHybridization())
                if len(neighbours) < 2 or ideal is None:
                    continue
                # Small rings are intentionally strained and should not fail validation.
                if ring_info.IsAtomInRingOfSize(center.GetIdx(), 3) or ring_info.IsAtomInRingOfSize(center.GetIdx(), 4):
                    continue
                target, tolerance = ideal
                for offset, first in enumerate(neighbours):
                    for second in neighbours[offset + 1:]:
                        angle = rdMolTransforms.GetAngleDeg(conf, first, center.GetIdx(), second)
                        if not math.isfinite(angle) or abs(angle - target) > tolerance:
                            errors.append(
                                f"implausible angle atoms {first + 1}-{center.GetIdx() + 1}-{second + 1}: "
                                f"{angle:.1f}° (expected about {target:.1f}°)"
                            )

    result = LigandValidationResult(
        path=str(ligand), valid=not errors, atoms=atoms, bonds=bonds,
        fragments=fragments, errors=errors, warnings=warnings,
    )
    if report_path:
        destination = Path(report_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")
    return result


def require_valid_ligand(path: str, *, report_path: Optional[str] = None) -> LigandValidationResult:
    result = validate_ligand(path, report_path=report_path)
    if not result.valid:
        raise LigandValidationError(result)
    return result
