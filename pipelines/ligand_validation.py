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

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms


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
    # True when the supplied coordinates are a 2D drawing (flagged 2D, or
    # flat despite having tetrahedral centres) and a 3D conformer must be
    # generated before the structure can be simulated.
    needs_3d: bool = False

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


# RMS out-of-plane deviation (Å) below which a conformer counts as flat.
_PLANARITY_TOLERANCE = 0.05


def _is_flat_drawing(mol: Chem.Mol, conf: Chem.Conformer) -> bool:
    """True for coordinates that are a 2D sketch saved in a 3D format.

    Sketchers exported through Open Babel write z = 0 for every atom into a
    PDB/MOL2 that RDKit then reports as 3D. Fully conjugated molecules
    (benzene, planar dyes) are legitimately flat, so a conformer only counts
    as a drawing when it also has a tetrahedral centre with three or more
    neighbours, which can never be coplanar in a real structure.
    """
    if mol.GetNumAtoms() < 4:
        return False
    coords = np.array(conf.GetPositions(), dtype=float)
    if not np.all(np.isfinite(coords)):
        return False
    centred = coords - coords.mean(axis=0)
    smallest = np.linalg.svd(centred, compute_uv=False)[-1]
    if smallest / math.sqrt(len(coords)) > _PLANARITY_TOLERANCE:
        return False
    return any(
        atom.GetHybridization() == Chem.HybridizationType.SP3 and atom.GetDegree() >= 3
        for atom in mol.GetAtoms()
    )


def validate_ligand(path: str, *, report_path: Optional[str] = None) -> LigandValidationResult:
    """Validate graph and 3D geometry; raise no exception and return a report."""
    ligand = Path(path).expanduser().resolve()
    errors: list[str] = []
    warnings: list[str] = []
    atoms = bonds = fragments = 0
    needs_3d = False
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
        sanitized = True
        try:
            Chem.SanitizeMol(mol)
        except Exception as exc:
            sanitized = False
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
                needs_3d = True
                warnings.append("coordinates are marked as 2D; downstream preparation will generate a 3D conformer")
            elif sanitized and _is_flat_drawing(mol, conf):
                validate_geometry = False
                needs_3d = True
                warnings.append(
                    "coordinates are flat (2D drawing saved as 3D) although the molecule has "
                    "tetrahedral centres; downstream preparation will generate a 3D conformer"
                )
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
        fragments=fragments, errors=errors, warnings=warnings, needs_3d=needs_3d,
    )
    if report_path:
        _write_report(result, report_path)
    return result


def _write_report(result: LigandValidationResult, report_path: str) -> None:
    destination = Path(report_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")


def require_valid_ligand(path: str, *, report_path: Optional[str] = None) -> LigandValidationResult:
    result = validate_ligand(path, report_path=report_path)
    if not result.valid:
        raise LigandValidationError(result)
    return result


def _embed_rdkit(mol: Chem.Mol) -> Optional[np.ndarray]:
    """ETKDG + force-field relaxation on the file's own graph (atom order kept)."""
    work = Chem.Mol(mol)
    work.RemoveAllConformers()
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xC0FFEE
    if AllChem.EmbedMolecule(work, params) != 0:
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(work, params) != 0:
            return None
    try:
        if AllChem.MMFFHasAllMoleculeParams(work):
            AllChem.MMFFOptimizeMolecule(work, maxIters=2000)
        else:
            AllChem.UFFOptimizeMolecule(work, maxIters=2000)
    except Exception:
        pass
    return np.array(work.GetConformer().GetPositions(), dtype=float)


def _embed_openbabel(path: Path, expected_atoms: int) -> Optional[np.ndarray]:
    """Fallback for graphs RDKit cannot embed; only accepted if atoms match 1:1."""
    executable = shutil.which("obabel")
    if not executable:
        return None
    try:
        with tempfile.TemporaryDirectory(prefix="chemlink-gen3d-") as temp_dir:
            sdf_path = os.path.join(temp_dir, "gen3d.sdf")
            result = subprocess.run(
                [executable, str(path), "-O", sdf_path, "--gen3d", "best"],
                capture_output=True, text=True, timeout=300,
            )
            if result.returncode != 0 or not os.path.isfile(sdf_path):
                return None
            supplier = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=False)
            mol = supplier[0] if supplier and len(supplier) else None
            if mol is None or mol.GetNumAtoms() != expected_atoms:
                return None
            return np.array(mol.GetConformer().GetPositions(), dtype=float)
    except Exception:
        return None


def _write_with_coordinates(source: Path, destination: Path, coords: np.ndarray, mol: Chem.Mol) -> None:
    """Write *coords* in the source format, keeping names/residues/CONECT intact."""
    suffix = source.suffix.lower()
    lines = source.read_text(errors="replace").splitlines()
    if suffix in {".pdb", ".pdbqt"}:
        out, index = [], 0
        for line in lines:
            if line.startswith(("ATOM", "HETATM")):
                if index >= len(coords):
                    raise ValueError("coordinate/atom count mismatch while writing 3D ligand")
                x, y, z = coords[index]
                line = line.ljust(54)
                line = f"{line[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}"
                index += 1
            out.append(line)
        if index != len(coords):
            raise ValueError("coordinate/atom count mismatch while writing 3D ligand")
        destination.write_text("\n".join(out) + "\n", encoding="utf-8")
        return
    if suffix == ".mol2":
        out, index, in_atoms = [], 0, False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("@<TRIPOS>"):
                in_atoms = stripped == "@<TRIPOS>ATOM"
            elif in_atoms and stripped:
                fields = line.split()
                x, y, z = coords[index]
                fields[2:5] = [f"{x:.4f}", f"{y:.4f}", f"{z:.4f}"]
                line = " ".join(fields)
                index += 1
            out.append(line)
        if index != len(coords):
            raise ValueError("coordinate/atom count mismatch while writing 3D ligand")
        destination.write_text("\n".join(out) + "\n", encoding="utf-8")
        return
    # SDF/MOL: re-read the original record so properties/charges survive.
    original = _read_ligand(source)
    conf = Chem.Conformer(original.GetNumAtoms())
    for i, (x, y, z) in enumerate(coords):
        conf.SetAtomPosition(i, (float(x), float(y), float(z)))
    conf.Set3D(True)
    original.RemoveAllConformers()
    original.AddConformer(conf, assignId=True)
    Chem.MolToMolFile(original, str(destination), kekulize=False)


def regenerate_3d_coordinates(path: str, output_path: str) -> str:
    """Build a 3D conformer for a 2D ligand file, preserving atom order and names.

    Raises ValueError when no valid 3D structure can be produced.
    """
    source = Path(path).expanduser().resolve()
    mol = _read_ligand(source)
    if mol is None:
        raise ValueError(f"cannot read {source} to generate 3D coordinates")
    Chem.SanitizeMol(mol)
    coords = _embed_rdkit(mol)
    if coords is None:
        coords = _embed_openbabel(source, mol.GetNumAtoms())
    if coords is None:
        raise ValueError(f"could not generate a 3D conformer for {source}")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_with_coordinates(source, destination, coords, mol)
    return str(destination)


def prepare_valid_ligand(
    path: str, work_dir: str, *, report_path: Optional[str] = None,
) -> tuple[str, LigandValidationResult]:
    """Validate a ligand and, if it is only a 2D drawing, replace it with 3D.

    Returns the path downstream steps must use (the original file, or a
    ``<stem>_3d<suffix>`` copy in *work_dir*) and its validation result.
    Chemically broken inputs still raise LigandValidationError.
    """
    result = validate_ligand(path)
    if result.valid and result.needs_3d:
        source = Path(path)
        suffix = source.suffix.lower()
        generated = regenerate_3d_coordinates(
            path, os.path.join(work_dir, f"{source.stem}_3d{suffix}"),
        )
        regenerated = validate_ligand(generated)
        regenerated.warnings = [
            f"input {source.name} had 2D coordinates; generated 3D conformer {Path(generated).name}",
            *regenerated.warnings,
        ]
        if regenerated.valid and regenerated.needs_3d:
            regenerated.valid = False
            regenerated.errors.append("generated conformer is still not three-dimensional")
        result, path = regenerated, generated
    if report_path:
        _write_report(result, report_path)
    if not result.valid:
        raise LigandValidationError(result)
    return path, result
