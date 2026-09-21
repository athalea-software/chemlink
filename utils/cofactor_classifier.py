"""Classification of MD cofactors (ions or small inorganic molecules).

A cofactor is never run through ``pdb2gmx``: like a ligand, but unlike the
protein/peptide/nucleic-acid chains, pdb2gmx only knows residues present in
its own .rtp database, so an ion or small inorganic guest is fused directly
into ``complex.gro``/``topol.top`` after the ligand step, mirroring how
``gmx genion`` adds ions by writing coordinates and a moleculetype count
without ever invoking pdb2gmx on them.

Two cofactor kinds are supported:

- ``ion``: a single heavy atom. Resolved against a ready-made
  ``[ moleculetype ]`` block already shipped in the target force field's
  ``ions.itp`` -- no parametrisation is invented here.
- ``molecule``: two or more heavy atoms (e.g. sulfate, phosphate,
  carbonate). Parametrised with GAFF2/ACPYPE exactly like the ligand.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class CofactorError(ValueError):
    """Raised when a cofactor input cannot be classified or resolved safely."""


@dataclass
class CofactorAtom:
    name: str
    element: str
    x: float
    y: float
    z: float


@dataclass
class CofactorClassification:
    kind: str  # "ion" | "molecule"
    resname: str
    atoms: list[CofactorAtom]


def _guess_element(atom_name: str) -> str:
    letters = "".join(char for char in atom_name if char.isalpha())
    if not letters:
        raise CofactorError(f"Could not infer an element from atom name {atom_name!r}")
    return letters[:1].upper() + letters[1:2].lower()


def _parse_atoms(path: Path) -> tuple[str, list[CofactorAtom]]:
    resname = None
    atoms: list[CofactorAtom] = []
    with path.open() as handle:
        for line in handle:
            if not line.startswith(("ATOM", "HETATM")) or len(line) < 54:
                continue
            atom_name = line[12:16].strip()
            raw_element = line[76:78].strip()
            element = (raw_element[:1].upper() + raw_element[1:].lower()) if raw_element else _guess_element(atom_name)
            resname = resname or line[17:20].strip().upper()
            atoms.append(
                CofactorAtom(
                    name=atom_name,
                    element=element,
                    x=float(line[30:38]),
                    y=float(line[38:46]),
                    z=float(line[46:54]),
                )
            )
    if not atoms:
        raise CofactorError(f"No ATOM/HETATM records found in cofactor file: {path}")
    heavy = [atom for atom in atoms if atom.element.upper() != "H"]
    if not heavy:
        raise CofactorError(f"Cofactor file contains only hydrogen atoms: {path}")
    return resname or heavy[0].element.upper(), heavy


def classify_cofactor(path: str) -> CofactorClassification:
    """Classify a cofactor input file as a monoatomic ion or a small molecule."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise CofactorError(f"Cofactor file does not exist: {resolved}")
    resname, atoms = _parse_atoms(resolved)
    kind = "ion" if len(atoms) == 1 else "molecule"
    return CofactorClassification(kind=kind, resname=resname, atoms=atoms)


def _iter_moleculetypes(itp_path: Path):
    """Yield ``(name, atom_lines)`` for every ``[ moleculetype ]`` block."""
    name: str | None = None
    section: str | None = None
    atom_lines: list[str] = []
    with itp_path.open() as handle:
        for raw in handle:
            line = raw.split(";", 1)[0].strip()
            if line.startswith("[") and line.endswith("]"):
                new_section = line.strip("[] ").lower()
                if new_section == "moleculetype" and name is not None:
                    yield name, atom_lines
                    name, atom_lines = None, []
                section = new_section
                continue
            if not line:
                continue
            if section == "moleculetype" and name is None:
                name = line.split()[0]
            elif section == "atoms":
                atom_lines.append(raw.rstrip("\n"))
    if name is not None:
        yield name, atom_lines


def find_ion_moleculetype(itp_path: Path, resname: str) -> tuple[str, str]:
    """Find a ready-made single-atom moleculetype matching the ion's residue name.

    Tries the residue name exactly as given first -- PDB ion HETATMs
    conventionally use the bare element as the residue name, and several
    GROMACS force fields (AMBER, CHARMM36) name their monoatomic-ion
    moleculetypes identically. Raises with the list of available
    moleculetypes so the user can rename the residue, or provide the
    cofactor as a GAFF2-parametrised molecule instead, or pick a force
    field that defines this ion.
    """
    candidates = [resname.upper()]
    element_guess = resname[:2].title() if len(resname) > 1 else resname
    if element_guess.upper() not in candidates:
        candidates.append(element_guess.upper())

    available: dict[str, str] = {}
    for name, atom_lines in _iter_moleculetypes(itp_path):
        if len(atom_lines) == 1:
            fields = atom_lines[0].split()
            if len(fields) >= 5:
                available[name.upper()] = fields[4]

    for candidate in candidates:
        if candidate in available:
            return candidate, available[candidate]

    raise CofactorError(
        f"No ready-made single-atom moleculetype named '{resname}' was found in {itp_path}. "
        f"Available ion moleculetypes: {', '.join(sorted(available)) or 'none'}. "
        "Rename the cofactor residue to match one of these, or supply the cofactor as a "
        "GAFF2-parametrised molecule instead (two or more heavy atoms)."
    )


def format_gro_atom_line(
    resnum: int, resname: str, atomname: str, atomnum: int,
    x_nm: float, y_nm: float, z_nm: float,
) -> str:
    """Format one fixed-width GROMACS .gro atom coordinate line (nm units)."""
    return (
        f"{resnum % 100000:5d}{resname[:5]:<5s}{atomname[:5]:>5s}{atomnum % 100000:5d}"
        f"{x_nm:8.3f}{y_nm:8.3f}{z_nm:8.3f}\n"
    )
