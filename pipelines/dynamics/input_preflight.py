"""Early validation and conservative cleanup for molecular-dynamics inputs."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path


PROTEIN_RESIDUES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "HID",
    "HIE", "HIP", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR",
    "TRP", "TYR", "VAL", "ASH", "GLH", "LYN", "CYX",
}


def _pdb_atoms(path: str | Path):
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith(("ATOM  ", "HETATM")) and len(line) >= 27:
                yield line


def inspect_pdb(path: str | Path) -> dict:
    """Return enough structural metadata to catch swapped CLI inputs."""
    atoms = list(_pdb_atoms(path))
    residues = {(line[21:22], line[22:26].strip(), line[26:27], line[17:20].strip().upper()) for line in atoms}
    protein_residues = {item for item in residues if item[3] in PROTEIN_RESIDUES}
    nonstandard = sorted({item[3] for item in residues if item[3] not in PROTEIN_RESIDUES})
    residue_atom_names = Counter(
        (line[21:22], line[22:26].strip(), line[26:27], line[17:20].strip(), line[12:16].strip())
        for line in atoms
    )
    duplicate_names = sum(count - 1 for count in residue_atom_names.values() if count > 1)
    hydrogen_names = Counter(
        (line[21:22], line[22:26].strip(), line[26:27], line[17:20].strip(), line[12:16].strip())
        for line in atoms
        if (line[76:78].strip().upper() if len(line) >= 78 else "") == "H"
        or line[12:16].strip().lstrip("0123456789").upper().startswith("H")
    )
    duplicate_hydrogens = sum(count - 1 for count in hydrogen_names.values() if count > 1)
    protein_fraction = len(protein_residues) / len(residues) if residues else 0.0
    return {
        "path": str(Path(path).resolve()),
        "atom_count": len(atoms),
        "residue_count": len(residues),
        "protein_residue_count": len(protein_residues),
        "protein_fraction": round(protein_fraction, 4),
        "classification": "protein" if len(protein_residues) >= 2 and protein_fraction >= 0.8 else "small_molecule",
        "nonstandard_residues": nonstandard,
        "duplicate_atom_names": duplicate_names,
        "duplicate_hydrogen_names": duplicate_hydrogens,
    }


def validate_input_roles(dyn_type: str, files: list[str], report_path: str | Path) -> dict:
    """Validate semantic input order and persist an actionable report."""
    inspected = [inspect_pdb(path) if Path(path).suffix.lower() == ".pdb" else {
        "path": str(Path(path).resolve()), "classification": "small_molecule"
    } for path in files]
    errors: list[str] = []
    warnings: list[str] = []
    receptor_count = 2 if dyn_type == "ppligand" else 1

    for index in range(min(receptor_count, len(inspected))):
        if inspected[index]["classification"] != "protein":
            errors.append(
                f"El receptor '{Path(files[index]).name}' no parece una proteína "
                f"(residuos no estándar: {', '.join(inspected[index].get('nonstandard_residues', [])) or 'ninguno'})."
            )
    if dyn_type in {"pligand", "ppligand"} and len(inspected) > receptor_count:
        ligand = inspected[receptor_count]
        if ligand["classification"] == "protein":
            errors.append(
                f"El ligando '{Path(files[receptor_count]).name}' parece una proteína. "
                "Es probable que receptor y ligando estén invertidos."
            )
    for item in inspected[:receptor_count]:
        if item.get("duplicate_hydrogen_names", 0):
            warnings.append(
                f"{Path(item['path']).name}: {item['duplicate_hydrogen_names']} nombres de hidrógeno duplicados; "
                "los hidrógenos se eliminarán y regenerarán."
            )

    report = {"dynamic_type": dyn_type, "inputs": inspected, "errors": errors, "warnings": warnings}
    Path(report_path).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if errors:
        raise ValueError(" ".join(errors))
    return report


def prepare_protein_pdb(source: str | Path, destination: str | Path) -> dict:
    """Strip input hydrogens and normalize standard amino-acid records.

    pdb2gmx is already invoked with ``-ignh``. Removing hydrogens before PDBFixer
    also prevents non-standard names such as YASARA's repeated 1HH/2HH pairs
    from being interpreted as duplicate atoms.
    """
    removed_hydrogens = 0
    converted_records = 0
    discarded_altlocs = 0
    output: list[str] = []
    with open(source, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")) or len(line) < 27:
                output.append(line)
                continue
            atom_name = line[12:16].strip()
            element = line[76:78].strip().upper() if len(line) >= 78 else ""
            inferred = atom_name.lstrip("0123456789")[:1].upper()
            if element == "H" or (not element and inferred == "H"):
                removed_hydrogens += 1
                continue
            altloc = line[16:17]
            if altloc not in {" ", "A"}:
                discarded_altlocs += 1
                continue
            if altloc == "A":
                line = line[:16] + " " + line[17:]
            residue = line[17:20].strip().upper()
            if line.startswith("HETATM") and residue in PROTEIN_RESIDUES:
                line = "ATOM  " + line[6:]
                converted_records += 1
            output.append(line)
    Path(destination).write_text("".join(output), encoding="utf-8")
    return {
        "source": str(Path(source).resolve()),
        "prepared": str(Path(destination).resolve()),
        "removed_hydrogens": removed_hydrogens,
        "converted_protein_records": converted_records,
        "discarded_alternate_locations": discarded_altlocs,
    }
