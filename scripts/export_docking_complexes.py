#!/usr/bin/env python3
"""Export ranked docking poses and receptor-ligand complexes for MD setup."""

import argparse
import csv
import re
import shutil
from pathlib import Path


def _element(atom_name: str, autodock_type: str) -> str:
    """Map common AutoDock atom types to PDB element symbols."""
    mapping = {
        "A": "C", "C": "C", "HD": "H", "H": "H", "N": "N",
        "NA": "N", "NS": "N", "OA": "O", "OS": "O", "O": "O",
        "P": "P", "S": "S", "SA": "S", "F": "F", "CL": "Cl",
        "BR": "Br", "I": "I", "MG": "Mg", "MN": "Mn", "ZN": "Zn",
        "CA": "Ca", "FE": "Fe",
    }
    atom_type = autodock_type.upper()
    if atom_type in mapping:
        return mapping[atom_type]
    match = re.search(r"[A-Za-z]+", atom_name)
    return (match.group(0)[0] if match else "C").upper()


def extract_pose(dlg_path: Path, run_number: int) -> list[dict]:
    """Extract atom metadata and coordinates for one run from an AutoDock DLG."""
    atoms: list[dict] = []
    in_target = False
    with dlg_path.open(encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            run_match = re.match(r"\s*Run:\s*(\d+)\s*/", line)
            if run_match:
                current_run = int(run_match.group(1))
                if in_target and current_run != run_number:
                    break
                in_target = current_run == run_number
                continue
            if not in_target:
                continue
            if line.strip().startswith("DOCKED: ENDMDL"):
                break
            match = re.match(r"\s*DOCKED:\s?(ATOM  |HETATM)(.*)$", line)
            if not match:
                continue
            record = match.group(1) + match.group(2)
            try:
                atoms.append({
                    "name": record[12:16].strip() or "C",
                    "x": float(record[30:38]),
                    "y": float(record[38:46]),
                    "z": float(record[46:54]),
                    "type": record.split()[-1],
                })
            except (ValueError, IndexError):
                continue
    if not atoms:
        raise RuntimeError(f"No atoms found for run {run_number} in {dlg_path}")
    return atoms


def write_ligand(atoms: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for serial, atom in enumerate(atoms, 1):
            element = _element(atom["name"], atom["type"])
            handle.write(
                f"HETATM{serial:5d} {atom['name'][:4]:>4s} LIG Z   1    "
                f"{atom['x']:8.3f}{atom['y']:8.3f}{atom['z']:8.3f}"
                f"  1.00  0.00          {element:>2s}\n"
            )
        handle.write("TER\nEND\n")


def cofactor_pdb_for(clean_receptors_dir: Path, protein: str) -> Path | None:
    """Cofactors (heme, metal ions) kept out of the Meeko receptor, if any."""
    cofactor = clean_receptors_dir / f"{protein}_cofactors.pdb"
    return cofactor if cofactor.is_file() else None


def write_complex(
    receptor: Path, ligand: Path, output: Path, cofactor: Path | None = None
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as dst:
        with receptor.open(encoding="utf-8", errors="ignore") as src:
            for line in src:
                if not line.startswith(("END", "CONECT")):
                    dst.write(line)
        dst.write("TER\n")
        if cofactor is not None:
            with cofactor.open(encoding="utf-8") as src:
                for line in src:
                    if line.startswith(("ATOM", "HETATM")):
                        dst.write(line)
            dst.write("TER\n")
        with ligand.open(encoding="utf-8") as src:
            for line in src:
                if line.startswith(("ATOM", "HETATM")):
                    dst.write(line)
        dst.write("TER\nEND\n")


def export(
    run_dir: Path,
    output_dir: Path,
    top_n: int,
    candidates_csv: Path | None = None,
) -> None:
    candidates = candidates_csv or run_dir / "docking_analysis" / "dynamics_candidates.csv"
    if not candidates.is_file():
        raise FileNotFoundError(f"Candidates CSV not found: {candidates}")

    receptor_dir = output_dir / "receptors"
    ligand_dir = output_dir / "ligands"
    complex_dir = output_dir / "complexes"
    for directory in (receptor_dir, ligand_dir, complex_dir):
        directory.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    with candidates.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))[:top_n]

    for row in rows:
        protein = row["protein"]
        ligand = row["ligand"]
        rank = int(row["rank"])
        run_number = int(row["best_run"])
        receptor = run_dir / "clean_receptors" / f"{protein}_h.pdb"
        if not receptor.is_file():
            receptor = run_dir / "clean_receptors" / f"{protein}.pdb"
        if not receptor.is_file():
            raise FileNotFoundError(f"Prepared receptor PDB not found for {protein}")

        receptor_out = receptor_dir / f"{protein}.pdb"
        if not receptor_out.exists():
            shutil.copy2(receptor, receptor_out)

        dlg = Path(row["dlg_file"])
        if not dlg.is_absolute() and not dlg.is_file():
            dlg = run_dir / "docking_results" / protein / "dlg" / f"{protein}_{ligand}.dlg"
        atoms = extract_pose(dlg, run_number)
        stem = f"rank_{rank:02d}_{protein}_{ligand}_run_{run_number}"
        ligand_out = ligand_dir / f"{stem}_ligand.pdb"
        complex_out = complex_dir / f"{stem}_complex.pdb"
        write_ligand(atoms, ligand_out)
        cofactor = cofactor_pdb_for(run_dir / "clean_receptors", protein)
        write_complex(receptor_out, ligand_out, complex_out, cofactor)
        manifest_rows.append({
            **row,
            "receptor_pdb": str(receptor_out),
            "ligand_pdb": str(ligand_out),
            "complex_pdb": str(complex_out),
            "ligand_atoms": len(atoms),
        })

    manifest = output_dir / "manifest.csv"
    if manifest_rows:
        with manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=manifest_rows[0].keys())
            writer.writeheader()
            writer.writerows(manifest_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Docking run directory")
    parser.add_argument("--output-dir", type=Path, help="Destination directory")
    parser.add_argument("--top", type=int, default=10, help="Maximum ranked candidates")
    args = parser.parse_args()
    output = args.output_dir or args.run_dir / "docking_analysis" / "dynamics_inputs"
    export(args.run_dir, output, max(0, args.top))


if __name__ == "__main__":
    main()
