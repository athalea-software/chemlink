"""Preparation and validation for relative ligand FEP simulations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib.util
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile

from rdkit import Chem
from rdkit.Chem import AllChem, rdFMCS, rdDetermineBonds


class FEPPreparationError(RuntimeError):
    pass


class FEPExecutionError(RuntimeError):
    pass


@dataclass
class PreparedLigand:
    label: str
    source: str
    sdf: str
    charge: int
    atoms: int
    heavy_atoms: int
    method: str


def _distance(a, b) -> float:
    return math.sqrt(sum((a[index] - b[index]) ** 2 for index in range(3)))


def _angle(a, b, c) -> float:
    first = [a[index] - b[index] for index in range(3)]
    second = [c[index] - b[index] for index in range(3)]
    denominator = math.sqrt(sum(x * x for x in first) * sum(x * x for x in second))
    if denominator == 0:
        return 0.0
    cosine = max(-1.0, min(1.0, sum(x * y for x, y in zip(first, second)) / denominator))
    return math.degrees(math.acos(cosine))


def _protein_anchor_candidates(protein: Path) -> list[dict]:
    atoms = []
    for line in protein.read_text(errors="replace").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")) or len(line) < 54:
            continue
        if line[12:16].strip() not in {"N", "CA", "C"}:
            continue
        try:
            xyz = tuple(float(line[start:end]) for start, end in ((30, 38), (38, 46), (46, 54)))
        except ValueError:
            continue
        atoms.append({
            "serial": int(line[6:11]), "name": line[12:16].strip(),
            "residue": line[17:20].strip(), "chain": line[21:22].strip() or "_",
            "residue_number": line[22:26].strip(), "xyz": xyz,
        })
    return atoms


def select_boresch_anchors(protein_path: str | Path, ligand: Chem.Mol) -> dict:
    """Select a conservative CA/N/C and rigid ligand anchor sextet."""
    protein_atoms = _protein_anchor_candidates(Path(protein_path))
    by_residue: dict[tuple[str, str], dict[str, dict]] = {}
    for atom in protein_atoms:
        by_residue.setdefault((atom["chain"], atom["residue_number"]), {})[atom["name"]] = atom
    residues = [atoms for atoms in by_residue.values() if {"N", "CA", "C"} <= atoms.keys()]
    conformer = ligand.GetConformer()
    ligand_candidates = [
        atom for atom in ligand.GetAtoms()
        if atom.GetAtomicNum() > 1 and (atom.IsInRing() or atom.GetDegree() >= 2)
    ]
    if len(ligand_candidates) < 3 or not residues:
        raise FEPPreparationError("Could not find enough rigid atoms for Boresch restraints.")
    ligand_xyz = {
        atom.GetIdx(): tuple(conformer.GetAtomPosition(atom.GetIdx())) for atom in ligand_candidates
    }
    closest = min(
        ((residue, atom) for residue in residues for atom in ligand_candidates),
        key=lambda pair: _distance(pair[0]["CA"]["xyz"], ligand_xyz[pair[1].GetIdx()]),
    )
    residue, ligand_1 = closest
    heavy_neighbors = [
        atom for atom in ligand_1.GetNeighbors() if atom.GetAtomicNum() > 1
    ]
    if not heavy_neighbors:
        raise FEPPreparationError("Ligand anchor L1 has no bonded heavy-atom neighbour.")
    ligand_2 = heavy_neighbors[0]
    third_candidates = [
        atom for atom in ligand_2.GetNeighbors()
        if atom.GetAtomicNum() > 1 and atom.GetIdx() != ligand_1.GetIdx()
    ]
    if not third_candidates:
        third_candidates = [
            atom for atom in ligand_candidates
            if atom.GetIdx() not in {ligand_1.GetIdx(), ligand_2.GetIdx()}
        ]
    ligand_3 = min(
        third_candidates,
        key=lambda atom: abs(_angle(
            ligand_xyz.get(ligand_1.GetIdx(), tuple(conformer.GetAtomPosition(ligand_1.GetIdx()))),
            tuple(conformer.GetAtomPosition(ligand_2.GetIdx())),
            tuple(conformer.GetAtomPosition(atom.GetIdx())),
        ) - 90.0),
    )
    p1, p2, p3 = residue["CA"], residue["N"], residue["C"]
    l1 = ligand_xyz[ligand_1.GetIdx()]
    l2 = tuple(conformer.GetAtomPosition(ligand_2.GetIdx()))
    l3 = tuple(conformer.GetAtomPosition(ligand_3.GetIdx()))
    geometry = {
        "distance_angstrom": round(_distance(p1["xyz"], l1), 4),
        "angle_p2_p1_l1_deg": round(_angle(p2["xyz"], p1["xyz"], l1), 4),
        "angle_p1_l1_l2_deg": round(_angle(p1["xyz"], l1, l2), 4),
        "angle_l1_l2_l3_deg": round(_angle(l1, l2, l3), 4),
    }
    problematic = [name for name, value in geometry.items() if "angle" in name and not 15 < value < 165]
    status = "valid" if 1.5 <= geometry["distance_angstrom"] <= 15 and not problematic else "manual_review"
    return {
        "status": status,
        "protein": [
            {key: atom[key] for key in ("serial", "name", "residue", "chain", "residue_number")}
            for atom in (p1, p2, p3)
        ],
        "ligand": [
            {"index": atom.GetIdx(), "name": atom.GetSymbol(), "atomic_number": atom.GetAtomicNum()}
            for atom in (ligand_1, ligand_2, ligand_3)
        ],
        "geometry": geometry,
        "force_constants": {
            "distance_kj_mol_nm2": 4184.0,
            "angle_kj_mol_rad2": 41.84,
            "dihedral_kj_mol_rad2": 41.84,
        },
        "validation": {"problematic_geometry": problematic, "requires_short_md": True},
    }


def _read_first_sdf(path: Path) -> Chem.Mol | None:
    supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=True)
    return supplier[0] if supplier and len(supplier) else None


def _pdb_to_molecule(path: Path, charge: int, smiles: str | None) -> tuple[Chem.Mol, str]:
    # Extracted ligands often retain CONECT records from the original complex;
    # those serials can reference protein atoms and create phantom atoms/bonds.
    source_lines = path.read_text(errors="replace").splitlines()
    is_pdbqt = any(
        line.startswith(("ROOT", "ENDROOT", "BRANCH", "ENDBRANCH", "TORSDOF"))
        for line in source_lines
    )
    if is_pdbqt and not smiles:
        try:
            from .ligand_charge import resolve_openbabel_binary
        except ImportError:
            from pipelines.dynamics.ligand_charge import resolve_openbabel_binary
        try:
            executable = resolve_openbabel_binary()
        except RuntimeError as exc:
            raise FEPPreparationError(str(exc)) from exc
        with tempfile.TemporaryDirectory(prefix="chemlink-fep-pdbqt-") as directory:
            converted = Path(directory) / "ligand.sdf"
            result = subprocess.run(
                [executable, "-ipdbqt", str(path), "-osdf", "-h", "-O", str(converted)],
                capture_output=True, text=True, timeout=180,
            )
            if result.returncode != 0:
                raise FEPPreparationError(
                    f"Open Babel could not convert {path.name}: "
                    + (result.stderr.strip() or "unknown conversion error")
                )
            molecule = _read_first_sdf(converted)
            if molecule is None:
                raise FEPPreparationError(
                    f"Open Babel produced an invalid SDF for {path.name}."
                )
            return molecule, "Open Babel PDBQT-formatted PDB to SDF"
    autodock_elements = {
        "A": "C", "C": "C", "N": "N", "NA": "N", "NS": "N",
        "O": "O", "OA": "O", "OS": "O", "S": "S", "SA": "S",
        "P": "P", "H": "H", "HD": "H", "HS": "H", "F": "F",
        "CL": "Cl", "BR": "Br", "I": "I", "MG": "Mg", "MN": "Mn",
        "ZN": "Zn", "CA": "Ca", "FE": "Fe",
    }
    atom_lines = []
    for line in source_lines:
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        if is_pdbqt:
            atom_type = line.split()[-1].upper()
            element = autodock_elements.get(atom_type)
            if element is None:
                raise FEPPreparationError(
                    f"Unsupported AutoDock atom type '{atom_type}' in {path.name}."
                )
            # Remove PDBQT charge/type columns and write a valid PDB element field.
            line = line[:66].ljust(76) + element.rjust(2)
        atom_lines.append(line + "\n")
    pdb_block = "".join(atom_lines) + "END\n"
    molecule = Chem.MolFromPDBBlock(
        pdb_block, removeHs=False, sanitize=False, proximityBonding=True,
    )
    if molecule is None or molecule.GetNumAtoms() == 0:
        raise FEPPreparationError(f"RDKit could not read PDB ligand: {path}")
    if smiles:
        template = Chem.MolFromSmiles(smiles)
        if template is None:
            raise FEPPreparationError(f"Invalid reference SMILES for {path.name}")
        try:
            molecule = AllChem.AssignBondOrdersFromTemplate(
                Chem.RemoveHs(template), Chem.RemoveHs(molecule),
            )
        except Exception as exc:
            raise FEPPreparationError(
                f"Reference SMILES does not match PDB ligand {path.name}: {exc}"
            ) from exc
        method = "RDKit PDB coordinates + reference-SMILES bond orders"
    else:
        try:
            rdDetermineBonds.DetermineBonds(
                molecule, charge=int(charge), allowChargedFragments=False,
            )
        except Exception as exc:
            raise FEPPreparationError(
                f"Could not infer PDB connectivity/bond orders for {path.name} at charge "
                f"{charge}: {exc}. Provide a reference SMILES or an SDF file."
            ) from exc
        source_kind = "PDBQT-formatted PDB" if is_pdbqt else "PDB"
        method = f"RDKit DetermineBonds from {source_kind} coordinates"
    Chem.SanitizeMol(molecule)
    return molecule, method


def prepare_fep_ligand(
    source: str | Path,
    destination: str | Path,
    *,
    label: str,
    charge: int | None,
    smiles: str | None = None,
) -> tuple[PreparedLigand, Chem.Mol]:
    source_path = Path(source).expanduser().resolve()
    destination_path = Path(destination).resolve()
    if not source_path.is_file():
        raise FEPPreparationError(f"Ligand {label} was not found: {source_path}")
    suffix = source_path.suffix.lower()
    if suffix == ".pdb":
        if charge is None:
            raise FEPPreparationError(
                f"PDB does not reliably encode net charge; provide --charge-{label.lower()}."
            )
        molecule, method = _pdb_to_molecule(source_path, charge, smiles)
    elif suffix in {".sdf", ".mol"}:
        molecule = _read_first_sdf(source_path) if suffix == ".sdf" else Chem.MolFromMolFile(
            str(source_path), removeHs=False, sanitize=True,
        )
        if molecule is None:
            raise FEPPreparationError(f"Could not read ligand {label}: {source_path}")
        detected = int(Chem.GetFormalCharge(molecule))
        if charge is None:
            charge = detected
        elif charge != detected:
            raise FEPPreparationError(
                f"Ligand {label} has formal charge {detected} in {suffix}, but {charge} was requested."
            )
        method = f"RDKit normalized {suffix[1:].upper()}"
    else:
        raise FEPPreparationError(
            f"FEP ligand {label} format {suffix or '(none)'} is unsupported; use PDB, SDF, or MOL."
        )
    if len(Chem.GetMolFrags(molecule)) != 1:
        raise FEPPreparationError(f"Ligand {label} must contain exactly one connected molecule.")
    if molecule.GetNumConformers() == 0 or not molecule.GetConformer().Is3D():
        raise FEPPreparationError(f"Ligand {label} must contain 3D coordinates.")
    actual_charge = int(Chem.GetFormalCharge(molecule))
    if actual_charge != int(charge):
        raise FEPPreparationError(
            f"Converted ligand {label} has charge {actual_charge}, expected {charge}."
        )
    electron_count = sum(
        atom.GetAtomicNum() + atom.GetNumImplicitHs() for atom in molecule.GetAtoms()
    ) - actual_charge
    if electron_count % 2:
        raise FEPPreparationError(
            f"Ligand {label} has an odd electron count ({electron_count}) at charge "
            f"{actual_charge}. Check protonation/net charge or provide a closed-shell SDF/SMILES; "
            "ACPYPE AM1-BCC uses singlet multiplicity by default."
        )
    molecule.SetProp("_Name", f"FEP ligand {label}")
    molecule.SetIntProp("CHEMLINK_NET_CHARGE", int(charge))
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(destination_path))
    writer.write(molecule)
    writer.close()
    prepared = PreparedLigand(
        label=label,
        source=str(source_path),
        sdf=str(destination_path),
        charge=int(charge),
        atoms=molecule.GetNumAtoms(),
        heavy_atoms=molecule.GetNumHeavyAtoms(),
        method=method,
    )
    return prepared, molecule


def compare_ligands(
    first: Chem.Mol,
    second: Chem.Mol,
    *,
    require_equal_charge: bool = True,
    minimum_coverage: float = 0.5,
) -> dict:
    if require_equal_charge and Chem.GetFormalCharge(first) != Chem.GetFormalCharge(second):
        raise FEPPreparationError(
            "Initial RBFE support requires equal net charges for ligands A and B."
        )
    result = rdFMCS.FindMCS(
        [Chem.RemoveHs(first), Chem.RemoveHs(second)],
        atomCompare=rdFMCS.AtomCompare.CompareElements,
        bondCompare=rdFMCS.BondCompare.CompareOrder,
        ringMatchesRingOnly=True,
        completeRingsOnly=True,
        timeout=30,
    )
    if result.canceled or result.numAtoms == 0:
        raise FEPPreparationError("No reliable common chemical substructure was found.")
    minimum = min(first.GetNumHeavyAtoms(), second.GetNumHeavyAtoms())
    coverage = result.numAtoms / minimum if minimum else 0.0
    if coverage < minimum_coverage:
        raise FEPPreparationError(
            f"Ligands are too dissimilar for the initial RBFE workflow (MCS coverage {coverage:.1%})."
        )
    query = Chem.MolFromSmarts(result.smartsString)
    match_a = list(Chem.RemoveHs(first).GetSubstructMatch(query))
    match_b = list(Chem.RemoveHs(second).GetSubstructMatch(query))
    return {
        "common_heavy_atoms": result.numAtoms,
        "minimum_heavy_atoms": minimum,
        "coverage": round(coverage, 4),
        "smarts": result.smartsString,
        "atom_mapping": [{"a": a, "b": b} for a, b in zip(match_a, match_b)],
    }


def dependency_status() -> dict:
    gromacs = shutil.which("gmx_mpi") or shutil.which("gmx")
    if not gromacs:
        for candidate in ("/usr/local/gromacs/bin/gmx_mpi", "/usr/local/gromacs/bin/gmx"):
            if Path(candidate).is_file():
                gromacs = candidate
                break
    return {
        "gromacs": gromacs,
        "pmx": shutil.which("pmx") or ("python-module" if importlib.util.find_spec("pmx") else None),
        "alchemlyb": "python-module" if importlib.util.find_spec("alchemlyb") else None,
        "pymbar": "python-module" if importlib.util.find_spec("pymbar") else None,
        "acpype": shutil.which("acpype"),
        "openbabel": shutil.which("obabel"),
        "rdkit": Chem.rdBase.rdkitVersion,
    }


def _run_fep_command(command: list[str], *, cwd: Path, input_text: str | None = None) -> None:
    """Run one setup command and retain useful diagnostics in the run directory."""
    result = subprocess.run(
        command, cwd=str(cwd), input=input_text, text=True,
        capture_output=True, timeout=900,
    )
    with (cwd / "setup_commands.log").open("a", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(command) + "\n")
        handle.write(result.stdout)
        handle.write(result.stderr)
        handle.write("\n")
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        raise FEPExecutionError(
            f"Command failed ({result.returncode}) in {cwd}: {' '.join(command)}\n{detail[-3000:]}"
        )


def _merge_gro(protein_gro: Path, ligand_gro: Path, output: Path) -> None:
    protein = protein_gro.read_text().splitlines(keepends=True)
    ligand = ligand_gro.read_text().splitlines(keepends=True)
    if len(protein) < 3 or len(ligand) < 3:
        raise FEPExecutionError("ACPYPE or pdb2gmx produced an invalid GRO file.")
    atoms = protein[2:-1] + ligand[2:-1]
    output.write_text(
        "ChemLink ABFE complex\n" + str(len(atoms)) + "\n" + "".join(atoms) + protein[-1],
        encoding="utf-8",
    )


def _split_ligand_topology(source: Path, atomtypes: Path, molecule: Path) -> None:
    """Split ACPYPE output so GROMACS sees atom types before molecule definitions."""
    lines = source.read_text().splitlines(keepends=True)
    start = next(
        (index for index, line in enumerate(lines) if line.strip().lower() == "[ atomtypes ]"),
        None,
    )
    if start is None:
        raise FEPExecutionError(f"No [ atomtypes ] section found in {source}.")
    end = next(
        (index for index in range(start + 1, len(lines))
         if lines[index].strip().startswith("[")),
        len(lines),
    )
    atomtypes.write_text("".join(lines[start:end]), encoding="utf-8")
    molecule.write_text("".join(lines[:start] + lines[end:]), encoding="utf-8")


def _patch_complex_topology(topology: Path, atomtypes_itp: Path, ligand_itp: Path) -> None:
    lines = topology.read_text().splitlines(keepends=True)
    forcefield = next(
        (index for index, line in enumerate(lines) if "forcefield.itp" in line), None,
    )
    if forcefield is None:
        raise FEPExecutionError(f"No forcefield.itp include found in {topology}.")
    lines[forcefield + 1:forcefield + 1] = [
        "\n; ChemLink GAFF2 atom types\n",
        f'#include "{atomtypes_itp.as_posix()}"\n',
    ]
    insertion = next(
        (index for index, line in enumerate(lines) if "water" in line.lower() and "#include" in line),
        next((index for index, line in enumerate(lines) if line.strip() == "[ system ]"), len(lines)),
    )
    lines[insertion:insertion] = [
        "\n; ChemLink GAFF2 ligand molecule\n",
        f'#include "{ligand_itp.as_posix()}"\n', "\n",
    ]
    molecules = next((index for index, line in enumerate(lines) if line.strip() == "[ molecules ]"), None)
    if molecules is None:
        raise FEPExecutionError(f"No [ molecules ] section found in {topology}.")
    lines.append("LIG                 1\n")
    topology.write_text("".join(lines), encoding="utf-8")


def _solvent_topology(
    protein_topology: Path, atomtypes_itp: Path, ligand_itp: Path, output: Path,
) -> None:
    includes = [
        line for line in protein_topology.read_text().splitlines()
        if line.lstrip().startswith("#include") and (
            "forcefield.itp" in line or "water" in line.lower()
            or any(model in line.lower() for model in ("tip3p.itp", "tip4p.itp", "spc.itp"))
            or "ions.itp" in line
        )
    ]
    forcefield = next((line for line in includes if "forcefield.itp" in line), None)
    water = next((
        line for line in includes
        if "water" in line.lower()
        or any(model in line.lower() for model in ("tip3p.itp", "tip4p.itp", "spc.itp"))
    ), None)
    ions = next((line for line in includes if "ions.itp" in line), None)
    if not forcefield or not water or not ions:
        raise FEPExecutionError("Could not derive solvent topology includes from pdb2gmx output.")
    output.write_text(
        "\n".join((
            forcefield, f'#include "{atomtypes_itp.as_posix()}"',
            f'#include "{ligand_itp.as_posix()}"', water, ions,
        ))
        + "\n\n[ system ]\nLigand in solvent\n\n[ molecules ]\nLIG 1\n",
        encoding="utf-8",
    )


def _ions_mdp() -> str:
    return (
        "integrator = steep\nnsteps = 500\ncutoff-scheme = Verlet\n"
        "coulombtype = cutoff\nrcoulomb = 1.0\nrvdw = 1.0\npbc = xyz\n"
    )


def repair_abfe_protein(source: Path, destination: Path, report_path: Path) -> dict:
    """Rebuild missing protein heavy atoms while preserving residue identifiers."""
    try:
        from pdbfixer import PDBFixer
        from openmm.app import PDBFile
    except ImportError as exc:
        raise FEPExecutionError(
            "Protein repair requires pdbfixer and openmm before ABFE system construction."
        ) from exc
    try:
        fixer = PDBFixer(filename=str(source))
        # Discover sequence gaps because PDBFixer requires this state, but do not
        # synthesize entire missing residues without an explicit modelling step.
        fixer.findMissingResidues()
        sequence_gaps = len(fixer.missingResidues)
        fixer.missingResidues = {}
        fixer.findMissingAtoms()
        missing = [
            {
                "chain": residue.chain.id,
                "residue": residue.name,
                "residue_number": residue.id,
                "atoms": [getattr(atom, "name", str(atom)) for atom in atoms],
            }
            for residue, atoms in fixer.missingAtoms.items()
        ]
        terminal = [
            {
                "chain": residue.chain.id,
                "residue": residue.name,
                "residue_number": residue.id,
                "atoms": [getattr(atom, "name", str(atom)) for atom in atoms],
            }
            for residue, atoms in fixer.missingTerminals.items()
        ]
        fixer.addMissingAtoms()
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            PDBFile.writeFile(fixer.topology, fixer.positions, handle, keepIds=True)
    except Exception as exc:
        raise FEPExecutionError(f"PDBFixer could not repair ABFE protein {source}: {exc}") from exc
    report = {
        "source": str(source), "output": str(destination),
        "missing_residue_gaps_detected_but_not_built": sequence_gaps,
        "repaired_residues": missing, "repaired_terminals": terminal,
        "added_heavy_atoms": sum(len(item["atoms"]) for item in missing + terminal),
        "hydrogens_added": 0,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def execute_abfe_system_setup(config: dict, report: dict) -> dict:
    """Parameterize ligands and build/solvate/neutralize all four ABFE systems."""
    if report.get("method") != "ABFE":
        raise FEPExecutionError("ABFE system setup requires an ABFE preparation report.")
    tools = dependency_status()
    missing = [name for name in ("gromacs", "acpype") if not tools.get(name)]
    if missing:
        raise FEPExecutionError("ABFE setup requires: " + ", ".join(missing))
    root = Path(config["work_dir"]).resolve()
    common = root / "common"
    gmx = str(tools["gromacs"])
    force_field = str(config.get("force_field", "amber99sb-ildn"))
    water = str(config.get("water_model", "tip3p"))
    protein_dir = common / "protein_topology"
    protein_dir.mkdir(exist_ok=True)
    repair_report = repair_abfe_protein(
        common / "protein.pdb", protein_dir / "protein.pdb", common / "protein_repair.json",
    )
    _run_fep_command([
        gmx, "pdb2gmx", "-f", "protein.pdb", "-o", "protein.gro", "-p", "topol.top",
        "-water", water, "-ignh", "-ff", force_field,
    ], cwd=protein_dir)
    ligand_outputs: dict[str, dict[str, str]] = {}
    for label in ("A", "B"):
        parameter_dir = common / f"ligand_{label}_parameters"
        parameter_dir.mkdir(exist_ok=True)
        source = common / f"ligand_{label}_normalized.sdf"
        shutil.copy2(source, parameter_dir / "ligand.sdf")
        charge = int(report["ligands"][label]["charge"])
        _run_fep_command([
            str(tools["acpype"]), "-i", "ligand.sdf", "-b", "LIG", "-c", "bcc",
            "-n", str(charge), "-a", "gaff2",
        ], cwd=parameter_dir)
        outputs = list(parameter_dir.glob("*.acpype/LIG_GMX.itp"))
        coordinates = list(parameter_dir.glob("*.acpype/LIG_GMX.gro"))
        if len(outputs) != 1 or len(coordinates) != 1:
            raise FEPExecutionError(f"ACPYPE outputs for ligand {label} are missing or ambiguous.")
        atomtypes = parameter_dir / "ligand_atomtypes.itp"
        molecule_itp = parameter_dir / "ligand.itp"
        _split_ligand_topology(outputs[0], atomtypes, molecule_itp)
        ligand_outputs[label] = {
            "atomtypes": str(atomtypes.resolve()), "itp": str(molecule_itp.resolve()),
            "gro": str(coordinates[0].resolve()),
        }
    setup = {"status": "building", "protein_repair": repair_report, "systems": {}}
    for label in ("A", "B"):
        ligand_itp = Path(ligand_outputs[label]["itp"])
        ligand_atomtypes = Path(ligand_outputs[label]["atomtypes"])
        ligand_gro = Path(ligand_outputs[label]["gro"])
        for leg in ("complex", "solvent"):
            system_dir = root / f"ligand_{label}" / leg / "system"
            system_dir.mkdir(parents=True, exist_ok=True)
            topology = system_dir / "topol.top"
            local_ligand_itp = system_dir / "ligand.itp"
            local_atomtypes_itp = system_dir / "ligand_atomtypes.itp"
            shutil.copy2(ligand_itp, local_ligand_itp)
            shutil.copy2(ligand_atomtypes, local_atomtypes_itp)
            if leg == "complex":
                shutil.copy2(protein_dir / "topol.top", topology)
                for protein_itp in protein_dir.glob("*.itp"):
                    shutil.copy2(protein_itp, system_dir / protein_itp.name)
                _patch_complex_topology(
                    topology, Path("ligand_atomtypes.itp"), Path("ligand.itp"),
                )
                _merge_gro(protein_dir / "protein.gro", ligand_gro, system_dir / "input.gro")
            else:
                _solvent_topology(
                    protein_dir / "topol.top", Path("ligand_atomtypes.itp"),
                    Path("ligand.itp"), topology,
                )
                shutil.copy2(ligand_gro, system_dir / "input.gro")
            (system_dir / "ions.mdp").write_text(_ions_mdp(), encoding="utf-8")
            _run_fep_command([
                gmx, "editconf", "-f", "input.gro", "-o", "boxed.gro", "-bt", "dodecahedron",
                "-d", "1.0", "-c",
            ], cwd=system_dir)
            _run_fep_command([
                gmx, "solvate", "-cp", "boxed.gro", "-cs", "spc216.gro",
                "-o", "solvated.gro", "-p", "topol.top",
            ], cwd=system_dir)
            _run_fep_command([
                gmx, "grompp", "-f", "ions.mdp", "-c", "solvated.gro", "-p", "topol.top",
                "-o", "ions.tpr", "-maxwarn", "1",
            ], cwd=system_dir)
            _run_fep_command([
                gmx, "genion", "-s", "ions.tpr", "-o", "ionized.gro", "-p", "topol.top",
                "-pname", "NA", "-nname", "CL", "-neutral",
            ], cwd=system_dir, input_text="SOL\n")
            key = f"ligand_{label}_{leg}"
            setup["systems"][key] = {"status": "ready", "directory": str(system_dir)}
    setup["status"] = "systems_ready_restraints_pending"
    setup["pilot_blocker"] = (
        "Boresch restraint atom indices must be mapped onto each solvated topology before pilot grompp."
    )
    (root / "abfe_setup.json").write_text(json.dumps(setup, indent=2) + "\n", encoding="utf-8")
    return setup


def _production_mdp(state: int, lambdas: list[float], config: dict, method: str) -> str:
    values = " ".join(f"{value:.8f}" for value in lambdas)
    nsteps = int(float(config["production_ns"]) * 500000)
    coupling = ""
    if method == "ABFE":
        restraint_values = " ".join(f"{min(1.0, value * 4):.8f}" for value in lambdas)
        coul_values = " ".join(f"{max(0.0, min(1.0, (value - 0.25) * 4)):.8f}" for value in lambdas)
        vdw_values = " ".join(f"{max(0.0, min(1.0, (value - 0.5) * 2)):.8f}" for value in lambdas)
        coupling = """couple-moltype           = LIG
couple-lambda0           = vdw-q
couple-lambda1           = none
couple-intramol          = yes
"""
        lambda_parameters = (
            f"restraint-lambdas        = {restraint_values}\n"
            f"coul-lambdas             = {coul_values}\n"
            f"vdw-lambdas              = {vdw_values}\n"
        )
    else:
        lambda_parameters = f"fep-lambdas              = {values}\n"
    return f"""; ChemLink {method} production template
integrator               = md
dt                       = 0.002
nsteps                   = {nsteps}
continuation             = yes
constraints              = h-bonds
cutoff-scheme            = Verlet
coulombtype              = PME
vdwtype                  = Cut-off
tcoupl                   = v-rescale
ref-t                    = {float(config['temperature_k']):g}
free-energy              = yes
init-lambda-state        = {state}
{lambda_parameters}calc-lambda-neighbors    = 1
sc-alpha                 = 0.5
sc-power                 = 1
sc-sigma                 = 0.3
nstdhdl                  = 100
dhdl-print-energy        = total
separate-dhdl-file       = yes
{coupling}
"""


def _create_window_layout(
    root: Path, replicas: int, windows: int, config: dict, method: str,
) -> list[dict]:
    states = []
    lambdas = [index / (windows - 1) for index in range(windows)]
    systems = (
        [(None, "complex"), (None, "solvent")]
        if method == "RBFE"
        else [(ligand, leg) for ligand in ("ligand_A", "ligand_B") for leg in ("complex", "solvent")]
    )
    for ligand, leg in systems:
        for replica in range(1, replicas + 1):
            for index, value in enumerate(lambdas):
                base = root / ligand / leg if ligand else root / leg
                directory = base / f"replica_{replica:02d}" / f"lambda_{index:02d}"
                directory.mkdir(parents=True, exist_ok=True)
                state = {
                    "ligand": ligand, "leg": leg, "replica": replica, "lambda_state": index,
                    "lambda": round(value, 8), "status": "pending", "directory": str(directory),
                }
                (directory / "state.json").write_text(
                    json.dumps(state, indent=2) + "\n", encoding="utf-8",
                )
                (directory / "production.mdp").write_text(
                    _production_mdp(index, lambdas, config, method), encoding="utf-8",
                )
                states.append(state)
    return states


def prepare_fep_run(config: dict) -> dict:
    root = Path(config["work_dir"]).resolve()
    common = root / "common"
    common.mkdir(parents=True, exist_ok=True)
    if int(config["lambda_windows"]) < 2:
        raise FEPPreparationError("RBFE requires at least two lambda windows.")
    if int(config["replicas"]) < 1:
        raise FEPPreparationError("RBFE requires at least one replica.")
    if float(config["production_ns"]) <= 0 or float(config["equilibration_ns"]) <= 0:
        raise FEPPreparationError("FEP equilibration and production times must be positive.")
    protein = Path(config["protein"]).expanduser().resolve()
    if not protein.is_file():
        raise FEPPreparationError(f"Protein was not found: {protein}")
    shutil.copy2(protein, common / "protein.pdb")
    ligand_a, mol_a = prepare_fep_ligand(
        config["ligand_a"], common / "ligand_A_normalized.sdf", label="A",
        charge=config.get("charge_a"), smiles=config.get("smiles_a"),
    )
    ligand_b, mol_b = prepare_fep_ligand(
        config["ligand_b"], common / "ligand_B_normalized.sdf", label="B",
        charge=config.get("charge_b"), smiles=config.get("smiles_b"),
    )
    requested_method = str(config.get("fep_type", "auto")).lower()
    if requested_method not in {"auto", "rbfe", "abfe"}:
        raise FEPPreparationError(f"Unsupported FEP type: {requested_method}")
    try:
        comparison = compare_ligands(
            mol_a, mol_b, require_equal_charge=False, minimum_coverage=0.0,
        )
    except FEPPreparationError:
        if requested_method == "rbfe":
            raise
        comparison = {
            "common_heavy_atoms": 0,
            "minimum_heavy_atoms": min(mol_a.GetNumHeavyAtoms(), mol_b.GetNumHeavyAtoms()),
            "coverage": 0.0,
            "smarts": None,
            "atom_mapping": [],
        }
    rbfe_compatible = (
        ligand_a.charge == ligand_b.charge and comparison["coverage"] >= 0.5
    )
    method = (
        "ABFE" if requested_method == "abfe"
        else "RBFE" if requested_method == "rbfe"
        else "RBFE" if rbfe_compatible else "ABFE"
    )
    if method == "RBFE":
        mapping = compare_ligands(mol_a, mol_b)
        (common / "atom_mapping.json").write_text(
            json.dumps(mapping, indent=2) + "\n", encoding="utf-8",
        )
    else:
        mapping = comparison
        anchor_sets = {
            "A": select_boresch_anchors(protein, mol_a),
            "B": select_boresch_anchors(protein, mol_b),
        }
        for label, anchors in anchor_sets.items():
            (common / f"ligand_{label}_boresch_anchors.json").write_text(
                json.dumps(anchors, indent=2) + "\n", encoding="utf-8",
            )
        restraints = {
            "status": (
                "anchors_selected_pending_short_md"
                if all(item["status"] == "valid" for item in anchor_sets.values())
                else "manual_anchor_review_required"
            ),
            "method": "Boresch-style orientational restraints",
            "standard_state_correction": "required",
            "ligands": anchor_sets,
        }
        (common / "abfe_restraints.json").write_text(
            json.dumps(restraints, indent=2) + "\n", encoding="utf-8",
        )
    tools = dependency_status()
    parameterization = {}
    for label, ligand in (("A", ligand_a), ("B", ligand_b)):
        command = [
            "acpype", "-i", ligand.sdf, "-b", "LIG", "-c", "bcc",
            "-n", str(ligand.charge), "-a", "gaff2",
        ]
        parameterization[label] = {
            "status": "pending", "engine": "ACPYPE/GAFF2", "command": command,
            "expected_outputs": ["LIG_GMX.itp", "LIG_GMX.gro"],
        }
        (common / f"ligand_{label}_parameterization.json").write_text(
            json.dumps(parameterization[label], indent=2) + "\n", encoding="utf-8",
        )
    states = _create_window_layout(
        root, config["replicas"], config["lambda_windows"], config, method,
    )
    report = {
        "status": "prepared" if config.get("prepare_only") else "awaiting_alchemical_engine",
        "method": method,
        "requested_method": requested_method.upper(),
        "selection_reason": (
            "explicit user selection" if requested_method != "auto"
            else "ligands satisfy initial RBFE compatibility criteria" if method == "RBFE"
            else "automatic ABFE selection: unequal charges or common-heavy-atom coverage below 50%"
        ),
        "protein": str(Path(config["protein"]).resolve()),
        "ligands": {"A": asdict(ligand_a), "B": asdict(ligand_b)},
        "mapping": mapping,
        "dependencies": tools,
        "legs": ["complex", "solvent"],
        "independent_ligands": ["A", "B"] if method == "ABFE" else [],
        "replicas": config["replicas"],
        "lambda_windows": config["lambda_windows"],
        "state_count": len(states),
        "supported_estimators": ["BAR", "MBAR", "TI"],
        "parameterization": parameterization,
    }
    if method == "ABFE":
        systems = {}
        for ligand in ("A", "B"):
            for leg in ("complex", "solvent"):
                key = f"ligand_{ligand}_{leg}"
                systems[key] = {
                    "status": "pending",
                    "components": ["protein", f"ligand_{ligand}", "TIP3P", "ions"]
                    if leg == "complex" else [f"ligand_{ligand}", "TIP3P", "ions"],
                    "stages": ["topology", "box", "solvation", "ions", "minimization", "NVT", "NPT"],
                }
        report["systems"] = systems
        pilot_count = min(12, int(config["lambda_windows"]))
        pilot_indices = sorted({
            round(index * (int(config["lambda_windows"]) - 1) / (pilot_count - 1))
            for index in range(pilot_count)
        }) if pilot_count > 1 else [0]
        pilot = {
            "status": "required_before_production",
            "ligand": "A", "leg": "complex", "replica": 1,
            "lambda_states": pilot_indices,
            "checks": ["grompp", "LINCS", "dhdl.xvg", "anchor stability", "BAR/MBAR parsing"],
        }
        (root / "pilot_plan.json").write_text(
            json.dumps(pilot, indent=2) + "\n", encoding="utf-8",
        )
        report["pilot"] = pilot
        report["production_locked"] = True
    required_tools = (["pmx"] if method == "RBFE" else []) + ["alchemlyb", "pymbar"]
    missing = [name for name in required_tools if not tools[name]]
    report["missing_dependencies"] = missing
    (root / "fep_preflight.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8",
    )
    (root / "run_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8",
    )
    (root / "transformation_summary.md").write_text(
        "# ChemLink FEP preparation\n\n"
        f"- Method: {method}\n"
        + ("- Calculations: absolute ligand A and absolute ligand B\n" if method == "ABFE" else "- Transformation: ligand A → ligand B\n")
        + f"- Common-heavy-atom coverage: {mapping['coverage']:.1%}\n"
        + "- Legs: complex and solvent\n"
        + f"- Replicas: {config['replicas']}\n"
        + f"- Lambda windows per replica and leg: {config['lambda_windows']}\n"
        + f"- Total thermodynamic states: {len(states)}\n"
        + f"- Status: {report['status']}\n",
        encoding="utf-8",
    )
    if not config.get("prepare_only"):
        if missing:
            raise FEPPreparationError(
                "FEP inputs were prepared, but execution requires: " + ", ".join(missing)
            )
    return report
