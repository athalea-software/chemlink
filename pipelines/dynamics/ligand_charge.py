"""Ligand net-charge detection and provenance for dynamics workflows."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import re
import subprocess
from typing import Optional


class LigandChargeError(RuntimeError):
    """Raised when a ligand charge cannot be resolved safely."""


@dataclass
class LigandChargeResult:
    charge: Optional[int]
    method: str
    confidence: str
    source_file: str
    prepared_file: str
    raw_total: Optional[float] = None
    ph: Optional[float] = None
    user_confirmed: bool = False
    estimated: bool = False
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _result(
    path: Path,
    charge: Optional[int],
    method: str,
    confidence: str,
    *,
    raw_total: Optional[float] = None,
) -> LigandChargeResult:
    return LigandChargeResult(
        charge=charge,
        method=method,
        confidence=confidence,
        source_file=str(path.resolve()),
        prepared_file=str(path.resolve()),
        raw_total=raw_total,
    )


def _parse_sdf_or_mol(path: Path) -> LigandChargeResult:
    lines = path.read_text(errors="replace").splitlines()

    v3000 = [
        int(match.group(1))
        for line in lines
        for match in re.finditer(r"\bCHG=([-+]?\d+)\b", line)
    ]
    if v3000:
        total = sum(v3000)
        return _result(path, total, "SDF V3000 formal charges", "high", raw_total=float(total))

    m_chg_total = 0
    found_m_chg = False
    for line in lines:
        if not line.startswith("M  CHG"):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            count = int(parts[2])
            values = parts[3 : 3 + count * 2]
            if len(values) != count * 2:
                raise ValueError
            m_chg_total += sum(int(values[i + 1]) for i in range(0, len(values), 2))
            found_m_chg = True
        except (ValueError, IndexError):
            return _result(path, None, "malformed SDF M  CHG records", "unknown")
    if found_m_chg:
        return _result(
            path, m_chg_total, "SDF V2000 M  CHG records", "high",
            raw_total=float(m_chg_total),
        )

    atom_count = 0
    if len(lines) >= 4:
        try:
            atom_count = int(lines[3][0:3])
        except ValueError:
            pass
    charge_code_map = {1: 3, 2: 2, 3: 1, 5: -1, 6: -2, 7: -3}
    total = 0
    found_code = False
    for line in lines[4 : 4 + atom_count]:
        parts = line.split()
        if len(parts) < 6:
            continue
        try:
            code = int(parts[5])
        except ValueError:
            continue
        if code in charge_code_map:
            total += charge_code_map[code]
            found_code = True
    if found_code:
        return _result(
            path, total, "SDF V2000 atom charge codes", "medium",
            raw_total=float(total),
        )

    # Absence of explicit charge records is not proof that a ligand is neutral.
    return _result(path, None, "no explicit formal charge in SDF/MOL", "unknown")


def _rounded_partial_charge(
    path: Path, charges: list[float], method: str, tolerance: float
) -> LigandChargeResult:
    if not charges:
        return _result(path, None, f"no readable {method} charges", "unknown")
    total = sum(charges)
    rounded = round(total)
    if not math.isclose(total, rounded, abs_tol=tolerance):
        return _result(
            path, None, f"{method} sum is not close to an integer", "unknown",
            raw_total=total,
        )
    return _result(path, rounded, method, "medium", raw_total=total)


def _parse_mol2(path: Path, tolerance: float) -> LigandChargeResult:
    charges: list[float] = []
    in_atoms = False
    for raw in path.read_text(errors="replace").splitlines():
        line = raw.strip()
        if line.startswith("@<TRIPOS>"):
            in_atoms = line.upper() == "@<TRIPOS>ATOM"
            continue
        if not in_atoms or not line:
            continue
        parts = line.split()
        if len(parts) >= 9:
            try:
                charges.append(float(parts[8]))
            except ValueError:
                pass
    return _rounded_partial_charge(path, charges, "MOL2 partial-charge sum", tolerance)


def _parse_pdbqt(path: Path, tolerance: float) -> LigandChargeResult:
    charges: list[float] = []
    for line in path.read_text(errors="replace").splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        parts = line.split()
        if len(parts) >= 2:
            try:
                charges.append(float(parts[-2]))
            except ValueError:
                pass
    return _rounded_partial_charge(path, charges, "PDBQT partial-charge sum", tolerance)


def _parse_pdb_formal_charge(path: Path) -> LigandChargeResult:
    """Read the optional PDB formal-charge field (columns 79-80)."""
    total = 0
    found = False
    for line in path.read_text(errors="replace").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")) or len(line) < 80:
            continue
        value = line[78:80].strip()
        match = re.fullmatch(r"(\d)([+-])", value)
        if not match:
            continue
        magnitude = int(match.group(1))
        total += magnitude if match.group(2) == "+" else -magnitude
        found = True
    if found:
        return _result(
            path, total, "PDB formal-charge columns", "high", raw_total=float(total)
        )
    return _result(path, None, "no PDB formal-charge columns", "unknown")


def _cross_validate_neutral_sdf(path: Path) -> Optional[LigandChargeResult]:
    """Confirm an implicit neutral SDF with independent RDKit/Open Babel reads.

    A valid V2000/V3000 file normally omits charge records when every formal
    charge is zero.  That convention is ambiguous after conversion from PDB,
    so only accept neutrality when both chemistry toolkits agree and RDKit can
    fully sanitize a single, non-metal-containing molecular graph.
    """
    try:
        from rdkit import Chem

        rdkit_mol = Chem.MolFromMolFile(str(path), removeHs=False, sanitize=True)
        if rdkit_mol is None or rdkit_mol.GetNumAtoms() == 0:
            return None
        if len(Chem.GetMolFrags(rdkit_mol)) != 1:
            return None
        # Keep the automatic confirmation within ordinary organic chemistry.
        # Metal coordination and uncommon elements require an explicit charge.
        supported = {1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53}
        if any(atom.GetAtomicNum() not in supported for atom in rdkit_mol.GetAtoms()):
            return None
        rdkit_charge = int(Chem.GetFormalCharge(rdkit_mol))
    except (ImportError, OSError, RuntimeError, ValueError):
        return None

    try:
        try:
            from openbabel import openbabel as ob
        except ImportError:
            import openbabel as ob  # type: ignore

        conversion = ob.OBConversion()
        if not conversion.SetInFormat("sdf"):
            return None
        ob_mol = ob.OBMol()
        if not conversion.ReadFile(ob_mol, str(path)) or ob_mol.NumAtoms() == 0:
            return None
        openbabel_charge = int(ob_mol.GetTotalCharge())
    except (ImportError, OSError, RuntimeError, ValueError):
        return None

    if rdkit_charge != 0 or openbabel_charge != 0:
        return None
    return _result(
        path,
        0,
        "implicit neutral SDF cross-validated by RDKit and Open Babel",
        "medium",
        raw_total=0.0,
    )


def _write_sdf_charge_metadata(path: Path, result: LigandChargeResult) -> None:
    """Persist ChemLink charge provenance as standard SDF data fields."""
    if path.suffix.lower() != ".sdf" or result.charge is None:
        return
    text = path.read_text(errors="replace")
    marker = ">  <CHEMLINK_NET_CHARGE>"
    if marker in text:
        return
    metadata = (
        f">  <CHEMLINK_NET_CHARGE>\n{result.charge}\n\n"
        f">  <CHEMLINK_CHARGE_METHOD>\n{result.method}\n\n"
        f">  <CHEMLINK_CHARGE_CONFIDENCE>\n{result.confidence}\n\n"
    )
    terminator = "$$$$"
    if terminator in text:
        text = text.replace(terminator, metadata + terminator, 1)
    else:
        text = text.rstrip() + "\n" + metadata + terminator + "\n"
    path.write_text(text, encoding="utf-8")


def _openbabel_input_format(path: Path) -> str:
    """Return the actual Open Babel format, including mislabeled PDBQT files."""
    if path.suffix.lower() == ".pdbqt":
        return "pdbqt"
    if path.suffix.lower() != ".pdb":
        return path.suffix.lower().lstrip(".")

    # Docking programs frequently produce PDBQT content with a .pdb filename.
    # Open Babel selects its reader from the extension unless told otherwise,
    # so inspect AutoDock-specific records before invoking it.
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith(("ROOT", "ENDROOT", "BRANCH", "ENDBRANCH", "TORSDOF")):
            return "pdbqt"
    return "pdb"


def detect_ligand_charge(path: str | Path, *, tolerance: float = 0.15) -> LigandChargeResult:
    ligand = Path(path)
    if not ligand.is_file():
        raise LigandChargeError(f"Ligand file not found: {ligand}")

    suffix = ligand.suffix.lower()
    if suffix in {".sdf", ".mol"}:
        return _parse_sdf_or_mol(ligand)
    if suffix == ".mol2":
        return _parse_mol2(ligand, tolerance)
    if suffix == ".pdbqt":
        return _parse_pdbqt(ligand, tolerance)
    if suffix == ".pdb":
        return _parse_pdb_formal_charge(ligand)
    return _result(ligand, None, f"charge cannot be inferred from {suffix or 'this format'}", "unknown")


def prepare_protonated_sdf(
    ligand_path: str | Path,
    work_dir: str | Path,
    *,
    ph: float = 7.4,
) -> Path:
    ligand = Path(ligand_path).resolve()
    input_format = _openbabel_input_format(ligand)
    output = Path(work_dir).resolve() / f"ligand_pH{str(ph).replace('.', '')}.sdf"
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        completed = subprocess.run(
            [
                "obabel", f"-i{input_format}", str(ligand),
                "-osdf", "-O", str(output), "-p", str(ph),
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
    except FileNotFoundError as exc:
        raise LigandChargeError(
            "OpenBabel is required to infer charge from PDB/PDBQT input."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise LigandChargeError("OpenBabel timed out while protonating the ligand.") from exc

    if completed.returncode != 0 or not output.is_file() or output.stat().st_size == 0:
        detail = completed.stderr.strip() or "no output file was produced"
        raise LigandChargeError(f"OpenBabel ligand protonation failed: {detail}")
    return output


def detect_or_prepare_ligand_charge(
    ligand_path: str | Path,
    work_dir: str | Path,
    *,
    ph: float = 7.4,
    tolerance: float = 0.15,
) -> LigandChargeResult:
    ligand = Path(ligand_path).resolve()
    suffix = ligand.suffix.lower()
    if suffix in {".sdf", ".mol", ".mol2"}:
        return detect_ligand_charge(ligand, tolerance=tolerance)

    # PDB and PDBQT lack reliable formal-charge/connectivity information for
    # topology generation.  Protonate once and use that same SDF for ACPYPE.
    if suffix in {".pdb", ".pdbqt"}:
        input_format = _openbabel_input_format(ligand)
        prepared = prepare_protonated_sdf(ligand, work_dir, ph=ph)
        result = detect_ligand_charge(prepared, tolerance=tolerance)
        if result.charge is None and input_format == "pdbqt":
            # The generated SDF may omit formal-charge records. Preserve the
            # charge information present in the AutoDock input in that case.
            source_result = _parse_pdbqt(ligand, tolerance)
            if source_result.charge is not None:
                result = source_result
        elif result.charge is None and input_format == "pdb":
            source_result = _parse_pdb_formal_charge(ligand)
            if source_result.charge is not None:
                result = source_result

        if result.charge is None:
            cross_validated = _cross_validate_neutral_sdf(prepared)
            if cross_validated is not None:
                result = cross_validated
        result.source_file = str(ligand)
        result.prepared_file = str(prepared)
        result.ph = ph
        result.method = (
            f"OpenBabel {input_format.upper()} to SDF preparation at pH {ph:g}; "
            f"{result.method}"
        )
        if result.confidence == "high":
            result.confidence = "medium"

        _write_sdf_charge_metadata(prepared, result)
        return result

    return detect_ligand_charge(ligand, tolerance=tolerance)


def explicit_charge_result(ligand_path: str | Path, charge: int) -> LigandChargeResult:
    ligand = Path(ligand_path).resolve()
    return LigandChargeResult(
        charge=int(charge),
        method="explicit CLI override",
        confidence="explicit",
        source_file=str(ligand),
        prepared_file=str(ligand),
        raw_total=float(charge),
        user_confirmed=True,
    )


def estimate_unresolved_charge(
    result: LigandChargeResult, *, fallback_charge: int = 0
) -> LigandChargeResult:
    """Turn an unresolved detection into a documented, low-confidence estimate.

    Neutrality is deliberately only the final fallback.  It is useful for
    unattended workflows, but remains visibly distinct from a charge obtained
    from formal-charge records or partial charges.
    """
    if result.charge is not None:
        return result
    previous_method = result.method
    warning = (
        "Ligand net charge could not be determined reliably; "
        f"using estimated charge {int(fallback_charge)}. "
        "Use --charge INTEGER to override."
    )
    result.charge = int(fallback_charge)
    result.raw_total = float(fallback_charge)
    result.method = f"neutral fallback estimate; {previous_method}"
    result.confidence = "low"
    result.estimated = True
    result.warnings.append(warning)
    return result


def write_charge_provenance(result: LigandChargeResult, work_dir: str | Path) -> Path:
    output = Path(work_dir) / "ligand_charge.json"
    output.write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")
    return output
