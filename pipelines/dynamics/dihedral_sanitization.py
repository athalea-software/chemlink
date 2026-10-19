"""Safe normalizations for GROMACS ligand dihedrals."""

import json
import math
import re
from pathlib import Path


def sanitize_ligand_dihedrals(itp_path, report_path):
    """Remove or merge ParmEd-incompatible torsions without changing forces.

    ACPYPE can emit zero-energy function-9 terms with periodicity zero.  They
    are removed instead of being assigned a real periodicity, which could
    create a false conflict with a physical term.  Terms with equal atom
    quartets, periodicity, and phase are exactly equivalent to one term whose
    force constant is their sum.
    """
    source = Path(itp_path)
    report_file = Path(report_path)
    lines = source.read_text(encoding="utf-8").splitlines(keepends=True)
    removed_null = []
    merged = []
    incompatible = []
    invalid = []
    entries = []
    in_dihedrals = False

    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped.startswith("[") and "]" in stripped:
            section = stripped[1:stripped.index("]")].strip().lower()
            in_dihedrals = section == "dihedrals"
            continue
        if not in_dihedrals or not stripped or stripped.startswith((";", "#")):
            continue

        data = line.split(";", 1)[0]
        tokens = list(re.finditer(r"\S+", data))
        if len(tokens) < 8:
            continue
        values = [match.group(0) for match in tokens]
        try:
            atoms = tuple(int(value) for value in values[:4])
            function = int(values[4])
            phase = float(values[5])
            force_constant = float(values[6])
            periodicity = float(values[7])
        except ValueError:
            continue
        if function != 9:
            continue

        entry = {
            "line": line_number,
            "atoms": list(atoms),
            "function": function,
            "phase": phase,
            "force_constant": force_constant,
            "periodicity": periodicity,
            "tokens": tokens,
        }
        entries.append(entry)
        if periodicity == 0:
            public_entry = {key: value for key, value in entry.items() if key != "tokens"}
            if force_constant != 0:
                invalid.append(public_entry)
                continue
            public_entry["reason"] = "zero periodicity and zero force constant"
            removed_null.append(public_entry)

    # ParmEd groups function-9 terms by atom quartet. Merge only terms whose
    # phase is also equal (modulo 360), which preserves the potential exactly.
    groups = {}
    periodicity_groups = {}
    removed_lines = {term["line"] for term in removed_null}
    for entry in entries:
        if entry["periodicity"] == 0:
            continue
        canonical_atoms = min(tuple(entry["atoms"]), tuple(reversed(entry["atoms"])))
        phase = entry["phase"] % 360.0
        key = (canonical_atoms, entry["periodicity"], round(phase, 10))
        groups.setdefault(key, []).append(entry)
        periodicity_groups.setdefault((canonical_atoms, entry["periodicity"]), []).append(entry)

    for terms in periodicity_groups.values():
        phases = {round(term["phase"] % 360.0, 10) for term in terms}
        if len(terms) > 1 and len(phases) > 1:
            incompatible.append({
                "atoms": terms[0]["atoms"],
                "periodicity": terms[0]["periodicity"],
                "source_lines": [term["line"] for term in terms],
                "phases": [term["phase"] for term in terms],
                "force_constants": [term["force_constant"] for term in terms],
            })

    for terms in groups.values():
        if len(terms) < 2:
            continue
        keeper = terms[0]
        total_force = math.fsum(term["force_constant"] for term in terms)
        token = keeper["tokens"][6]
        line_index = keeper["line"] - 1
        line = lines[line_index]
        start, end = token.span()
        lines[line_index] = line[:start] + f"{total_force:.8g}" + line[end:]
        duplicate_lines = [term["line"] for term in terms[1:]]
        removed_lines.update(duplicate_lines)
        merged.append({
            "atoms": keeper["atoms"],
            "periodicity": keeper["periodicity"],
            "phase": keeper["phase"],
            "source_lines": [term["line"] for term in terms],
            "force_constants": [term["force_constant"] for term in terms],
            "merged_force_constant": total_force,
        })

    if removed_lines:
        lines = [line for number, line in enumerate(lines, start=1) if number not in removed_lines]

    report = {
        "source": str(source.resolve()),
        "removed_null_count": len(removed_null),
        "merged_count": len(merged),
        "incompatible_count": len(incompatible),
        "invalid_count": len(invalid),
        "removed_null_terms": removed_null,
        "merged": merged,
        "incompatible": incompatible,
        "invalid": invalid,
    }
    report_file.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if invalid:
        raise ValueError(
            f"Found {len(invalid)} ligand dihedral(s) with periodicity 0 and "
            f"non-zero force constant; inspect {report_file}"
        )
    if incompatible:
        raise ValueError(
            f"Found {len(incompatible)} ParmEd-incompatible ligand dihedral "
            f"series with equal periodicity but different phases; inspect {report_file}"
        )
    if removed_null or merged:
        source.write_text("".join(lines), encoding="utf-8")
    return report
