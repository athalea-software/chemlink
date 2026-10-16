"""Input and output locations resolved from the directory a command runs in.

Every path option is optional. When one is omitted, ChemLink looks in the
current working directory with a single convention shared by every command:

    receptors/  ligands/  peptides/    input subfolders, used when present
    *.pdb  *.sdf  *.mol2 ...           otherwise, loose files in the directory
    results/<pipeline>/                where new runs are written

Loose files are linked into ``<run>/inputs/<role>/`` before a pipeline reads
them, so a directory holding receptors and ligands side by side never feeds a
receptor PDB to ligand preparation (or vice versa).
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

RESULTS_DIR = "results"

RECEPTOR_EXTS = (".pdb",)
LIGAND_EXTS = (".sdf", ".mol2", ".mol", ".pdbqt")
PEPTIDE_EXTS = (".sdf",)

# role -> (subfolders searched in order, loose-file extensions, CLI flag)
_ROLES: Dict[str, Tuple[Tuple[str, ...], Tuple[str, ...], str]] = {
    "receptors": (("receptors", os.path.join("data", "input", "receptors")), RECEPTOR_EXTS, "-ir"),
    "ligands":   (("ligands", os.path.join("data", "input", "ligands")), LIGAND_EXTS, "-il"),
    "peptides":  (("peptides",), PEPTIDE_EXTS, "-ip"),
}


def _files_with(directory: str, exts: Sequence[str]) -> List[str]:
    try:
        names = sorted(os.listdir(directory))
    except OSError:
        return []
    return [
        os.path.abspath(os.path.join(directory, name))
        for name in names
        if name.lower().endswith(tuple(exts)) and os.path.isfile(os.path.join(directory, name))
    ]


@dataclass
class InputSource:
    """Where a pipeline's inputs for one role come from."""

    role: str
    directory: str
    origin: str                                  # "explicit" | "subfolder" | "cwd"
    files: List[str] = field(default_factory=list)

    def staged_path(self, run_dir: str) -> str:
        """The directory ``materialize(run_dir)`` returns, without creating it."""
        if self.origin != "cwd":
            return self.directory
        return os.path.join(run_dir, "inputs", self.role)

    def materialize(self, run_dir: str, *, copy: bool = False) -> str:
        """Return a directory the pipeline can read. Loose files from the
        working directory are linked (or, with ``copy``, copied: SLURM scripts
        list inputs with ``find -type f``) into ``run_dir/inputs/<role>/`` first."""
        dest = self.staged_path(run_dir)
        if self.origin != "cwd":
            return dest
        os.makedirs(dest, exist_ok=True)
        for path in self.files:
            target = os.path.join(dest, os.path.basename(path))
            if os.path.lexists(target):
                continue
            if copy:
                shutil.copy2(path, target)
            else:
                os.symlink(path, target)
        return dest

    def describe(self) -> str:
        if self.origin == "cwd":
            return f"{self.directory}  ({len(self.files)} file(s) in the current directory)"
        return self.directory


def find_inputs(role: str, explicit: Optional[str] = None) -> InputSource:
    """Resolve the input directory for ``role`` (receptors, ligands, peptides)."""
    subfolders, exts, flag = _ROLES[role]
    if explicit:
        path = os.path.abspath(os.path.expanduser(explicit))
        if not os.path.isdir(path):
            raise FileNotFoundError(f"{role.capitalize()} directory not found: {path}")
        return InputSource(role, path, "explicit")
    for sub in subfolders:
        if os.path.isdir(sub):
            return InputSource(role, os.path.abspath(sub), "subfolder")
    loose = _files_with(".", exts)
    if loose:
        return InputSource(role, os.path.abspath("."), "cwd", loose)
    raise FileNotFoundError(
        f"No {role} found in {os.path.abspath('.')}. Pass {flag} DIR, or run the command "
        f"from a directory with a {subfolders[0]}/ folder or loose {'/'.join('*' + e for e in exts)} files."
    )


def find_receptor_file(explicit: Optional[str] = None) -> str:
    """A single receptor PDB: ``explicit``, else the only PDB in receptors/ or the cwd."""
    if explicit:
        path = os.path.abspath(os.path.expanduser(explicit))
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Receptor file not found: {path}")
        return path
    for directory in ("receptors", "."):
        found = _files_with(directory, RECEPTOR_EXTS)
        if len(found) == 1:
            return found[0]
        if len(found) > 1:
            names = ", ".join(os.path.basename(p) for p in found)
            raise FileNotFoundError(
                f"Several receptor PDBs in {os.path.abspath(directory)} ({names}); choose one with -ir FILE."
            )
    raise FileNotFoundError(
        f"No receptor PDB found in {os.path.abspath('.')}; pass -ir FILE or put it in receptors/."
    )


def find_restraints(explicit: Optional[str] = None, *, required: bool = False) -> Optional[str]:
    """``explicit``, else the only ``restraints*`` file in the cwd."""
    if explicit:
        path = os.path.abspath(os.path.expanduser(explicit))
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Restraints file not found: {path}")
        return path
    found = sorted(
        os.path.abspath(name) for name in os.listdir(".")
        if name.lower().startswith("restraints") and os.path.isfile(name)
    )
    if len(found) == 1:
        return found[0]
    if len(found) > 1:
        raise FileNotFoundError(
            "Several restraints* files in the current directory; choose one with --restraints FILE."
        )
    if required:
        raise FileNotFoundError(
            "A restraints file is required: pass --restraints FILE or add a restraints.txt "
            "to the current directory."
        )
    return None


def output_base(pipeline: str, explicit: Optional[str] = None) -> str:
    """Base directory new runs of ``pipeline`` are written under."""
    return os.path.abspath(os.path.expanduser(explicit or os.path.join(RESULTS_DIR, pipeline)))


def _has_any(directory: str, markers: Iterable[str]) -> bool:
    return any(os.path.exists(os.path.join(directory, marker)) for marker in markers)


def find_run_dir(explicit: Optional[str], pipeline: str, markers: Sequence[str]) -> str:
    """An existing run directory: ``explicit``, else the cwd when it is a run,
    else the most recently modified run under ``results/<pipeline>/``."""
    if explicit:
        path = os.path.abspath(os.path.expanduser(explicit))
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Run directory not found: {path}")
        return path
    if _has_any(".", markers):
        return os.path.abspath(".")
    base = output_base(pipeline)
    try:
        candidates = [
            os.path.join(base, name) for name in os.listdir(base)
            if os.path.isdir(os.path.join(base, name)) and _has_any(os.path.join(base, name), markers)
        ]
    except OSError:
        candidates = []
    if not candidates:
        raise FileNotFoundError(
            f"No run directory given and none found under {base}; pass the run directory explicitly."
        )
    return max(candidates, key=os.path.getmtime)


# ── Dynamics input detection ──────────────────────────────────────────────────

_AMINO = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS",
    "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "HID", "HIE", "HIP", "HSD",
    "HSE", "HSP", "CYX", "CYM", "ASH", "GLH", "LYN", "MSE", "SEC", "PYL",
}
_NUCLEIC = {
    "A", "C", "G", "U", "T", "DA", "DC", "DG", "DT", "DU", "RA", "RC", "RG", "RU",
    "DA5", "DA3", "DC5", "DC3", "DG5", "DG3", "DT5", "DT3", "RA5", "RA3", "RC5", "RC3",
    "RG5", "RG3", "RU5", "RU3",
}


def classify_structure(path: str) -> Tuple[str, int]:
    """("protein" | "nucleic" | "small", residue count) for a dynamics input."""
    if not path.lower().endswith(".pdb"):
        return "small", 1
    amino, nucleic = set(), set()
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if not line.startswith(("ATOM  ", "HETATM")):
                    continue
                resname = line[17:20].strip().upper()
                key = (line[21:22], line[22:27])
                if resname in _AMINO:
                    amino.add(key)
                elif resname in _NUCLEIC:
                    nucleic.add(key)
    except OSError:
        return "small", 0
    if len(nucleic) > len(amino):
        return "nucleic", len(nucleic)
    if len(amino) >= 2:
        return "protein", len(amino)
    return "small", 1


# receptors, partner role for the "ligand" slot
_DYN_PARTNER = {
    "oprotein": (1, None),
    "pligand":  (1, "small"),
    "ppeptide": (1, "protein"),
    "pacid":    (1, "nucleic"),
    "pprotein": (1, "protein"),
    "ppligand": (2, "small"),
    "oligand":  (0, "small"),
}


def detect_dynamics_inputs(
    dyn_type: str,
    receptors: Sequence[str],
    ligands: Sequence[str],
    *,
    exclude: Sequence[str] = (),
    multiple_ligands: bool = False,
) -> Tuple[List[str], List[str]]:
    """Fill in whichever of receptors/ligands was not given, from the cwd.

    Candidates are the .pdb and ligand files in the cwd, receptors/ and
    ligands/. PDBs are classified as protein, nucleic acid or small molecule
    from their residues; partner proteins (peptides) are the smaller of two.
    """
    receptors, ligands = list(receptors), list(ligands)
    n_receptors, partner = _DYN_PARTNER[dyn_type]
    need_receptors = n_receptors and not receptors
    need_ligands = partner is not None and not ligands
    if not (need_receptors or need_ligands):
        return receptors, ligands

    taken = {os.path.abspath(p) for p in (*receptors, *ligands, *exclude)}
    candidates: List[str] = []
    for directory in (".", "receptors", "ligands"):
        for path in _files_with(directory, RECEPTOR_EXTS + LIGAND_EXTS):
            if path not in taken and path not in candidates:
                candidates.append(path)
    kinds: Dict[str, List[Tuple[str, int]]] = {"protein": [], "nucleic": [], "small": []}
    for path in candidates:
        kind, size = classify_structure(path)
        kinds[kind].append((path, size))
    proteins = [p for p, _ in sorted(kinds["protein"], key=lambda item: (-item[1], item[0]))]
    nucleic = [p for p, _ in kinds["nucleic"]]
    smalls = [p for p, _ in kinds["small"]]

    def fail(requirement: str) -> None:
        found = ", ".join(
            f"{len(paths)} {label} ({', '.join(os.path.basename(p) for p in paths)})"
            for label, paths in (("protein", proteins), ("nucleic acid", nucleic), ("small molecule", smalls))
            if paths
        ) or "no structure files"
        raise ValueError(
            f"{dyn_type} needs {requirement}; found {found} in {os.path.abspath('.')}. "
            f"Choose the files with -ir/-il."
        )

    partner_pool = {"small": smalls, "nucleic": nucleic, "protein": proteins}.get(partner or "", [])
    if partner == "protein" and need_receptors and need_ligands:
        if len(proteins) != 2:
            fail("2 protein structures (the larger is used as the receptor)")
        return proteins[:1], proteins[1:]
    if need_receptors:
        if len(proteins) != n_receptors:
            fail(f"{n_receptors} protein structure(s)")
        receptors = proteins[:n_receptors]
        if partner == "protein":
            partner_pool = proteins[n_receptors:]
    if need_ligands:
        label = {"small": "ligand", "nucleic": "nucleic acid", "protein": "partner protein"}[partner]
        if multiple_ligands and partner == "small":
            if not partner_pool:
                fail("at least 1 ligand")
            ligands = list(partner_pool)
        else:
            if len(partner_pool) != 1:
                fail(f"1 {label}")
            ligands = list(partner_pool)
    return receptors, ligands
