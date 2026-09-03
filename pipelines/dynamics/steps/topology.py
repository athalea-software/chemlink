import os
from pathlib import Path
import shutil
import json
from tqdm import tqdm
from pdbfixer import PDBFixer
from openmm.app import PDBFile

from ..logger import get_step_logger
from ..input_preflight import prepare_protein_pdb
from utils.retry import SubprocessError, run_subprocess

_PDB2GMX_TIMEOUT = 300
_BACKBONE_ATOMS = {"N", "CA", "C", "O"}


def classify_pdb2gmx_error(stderr: str) -> str:
    text = stderr.lower()
    if "atom c1 not found" in text and "met" in text and "tdb and rtp" in text:
        return "terminal_patch_conflict"
    if "residue" in text and ("not found in residue topology" in text or "not found in rtp" in text):
        return "unknown_residue"
    if "atom" in text and "not found in the input file" in text:
        return "missing_protein_atom"
    if "atom" in text:
        return "atom_topology_error"
    return "pdb2gmx_error"


def force_field_search_dirs(gmx_bin: str, custom_dir: str | None = None) -> list[Path]:
    """Return ordered GROMACS force-field roots, including ChemFusion layouts."""
    candidates = [
        custom_dir,
        os.environ.get("CHEMLINK_FORCEFIELD_DIR"),
        os.environ.get("GMXLIB"),
        str(Path(__file__).resolve().parents[3] / "forcefields"),
        "/usr/local/gromacs/share/gromacs/top",
        "/opt/gromacs/share/gromacs/top",
        "/usr/share/gromacs/top",
    ]
    resolved_gmx = shutil.which(gmx_bin) or gmx_bin
    if resolved_gmx:
        gmx_path = Path(resolved_gmx).resolve()
        candidates.insert(3, str(gmx_path.parent.parent / "share/gromacs/top"))
    result: list[Path] = []
    for candidate in candidates:
        if candidate:
            path = Path(candidate).expanduser().resolve()
            if path.is_dir() and path not in result:
                result.append(path)
    return result


def resolve_force_field(requested: str, search_dirs: list[Path]) -> tuple[str, Path]:
    """Resolve a field name; ``chemfusion`` means the newest CHARMM36 variant."""
    aliases = {"chemfusion", "charmm36m"}
    if requested.lower() not in aliases:
        for root in search_dirs:
            if (root / f"{requested}.ff").is_dir():
                return requested, root
        raise FileNotFoundError(f"GROMACS force field '{requested}.ff' was not found")

    exact_names = ("charmm36m", "charmm36-jul2022")
    for name in exact_names:
        for root in search_dirs:
            if (root / f"{name}.ff").is_dir():
                return name, root
    matches: list[tuple[str, Path]] = []
    for root in search_dirs:
        for directory in root.glob("charmm36*.ff"):
            if directory.is_dir():
                matches.append((directory.name[:-3], root))
    if matches:
        return sorted(matches, key=lambda item: item[0], reverse=True)[0]
    searched = ", ".join(str(path) for path in search_dirs) or "no valid directories"
    raise FileNotFoundError(
        "ChemFusion's CHARMM36m force field was not found. "
        "Use --force-field-dir DIR or CHEMLINK_FORCEFIELD_DIR. Searched: " + searched
    )


class TopologyStep:
    def __init__(self, config, gmx_bin):
        self.config        = config
        self.gmx_bin       = gmx_bin
        self.pdb_input_abs = os.path.abspath(self.config["pdb_input"])
        self.output_gro    = "processed.gro"
        self.logger        = get_step_logger(
            __name__, os.path.join(self.config["work_dir"], "simulation.log")
        )

    @staticmethod
    def _residue_count(topology) -> int:
        return sum(1 for _ in topology.residues())

    @staticmethod
    def _missing_atom_records(fixer) -> list[dict]:
        records = []
        for residue, atoms in fixer.missingAtoms.items():
            names = sorted(atom.name for atom in atoms)
            records.append({
                "chain": residue.chain.id or "_",
                "residue": residue.name,
                "number": residue.id,
                "atoms": names,
                "backbone_atoms": sorted(set(names) & _BACKBONE_ATOMS),
            })
        return records

    def _preflight_repair_pdb(self, input_pdb: str, output_pdb_path: str) -> str:
        """Repair missing side-chain heavy atoms without modelling missing residues."""
        report_path = Path(self.config["work_dir"]) / "protein_repair.json"
        report = {
            "status": "not_needed", "source": str(Path(input_pdb).resolve()),
            "output": None, "missing_residues": [], "missing_heavy_atoms": [],
            "added_atoms": [], "residue_count_before": 0, "residue_count_after": 0,
        }
        try:
            fixer = PDBFixer(filename=input_pdb)
            before = self._residue_count(fixer.topology)
            report["residue_count_before"] = before
            fixer.findMissingResidues()
            report["missing_residues"] = [
                {"chain_index": key[0], "position": key[1], "residues": value}
                for key, value in fixer.missingResidues.items()
            ]
            # Never model absent residues/loops automatically.
            fixer.missingResidues = {}
            fixer.findMissingAtoms()
            missing = self._missing_atom_records(fixer)
            report["missing_heavy_atoms"] = missing
            backbone = [item for item in missing if item["backbone_atoms"]]
            if backbone:
                report["status"] = "blocked_missing_backbone"
                report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
                details = ", ".join(
                    f"{item['chain']}:{item['residue']}{item['number']}:{'/'.join(item['backbone_atoms'])}"
                    for item in backbone
                )
                raise RuntimeError(
                    "Missing protein-backbone atoms require manual review: " + details
                )
            if not missing:
                report["residue_count_after"] = before
                report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
                return input_pdb

            # Terminal atoms are generated by pdb2gmx's terminal patches.
            fixer.missingTerminals = {}
            atom_labels = [
                f"{item['chain']}:{item['residue']}{item['number']}:{name}"
                for item in missing for name in item["atoms"]
            ]
            print(
                f"  Missing {len(atom_labels)} side-chain heavy atom(s); "
                "repairing before pdb2gmx: " + ", ".join(atom_labels)
            )
            fixer.addMissingAtoms()
            after = self._residue_count(fixer.topology)
            if after != before:
                raise RuntimeError(
                    f"PDBFixer changed the residue count ({before} -> {after}); repair rejected."
                )
            with open(output_pdb_path, "w") as fh:
                PDBFile.writeFile(fixer.topology, fixer.positions, fh, keepIds=True)

            verification = PDBFixer(filename=output_pdb_path)
            verification.findMissingResidues()
            verification.missingResidues = {}
            verification.findMissingAtoms()
            remaining = self._missing_atom_records(verification)
            if remaining:
                raise RuntimeError("PDBFixer left missing heavy atoms after repair.")
            report.update({
                "status": "repaired", "output": str(Path(output_pdb_path).resolve()),
                "added_atoms": atom_labels, "residue_count_after": after,
            })
            report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            self.logger.info(
                "Preventive protein repair completed: added %d side-chain heavy atoms; "
                "residue count unchanged (%d)", len(atom_labels), before,
            )
            return output_pdb_path
        except RuntimeError:
            if report["status"] != "blocked_missing_backbone":
                report["status"] = "repair_failed"
                report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            raise
        except Exception as exc:
            report["status"] = "repair_failed"
            report["error"] = str(exc)
            report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            raise RuntimeError(f"Preventive protein repair failed: {exc}") from exc

    def _build_pdb2gmx_cmd(self, pdb_path: str) -> list:
        sim_type = self.config.get("sim_type")
        requested = self.config.get("force_field", "default")
        water = self.config.get("water_model", "tip3p")
        base = [
            self.gmx_bin, "pdb2gmx",
            "-f", pdb_path,
            "-o", self.output_gro,
            "-water", water,
            "-ignh",
        ]
        if requested != "default":
            resolved, root = resolve_force_field(
                requested,
                force_field_search_dirs(self.gmx_bin, self.config.get("force_field_dir")),
            )
            self.config["resolved_force_field"] = resolved
            self.config["resolved_force_field_dir"] = str(root)
            self.config["water_model"] = water
            os.environ["GMXLIB"] = str(root)
            config_path = Path(self.config["work_dir"]) / "run_config.json"
            if config_path.is_file():
                config_path.write_text(json.dumps(self.config, indent=2) + "\n", encoding="utf-8")
            return base + ["-ff", resolved, "-chainsep", "id"]
        if sim_type in ["3", "4", "5", "6"]:
            return base + ["-ff", "amber99sb-ildn", "-chainsep", "id"]
        return base + ["-ff", "amber03"]

    def _chain_count(self, pdb_path: str) -> int:
        chains = set()
        with open(pdb_path) as handle:
            for line in handle:
                if line.startswith(("ATOM  ", "HETATM")) and len(line) >= 22:
                    chains.add(line[21] or "_")
        return max(1, len(chains))

    def _chain_terminal_residues(self, pdb_path: str) -> list[tuple[str, str, str]]:
        chains = {}
        with open(pdb_path) as handle:
            for line in handle:
                if not line.startswith(("ATOM  ", "HETATM")) or len(line) < 26:
                    continue
                chain = line[21] or "_"
                residue = line[17:20].strip().upper()
                chains.setdefault(chain, [residue, residue])[1] = residue
        return [(chain, ends[0], ends[1]) for chain, ends in chains.items()]

    def _charged_termini_input(self, pdb_path: str) -> str:
        """Select charged protein patches from pdb2gmx's filtered menu.

        CHARMM36 normally presents the residue-appropriate charged protein
        patch first. For N-terminal MET it incorrectly promotes the unrelated
        carbohydrate patch MET1 to index 0, moving NH3+ to index 1.
        """
        answers = []
        for _chain, first, _last in self._chain_terminal_residues(pdb_path):
            answers.extend([
                "1" if first == "MET" else "0",
                "0",
            ])
        return "\n".join(answers) + "\n"

    @staticmethod
    def _n_terminal_patch(residue: str) -> str:
        if residue == "GLY":
            return "GLY-NH3+"
        if residue == "PRO":
            return "PRO-NH2+"
        return "NH3+"

    def _write_topology_preflight(self, pdb_path: str, *, error_class=None) -> None:
        chains = {}
        with open(pdb_path) as handle:
            for line in handle:
                if not line.startswith(("ATOM  ", "HETATM")) or len(line) < 26:
                    continue
                chain = line[21] or "_"
                residue = f"{line[17:20].strip()} {line[22:26].strip()}"
                chains.setdefault(chain, {"first_residue": residue, "last_residue": residue})
                chains[chain]["last_residue"] = residue
        payload = {
            "chains": chains,
            "termini_policy": "charged" if self._uses_charmm36() else "automatic",
            "selected_termini": {
                chain: {"n": self._n_terminal_patch(ends["first_residue"].split()[0]), "c": "COO-"}
                for chain, ends in chains.items()
            } if self._uses_charmm36() else {},
            "error_class": error_class,
        }
        path = Path(self.config["work_dir"]) / "topology_preflight.json"
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def _uses_charmm36(self) -> bool:
        selected = str(self.config.get("resolved_force_field") or self.config.get("force_field", ""))
        return selected.lower() in {"chemfusion", "charmm36m"} or selected.lower().startswith("charmm36")

    def run(self) -> None:
        work_dir         = self.config["work_dir"]
        repaired_pdb_abs = os.path.join(os.path.abspath(work_dir), "complex_repaired.pdb")
        prepared_pdb_abs = os.path.join(os.path.abspath(work_dir), "protein_prepared.pdb")
        preparation = prepare_protein_pdb(self.pdb_input_abs, prepared_pdb_abs)
        Path(work_dir, "protein_preparation.json").write_text(
            json.dumps(preparation, indent=2) + "\n", encoding="utf-8"
        )
        self.logger.info(
            "Protein input prepared: removed %d hydrogens, normalized %d protein records",
            preparation["removed_hydrogens"], preparation["converted_protein_records"],
        )
        topology_input = self._preflight_repair_pdb(prepared_pdb_abs, repaired_pdb_abs)
        cmd              = self._build_pdb2gmx_cmd(topology_input)
        termini_input = None
        if self._uses_charmm36():
            cmd.append("-ter")
            termini_input = self._charged_termini_input(topology_input)
        self._write_topology_preflight(topology_input)
        environment      = os.environ.copy()
        if self.config.get("resolved_force_field_dir"):
            environment["GMXLIB"] = self.config["resolved_force_field_dir"]

        try:
            run_subprocess(
                cmd,
                timeout=_PDB2GMX_TIMEOUT,
                retries=1,
                cwd=work_dir,
                env=environment,
                input_data=termini_input,
                logger=self.logger,
            )
        except SubprocessError as exc:
            error_class = classify_pdb2gmx_error(exc.stderr)
            self._write_topology_preflight(topology_input, error_class=error_class)
            if error_class == "terminal_patch_conflict":
                raise RuntimeError(
                    "CHARMM36 selected an incompatible MET1 carbohydrate terminal patch. "
                    "Explicit NH3+/COO- selection was attempted; inspect topology_preflight.json "
                    "and pdb2gmx output."
                ) from exc
            # PDBFixer is appropriate only for genuinely missing protein atoms.
            if error_class == "missing_protein_atom":
                raise RuntimeError(
                    "pdb2gmx still reports a missing protein atom after preventive repair; "
                    "inspect protein_repair.json and the input structure."
                ) from exc
            self.logger.error("pdb2gmx failed:\n%s", exc.stderr)
            raise RuntimeError(str(exc)) from exc
