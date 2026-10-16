"""PyRosetta/FlexPepDock refinement step for the robustness pipeline.

For every gate candidate, runs N independent-seed FlexPepDock refinement
trials on the LightDock-docked complex and reports the mean interface dG
(InterfaceAnalyzerMover.get_interface_dG(), the correct binding metric --
NOT pose.energies().total_energy(), which mixes in the whole receptor's
internal energy). Trials with interface_dG > 0 are excluded as outliers:
physically impossible for a real bound interface, but observed in ~0.8%
of trials from local steric clashes FastRelax's single cycle didn't fully
resolve.

Only the mean/min interface_dG was originally kept and every trial's pose
discarded -- for a non-canonical candidate this meant the only structure
ever written to disk was _flexpepdock_ncaa_assemble_worker.py's raw
assembly scaffold (backbone torsions copied from the docked pose, real
side-chain rotamers left at pose_from_sequence()'s untouched defaults,
dumped with no repacking/minimization). Confirmed via a real report
export of that scaffold: severe self-clashes near the grafted residue
(two atoms 0.09-0.11A apart -- effectively coincident), since nothing
after the graft ever relaxes it. FastRelax + FlexPepDockingProtocol
already resolves this every trial as a side effect of scoring; _run_trial
now saves that relaxed pose too, and _refine_one keeps only the best
(lowest interface_dG) trial's file as best_refined_complex.pdb, deleting
the rest -- so the CSV's "best_refined_pdb" column points at an actually
clash-checked structure instead of the pre-relaxation scaffold.

Candidates whose standardization audit found modified/non-canonical
residues don't get scored on the standardized (recorted-to-ALA/GLY) pose
-- their real side-chain chemistry is extracted from the source SDF,
turned into a custom .params residue, and assembled onto the real
docked backbone before refinement (ported from this project's proven
process_noncanonical.py / build_all_params.py / assemble_noncanonical.py
scratch scripts). Two failure modes are detected and skipped rather than
silently mis-processed: a fragment implausibly large (>~55 atoms) or two
modified positions whose fragments share atoms both mean the "modification"
is actually a real backbone-to-backbone crosslink (a stapled/macrocyclic
peptide), which this extraction method cannot parameterize correctly.
"""

import csv
import json
import logging
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

from .....adapters.pyrosetta.rosetta_env import RosettaEnvironment
from .....storage.file_manager import create_folder
from .....utils.logger import setup_logger
from .....utils.progress import step_bar_iter
from ._pdb_utils import chain_letters, filter_chains

logger = setup_logger(__name__, level=logging.INFO)

_UNSUPPORTED_ATOM_THRESHOLD = 55
_MAX_LIGAND_CHAINS = 2  # see _pdb_utils.chain_letters()

_THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}


class FlexPepDockRefinement:
    """Refine every gate candidate through N independent FlexPepDock trials."""

    def __init__(
        self,
        lightdock_results_dir: str,
        gate_rows: List[dict],
        audit_rows: Dict[str, dict],
        peptide_input_path: str,
        receptor_pdb: str,
        output_path: str,
        gso_steps: int = 10,
        glowworms: int = 20,
        n_trials: int = 10,
        rosetta_python: Optional[str] = None,
        assemble_timeout_seconds: int = 900,
        trial_timeout_seconds: int = 1800,
    ):
        self.lightdock_results_dir = lightdock_results_dir
        self.gate_rows = gate_rows
        self.audit_rows = audit_rows
        self.peptide_input_path = peptide_input_path
        self.receptor_pdb = os.path.abspath(receptor_pdb)
        self.output_path = output_path
        self.gso_steps = gso_steps
        self.glowworms = glowworms
        self.n_trials = n_trials
        self.assemble_timeout_seconds = assemble_timeout_seconds
        self.trial_timeout_seconds = trial_timeout_seconds
        self.rosetta_env = RosettaEnvironment(rosetta_python)
        self.jobs_dir = os.path.join(self.output_path, "rosetta_jobs")
        self.csv_path = os.path.join(self.output_path, "flexpepdock_results.csv")

    # ── pose resolution / phantom-chain cleanup ─────────────────────────────

    def _pose_pdb(self, cid: str, swarm: str, glowworm: str) -> Optional[str]:
        job_dir = os.path.join(self.lightdock_results_dir, cid)
        swarm_dir = os.path.join(job_dir, f"swarm_{swarm}")
        pose_pdb = os.path.join(swarm_dir, f"lightdock_{glowworm}.pdb")
        if not os.path.isfile(pose_pdb):
            subprocess.run(
                ["lgd_generate_conformations.py", "../receptor.pdb", "../ligand.pdb",
                 f"gso_{self.gso_steps}.out", str(self.glowworms), "--setup", "../setup.json"],
                cwd=swarm_dir, capture_output=True, text=True,
            )
        return pose_pdb if os.path.isfile(pose_pdb) else None

    # ── non-canonical fragment extraction (in-process, RDKit only) ─────────

    @staticmethod
    def _real_side_chain(mol, chain, pos0idx: int) -> set:
        """BFS from CA, unrestricted except for this residue's own two peptide
        bonds -- see this session's finding that side_chain_atoms() truncates
        a quaternary-CA/large side chain by stopping at its own mis-detected
        "phantom residue" boundary. Ported from process_noncanonical.py.
        """
        residue = chain[pos0idx]
        prev_c = chain[pos0idx - 1]["c"] if pos0idx > 0 else None
        next_n = chain[pos0idx + 1]["n"] if pos0idx + 1 < len(chain) else None
        blocked = set()
        if prev_c is not None:
            blocked.add((residue["n"], prev_c)); blocked.add((prev_c, residue["n"]))
        if next_n is not None:
            blocked.add((residue["c"], next_n)); blocked.add((next_n, residue["c"]))

        cb_candidates = [nb.GetIdx() for nb in mol.GetAtomWithIdx(residue["ca"]).GetNeighbors()
                          if nb.GetIdx() not in (residue["n"], residue["c"])]
        visited = set(cb_candidates)
        frontier = list(cb_candidates)
        while frontier:
            nxt = []
            for node in frontier:
                for nb in mol.GetAtomWithIdx(node).GetNeighbors():
                    j = nb.GetIdx()
                    if (node, j) in blocked:
                        continue
                    if j != residue["ca"] and j not in visited:
                        visited.add(j)
                        nxt.append(j)
            frontier = nxt
        return visited

    @staticmethod
    def _extract_fragment(mol, chain, pos0idx: int, side_atoms: set, out_sdf: str) -> None:
        """Write a standalone fragment SDF (backbone N/CA/C/O + side chain,
        capped with dummy '*' atoms at the two peptide-bond attachment
        points) for rdkit_to_params to turn into a custom residue."""
        from rdkit import Chem

        residue = chain[pos0idx]
        keep = {residue["n"], residue["ca"], residue["c"], residue["o"]} | side_atoms

        emol = Chem.RWMol()
        old_to_new = {}
        conf = mol.GetConformer()
        positions = {}
        for old_idx in sorted(keep):
            atom = mol.GetAtomWithIdx(old_idx)
            new_idx = emol.AddAtom(Chem.Atom(atom.GetSymbol()))
            old_to_new[old_idx] = new_idx
            positions[new_idx] = conf.GetAtomPosition(old_idx)

        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if i in keep and j in keep:
                emol.AddBond(old_to_new[i], old_to_new[j], bond.GetBondType())

        # The dummy atoms mark connection points for rdkit_to_params, which
        # derives each residue's LOWER_CONNECT/UPPER_CONNECT ICOOR_INTERNAL
        # (the actual bond angle/dihedral/distance FastRelax uses to place
        # this residue against its real neighbors) from the dummy atom's 3D
        # position -- NOT just its presence. Placing it at a fictitious
        # "-1.3 along X" offset (the previous version) gives a bond-length-
        # looking distance but a chemically meaningless angle/dihedral, so
        # pose_from_sequence() places this residue badly distorted against
        # its neighbor from the moment it's built (confirmed: a C(i)-CA(i+1)
        # distance of ~0.56A -- should be ~2.4A -- present immediately after
        # construction, before any torsion copying, and unmoved by FastRelax
        # since nothing here is "unrelaxed geometry", it's what the residue
        # type itself considers correct). Using the REAL neighboring atom's
        # position (already available via `chain`) instead gives
        # rdkit_to_params the true local bond direction, so the derived
        # ICOOR is chemically sound.
        n_new, c_new = old_to_new[residue["n"]], old_to_new[residue["c"]]
        dummy_prev = emol.AddAtom(Chem.Atom(0))
        emol.AddBond(n_new, dummy_prev, Chem.BondType.SINGLE)
        if pos0idx > 0:
            p = conf.GetAtomPosition(chain[pos0idx - 1]["c"])
            positions[dummy_prev] = (p.x, p.y, p.z)
        else:
            prev_pos = conf.GetAtomPosition(residue["n"])
            positions[dummy_prev] = (prev_pos.x - 1.3, prev_pos.y, prev_pos.z)  # N-terminus: no real neighbor

        dummy_next = emol.AddAtom(Chem.Atom(0))
        emol.AddBond(c_new, dummy_next, Chem.BondType.SINGLE)
        if pos0idx + 1 < len(chain):
            p = conf.GetAtomPosition(chain[pos0idx + 1]["n"])
            positions[dummy_next] = (p.x, p.y, p.z)
        else:
            next_pos = conf.GetAtomPosition(residue["c"])
            positions[dummy_next] = (next_pos.x + 1.3, next_pos.y, next_pos.z)  # C-terminus: no real neighbor

        new_conf = Chem.Conformer(emol.GetNumAtoms())
        for idx, pos in positions.items():
            new_conf.SetAtomPosition(idx, pos)

        frag = emol.GetMol()
        frag.AddConformer(new_conf, assignId=True)
        Chem.SanitizeMol(frag)  # may raise Chem.KekulizeException -- caller catches it
        frag = Chem.AddHs(frag, addCoords=True)
        Chem.MolToMolFile(frag, out_sdf)

    def _prepare_ncaa_inputs(self, cid: str, positions: List[int], job_dir: str) -> Optional[str]:
        """Extract every modified position's real chemistry and write
        seq.json + frag_<cid>_<pos>.sdf into job_dir.

        Returns None on success, or a human-readable "unsupported:..."
        reason string when the case can't be handled by this extraction
        method (stapled/macrocyclic crosslink, or a cut that breaks an
        aromatic ring).
        """
        from rdkit import Chem
        from .....utils.peptide_processor import NonCanonicalResidueError, PeptideProcessor

        sdf_path = os.path.join(self.peptide_input_path, f"{cid}.sdf")
        if not os.path.isfile(sdf_path):
            return f"unsupported:source_sdf_missing ({sdf_path})"

        processor = PeptideProcessor(standardize=True)
        mol = processor.read_molecule(sdf_path)
        mol = processor.embed_3d(mol)
        Chem.GetSymmSSSR(mol)
        ring_info = mol.GetRingInfo()
        residues = processor.find_residues(mol)
        backbone_atoms = set()
        for r in residues:
            backbone_atoms.update((r["n"], r["ca"], r["c"], r["o"]))
        chain = processor.order_chain(mol, residues, backbone_atoms)[0]

        side_atoms_by_pos = {}
        for pos1 in positions:
            if pos1 < 1 or pos1 > len(chain):
                return f"unsupported:position_{pos1}_out_of_range"
            side_atoms_by_pos[pos1] = self._real_side_chain(mol, chain, pos1 - 1)

        for atoms in side_atoms_by_pos.values():
            if len(atoms) > _UNSUPPORTED_ATOM_THRESHOLD:
                return (f"unsupported:stapled_macrocyclic (fragment has "
                        f"{max(len(a) for a in side_atoms_by_pos.values())} atoms, "
                        f">{_UNSUPPORTED_ATOM_THRESHOLD})")
        pos_list = list(side_atoms_by_pos)
        for i in range(len(pos_list)):
            for j in range(i + 1, len(pos_list)):
                if side_atoms_by_pos[pos_list[i]] & side_atoms_by_pos[pos_list[j]]:
                    return (f"unsupported:stapled_macrocyclic (positions {pos_list[i]} "
                            f"and {pos_list[j]} share atoms -- real crosslink)")

        seq_parts = []
        residue_names = {}
        used_names = set()
        for i, residue in enumerate(chain):
            pos1 = i + 1
            if pos1 in side_atoms_by_pos:
                name = (f"L{pos1}{cid[-2:]}").upper()[:3]
                alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                k = 0
                while name in used_names:
                    name = (name[:2] + alphabet[k % len(alphabet)]).upper()
                    k += 1
                used_names.add(name)
                seq_parts.append(f"X[{name}]")
                residue_names[pos1] = name
                out_sdf = os.path.join(job_dir, f"frag_{cid}_{pos1}.sdf")
                try:
                    self._extract_fragment(mol, chain, i, side_atoms_by_pos[pos1], out_sdf)
                except Exception as exc:  # noqa: BLE001 - RDKit KekulizeException et al.
                    return f"unsupported:aromatic_ring_cut ({type(exc).__name__}: {exc})"
            else:
                try:
                    resname, _ = processor.classify_residue(mol, residue, ring_info, backbone_atoms)
                    seq_parts.append(_THREE_TO_ONE.get(resname, "A"))
                except NonCanonicalResidueError:
                    seq_parts.append("A")  # not flagged by the audit but non-canonical anyway

        with open(os.path.join(job_dir, "seq.json"), "w") as fh:
            json.dump({"cid": cid, "seq": "".join(seq_parts),
                       "residue_names": residue_names, "n_residues": len(chain)}, fh)
        return None

    # ── per-candidate pipeline ──────────────────────────────────────────────

    @staticmethod
    def _blas_pinned_env() -> Dict[str, str]:
        """Env for a PyRosetta subprocess, BLAS/OpenMP threads capped to 1.

        Measured on this project's own hardware: a single FlexPepDock trial
        imports numpy at startup and ends up with ~24 live OS threads (one
        per visible core) despite using only ~100% CPU total -- Rosetta's
        Monte Carlo sampling (FastRelax, FlexPepDockingProtocol) is
        inherently sequential, so those threads never do real parallel work.
        Confirmed via /usr/bin/time -v side by side: with vs. without this
        pin, CPU% and wall-clock were identical (~100%, ~2:50) -- so unlike
        HADDOCK3/CNS (genuinely process-parallel, see haddock3_refinement.py
        and its `ncores`), there's no per-job core budget to divide here.
        This pin costs nothing measured and removes N_workers x ~24 idle
        threads that serve no purpose.
        """
        env = dict(os.environ)
        for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[_var] = "1"
        return env

    def _run_assemble(self, job_dir: str, complex_pdb: str) -> dict:
        # Invoked by script path, not `-m chemlink...` -- the worker itself only
        # imports pyrosetta/rdkit (see its module docstring), but `-m` first
        # imports the whole chemlink.pipelines.docking.peptide_docking.steps package
        # (numpy, rich, every other step's adapter, ...) as a side effect of
        # importing its parent packages, and the minimal `pyrosetta` conda env
        # doesn't have those installed -- see this project's rich
        # ModuleNotFoundError when this was still `-m`-invoked.
        worker = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "_flexpepdock_ncaa_assemble_worker.py")
        cmd = [
            self.rosetta_env.python_executable, worker,
            "--job-dir", job_dir, "--complex-pdb", complex_pdb,
            "--seq-json", os.path.join(job_dir, "seq.json"),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=self.assemble_timeout_seconds, env=self._blas_pinned_env())
        for line in reversed(proc.stdout.splitlines()):
            line = line.strip()
            if line.startswith("{"):
                return json.loads(line)
        return {"success": False, "error": f"no result line (exit {proc.returncode}): "
                                            f"{proc.stderr[-500:] if proc.stderr else '(no stderr)'}"}

    def _run_trial(self, complex_pdb: str, seed: int, params_files: List[str], out_pdb: str) -> dict:
        # See _run_assemble() -- same script-path invocation, same reason.
        worker = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "_flexpepdock_trial_worker.py")
        cmd = [self.rosetta_env.python_executable, worker,
               "--complex-pdb", complex_pdb, "--seed", str(seed), "--out-pdb", out_pdb]
        if params_files:
            cmd += ["--params", ",".join(params_files)]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                   timeout=self.trial_timeout_seconds, env=self._blas_pinned_env())
        except subprocess.TimeoutExpired:
            return {"seed": seed, "error": f"exceeded {self.trial_timeout_seconds}s timeout"}
        for line in reversed(proc.stdout.splitlines()):
            line = line.strip()
            if line.startswith("{"):
                return json.loads(line)
        return {"seed": seed, "error": f"no result line (exit {proc.returncode}): "
                                        f"{proc.stderr[-500:] if proc.stderr else '(no stderr)'}"}

    def _refine_one(self, row: dict) -> dict:
        cid = row["peptide"]
        job_dir = os.path.join(self.jobs_dir, cid)
        create_folder(job_dir)

        pose_pdb = self._pose_pdb(cid, row["best_swarm"], row["best_glowworm"])
        if pose_pdb is None:
            return {"peptide": cid, "chemistry": "", "n_trials_ok": 0,
                    "mean_interface_dG": "", "min_interface_dG": "",
                    "unsupported_reason": "", "error": "best pose PDB not found/generated"}

        n_ligand_chains = len(chain_letters(pose_pdb) - {"R"})
        if n_ligand_chains > _MAX_LIGAND_CHAINS:
            return {"peptide": cid, "chemistry": "", "n_trials_ok": 0,
                    "mean_interface_dG": "", "min_interface_dG": "",
                    "unsupported_reason": f"unsupported:multi_chain_peptide ({n_ligand_chains} PDB "
                                          f"chains -- the phantom-chain filter is only validated for "
                                          f"the single-extra-chain case, see _pdb_utils.chain_letters)",
                    "error": ""}

        complex_input = os.path.join(job_dir, "complex_input.pdb")
        filter_chains(pose_pdb, complex_input, keep_chains={"R", "A"})

        audit = self.audit_rows.get(cid)
        if not audit or audit.get("success") not in ("True", True, "1"):
            return {"peptide": cid, "chemistry": "", "n_trials_ok": 0,
                    "mean_interface_dG": "", "min_interface_dG": "",
                    "unsupported_reason": "", "error": "no successful standardization audit data for this candidate"}
        n_standardized = int(audit.get("n_standardized") or 0)

        params_files: List[str] = []
        complex_for_trials = complex_input
        chemistry = "standard"

        if n_standardized > 0:
            chemistry = "non_canonical"
            try:
                entries = json.loads(audit.get("standardized_residues") or "[]")
                # Chain-A positions only. PeptideProcessor.order_chain() numbers
                # EVERY chain from 1, and a lariat/isopeptide branch (its own
                # chain B, C, ...) always gets one when its attaching residue's
                # side chain is large enough to trip a second N-CA-C(=O) match
                # -- see this project's semaglutide case, where chain B's lone
                # pseudo-residue landed at resnum 1 and was mistaken for chain
                # A's real, canonical His at resnum 1, since the old 2-tuple
                # format didn't record which chain a flagged position was in.
                # A branch entry needs no separate handling here either way:
                # its atoms are already captured whole by its parent chain-A
                # residue's own unrestricted side-chain BFS (_real_side_chain
                # below doesn't stop at other residues' backbones).
                positions = [resnum for chain_letter, resnum, _resname in entries if chain_letter == "A"]
            except (json.JSONDecodeError, ValueError):
                positions = []
            if not positions:
                return {"peptide": cid, "chemistry": chemistry, "n_trials_ok": 0,
                        "mean_interface_dG": "", "min_interface_dG": "",
                        "unsupported_reason": "unsupported:no_standardized_positions_recorded", "error": ""}

            unsupported = self._prepare_ncaa_inputs(cid, positions, job_dir)
            if unsupported:
                return {"peptide": cid, "chemistry": chemistry, "n_trials_ok": 0,
                        "mean_interface_dG": "", "min_interface_dG": "",
                        "unsupported_reason": unsupported, "error": ""}

            try:
                assemble_result = self._run_assemble(job_dir, complex_input)
            except subprocess.TimeoutExpired:
                return {"peptide": cid, "chemistry": chemistry, "n_trials_ok": 0,
                        "mean_interface_dG": "", "min_interface_dG": "", "unsupported_reason": "",
                        "error": f"assembly exceeded {self.assemble_timeout_seconds}s timeout"}
            if not assemble_result.get("success"):
                return {"peptide": cid, "chemistry": chemistry, "n_trials_ok": 0,
                        "mean_interface_dG": "", "min_interface_dG": "", "unsupported_reason": "",
                        "error": f"assembly failed: {assemble_result.get('error')}"}
            complex_for_trials = assemble_result["out_pdb"]
            params_files = assemble_result["params_files"]

        dgs = []
        pdbs = []
        errors = []
        for seed in range(1, self.n_trials + 1):
            trial_pdb = os.path.join(job_dir, f"trial_{seed}.pdb")
            trial = self._run_trial(complex_for_trials, seed, params_files, trial_pdb)
            if "error" in trial:
                errors.append(trial["error"])
                continue
            dgs.append(trial["interface_dG"])
            pdbs.append(trial_pdb)

        # Outlier filter: interface_dG > 0 is physically impossible for a real
        # bound interface -- a residual local clash FastRelax's single cycle
        # didn't fully resolve, not a genuine result. See module docstring.
        clean = [(dg, pdb) for dg, pdb in zip(dgs, pdbs) if dg <= 0]
        # Discard every trial PDB up front -- either it's an outlier we never
        # want, or it's a clean trial whose file gets kept/renamed below. Keeps
        # a failed/all-outlier candidate from leaving N stray trial_*.pdb
        # files behind (the whole point of this loop is to end up with at
        # most one saved structure per candidate, see module docstring).
        best_pdb = ""
        if clean:
            best_dg, best_src = min(clean, key=lambda pair: pair[0])
            best_pdb = os.path.join(job_dir, "best_refined_complex.pdb")
            os.replace(best_src, best_pdb)
        for pdb in pdbs:
            if os.path.exists(pdb):
                os.remove(pdb)

        clean_dgs = [dg for dg, _ in clean]
        if not clean_dgs:
            return {"peptide": cid, "chemistry": chemistry, "n_trials_ok": 0,
                    "mean_interface_dG": "", "min_interface_dG": "", "best_refined_pdb": "",
                    "unsupported_reason": "",
                    "error": f"no usable trial (0/{self.n_trials} ok); sample errors: {errors[:2]}"}

        return {
            "peptide": cid, "chemistry": chemistry, "n_trials_ok": len(clean_dgs),
            "mean_interface_dG": sum(clean_dgs) / len(clean_dgs),
            "min_interface_dG": min(clean_dgs), "best_refined_pdb": best_pdb,
            "unsupported_reason": "", "error": "",
        }

    def _existing_rows(self) -> Dict[str, dict]:
        if not os.path.isfile(self.csv_path):
            return {}
        with open(self.csv_path, newline="") as fh:
            return {row["peptide"]: row for row in csv.DictReader(fh)}

    def run(self, n_workers: Optional[int] = None) -> Dict[str, int]:
        """Refine every gate candidate not already completed (see --resume)."""
        create_folder(self.output_path)
        create_folder(self.jobs_dir)

        fieldnames = ["peptide", "chemistry", "n_trials_ok", "mean_interface_dG",
                      "min_interface_dG", "best_refined_pdb", "unsupported_reason", "error"]

        if not self.rosetta_env.is_available():
            reason = self.rosetta_env.unavailable_reason()
            logger.warning(f"PyRosetta not available ({reason}) -- skipping Rosetta refinement entirely.")
            with open(self.csv_path, "w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                writer.writeheader()
                for row in self.gate_rows:
                    writer.writerow({"peptide": row["peptide"], "chemistry": "", "n_trials_ok": 0,
                                      "mean_interface_dG": "", "min_interface_dG": "",
                                      "unsupported_reason": "no_license_skipped", "error": reason})
            return {"refined": len(self.gate_rows), "successful": 0,
                    "csv": self.csv_path, "skipped_no_pyrosetta": True}

        existing = {cid: row for cid, row in self._existing_rows().items()
                    if row.get("error") == "" and row.get("unsupported_reason") == "" and row.get("n_trials_ok", "0") != "0"}
        pending = [row for row in self.gate_rows if row["peptide"] not in existing]

        rows: Dict[str, dict] = dict(existing)
        if pending:
            if n_workers is None:
                n_workers = 4
            n_workers = max(1, n_workers)
            logger.info(f"FlexPepDock refinement: {len(pending)} candidate(s), {n_workers} worker(s), "
                        f"{self.n_trials} trial(s) each")
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                results = list(step_bar_iter(
                    pool.map(self._refine_one, pending), "FlexPepDock Refinement",
                    total=len(pending), unit="peptide", colour="green",
                ))
            for result in results:
                rows[result["peptide"]] = result

        with open(self.csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.gate_rows:
                cid = row["peptide"]
                writer.writerow(rows.get(cid, {"peptide": cid, "chemistry": "", "n_trials_ok": 0,
                                                "mean_interface_dG": "", "min_interface_dG": "",
                                                "unsupported_reason": "", "error": "not run"}))

        n_ok = sum(1 for row in self.gate_rows if rows.get(row["peptide"], {}).get("mean_interface_dG") not in (None, ""))
        return {"refined": len(self.gate_rows), "successful": n_ok, "csv": self.csv_path}
