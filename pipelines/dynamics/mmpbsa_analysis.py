"""Robust MM-GBSA/MM-PBSA analysis for biomolecular MD trajectories."""

from __future__ import annotations

import json
import logging
import math
import os
import csv
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import Optional

from utils.progress import make_analysis_progress

logger = logging.getLogger(__name__)


def find_mmpbsa_tool(name: str, env_var: str) -> Optional[str]:
    override = os.environ.get(env_var)
    candidates = [
        override,
        str(Path.home() / ".conda/envs/gmx_mmpbsa/bin" / name),
        str(Path.home() / "miniconda3/envs/gmx_mmpbsa/bin" / name),
        f"/opt/conda/envs/gmx_mmpbsa/bin/{name}",
        f"/opt/miniconda/envs/gmx_mmpbsa/bin/{name}",
        shutil.which(name),
    ]
    return next((path for path in candidates if path and Path(path).is_file() and os.access(path, os.X_OK)), None)


def parse_gmx_frame_count(output: str) -> Optional[int]:
    """Extract a trajectory frame count from different GROMACS versions."""
    counts: list[int] = []
    indices: list[int] = []
    for line in output.replace("\r", "\n").splitlines():
        match = re.match(r"\s*Coords\s+(\d+)\b", line)
        if match:
            counts.append(int(match.group(1)))
        match = re.search(r"\bLast frame\s+(\d+)\b", line, re.IGNORECASE)
        if match:
            indices.append(int(match.group(1)))
        match = re.search(r"\bReading frame\s+(\d+)\s+time\b", line, re.IGNORECASE)
        if match:
            indices.append(int(match.group(1)))
    if counts:
        return max(counts)
    if indices:
        return max(indices) + 1
    return None


def parse_mmpbsa_frame_progress(output: str) -> Optional[tuple[int, int]]:
    """Extract ``current/total`` from a gmx_MMPBSA tqdm progress update."""
    matches = re.findall(r"\b(\d+)\s*/\s*(\d+)\b", output)
    if not matches:
        return None
    current, total = (int(value) for value in matches[-1])
    if total < 1 or current < 0 or current > total:
        return None
    return current, total


def parse_gmx_trajectory_timing(output: str) -> dict:
    """Extract trajectory start time and time step (picoseconds)."""
    samples: list[tuple[int, float]] = []
    for match in re.finditer(
        r"Reading frame\s+(\d+)\s+time\s+([-+]?\d+(?:\.\d+)?)",
        output.replace("\r", "\n"),
        re.IGNORECASE,
    ):
        samples.append((int(match.group(1)), float(match.group(2))))
    start_ps = next((time_ps for frame, time_ps in samples if frame == 0), 0.0)
    step_ps: Optional[float] = None
    if len(samples) >= 2:
        first_frame, first_time = samples[0]
        for frame, time_ps in samples[1:]:
            if frame != first_frame:
                step_ps = (time_ps - first_time) / (frame - first_frame)
                break
    if step_ps is None:
        match = re.search(r"^\s*Time\s+\d+\s+([-+]?\d+(?:\.\d+)?)\s*$", output, re.MULTILINE)
        if match:
            step_ps = float(match.group(1))
    return {"start_ps": start_ps, "step_ps": step_ps}


def calculate_sampling(total_frames: int, target_frames: int = 1000, discard_percent: float = 20.0) -> dict:
    if total_frames < 1:
        raise ValueError("total_frames must be positive")
    if target_frames < 1:
        raise ValueError("target_frames must be positive")
    if not 0 <= discard_percent < 100:
        raise ValueError("discard_percent must be between 0 and 100")
    discarded = min(total_frames - 1, int(total_frames * discard_percent / 100.0))
    start = discarded + 1
    end = total_frames
    available = end - start + 1
    interval = max(1, math.ceil(available / target_frames))
    analyzed = math.ceil(available / interval)
    return {
        "total_frames": total_frames,
        "discard_percent": discard_percent,
        "discarded_frames": discarded,
        "start_frame": start,
        "end_frame": end,
        "interval": interval,
        "analyzed_frames": analyzed,
    }


def parse_index(path: Path) -> list[tuple[str, list[int]]]:
    groups: list[tuple[str, list[int]]] = []
    name: Optional[str] = None
    atoms: list[int] = []
    for raw in path.read_text(errors="replace").splitlines():
        line = raw.strip()
        if line.startswith("[") and line.endswith("]"):
            if name is not None:
                groups.append((name, atoms))
            name, atoms = line[1:-1].strip(), []
        elif name is not None and line:
            try:
                atoms.extend(int(value) for value in line.split())
            except ValueError as exc:
                raise ValueError(f"Malformed atom list in index group {name!r}") from exc
    if name is not None:
        groups.append((name, atoms))
    return groups


def _write_index_group(handle, name: str, atoms: list[int]) -> None:
    handle.write(f"[ {name} ]\n")
    for offset in range(0, len(atoms), 15):
        handle.write(" ".join(str(atom) for atom in atoms[offset:offset + 15]) + "\n")
    handle.write("\n")


class GMX_MMPBSA_Analyzer:
    def __init__(
        self,
        results_dir: str,
        gmx_bin: Optional[str] = None,
        *,
        method: str = "both",
        target_frames: int = 1000,
        discard_percent: float = 20.0,
        n_cores: int = 1,
        salt_concentration: float = 0.150,
        timeout: Optional[int] = None,
        binding_system: str = "protein-ligand",
    ):
        self.results_dir = Path(results_dir).resolve()
        self.output_dir = self.results_dir / "binding_energy_analysis" / "gmx_MMPBSA"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.gmx_bin = (shutil.which(gmx_bin) or gmx_bin) if gmx_bin else (shutil.which("gmx_mpi") or shutil.which("gmx"))
        if binding_system not in {"protein-ligand", "protein-protein"}:
            raise ValueError(f"Unsupported binding system: {binding_system}")
        self.binding_system = binding_system
        # ChemFusion uses GB only for protein-protein systems.  Besides being
        # substantially cheaper for large partners, this avoids its known PB
        # preparation failure mode for two macromolecular receptor groups.
        self.method = "gb" if binding_system == "protein-protein" else method
        self.target_frames = int(target_frames)
        self.discard_percent = float(discard_percent)
        self.n_cores = max(1, int(n_cores))
        self.salt = float(salt_concentration)
        self.timeout = timeout
        self.inputs = {
            "structure": self.results_dir / "md.tpr",
            "trajectory": self.results_dir / "md_center.xtc",
            "topology": self.results_dir / "topol.top",
            "coordinates": self.results_dir / "md.pdb",
            "source_index": self.results_dir / "index.ndx",
            "index": self.output_dir / "index_mmpbsa.ndx",
        }
        self.metadata: dict = {
            "status": "pending",
            "method": self.method,
            "binding_system": binding_system,
            "mpi_cores": self.n_cores,
            "salt_concentration_molar": self.salt,
            "inputs": {key: str(value) for key, value in self.inputs.items()},
        }

    def _write_metadata(self) -> None:
        (self.output_dir / "analysis_metadata.json").write_text(
            json.dumps(self.metadata, indent=2) + "\n", encoding="utf-8"
        )

    def validate_inputs(self) -> tuple[bool, list[str]]:
        required = ("structure", "trajectory", "topology", "source_index")
        missing = [key for key in required if not self.inputs[key].is_file() or self.inputs[key].stat().st_size == 0]
        return not missing, missing

    def create_binding_index(self) -> tuple[int, int]:
        groups = parse_index(self.inputs["source_index"])
        by_name = {name.lower(): (name, atoms) for name, atoms in groups}
        protein = by_name.get("protein")
        partner_names = (
            ("other", "partner", "protein_b")
            if self.binding_system == "protein-protein"
            else ("lig", "mol", "ligand")
        )
        partner = next(
            (by_name[name] for name in partner_names if name in by_name),
            None,
        )
        if self.binding_system == "protein-protein" and partner is None:
            protein, partner = self._protein_partners_from_coordinates()
        if protein is None:
            protein = next(((name, atoms) for name, atoms in groups if name.lower().startswith("protein") and "_" not in name), None)
        if protein is None or partner is None:
            available = ", ".join(name for name, _ in groups)
            expected = "partner" if self.binding_system == "protein-protein" else "ligand"
            raise ValueError(f"Could not identify exact Protein and {expected} groups. Available: {available}")
        protein_atoms = sorted(set(protein[1]))
        partner_atoms = sorted(set(partner[1]))
        if not protein_atoms or not partner_atoms:
            raise ValueError("Protein or binding-partner index group is empty")
        overlap = set(protein_atoms).intersection(partner_atoms)
        if overlap:
            raise ValueError(f"Protein and ligand groups overlap by {len(overlap)} atoms")
        with self.inputs["index"].open("w", encoding="utf-8") as handle:
            _write_index_group(handle, "Protein", protein_atoms)
            _write_index_group(
                handle,
                "Other" if self.binding_system == "protein-protein" else "LIG",
                partner_atoms,
            )
        self.metadata["groups"] = {
            "protein": {"source_name": protein[0], "index": 0, "atoms": len(protein_atoms)},
            "partner": {"source_name": partner[0], "index": 1, "atoms": len(partner_atoms)},
        }
        return 0, 1

    def _protein_partners_from_coordinates(self) -> tuple[tuple[str, list[int]], tuple[str, list[int]]]:
        """Build ChemFusion-style receptor/partner groups from the two largest chains."""
        path = self.inputs["coordinates"]
        if not path.is_file():
            raise ValueError(
                "Protein-protein groups are absent from index.ndx and md.pdb is unavailable"
            )
        chains: dict[str, list[int]] = {}
        for line in path.read_text(errors="replace").splitlines():
            if not line.startswith("ATOM") or len(line) < 22:
                continue
            chain = line[21].strip()
            if not chain:
                continue
            try:
                atom = int(line[6:11].strip())
            except ValueError:
                continue
            chains.setdefault(chain, []).append(atom)
        ordered = sorted(chains.items(), key=lambda item: len(item[1]), reverse=True)
        if len(ordered) < 2:
            available = ", ".join(chains) or "none"
            raise ValueError(f"Could not identify two protein chains in md.pdb. Available: {available}")
        first, second = ordered[:2]
        self.metadata["chain_detection"] = {
            "strategy": "two-largest-pdb-chains",
            "receptor_chain": first[0],
            "partner_chain": second[0],
        }
        return (f"chain_{first[0]}", first[1]), (f"chain_{second[0]}", second[1])

    def detect_available_frames(self) -> int:
        if not self.gmx_bin:
            raise RuntimeError("GROMACS executable was not found")
        result = subprocess.run(
            [self.gmx_bin, "check", "-f", str(self.inputs["trajectory"])],
            capture_output=True, text=True, timeout=120,
        )
        output = (result.stdout or "") + "\n" + (result.stderr or "")
        count = parse_gmx_frame_count(output)
        if count is None:
            raise RuntimeError("GROMACS did not report the trajectory frame count")
        self.metadata["trajectory_timing"] = parse_gmx_trajectory_timing(output)
        return count

    def find_ligand_files(self) -> tuple[Path, Optional[Path]]:
        roots = sorted((self.results_dir / "acpype_work").glob("*.acpype"))
        patterns = ("*_gaff2.mol2", "*_gaff.mol2", "*_AC.mol2", "LIG.mol2")
        for root in roots:
            for pattern in patterns:
                matches = sorted(root.glob(pattern))
                if matches:
                    frcmods = sorted(root.glob("*.frcmod"))
                    return matches[0], frcmods[0] if frcmods else None
        raise FileNotFoundError("No ACPYPE MOL2 file was found for the ligand")

    def create_input_file(self, sampling: dict) -> Path:
        use_gb = self.method in {"gb", "both"}
        use_pb = self.method in {"pb", "both"}
        lines = [
            "&general",
            f"startframe={sampling['start_frame']},",
            f"endframe={sampling['end_frame']},",
            f"interval={sampling['interval']},",
            "/", "",
        ]
        if use_gb:
            lines.extend(["&gb", "igb=5,", f"saltcon={self.salt:.3f},", "/", ""])
        if use_pb:
            lines.extend(["&pb", f"istrng={self.salt:.3f},", "/", ""])
        path = self.output_dir / "mmpbsa.in"
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def _command(
        self,
        input_file: Path,
        ligand_mol2: Optional[Path],
        protein_group: int,
        ligand_group: int,
    ) -> list[str]:
        binary = find_mmpbsa_tool("gmx_MMPBSA", "CHEMLINK_MMPBSA_BIN")
        if not binary:
            raise FileNotFoundError("gmx_MMPBSA was not found; install it or set CHEMLINK_MMPBSA_BIN")
        base = [
            binary, "-O", "-nogui", "-i", str(input_file),
            "-eo", str(self.output_dir / "energy_terms_per_frame.csv"),
            "-cs", str(self.inputs["structure"]), "-ci", str(self.inputs["index"]),
            "-cg", str(protein_group), str(ligand_group),
            "-ct", str(self.inputs["trajectory"]), "-cp", str(self.inputs["topology"]),
        ]
        if ligand_mol2 is not None:
            base.extend(["-lm", str(ligand_mol2)])
        mpirun = find_mmpbsa_tool("mpirun", "CHEMLINK_MMPBSA_MPIRUN") or shutil.which("mpirun")
        if self.n_cores > 1 and mpirun:
            self.metadata["mpi_cores"] = self.n_cores
            self.metadata["execution_mode"] = "mpi"
            return [mpirun, "-np", str(self.n_cores), binary, "MPI"] + base[1:]
        self.metadata["mpi_cores"] = 1
        self.metadata["execution_mode"] = "serial"
        if self.n_cores > 1:
            self.metadata["parallel_fallback"] = (
                f"Requested {self.n_cores} MPI processes, but mpirun was not found; "
                "running serially."
            )
        return base

    def extract_energies(self) -> dict:
        energies = {"GB": {}, "PB": {}}
        result_file = self.output_dir / "FINAL_RESULTS_MMPBSA.dat"
        if not result_file.is_file():
            return energies
        content = result_file.read_text(errors="replace")
        sections = {"GB": "GENERALIZED BORN", "PB": "POISSON BOLTZMANN"}
        for key, heading in sections.items():
            if heading not in content:
                continue
            section = content.split(heading, 1)[1]
            if key == "GB" and sections["PB"] in section:
                section = section.split(sections["PB"], 1)[0]
            for line in section.splitlines():
                normalized = line.upper().replace("Δ", "DELTA ")
                if "DELTA TOTAL" not in normalized and "DELTA G BINDING" not in normalized:
                    continue
                values = re.findall(r"[-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?", line)
                if len(values) >= 2:
                    # gmx_MMPBSA columns are Average, SD(Prop.), SD,
                    # SEM(Prop.), SEM. Report Average ± sample SD.
                    std_index = 2 if len(values) >= 5 else 1
                    energies[key] = {
                        "total_mean": float(values[0]),
                        "total_std": float(values[std_index]),
                    }
                    break
        return energies

    def _write_results(self, energies: dict) -> None:
        methods, means, deviations = [], [], []
        for key, label in (("GB", "MM-GBSA"), ("PB", "MM-PBSA")):
            if "total_mean" in energies[key]:
                methods.append(label)
                means.append(energies[key]["total_mean"])
                deviations.append(energies[key]["total_std"])
        if methods:
            mpl_cache = self.output_dir / ".matplotlib"
            mpl_cache.mkdir(exist_ok=True)
            os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axis = plt.subplots(figsize=(8, 5))
            positions = list(range(len(methods)))
            bars = axis.bar(positions, means, yerr=deviations, capsize=8)
            axis.set_xticks(positions, methods)
            axis.set_ylabel("ΔG_bind (kcal/mol)")
            axis.axhline(0, color="black", linewidth=0.8)
            for bar, mean, std in zip(bars, means, deviations):
                axis.text(bar.get_x() + bar.get_width() / 2, mean, f"{mean:.2f} ± {std:.2f}", ha="center")
            fig.tight_layout()
            fig.savefig(self.output_dir / "binding_energy.png", dpi=300)
            plt.close(fig)
        report = ["gmx_MMPBSA BINDING ENERGY REPORT", "=" * 50]
        for key, label in (("GB", "MM-GBSA"), ("PB", "MM-PBSA")):
            if "total_mean" in energies[key]:
                report.append(f"{label}: ΔG_bind = {energies[key]['total_mean']:.2f} ± {energies[key]['total_std']:.2f} kcal/mol")
        if len(report) == 2:
            report.append("No binding energies could be extracted; inspect FINAL_RESULTS_MMPBSA.dat.")
        report.extend(["", "Negative values indicate more favorable estimated binding."])
        (self.output_dir / "SUMMARY_REPORT.txt").write_text("\n".join(report) + "\n", encoding="utf-8")

    def _write_timeseries_outputs(self) -> int:
        """Create a tidy per-frame binding-energy CSV and time-series plot."""
        source = self.output_dir / "energy_terms_per_frame.csv"
        destination = self.output_dir / "binding_energy_timeseries.csv"
        if not source.is_file():
            return 0

        method: Optional[str] = None
        block: Optional[str] = None
        header: Optional[list[str]] = None
        records: list[dict] = []
        with source.open(newline="", encoding="utf-8-sig", errors="replace") as handle:
            for row in csv.reader(handle):
                row = [cell.strip() for cell in row]
                if not any(row):
                    header = None
                    continue
                label = row[0].upper()
                if len(row) == 1 and "GENERALIZED BORN" in label:
                    method, block, header = "MM-GBSA", None, None
                    continue
                if len(row) == 1 and "POISSON BOLTZMANN" in label:
                    method, block, header = "MM-PBSA", None, None
                    continue
                if len(row) == 1 and "ENERGY TERMS" in label:
                    block, header = label, None
                    continue
                if row[0].lower() == "frame #":
                    header = row
                    continue
                if method and block and block.startswith("DELTA ") and header and len(row) >= 2:
                    try:
                        frame = int(float(row[0]))
                        values = {name: float(value) for name, value in zip(header[1:], row[1:]) if value}
                    except ValueError:
                        continue
                    timing = self.metadata.get("trajectory_timing", {})
                    start_ps = float(timing.get("start_ps") or 0.0)
                    step_ps = timing.get("step_ps")
                    time_ps = start_ps + (frame - 1) * float(step_ps) if step_ps is not None else None
                    records.append({
                        "method": method,
                        "frame": frame,
                        "time_ps": time_ps,
                        "time_ns": time_ps / 1000.0 if time_ps is not None else None,
                        **values,
                    })
        if not records:
            return 0

        preferred = ["method", "frame", "time_ps", "time_ns"]
        energy_fields = sorted({key for record in records for key in record if key not in preferred})
        fields = preferred + energy_fields
        with destination.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)

        total_key = next((name for name in energy_fields if name.upper() == "TOTAL"), None)
        if total_key:
            mpl_cache = self.output_dir / ".matplotlib"
            mpl_cache.mkdir(exist_ok=True)
            os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axis = plt.subplots(figsize=(10, 5.5))
            for current_method in ("MM-GBSA", "MM-PBSA"):
                selected = [record for record in records if record["method"] == current_method]
                if not selected:
                    continue
                x_key = "time_ns" if selected[0]["time_ns"] is not None else "frame"
                axis.plot(
                    [record[x_key] for record in selected],
                    [record[total_key] for record in selected],
                    linewidth=1.4,
                    label=current_method,
                )
            axis.axhline(0, color="black", linewidth=0.8, alpha=0.6)
            axis.set_xlabel("Simulation time (ns)" if records[0]["time_ns"] is not None else "Frame")
            axis.set_ylabel("ΔG binding (kcal/mol)")
            axis.set_title("Binding energy over simulation time")
            axis.grid(alpha=0.25)
            axis.legend()
            fig.tight_layout()
            fig.savefig(self.output_dir / "binding_energy_timeseries.png", dpi=300)
            plt.close(fig)
        return len(records)

    def _run_with_frame_progress(self, command: list[str], environment: dict, expected_frames: int) -> int:
        """Run gmx_MMPBSA while mirroring its per-frame progress in ChemLink."""
        stdout_path = self.output_dir / "gmx_mmpbsa.log"
        stderr_path = self.output_dir / "gmx_mmpbsa.err"
        method = "MM-PBSA" if self.method == "pb" else "MM-GBSA"
        component = "system"
        started = time.monotonic()

        frames = make_analysis_progress()
        with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr, frames:
            current_progress = 0
            task = frames.add_task(
                f"[bold magenta]◆ Binding Energy · {method} {component}[/]",
                total=expected_frames,
            )
            process = subprocess.Popen(
                command,
                cwd=self.output_dir,
                stdout=stdout,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=environment,
            )
            assert process.stderr is not None
            try:
                # tqdm uses carriage returns for updates. Text mode translates
                # them into line boundaries, allowing progress to be streamed.
                for line in process.stderr:
                    stderr.write(line)
                    stderr.flush()
                    normalized = line.strip().lower()
                    if "beginning gb calculations" in normalized:
                        method = "MM-GBSA"
                    elif "beginning pb calculations" in normalized:
                        method = "MM-PBSA"
                    contribution = re.search(r"calculating\s+(complex|receptor|ligand)\s+contribution", normalized)
                    if contribution:
                        component = contribution.group(1)
                        if component == "ligand" and self.binding_system == "protein-protein":
                            component = "partner"
                        current_progress = 0
                        frames.reset(task, total=expected_frames)
                        frames.update(
                            task,
                            description=f"[bold magenta]◆ Binding Energy · {method} {component}[/]",
                        )
                    parsed = parse_mmpbsa_frame_progress(line)
                    if parsed:
                        current, reported_total = parsed
                        if current < current_progress:
                            frames.reset(task, total=reported_total)
                        frames.update(task, total=reported_total, completed=current)
                        current_progress = current
                    if self.timeout is not None and time.monotonic() - started > self.timeout:
                        process.kill()
                        raise subprocess.TimeoutExpired(command, self.timeout)
                return process.wait()
            except BaseException:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                raise

    def run_analysis(self, **_compat) -> bool:
        try:
            valid, missing = self.validate_inputs()
            if not valid:
                raise FileNotFoundError("Missing required MM-PBSA inputs: " + ", ".join(missing))
            protein_group, ligand_group = self.create_binding_index()
            local_mol2: Optional[Path] = None
            if self.binding_system == "protein-ligand":
                mol2, frcmod = self.find_ligand_files()
                local_mol2 = self.output_dir / mol2.name
                shutil.copy2(mol2, local_mol2)
                if frcmod:
                    shutil.copy2(frcmod, self.output_dir / frcmod.name)
                self.metadata["ligand_mol2"] = str(mol2)
            total_frames = self.detect_available_frames()
            sampling = calculate_sampling(total_frames, self.target_frames, self.discard_percent)
            self.metadata["sampling"] = sampling
            input_file = self.create_input_file(sampling)
            command = self._command(input_file, local_mol2, protein_group, ligand_group)
            self.metadata["command"] = command
            self._write_metadata()
            environment = os.environ.copy()
            environment["GMX_BIN"] = str(self.gmx_bin)
            mpl_cache = self.output_dir / ".matplotlib"
            mpl_cache.mkdir(exist_ok=True)
            environment["MPLCONFIGDIR"] = str(mpl_cache)
            mmpbsa_binary = find_mmpbsa_tool("gmx_MMPBSA", "CHEMLINK_MMPBSA_BIN")
            if mmpbsa_binary:
                mmpbsa_bin_dir = Path(mmpbsa_binary).resolve().parent
                environment["AMBERHOME"] = str(mmpbsa_bin_dir.parent)
                environment["PATH"] = str(mmpbsa_bin_dir) + os.pathsep + environment.get("PATH", "")
            if self.gmx_bin:
                environment["PATH"] = str(Path(self.gmx_bin).resolve().parent) + os.pathsep + environment.get("PATH", "")
            returncode = self._run_with_frame_progress(command, environment, sampling["analyzed_frames"])
            if returncode != 0:
                raise RuntimeError(f"gmx_MMPBSA exited with status {returncode}")
            energies = self.extract_energies()
            self._write_results(energies)
            timeseries_rows = self._write_timeseries_outputs()
            has_energies = any("total_mean" in result for result in energies.values())
            self.metadata.update(
                status="completed" if has_energies else "completed_with_warnings",
                energies=energies,
                timeseries_rows=timeseries_rows,
                energy_terms_csv=str(self.output_dir / "energy_terms_per_frame.csv"),
                binding_timeseries_csv=str(self.output_dir / "binding_energy_timeseries.csv"),
                binding_timeseries_plot=str(self.output_dir / "binding_energy_timeseries.png"),
            )
            self._write_metadata()
            return True
        except Exception as exc:
            self.metadata.update(status="failed", error=str(exc))
            self._write_metadata()
            logger.exception("Binding-energy analysis failed")
            return False
