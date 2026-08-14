"""CPU docking execution step using AutoDock Vina.

This is the CPU counterpart to `docking_execution.py` (AutoDock-GPU). It is
deliberately shaped like `DockingExecution` -- same orchestration, logging,
progress bar, and worker model -- but with two engine-specific differences:

  1. No AutoGrid. AutoDock-GPU consumes precomputed `.maps.fld` grids; Vina
     instead takes the prepared receptor PDBQT plus an explicit search box
     (center + size in angstroms). So this step reads receptor PDBQTs directly
     from `prepared_receptors_pdbqt/` -- there is no map-generation phase.

  2. Box units. The box config (boxes.json) stores the box the AutoGrid way:
     a center plus `npts` = the number of grid points per axis. Vina wants the
     box *size in angstroms*, where size = npts * grid_spacing. That conversion
     is `_npts_to_box_size`; the spacing default matches
     ActiveSiteDetection.GRID_SPACING (0.375 A) so the two engines describe the
     same physical box.
"""

import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ....adapters.autodock_cpu.autodock_vina_adapter import AutoDockVinaAdapter
from ....scripts.export_docking_complexes import write_ligand, write_complex
from ....storage.file_manager import create_folder, list_files_in_directory, find_compound_name
from ....utils.logger import setup_logger
from ....utils.progress import job_bar, CTRL_C_HINT, console as _progress_console
from .active_site_detection import ActiveSiteDetection, load_box_config

logger = setup_logger(__name__)


@dataclass(frozen=True)
class VinaJob:
	"""Single receptor/ligand docking task for Vina."""

	receptor_file: str
	ligand_file: str
	receptor_name: str
	ligand_name: str
	center: Tuple[float, float, float]
	box_size: Tuple[float, float, float]


class VinaExecution:
	"""Execute AutoDock Vina (CPU) docking jobs from prepared PDBQT inputs.

	Like DockingExecution, this class only orchestrates: it pairs prepared
	receptors with ligands, resolves each receptor's search box, runs Vina, and
	records the generated files in a structured output layout.
	"""

	DEFAULT_VINA_EXECUTABLE = os.environ.get("VINA_BIN", "/usr/local/bin/vina")

	def __init__(
		self,
		receptor_dir: str,
		ligand_dir: str,
		output_path: str,
		box_config_path: Optional[str] = None,
		vina_executable: Optional[str] = None,
		receptor_ext: str = ".pdbqt",
		ligand_ext: str = ".pdbqt",
		exhaustiveness: int = 8,
		num_modes: int = 9,
		cpu_per_job: Optional[int] = None,
		seed: Optional[int] = None,
		timeout_seconds: int = 600,
		grid_spacing: float = ActiveSiteDetection.GRID_SPACING,
		clean_receptors_dir: Optional[str] = None,
	):
		self.receptor_dir = receptor_dir
		self.ligand_dir = ligand_dir
		self.output_path = output_path
		self.box_config_path = box_config_path
		# Where ReceptorPreparation wrote the clean, protonated receptor PDBs used
		# to build the receptor+ligand complex (see _write_complex). Defaults to a
		# sibling of output_path, matching ReceptorPreparation's own layout; pass
		# this explicitly when output_path is an isolated subfolder (e.g. the
		# engine="both" run puts CPU results under output_path/cpu_validation,
		# while clean_receptors/ -- shared prep, not engine-specific -- stays at
		# the pipeline root).
		self.clean_receptors_dir = clean_receptors_dir or os.path.join(self.output_path, "clean_receptors")
		self.vina = AutoDockVinaAdapter(vina_executable)
		self.vina_executable = vina_executable or self.DEFAULT_VINA_EXECUTABLE
		self.receptor_ext = receptor_ext
		self.ligand_ext = ligand_ext
		self.exhaustiveness = exhaustiveness
		self.num_modes = num_modes
		self.cpu_per_job = cpu_per_job
		self.seed = seed
		self.timeout_seconds = timeout_seconds
		self.grid_spacing = grid_spacing

		self.results_dir = os.path.join(self.output_path, "docking_results")
		self.temp_dir = os.path.join(self.results_dir, "temp")
		self.log_file = os.path.join(self.output_path, "docking_vina.log")
		self.config_file = os.path.join(self.output_path, "docking_config.txt")
		self.affinity_file = os.path.join(self.results_dir, "affinities.csv")

	# --- Box handling ---------------------------------------------------------

	def _npts_to_box_size(self, npts: Tuple[int, int, int]) -> Tuple[float, float, float]:
		"""Convert an AutoGrid npts triple (grid points) to a Vina box size (A).

		size_axis = number_of_points * grid_spacing. The center is already in
		angstroms and passes through unchanged elsewhere; only the size needs
		this conversion.
		"""
		nx, ny, nz = npts
		return (nx * self.grid_spacing, ny * self.grid_spacing, nz * self.grid_spacing)

	def _load_boxes(self) -> Dict[str, Dict[str, Any]]:
		"""Load per-receptor boxes from a JSON config, or fall back to GPF files.

		A JSON box config (boxes.json), when provided, wins and is used as-is.
		Otherwise the boxes come from the ``.gpf`` files that active-site
		detection writes next to the prepared receptors -- each GPF carries a
		``gridcenter`` and ``npts`` line, which is exactly what we need.
		"""
		if self.box_config_path and os.path.isfile(self.box_config_path):
			return load_box_config(self.box_config_path)

		boxes: Dict[str, Dict[str, Any]] = {}
		for gpf in list_files_in_directory(self.receptor_dir, ["*.gpf"]):
			parsed = self._parse_gpf(gpf)
			if parsed:
				boxes[Path(gpf).stem] = parsed
		return boxes

	@staticmethod
	def _parse_gpf(gpf_path: str) -> Optional[Dict[str, Any]]:
		"""Extract center + npts from an AutoGrid GPF file.

		Relevant lines look like:
		    npts 60 60 60
		    gridcenter 8.605 20.908 -0.310
		"""
		center: Optional[Tuple[float, float, float]] = None
		npts: Optional[Tuple[int, int, int]] = None
		with open(gpf_path, "r", encoding="utf-8", errors="ignore") as handle:
			for line in handle:
				parts = line.split()
				if len(parts) < 4:
					continue
				try:
					if parts[0] == "gridcenter":
						center = (float(parts[1]), float(parts[2]), float(parts[3]))
					elif parts[0] == "npts":
						npts = (int(parts[1]), int(parts[2]), int(parts[3]))
				except ValueError:
					# e.g. "gridcenter auto" -- not usable for a fixed Vina box.
					continue
		if center and npts:
			return {"center": center, "npts": npts}
		return None

	@staticmethod
	def _match_box(receptor_name: str, boxes: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
		"""Find the box config entry for a receptor.

		Tries an exact filename-stem match first (boxes.json keys look like
		``LolA(7Z6W)_clean``), then a looser compound-name match.
		"""
		if receptor_name in boxes:
			return boxes[receptor_name]
		compound = find_compound_name(receptor_name)
		for key, entry in boxes.items():
			if find_compound_name(key) == compound:
				return entry
		return None

	# --- Logging / config -----------------------------------------------------

	def _log(self, level: str, message: str, echo: bool = True) -> None:
		"""Write a timestamped message to the step log and optionally stdout."""
		create_folder(self.output_path)
		timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
		line = f"[{timestamp}] [{level}] {message}"
		if echo:
			print(line)
		with open(self.log_file, "a", encoding="utf-8") as handle:
			handle.write(line + "\n")

	def _write_config_header(self, total_receptors: int, total_ligands: int, total_jobs: int) -> None:
		"""Write the top-level configuration summary."""
		create_folder(self.output_path)
		with open(self.config_file, "w", encoding="utf-8") as handle:
			handle.write("CONFIGURACIÓN DE DOCKING MOLECULAR (AutoDock Vina / CPU)\n")
			handle.write(f"Fecha de inicio: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
			handle.write(f"Receptores procesados: {total_receptors}\n")
			handle.write(f"Ligandos procesados: {total_ligands}\n")
			handle.write(f"Total de dockings: {total_jobs}\n")
			handle.write(f"Exhaustiveness: {self.exhaustiveness}  |  num_modes: {self.num_modes}\n")
			handle.write("=============================================\n\n")

	# --- Job matrix -----------------------------------------------------------

	def _build_jobs(self) -> List[VinaJob]:
		"""Build the receptor/ligand job matrix, resolving each box."""
		receptors = list_files_in_directory(self.receptor_dir, [f"*{self.receptor_ext}"])
		ligands = list_files_in_directory(self.ligand_dir, [f"*{self.ligand_ext}"])

		if not receptors:
			raise RuntimeError(f"No receptor PDBQT files found in: {self.receptor_dir}")
		if not ligands:
			raise RuntimeError(f"No ligand PDBQT files found in: {self.ligand_dir}")

		boxes = self._load_boxes()
		if not boxes:
			raise RuntimeError(
				"No box definitions found: pass a --box-config JSON, or run active-site "
				f"detection first so .gpf files exist in {self.receptor_dir}"
			)

		jobs: List[VinaJob] = []
		for receptor_file in receptors:
			receptor_name = Path(receptor_file).stem
			entry = self._match_box(receptor_name, boxes)
			if entry is None:
				# Vina cannot run without a box; skip this receptor loudly.
				self._log("ERROR", f"No box config for receptor '{receptor_name}' -- skipping", echo=False)
				continue

			center = tuple(entry["center"])
			box_size = self._npts_to_box_size(tuple(entry["npts"]))
			for ligand_file in ligands:
				ligand_name = find_compound_name(ligand_file)
				jobs.append(
					VinaJob(
						receptor_file=receptor_file,
						ligand_file=ligand_file,
						receptor_name=receptor_name,
						ligand_name=ligand_name,
						center=center,
						box_size=box_size,
					)
				)

		if not jobs:
			raise RuntimeError("No docking jobs built: no receptor matched a box config entry")
		return jobs

	# --- Output layout / execution -------------------------------------------

	def _create_output_layout(self, receptor_name: str) -> Dict[str, str]:
		"""Create and return the output layout for a receptor."""
		receptor_output = os.path.join(self.results_dir, receptor_name)
		pose_dir = os.path.join(receptor_output, "poses")
		log_dir = os.path.join(receptor_output, "logs")
		create_folder(receptor_output)
		create_folder(pose_dir)
		create_folder(log_dir)
		return {"receptor_output": receptor_output, "pose_dir": pose_dir, "log_dir": log_dir}

	def _run_single(self, job: VinaJob) -> Dict[str, Any]:
		"""Run one Vina docking job."""
		layout = self._create_output_layout(job.receptor_name)
		output_name = f"{job.receptor_name}_{job.ligand_name}"
		output_prefix = os.path.join(self.temp_dir, output_name)

		result = self.vina.dock(
			receptor_pdbqt=job.receptor_file,
			ligand_pdbqt=job.ligand_file,
			output_prefix=output_prefix,
			center=job.center,
			box_size=job.box_size,
			exhaustiveness=self.exhaustiveness,
			num_modes=self.num_modes,
			cpu=self.cpu_per_job,
			seed=self.seed,
			timeout_seconds=self.timeout_seconds,
		)

		return {
			"receptor": job.receptor_name,
			"ligand": job.ligand_name,
			"out_file": result.out_file,
			"log_file": result.log_file,
			"best_affinity": result.best_affinity,
			"num_poses": len(result.poses),
			"receptor_output": layout["receptor_output"],
		}

	@staticmethod
	def _extract_best_pose_atoms(out_pdbqt: str) -> List[Dict[str, Any]]:
		"""Extract atoms of the best (first) MODEL block from a Vina output PDBQT.

		Vina ranks poses best-first, so MODEL 1 is the top pose -- the same
		best-first convention export_docking_complexes.py relies on for
		AutoDock-GPU's DLG runs.
		"""
		atoms: List[Dict[str, Any]] = []
		in_first_model = False
		seen_model = False
		with open(out_pdbqt, "r", encoding="utf-8", errors="ignore") as handle:
			for line in handle:
				if line.startswith("MODEL"):
					if seen_model:
						break  # stop after the first model
					seen_model = True
					in_first_model = True
					continue
				if not in_first_model:
					continue
				if line.startswith("ENDMDL"):
					break
				if line.startswith(("ATOM", "HETATM")):
					try:
						atoms.append({
							"name": line[12:16].strip() or "C",
							"x": float(line[30:38]),
							"y": float(line[38:46]),
							"z": float(line[46:54]),
							"type": line.split()[-1],
						})
					except (ValueError, IndexError):
						continue
		return atoms

	def _write_complex(self, result: Dict[str, Any], out_pdbqt_path: str) -> Optional[str]:
		"""Build the receptor+ligand complex PDB for the best Vina pose (MD input).

		Mirrors export_docking_complexes.py's GPU/DLG complex export -- same
		clean-receptor lookup (`<name>_h.pdb` then `<name>.pdb`) and the same
		ligand/complex PDB writers -- so both engines produce comparable complex
		files for a double-validation run.
		"""
		receptor_name = result["receptor"]
		clean_receptor = os.path.join(self.clean_receptors_dir, f"{receptor_name}_h.pdb")
		if not os.path.isfile(clean_receptor):
			clean_receptor = os.path.join(self.clean_receptors_dir, f"{receptor_name}.pdb")
		if not os.path.isfile(clean_receptor):
			logger.warning(
				f"No clean receptor PDB found for '{receptor_name}' in "
				f"{self.clean_receptors_dir}; skipping complex export for this pair"
			)
			return None

		atoms = self._extract_best_pose_atoms(out_pdbqt_path)
		if not atoms:
			logger.warning(f"No atoms extracted from best Vina pose in {out_pdbqt_path}; skipping complex export")
			return None

		complex_dir = os.path.join(result["receptor_output"], "complex")
		create_folder(complex_dir)
		stem = f"{receptor_name}_{result['ligand']}"
		ligand_out = Path(complex_dir) / f"{stem}_ligand.pdb"
		complex_out = Path(complex_dir) / f"{stem}_complex.pdb"
		write_ligand(atoms, ligand_out)
		write_complex(Path(clean_receptor), ligand_out, complex_out)
		return str(complex_out)

	def _move_outputs(self, result: Dict[str, Any]) -> None:
		"""Move generated pose/log files, then export the receptor+ligand complex."""
		receptor_output = result["receptor_output"]
		pose_dir = os.path.join(receptor_output, "poses")
		log_dir = os.path.join(receptor_output, "logs")

		final_out_file = os.path.join(pose_dir, os.path.basename(result["out_file"]))
		shutil.move(result["out_file"], final_out_file)
		if result.get("log_file"):
			shutil.move(result["log_file"], os.path.join(log_dir, os.path.basename(result["log_file"])))

		result["complex_pdb"] = self._write_complex(result, final_out_file)

	def _append_affinity(self, result: Dict[str, Any]) -> None:
		"""Append one best-affinity row to the results CSV."""
		new_file = not os.path.isfile(self.affinity_file)
		with open(self.affinity_file, "a", encoding="utf-8") as handle:
			if new_file:
				handle.write("receptor,ligand,best_affinity_kcal_mol,num_poses,complex_pdb\n")
			affinity = result["best_affinity"]
			affinity_str = f"{affinity:.3f}" if affinity is not None else "NA"
			handle.write(
				f"{result['receptor']},{result['ligand']},{affinity_str},"
				f"{result['num_poses']},{result.get('complex_pdb') or ''}\n"
			)

	def run(self, n_workers: Optional[int] = None) -> Dict[str, int]:
		"""Run all Vina docking jobs."""
		self.vina.validate()
		create_folder(self.results_dir)
		create_folder(self.temp_dir)
		with open(self.log_file, "w", encoding="utf-8") as handle:
			handle.write("=== INICIO DE DOCKING MOLECULAR (Vina / CPU) ===\n")

		jobs = self._build_jobs()
		receptor_names = sorted({job.receptor_name for job in jobs})
		ligand_names = sorted({job.ligand_name for job in jobs})
		self._write_config_header(len(receptor_names), len(ligand_names), len(jobs))

		if n_workers is None:
			n_workers = 1
		n_workers = max(1, min(n_workers, len(jobs)))

		# Auto-split cores across parallel jobs unless the caller pinned --vina-cpu.
		# total threads = n_workers * cpu_per_job, kept <= available cores so
		# concurrent Vina processes never oversubscribe the machine.
		cpu_mode = "explícito"
		if self.cpu_per_job is None:
			self.cpu_per_job = max(1, cpu_count() // n_workers)
			cpu_mode = "auto"

		self._log(
			"INFO",
			f"Iniciando docking Vina con {len(jobs)} combinación(es) usando {n_workers} worker(s), "
			f"{self.cpu_per_job} CPU(s) por trabajo ({cpu_mode}; {cpu_count()} núcleos disponibles)",
		)

		successful = 0
		failed = 0
		results: List[Dict[str, Any]] = []

		_progress_console.print(CTRL_C_HINT, end="")
		with job_bar("  Docking (Vina)", len(jobs)) as progress:
			def _record_success(job: VinaJob, result: Dict[str, Any]) -> None:
				nonlocal successful
				self._move_outputs(result)
				self._append_affinity(result)
				results.append(result)
				successful += 1
				affinity = result["best_affinity"]
				affinity_str = f"{affinity:.2f}" if affinity is not None else "NA"
				status_text = f"OK {job.receptor_name} vs {job.ligand_name} ({affinity_str} kcal/mol)"
				progress.set_postfix_str(status_text)
				self._log("INFO", status_text, echo=False)

			def _record_failure(job: VinaJob, exc: Exception) -> None:
				nonlocal failed
				failed += 1
				import traceback
				full_error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
				self._log("ERROR", f"FAILED {job.receptor_name} vs {job.ligand_name}:\n{full_error}", echo=False)
				exc_str = str(exc)
				if len(exc_str) > 100:
					exc_str = exc_str[:97] + "..."
				status_text = f"FAILED {job.receptor_name} vs {job.ligand_name}: {exc_str}"
				progress.set_postfix_str(status_text)

			if n_workers == 1:
				for job in jobs:
					try:
						result = self._run_single(job)
						_record_success(job, result)
					except Exception as exc:
						_record_failure(job, exc)
					finally:
						progress.update(1)
			else:
				with ThreadPoolExecutor(max_workers=n_workers) as executor:
					future_map = {executor.submit(self._run_single, job): job for job in jobs}
					for future in as_completed(future_map):
						job = future_map[future]
						try:
							result = future.result()
							_record_success(job, result)
						except Exception as exc:
							_record_failure(job, exc)
						finally:
							progress.update(1)

		self._log("INFO", f"Docking Vina finalizado: {successful} exitosos, {failed} fallidos")
		return {"successful": successful, "failed": failed, "total": len(jobs)}
