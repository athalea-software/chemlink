"""Docking execution step for molecular docking."""

import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ....adapters.autogrid.autogrid_adapter import AutoGridAdapter
from ....adapters.autodock_gpu.autodock_gpu_adapter import AutoDockGPUAdapter
from ....storage.file_manager import create_folder, list_files_in_directory, find_compound_name
from ....utils.logger import setup_logger
from ....utils.progress import job_bar, console as _progress_console, Checklist
from ....utils.style import log_line

logger = setup_logger(__name__)


@dataclass(frozen=True)
class DockingJob:
	"""Single protein/ligand docking task."""

	protein_map: str
	ligand_file: str
	protein_name: str
	ligand_name: str
	gpu_id: Optional[int] = None
	flex_file: Optional[str] = None


class DockingExecution:
	"""Execute AutoGrid/AutoDock-GPU docking jobs from prepared inputs.

	The class is intentionally focused on orchestration: it finds prepared maps and
	ligands, executes the docking binary, and records the generated files in a
	structured output layout.
	"""

	DEFAULT_AUTODOCK_GPU_EXECUTABLE = os.environ.get("AUTODOCK_GPU_BIN", "/usr/local/bin/autodock-gpu")
	DEFAULT_AUTOGRID_EXECUTABLE = os.environ.get("AUTOGRID4_BIN", "/usr/local/bin/autogrid4")
	DEFAULT_AUTODOCK_GPU_FALLBACKS = (
		"/usr/local/bin/autodock-gpu",
		"/usr/local/bin/adgpu-v1.6_linux_x64_cuda12_128wi",
	)

	def __init__(
		self,
		protein_maps_dir: str,
		ligand_dir: str,
		output_path: str,
		autogrid_executable: Optional[str] = None,
		autodock_gpu_executable: Optional[str] = None,
		grid_ext: str = ".maps.fld",
		ligand_ext: str = ".pdbqt",
		run_count: int = 1000,
		lsmet: str = "ad",
		nev: int = 2_500_000,
		timeout_seconds: int = 600,
	):
		self.protein_maps_dir = protein_maps_dir
		self.ligand_dir = ligand_dir
		self.output_path = output_path
		self.autogrid = AutoGridAdapter(autogrid_executable)
		self.autodock_gpu = AutoDockGPUAdapter(autodock_gpu_executable)
		self.autogrid_executable = autogrid_executable or self.DEFAULT_AUTOGRID_EXECUTABLE
		self.autodock_gpu_executable = autodock_gpu_executable or self.DEFAULT_AUTODOCK_GPU_EXECUTABLE
		self.grid_ext = grid_ext
		self.ligand_ext = ligand_ext
		self.run_count = run_count
		self.lsmet = lsmet
		self.nev = nev
		self.timeout_seconds = timeout_seconds

		self.results_dir = os.path.join(self.output_path, "docking_results")
		self.temp_dir = os.path.join(self.results_dir, "temp")
		self.log_file = os.path.join(self.output_path, "docking_molecular.log")
		self.config_file = os.path.join(self.output_path, "docking_config.txt")
		self.summary_file = os.path.join(self.results_dir, "resumen_docking.txt")

	def _validate_dependencies(self, require_autogrid: bool = True) -> None:
		"""Validate external executables required by this step."""
		if require_autogrid:
			self.autogrid.validate()

		self.autodock_gpu.validate()

	def _log(self, level: str, message: str, echo: bool = True) -> None:
		"""Write a timestamped file entry and a concise console message."""
		create_folder(self.output_path)
		timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
		line = f"[{timestamp}] [{level}] {message}"
		if echo:
			_progress_console.print(log_line(level, message))
		with open(self.log_file, "a", encoding="utf-8") as handle:
			handle.write(line + "\n")

	def _build_docking_jobs(self) -> List[DockingJob]:
		"""Build the protein/ligand job matrix."""
		protein_maps = list_files_in_directory(self.protein_maps_dir, [f"*{self.grid_ext}"])
		ligands = list_files_in_directory(self.ligand_dir, [f"*{self.ligand_ext}"])

		if not protein_maps:
			raise RuntimeError(f"No map files found in: {self.protein_maps_dir}")
		if not ligands:
			raise RuntimeError(f"No ligand files found in: {self.ligand_dir}")

		jobs: List[DockingJob] = []
		for protein_map in protein_maps:
			protein_name = Path(protein_map).name.removesuffix(self.grid_ext)
			base_name = protein_name.removesuffix("_rigid")
			flex_candidate = os.path.join(self.protein_maps_dir, f"{base_name}_flex.pdbqt")
			flex_file = flex_candidate if os.path.isfile(flex_candidate) else None
			for ligand_file in ligands:
				ligand_name = find_compound_name(ligand_file)
				jobs.append(
					DockingJob(
						protein_map=protein_map,
						ligand_file=ligand_file,
						protein_name=protein_name,
						ligand_name=ligand_name,
						flex_file=flex_file,
					)
				)
		return jobs

	def _generate_maps_from_gpf(self, checklist: Optional[Checklist] = None) -> int:
		"""Generate `.maps.fld` files from available `.gpf` files using AutoGrid4."""
		gpf_files = list_files_in_directory(self.protein_maps_dir, ["*.gpf"])
		if not gpf_files:
			raise RuntimeError(
				f"No map files ({self.grid_ext}) and no .gpf files found in: {self.protein_maps_dir}"
			)

		item = checklist.step("Preparing docking grid", "AutoGrid4") if checklist is not None else None
		generated = 0
		try:
			for gpf_file in gpf_files:
				map_file = self.autogrid.generate_maps(gpf_file)
				generated += 1
				if item is not None:
					item.update(f"{generated}/{len(gpf_files)} maps generated")
		except BaseException as exc:
			if item is not None:
				item.finish(str(exc), ok=False)
			raise
		else:
			if item is not None:
				item.finish()

		return generated

	def _create_output_layout(self, protein_name: str) -> Dict[str, str]:
		"""Create and return the output layout for a protein."""
		protein_output = os.path.join(self.results_dir, protein_name)
		dlg_dir = os.path.join(protein_output, "dlg")
		xml_dir = os.path.join(protein_output, "xml")
		create_folder(protein_output)
		create_folder(dlg_dir)
		create_folder(xml_dir)
		return {
			"protein_output": protein_output,
			"dlg_dir": dlg_dir,
			"xml_dir": xml_dir,
			"affinity_file": os.path.join(protein_output, "affinity.dat"),
		}

	def _run_single(self, job: DockingJob) -> Dict[str, Any]:
		"""Run one docking job."""
		layout = self._create_output_layout(job.protein_name)
		output_name = f"{job.protein_name}_{job.ligand_name}"
		output_prefix = os.path.join(self.temp_dir, output_name)

		result = self.autodock_gpu.dock(
			protein_map=job.protein_map,
			ligand_file=job.ligand_file,
			output_prefix=output_prefix,
			run_count=self.run_count,
			lsmet=self.lsmet,
			nev=self.nev,
			timeout_seconds=self.timeout_seconds,
			gpu_id=job.gpu_id,
			flexres_file=job.flex_file,
		)

		return {
			"protein": job.protein_name,
			"ligand": job.ligand_name,
			"dlg_file": result.dlg_file,
			"xml_file": result.xml_file,
			"affinity": result.affinity,
			"run": result.run,
			"rmsd": result.rmsd,
			"protein_output": layout["protein_output"],
		}

	def _move_outputs(self, result: Dict[str, Any]) -> None:
		"""Move generated output files into the protein-specific folders."""
		protein_output = result["protein_output"]
		dlg_dir = os.path.join(protein_output, "dlg")
		xml_dir = os.path.join(protein_output, "xml")

		shutil.move(result["dlg_file"], os.path.join(dlg_dir, os.path.basename(result["dlg_file"])))
		if result.get("xml_file"):
			shutil.move(result["xml_file"], os.path.join(xml_dir, os.path.basename(result["xml_file"])))

	def _write_config_header(self, total_proteins: int, total_ligands: int, total_jobs: int) -> None:
		"""Write the top-level configuration summary."""
		create_folder(self.output_path)
		with open(self.config_file, "w", encoding="utf-8") as handle:
			handle.write("CONFIGURACIÓN DE DOCKING MOLECULAR\n")
			handle.write(f"Fecha de inicio: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
			handle.write(f"Proteínas procesadas: {total_proteins}\n")
			handle.write(f"Ligandos procesados: {total_ligands}\n")
			handle.write(f"Total de dockings: {total_jobs}\n")
			handle.write("=============================================\n\n")

	def _append_protein_summary(self, protein_name: str, success: int, failures: int) -> None:
		"""Append a per-protein summary to the config file."""
		with open(self.config_file, "a", encoding="utf-8") as handle:
			handle.write(f"Proteína: {protein_name}\n")
			handle.write(f"  Dockings exitosos: {success}\n")
			handle.write(f"  Dockings fallidos: {failures}\n")
			handle.write(f"  Total: {success + failures}\n\n")

	def run(
		self,
		n_workers: Optional[int] = None,
		gpu_ids: Optional[Sequence[int]] = None,
		checklist: Optional[Checklist] = None,
	) -> Dict[str, int]:
		"""Run all docking jobs.

		checklist: When provided, report the grid-generation and search phases
			as named checklist items with live counts instead of the standalone
			job_bar. Omit for today's behaviour unchanged.
		"""
		initial_map_files = list_files_in_directory(self.protein_maps_dir, [f"*{self.grid_ext}"])
		self._validate_dependencies(require_autogrid=not bool(initial_map_files))
		create_folder(self.results_dir)
		create_folder(self.temp_dir)
		with open(self.log_file, "w", encoding="utf-8") as handle:
			handle.write("=== INICIO DE DOCKING MOLECULAR ===\n")

		map_files = initial_map_files
		if not map_files:
			generated = self._generate_maps_from_gpf(checklist=checklist)
			self._log("INFO", f"AutoGrid generated {generated} map file(s)")

		jobs = self._build_docking_jobs()
		protein_names = sorted({job.protein_name for job in jobs})
		ligand_names = sorted({job.ligand_name for job in jobs})
		self._write_config_header(len(protein_names), len(ligand_names), len(jobs))

		if gpu_ids:
			gpu_ids = [int(gpu_id) for gpu_id in gpu_ids]

		if n_workers is None:
			n_workers = 1
		if gpu_ids:
			n_workers = len(gpu_ids)
		n_workers = max(1, min(n_workers, len(jobs)))

		if gpu_ids:
			jobs = [
				DockingJob(
					protein_map=job.protein_map,
					ligand_file=job.ligand_file,
					protein_name=job.protein_name,
					ligand_name=job.ligand_name,
					gpu_id=gpu_ids[index % len(gpu_ids)],
					flex_file=job.flex_file,
				)
				for index, job in enumerate(jobs)
			]

		gpu_note = f" en GPU(s) {','.join(str(g) for g in gpu_ids)}" if gpu_ids else ""
		self._log("INFO", f"Iniciando docking con {len(jobs)} combinación(es) usando {n_workers} worker(s){gpu_note}")

		successful = 0
		failed = 0
		results: List[Dict[str, Any]] = []

		if checklist is not None:
			gpu_note = f"{n_workers} GPU(s)" if gpu_ids else f"{n_workers} worker(s)"
			item = checklist.step(f"Running GPU search · {gpu_note}", "AutoDock-GPU")
			completed = 0

			def _record_success(job: DockingJob, result: Dict[str, Any]) -> None:
				nonlocal successful, completed
				self._move_outputs(result)
				results.append(result)
				successful += 1
				completed += 1
				self._log("INFO", f"OK {job.protein_name} vs {job.ligand_name}", echo=False)
				item.update(f"{completed:,} / {len(jobs):,} ligands docked")

			def _record_failure(job: DockingJob, exc: Exception) -> None:
				nonlocal failed, completed
				failed += 1
				completed += 1
				import traceback
				full_error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
				self._log("ERROR", f"FAILED {job.protein_name} vs {job.ligand_name}:\n{full_error}", echo=False)
				item.update(f"{completed:,} / {len(jobs):,} ligands docked")

			if n_workers == 1:
				for job in jobs:
					try:
						result = self._run_single(job)
						_record_success(job, result)
					except Exception as exc:
						_record_failure(job, exc)
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

			pct = 100 * successful // len(jobs) if jobs else 100
			item.finish(f"{len(jobs):,} / {len(jobs):,} ligands docked  ·  {pct}%")
		else:
			with job_bar("  Docking", len(jobs)) as progress:
				def _record_success(job: DockingJob, result: Dict[str, Any]) -> None:
					nonlocal successful
					self._move_outputs(result)
					results.append(result)
					successful += 1
					status_text = f"OK {job.protein_name} vs {job.ligand_name}"
					progress.set_postfix_str(status_text)
					self._log("INFO", status_text, echo=False)

				def _record_failure(job: DockingJob, exc: Exception) -> None:
					nonlocal failed
					failed += 1
					# Log full error message with traceback to file
					import traceback
					full_error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
					self._log("ERROR", f"FAILED {job.protein_name} vs {job.ligand_name}:\n{full_error}", echo=False)
					# Truncate for progress bar display
					exc_str = str(exc)
					if len(exc_str) > 100:
						exc_str = exc_str[:97] + "..."
					status_text = f"FAILED {job.protein_name} vs {job.ligand_name}: {exc_str}"
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

		for protein_name in protein_names:
			protein_results = [r for r in results if r["protein"] == protein_name]
			self._append_protein_summary(protein_name, len(protein_results), len(ligand_names) - len(protein_results))

		self._log("INFO", f"Docking finalizado: {successful} exitosos, {failed} fallidos")
		return {"successful": successful, "failed": failed, "total": len(jobs)}
