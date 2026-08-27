"""High-level orchestration for the docking preparation workflow."""

import os
import csv
import json
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from ...adapters.autodock_gpu.autodock_gpu_adapter import AutoDockGPUAdapter
from ...storage.file_manager import create_folder, list_files_in_directory
from ...pipelines.ligand_validation import require_valid_ligand
from .steps import (
	ActiveSiteDetection,
	DockingAnalysis,
	DockingExecution,
	LigandPreparation,
	ReceptorPreparation,
	VinaExecution,
	load_box_config,
)


@dataclass
class DockingPipelineResult:
	"""Structured result for the preparation pipeline."""

	receptor_preparation: Dict[str, int]
	ligand_preparation: Dict[str, Any]
	active_site_detection: Dict[str, int]
	docking_execution: Optional[Dict[str, Any]] = None
	docking_analysis: Optional[Dict[str, Any]] = None
	executed_steps: Tuple[str, ...] = ()


class DockingPipeline:
	"""Compose receptor preparation, ligand preparation, and active-site detection.

	This pipeline supports both preparation-only execution and full docking runs
	that include execution and analysis stages.
	"""

	STEP_ORDER = ("receptor", "ligand", "active-site", "execution", "analysis")
	STEP_ALIASES = {
		"receptor": "receptor",
		"receptor-preparation": "receptor",
		"ligand": "ligand",
		"ligand-preparation": "ligand",
		"active-site": "active-site",
		"active-site-detection": "active-site",
		"execution": "execution",
		"docking-execution": "execution",
		"analysis": "analysis",
		"docking-analysis": "analysis",
	}

	def __init__(
		self,
		receptor_input_path: str,
		ligand_input_path: str,
		output_path: str,
		fpocket_path: Optional[str] = None,
		manual_center: Optional[Tuple[float, float, float]] = None,
		manual_npts: Optional[Tuple[int, int, int]] = None,
		box_config_path: Optional[str] = None,
		flex_radius: float = 0.0,
	):
		self.receptor_input_path = receptor_input_path
		self.ligand_input_path = ligand_input_path
		self.output_path = output_path
		self.fpocket_path = fpocket_path
		self.manual_center = manual_center
		self.manual_npts = manual_npts
		self.box_config_path = box_config_path
		self.flex_radius = flex_radius
		self.manual_boxes = load_box_config(box_config_path) if box_config_path else None

	@property
	def prepared_receptor_path(self) -> str:
		"""Return the output directory used by receptor preparation.
		
		If the receptor_input_path contains .pdbqt files, it's treated as already-prepared
		receptors (useful when starting from a later step in the pipeline).
		Otherwise, return the standard output path for prepared receptors.
		When inside a run_<timestamp> subdirectory, also check the parent directory.
		"""
		import os
		# Check if receptor_input_path already contains prepared receptors
		if list_files_in_directory(self.receptor_input_path, ["*.pdbqt"]):
			return self.receptor_input_path
		
		# Try the standard location
		standard_path = f"{self.output_path}/prepared_receptors_pdbqt"
		if list_files_in_directory(standard_path, ["*.pdbqt"]):
			return standard_path
		
		# If inside a run_<timestamp> subdirectory, check parent
		parent_path = os.path.dirname(self.output_path)
		if parent_path != self.output_path:  # Avoid infinite recursion
			parent_candidate = f"{parent_path}/prepared_receptors_pdbqt"
			if os.path.isdir(parent_candidate) and list_files_in_directory(parent_candidate, ["*.pdbqt"]):
				return parent_candidate
		
		# Return the standard path (will error later if not found)
		return standard_path

	@property
	def prepared_ligand_path(self) -> str:
		"""Return the output directory used by ligand preparation.
		
		If the ligand_input_path contains .pdbqt files, it's treated as already-prepared
		ligands (useful when starting from a later step in the pipeline).
		Otherwise, return the standard output path for prepared ligands.
		When inside a run_<timestamp> subdirectory, also check the parent directory.
		"""
		import os
		# Check if ligand_input_path already contains prepared ligands
		if list_files_in_directory(self.ligand_input_path, ["*.pdbqt"]):
			return self.ligand_input_path
		
		# Try the standard location
		standard_path = f"{self.output_path}/prepared_ligands_pdbqt"
		if list_files_in_directory(standard_path, ["*.pdbqt"]):
			return standard_path
		
		# If inside a run_<timestamp> subdirectory, check parent
		parent_path = os.path.dirname(self.output_path)
		if parent_path != self.output_path:  # Avoid infinite recursion
			parent_candidate = f"{parent_path}/prepared_ligands_pdbqt"
			if os.path.isdir(parent_candidate) and list_files_in_directory(parent_candidate, ["*.pdbqt"]):
				return parent_candidate
		
		# Return the standard path (will error later if not found)
		return standard_path

	def _build_receptor_preparation(self) -> ReceptorPreparation:
		return ReceptorPreparation(
			input_path=self.receptor_input_path,
			output_path=self.output_path,
		)

	def _build_ligand_preparation(self) -> LigandPreparation:
		return LigandPreparation(
			input_path=self.ligand_input_path,
			output_path=self.output_path,
		)

	def _build_active_site_detection(self) -> ActiveSiteDetection:
		return ActiveSiteDetection(
			receptor_path=self.prepared_receptor_path,
			ligand_path=self.prepared_ligand_path,
			output_path=self.output_path,
			fpocket_path=self.fpocket_path,
			manual_center=self.manual_center,
			manual_npts=self.manual_npts,
			manual_boxes=self.manual_boxes,
			flex_radius=self.flex_radius,
		)

	def _ensure_prepared_inputs(self) -> None:
		"""Validate that preparation stages produced usable receptor and ligand files."""
		receptor_files = list_files_in_directory(self.prepared_receptor_path, ["*.pdbqt"])
		ligand_files = list_files_in_directory(self.prepared_ligand_path, ["*.pdbqt"])

		if not receptor_files:
			raise RuntimeError(
				f"No prepared receptor .pdbqt files found in {self.prepared_receptor_path}. "
				"Check the receptor input path and preparation logs."
			)

		if not ligand_files:
			raise RuntimeError(
				f"No prepared ligand .pdbqt files found in {self.prepared_ligand_path}. "
				"Check the ligand input path and preparation logs."
			)
		invalid = []
		for ligand in ligand_files:
			try:
				require_valid_ligand(ligand)
			except Exception as exc:
				invalid.append(f"{ligand}: {exc}")
		if invalid:
			raise RuntimeError(
				"Prepared ligand validation failed before docking:\n  - "
				+ "\n  - ".join(invalid)
			)

	@classmethod
	def _resolve_step_name(cls, step_name: str) -> str:
		normalized = step_name.strip().lower().replace("_", "-")
		if normalized not in cls.STEP_ALIASES:
			raise ValueError(
				f"Unknown step '{step_name}'. Valid steps are: {', '.join(cls.STEP_ORDER)}"
			)
		return cls.STEP_ALIASES[normalized]

	@classmethod
	def _selected_steps(cls, from_step: str, to_step: str) -> Tuple[str, ...]:
		start_step = cls._resolve_step_name(from_step)
		end_step = cls._resolve_step_name(to_step)
		start_index = cls.STEP_ORDER.index(start_step)
		end_index = cls.STEP_ORDER.index(end_step)
		if start_index > end_index:
			raise ValueError(
				f"from_step '{from_step}' must come before or equal to to_step '{to_step}'"
			)
		return cls.STEP_ORDER[start_index : end_index + 1]

	def run_receptor_preparation(self, n_workers: Optional[int] = None) -> Dict[str, int]:
		"""Prepare receptor inputs and convert them to PDBQT."""
		return self._build_receptor_preparation().prepare(n_workers=n_workers)

	def run_ligand_preparation(self, n_workers: Optional[int] = None) -> Dict[str, Any]:
		"""Prepare ligand inputs and convert them to docking-ready files."""
		return self._build_ligand_preparation().prepare(n_workers=n_workers)

	def run_active_site_detection(
		self, n_workers: Optional[int] = None, checklist=None,
	) -> Dict[str, int]:
		"""Detect binding pockets and generate GPF files."""
		return self._build_active_site_detection().prepare(n_workers=n_workers, checklist=checklist)

	def run_preparation_pipeline(
		self,
		receptor_workers: Optional[int] = None,
		ligand_workers: Optional[int] = None,
		active_site_workers: Optional[int] = None,
		checklist=None,
	) -> DockingPipelineResult:
		"""Run only the preparation stages in order."""
		return self.run_step_range(
			from_step="receptor",
			to_step="active-site",
			receptor_workers=receptor_workers,
			ligand_workers=ligand_workers,
			active_site_workers=active_site_workers,
			checklist=checklist,
		)

	def run_full_pipeline(
		self,
		receptor_workers: Optional[int] = None,
		ligand_workers: Optional[int] = None,
		active_site_workers: Optional[int] = None,
		docking_workers: Optional[int] = None,
		autogrid_executable: Optional[str] = None,
		autodock_gpu_executable: Optional[str] = None,
		pdb_export_limit: int = 10,
		max_workers: int = 4,
		engine: str = "both",
		exhaustiveness: int = 8,
		num_modes: int = 9,
		vina_cpu: Optional[int] = None,
		vina_executable: Optional[str] = None,
		seed: Optional[int] = None,
		checklist=None,
	) -> DockingPipelineResult:
		"""Run preparation, docking execution, and analysis in sequence."""
		return self.run_step_range(
			from_step="receptor",
			to_step="analysis",
			receptor_workers=receptor_workers,
			ligand_workers=ligand_workers,
			active_site_workers=active_site_workers,
			docking_workers=docking_workers,
			autogrid_executable=autogrid_executable,
			autodock_gpu_executable=autodock_gpu_executable,
			pdb_export_limit=pdb_export_limit,
			max_workers=max_workers,
			engine=engine,
			exhaustiveness=exhaustiveness,
			num_modes=num_modes,
			vina_cpu=vina_cpu,
			vina_executable=vina_executable,
			seed=seed,
			checklist=checklist,
		)

	def run_step_range(
		self,
		from_step: str,
		to_step: str,
		receptor_workers: Optional[int] = None,
		ligand_workers: Optional[int] = None,
		active_site_workers: Optional[int] = None,
		docking_workers: Optional[int] = None,
		autogrid_executable: Optional[str] = None,
		autodock_gpu_executable: Optional[str] = None,
		pdb_export_limit: int = 10,
		max_workers: int = 4,
		engine: str = "both",
		exhaustiveness: int = 8,
		num_modes: int = 9,
		vina_cpu: Optional[int] = None,
		vina_executable: Optional[str] = None,
		seed: Optional[int] = None,
		checklist=None,
	) -> DockingPipelineResult:
		"""Run a contiguous range of docking steps.

		This supports independent steps, the full pipeline, or suffixes like
		"everything except the first step".

		checklist: When provided, the active-site and execution stages report
			progress as named checklist items instead of their standalone
			progress bars (see ActiveSiteDetection.prepare/DockingExecution.run/
			VinaExecution.run). Not forwarded to the dual-engine "both" path.
			Receptor/ligand preparation ignore it -- unaffected either way.
		"""
		selected_steps = self._selected_steps(from_step, to_step)
		selected = set(selected_steps)

		result = DockingPipelineResult(
			receptor_preparation={},
			ligand_preparation={},
			active_site_detection={},
			executed_steps=selected_steps,
		)

		if "receptor" in selected:
			receptor_stats = self.run_receptor_preparation(n_workers=receptor_workers)
			if receptor_stats.get("successful", 0) <= 0:
				raise RuntimeError(
					"Receptor preparation finished without successful outputs, so the selected flow cannot continue."
				)
			result.receptor_preparation = receptor_stats

		if "ligand" in selected:
			ligand_stats = self.run_ligand_preparation(n_workers=ligand_workers)
			if ligand_stats.get("successful", 0) <= 0:
				raise RuntimeError(
					"Ligand preparation finished without successful outputs, so the selected flow cannot continue."
				)
			result.ligand_preparation = ligand_stats

		if "active-site" in selected:
			self._ensure_prepared_inputs()
			active_site_stats = self.run_active_site_detection(
				n_workers=active_site_workers, checklist=checklist,
			)
			if active_site_stats.get("successful", 0) <= 0:
				raise RuntimeError(
					"Active-site detection did not generate outputs, so the selected flow cannot continue."
				)
			result.active_site_detection = active_site_stats

		if "execution" in selected:
			self._record_selected_engines(engine)
			self._ensure_prepared_inputs()
			docking_execution = self.run_docking_execution(
				n_workers=docking_workers,
				engine=engine,
				autogrid_executable=autogrid_executable,
				autodock_gpu_executable=autodock_gpu_executable,
				vina_executable=vina_executable,
				exhaustiveness=exhaustiveness,
				num_modes=num_modes,
				vina_cpu=vina_cpu,
				seed=seed,
				checklist=checklist,
			)
			result.docking_execution = docking_execution

		if "analysis" in selected:
			docking_analysis = self.run_docking_analysis(
				pdb_export_limit=pdb_export_limit,
				max_workers=max_workers,
				engine=engine,
			)
			result.docking_analysis = docking_analysis

		return result

	def run_docking_execution(self, *args, **kwargs):
		"""Run the docking execution stage on the GPU (AutoDock-GPU), CPU (Vina), or both.

		Default is ``engine="both"``: GPU and CPU run concurrently (different
		hardware, no reason to wait on one before starting the other) against the
		same prepared receptor/ligand set (double validation), and a per-pair
		``docking_comparison.csv`` is written with both affinities side by side --
		the two engines' scores are on a comparable kcal/mol-ish scale, unlike
		LightDock's DFIRE, so a direct comparison is meaningful here.
		``engine="cpu"`` routes to Vina alone, which takes the prepared receptor
		PDBQT plus a search box (from --box-config or the generated GPF files);
		``engine="gpu"`` keeps the AutoGrid/AutoDock-GPU path alone.
		"""
		engine = (kwargs.pop("engine", "both") or "both").lower()
		# Popped centrally: the "both" (dual-engine) branch never gets a checklist,
		# since two threads independently animating items on the same Checklist's
		# Live isn't a combination this design covers -- it stays on its current
		# job_bar/print_stats output. Only the single-engine branches receive it.
		checklist = kwargs.pop("checklist", None)

		# Vina-only options (popped unconditionally so they never leak to the GPU .run).
		vina_executable = kwargs.pop("vina_executable", None)
		exhaustiveness = kwargs.pop("exhaustiveness", 8)
		num_modes = kwargs.pop("num_modes", 9)
		vina_cpu = kwargs.pop("vina_cpu", None)
		seed = kwargs.pop("seed", None)

		# GPU-only options.
		autogrid_executable = kwargs.pop("autogrid_executable", None)
		autodock_gpu_executable = kwargs.pop("autodock_gpu_executable", None)

		def _run_gpu(output_path: str, *, checklist=None) -> Dict[str, Any]:
			return DockingExecution(
				protein_maps_dir=self.prepared_receptor_path,
				ligand_dir=self.prepared_ligand_path,
				output_path=output_path,
				autogrid_executable=autogrid_executable,
				autodock_gpu_executable=autodock_gpu_executable,
			).run(*args, checklist=checklist, **kwargs)

		def _run_cpu(output_path: str, *, checklist=None) -> Dict[str, Any]:
			return VinaExecution(
				receptor_dir=self.prepared_receptor_path,
				ligand_dir=self.prepared_ligand_path,
				output_path=output_path,
				box_config_path=self.box_config_path,
				vina_executable=vina_executable,
				exhaustiveness=exhaustiveness,
				num_modes=num_modes,
				cpu_per_job=vina_cpu,
				seed=seed,
				# Shared prep, not engine-specific -- always at the pipeline root,
				# even when output_path is the isolated cpu_validation/ subfolder
				# engine="both" uses.
				clean_receptors_dir=os.path.join(self.output_path, "clean_receptors"),
			# gpu_ids (GPU-only) isn't a Vina argument -- drop it here even if the
			# caller passed one through *args/**kwargs for the GPU run above.
			).run(n_workers=kwargs.get("n_workers"), checklist=checklist)

		if engine == "both":
			# Isolated subfolder: DockingExecution and VinaExecution otherwise both
			# write "docking_config.txt" straight into output_path, so running them
			# at the same time in the same directory would race on that file (and
			# on docking_results/temp/). This isolation is also what makes running
			# them concurrently safe.
			cpu_output_path = os.path.join(self.output_path, "cpu_validation")

			# GPU and CPU docking use different hardware (the GPU device vs. CPU
			# cores) and don't share state once cpu_output_path is isolated, so
			# there's no reason to make one wait for the other -- run both at once.
			with ThreadPoolExecutor(max_workers=2) as executor:
				gpu_future = executor.submit(_run_gpu, self.output_path)
				cpu_future = executor.submit(_run_cpu, cpu_output_path)
				gpu_result, gpu_error = self._future_result(gpu_future)
				cpu_result, cpu_error = self._future_result(cpu_future)

			if gpu_error is not None and cpu_error is not None:
				raise RuntimeError(
					f"Both GPU and CPU docking failed. GPU: {gpu_error}. CPU: {cpu_error}."
				)

			# Best-effort even if one engine failed: the readers just see an empty
			# results dir for that side and the comparison falls back to "NA" there.
			comparison_file = self._write_dual_engine_comparison(
				self.output_path, self.output_path, cpu_output_path
			)
			result: Dict[str, Any] = {"comparison_file": comparison_file}
			result["gpu"] = gpu_result if gpu_error is None else {"error": str(gpu_error)}
			result["cpu"] = cpu_result if cpu_error is None else {"error": str(cpu_error)}
			return result

		if engine == "cpu":
			return _run_cpu(self.output_path, checklist=checklist)

		return _run_gpu(self.output_path, checklist=checklist)

	@staticmethod
	def _future_result(future: "Future") -> Tuple[Optional[Dict[str, Any]], Optional[BaseException]]:
		"""Resolve a future into (result, None) or (None, exception) without raising.

		Used so that engine="both" can wait on GPU and CPU concurrently and still
		get whichever one succeeded even if the other's run() raised.
		"""
		try:
			return future.result(), None
		except Exception as exc:  # noqa: BLE001 - surfaced to the caller, not swallowed
			return None, exc

	@staticmethod
	def _read_gpu_affinities(gpu_output_path: str) -> Dict[Tuple[str, str], str]:
		"""Read AutoDock-GPU affinities from the .dlg files DockingExecution wrote.

		Returns {(receptor, ligand): affinity}, reusing the adapter's own DLG
		parser instead of re-implementing the RMSD-table scraping here.
		"""
		affinities: Dict[Tuple[str, str], str] = {}
		results_dir = os.path.join(gpu_output_path, "docking_results")
		if not os.path.isdir(results_dir):
			return affinities

		for protein_name in os.listdir(results_dir):
			dlg_dir = os.path.join(results_dir, protein_name, "dlg")
			if not os.path.isdir(dlg_dir):
				continue
			prefix = f"{protein_name}_"
			for fname in os.listdir(dlg_dir):
				if not fname.endswith(".dlg"):
					continue
				stem = fname[: -len(".dlg")]
				ligand_name = stem[len(prefix):] if stem.startswith(prefix) else stem
				summary = AutoDockGPUAdapter._parse_rmsd_table(os.path.join(dlg_dir, fname))
				affinities[(protein_name, ligand_name)] = summary["affinity"]
		return affinities

	@staticmethod
	def _read_cpu_affinities(cpu_output_path: str) -> Dict[Tuple[str, str], str]:
		"""Read Vina affinities from the affinities.csv VinaExecution wrote.

		Returns {(receptor, ligand): best_affinity_kcal_mol}.
		"""
		affinities: Dict[Tuple[str, str], str] = {}
		csv_path = os.path.join(cpu_output_path, "docking_results", "affinities.csv")
		if not os.path.isfile(csv_path):
			return affinities

		with open(csv_path, "r", encoding="utf-8") as handle:
			next(handle, None)  # header
			for line in handle:
				parts = line.strip().split(",")
				if len(parts) < 3:
					continue
				receptor, ligand, affinity = parts[0], parts[1], parts[2]
				affinities[(receptor, ligand)] = affinity
		return affinities

	@staticmethod
	def _write_dual_engine_comparison(comparison_output_path: str, gpu_output_path: str, cpu_output_path: str) -> str:
		"""Merge GPU and CPU affinities for the same receptor/ligand pairs into one CSV."""
		gpu_affinities = DockingPipeline._read_gpu_affinities(gpu_output_path)
		cpu_affinities = DockingPipeline._read_cpu_affinities(cpu_output_path)
		pairs = sorted(set(gpu_affinities) | set(cpu_affinities))

		# Normally created as a side effect of either engine's own results_dir
		# creation, but guard it directly too in case both engines failed before
		# getting that far (e.g. both validate() calls failed immediately).
		create_folder(comparison_output_path)
		comparison_path = os.path.join(comparison_output_path, "docking_comparison.csv")
		with open(comparison_path, "w", encoding="utf-8") as handle:
			handle.write("receptor,ligand,gpu_affinity,cpu_affinity_kcal_mol\n")
			for receptor, ligand in pairs:
				gpu_value = gpu_affinities.get((receptor, ligand), "NA")
				cpu_value = cpu_affinities.get((receptor, ligand), "NA")
				handle.write(f"{receptor},{ligand},{gpu_value},{cpu_value}\n")
		return comparison_path

	def _record_selected_engines(self, engine: str) -> None:
		"""Persist the engines selected for this run in docking_inputs.json."""
		path = os.path.join(self.output_path, "docking_inputs.json")
		data: Dict[str, Any] = {}
		if os.path.isfile(path):
			try:
				with open(path, "r", encoding="utf-8") as handle:
					data = json.load(handle)
			except (OSError, json.JSONDecodeError):
				data = {}
		data["engines"] = ["autodock_gpu", "vina"] if engine == "both" else [
			"vina" if engine == "cpu" else "autodock_gpu"
		]
		with open(path, "w", encoding="utf-8") as handle:
			json.dump(data, handle, indent=2)

	@staticmethod
	def _read_analysis_ranking(path: Optional[str]) -> Dict[Tuple[str, str], Dict[str, Any]]:
		rows: Dict[Tuple[str, str], Dict[str, Any]] = {}
		if not path or not os.path.isfile(path):
			return rows
		with open(path, "r", encoding="utf-8", newline="") as handle:
			for row in csv.DictReader(handle):
				key = (row["protein"], row["ligand"])
				rows[key] = {
					"rank": int(row["rank"]),
					"affinity": float(row["best_affinity_kcal_mol"]),
				}
		return rows

	@classmethod
	def _write_analysis_consensus(
		cls, analysis_dir: str, gpu_result: Dict[str, Any], vina_result: Dict[str, Any]
	) -> str:
		"""Write rank-normalized dual-engine comparison and consensus."""
		gpu = cls._read_analysis_ranking(gpu_result.get("outputs", {}).get("ranking_csv"))
		vina = cls._read_analysis_ranking(vina_result.get("outputs", {}).get("ranking_csv"))
		pairs = sorted(set(gpu) | set(vina))
		gpu_count = max(1, len(gpu))
		vina_count = max(1, len(vina))
		rows = []
		for protein, ligand in pairs:
			g, v = gpu.get((protein, ligand)), vina.get((protein, ligand))
			normalized = []
			if g:
				normalized.append((g["rank"] - 1) / max(1, gpu_count - 1))
			if v:
				normalized.append((v["rank"] - 1) / max(1, vina_count - 1))
			rows.append({
				"receptor": protein,
				"ligand": ligand,
				"gpu_affinity_kcal_mol": g["affinity"] if g else "NA",
				"vina_affinity_kcal_mol": v["affinity"] if v else "NA",
				"gpu_rank": g["rank"] if g else "NA",
				"vina_rank": v["rank"] if v else "NA",
				"consensus_score": sum(normalized) / len(normalized),
				"engines": 2 if g and v else 1,
			})
		rows.sort(key=lambda row: (-row["engines"], row["consensus_score"], row["ligand"]))
		path = os.path.join(analysis_dir, "engine_comparison.csv")
		create_folder(analysis_dir)
		with open(path, "w", encoding="utf-8", newline="") as handle:
			fieldnames = [
				"consensus_rank", "receptor", "ligand", "gpu_affinity_kcal_mol",
				"vina_affinity_kcal_mol", "gpu_rank", "vina_rank",
				"consensus_score", "engines",
			]
			writer = csv.DictWriter(handle, fieldnames=fieldnames)
			writer.writeheader()
			for rank, row in enumerate(rows, 1):
				writer.writerow({"consensus_rank": rank, **row})
		return path

	def run_docking_analysis(self, *args, **kwargs):
		"""Run engine-aware analysis for GPU, CPU, or both outputs."""
		pdb_export_limit = kwargs.pop("pdb_export_limit", 10)
		max_workers = kwargs.pop("max_workers", 4)
		engine = (kwargs.pop("engine", "auto") or "auto").lower()
		if engine == "both":
			analysis_root = os.path.join(self.output_path, "docking_analysis")
			gpu_result = DockingAnalysis(
				self.output_path, pdb_export_limit, max_workers,
				engine="gpu", analysis_dir=os.path.join(analysis_root, "autodock_gpu"),
			).run(*args, **kwargs)
			vina_result = DockingAnalysis(
				os.path.join(self.output_path, "cpu_validation"),
				pdb_export_limit, max_workers,
				engine="vina", analysis_dir=os.path.join(analysis_root, "vina"),
			).run(*args, **kwargs)
			comparison = self._write_analysis_consensus(analysis_root, gpu_result, vina_result)
			return {"gpu": gpu_result, "cpu": vina_result, "comparison_file": comparison}

		analysis_engine = "vina" if engine == "cpu" else engine
		return DockingAnalysis(
			self.output_path,
			pdb_export_limit,
			max_workers,
			engine=analysis_engine,
		).run(*args, **kwargs)
