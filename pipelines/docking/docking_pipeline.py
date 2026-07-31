"""High-level orchestration for the docking preparation workflow."""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from ...storage.file_manager import list_files_in_directory
from .steps import (
	ActiveSiteDetection,
	DockingAnalysis,
	DockingExecution,
	LigandPreparation,
	ReceptorPreparation,
	load_box_config,
)


@dataclass
class DockingPipelineResult:
	"""Structured result for the preparation pipeline."""

	receptor_preparation: Dict[str, int]
	ligand_preparation: Dict[str, int]
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
	):
		self.receptor_input_path = receptor_input_path
		self.ligand_input_path = ligand_input_path
		self.output_path = output_path
		self.fpocket_path = fpocket_path
		self.manual_center = manual_center
		self.manual_npts = manual_npts
		self.box_config_path = box_config_path
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

	def run_ligand_preparation(self, n_workers: Optional[int] = None) -> Dict[str, int]:
		"""Prepare ligand inputs and convert them to docking-ready files."""
		return self._build_ligand_preparation().prepare(n_workers=n_workers)

	def run_active_site_detection(self, n_workers: Optional[int] = None) -> Dict[str, int]:
		"""Detect binding pockets and generate GPF files."""
		return self._build_active_site_detection().prepare(n_workers=n_workers)

	def run_preparation_pipeline(
		self,
		receptor_workers: Optional[int] = None,
		ligand_workers: Optional[int] = None,
		active_site_workers: Optional[int] = None,
	) -> DockingPipelineResult:
		"""Run only the preparation stages in order."""
		return self.run_step_range(
			from_step="receptor",
			to_step="active-site",
			receptor_workers=receptor_workers,
			ligand_workers=ligand_workers,
			active_site_workers=active_site_workers,
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
	) -> DockingPipelineResult:
		"""Run a contiguous range of docking steps.

		This supports independent steps, the full pipeline, or suffixes like
		"everything except the first step".
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
			active_site_stats = self.run_active_site_detection(n_workers=active_site_workers)
			if active_site_stats.get("successful", 0) <= 0:
				raise RuntimeError(
					"Active-site detection did not generate outputs, so the selected flow cannot continue."
				)
			result.active_site_detection = active_site_stats

		if "execution" in selected:
			self._ensure_prepared_inputs()
			docking_execution = self.run_docking_execution(
				n_workers=docking_workers,
				autogrid_executable=autogrid_executable,
				autodock_gpu_executable=autodock_gpu_executable,
			)
			result.docking_execution = docking_execution

		if "analysis" in selected:
			docking_analysis = self.run_docking_analysis(pdb_export_limit=pdb_export_limit, max_workers=max_workers)
			result.docking_analysis = docking_analysis

		return result

	def run_docking_execution(self, *args, **kwargs):
		"""Run the docking execution stage."""
		return DockingExecution(
			protein_maps_dir=self.prepared_receptor_path,
			ligand_dir=self.prepared_ligand_path,
			output_path=self.output_path,
			autogrid_executable=kwargs.pop("autogrid_executable", None),
			autodock_gpu_executable=kwargs.pop("autodock_gpu_executable", None),
		).run(*args, **kwargs)

	def run_docking_analysis(self, *args, **kwargs):
		"""Run the docking analysis stage."""
		return DockingAnalysis(
			output_path=self.output_path,
			pdb_export_limit=kwargs.pop("pdb_export_limit", 10),
			max_workers=kwargs.pop("max_workers", 4),
		).run(*args, **kwargs)