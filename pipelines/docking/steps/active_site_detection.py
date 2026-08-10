"""Active site detection pipeline step for molecular docking."""

import json
import math
import os
import shutil
import subprocess
import traceback
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
from typing import Optional, Dict, List, Tuple, Any

from ....utils.progress import step_bar_iter


from ....storage.file_manager import (
	create_folder,
	list_files_in_directory,
	find_compound_name,
	create_out_file,
)
from ....adapters.fpocket.fpocket_adapter import FpocketAdapter
from ....utils.logger import setup_logger

logger = setup_logger(__name__)


def load_box_config(path: str) -> Dict[str, Dict[str, Any]]:
	"""Load a per-receptor coordinate box config from a JSON file.

	Expected format::

		{
			"receptor_A": {"center": [x, y, z], "npts": [nx, ny, nz]},
			"receptor_B": {"center": [x, y, z], "npts": [nx, ny, nz]}
		}

	Keys must match the receptor's filename stem (see
	``storage.file_manager.find_compound_name``), e.g. "receptor_A" for
	"receptor_A.pdbqt". Returns a dict mapping receptor name to
	{"center": (x, y, z), "npts": (nx, ny, nz)}.
	"""
	if not os.path.isfile(path):
		raise FileNotFoundError(f"Box config file not found: {path}")

	with open(path, "r") as handle:
		try:
			raw = json.load(handle)
		except json.JSONDecodeError as exc:
			raise ValueError(f"Invalid JSON in box config file {path}: {exc}") from exc

	if not isinstance(raw, dict) or not raw:
		raise ValueError(
			f"Box config {path} must be a non-empty JSON object mapping receptor name -> box"
		)

	boxes: Dict[str, Dict[str, Any]] = {}
	for receptor_name, entry in raw.items():
		if not isinstance(entry, dict) or "center" not in entry or "npts" not in entry:
			raise ValueError(
				f"Box config entry '{receptor_name}' must be an object with 'center' and 'npts'"
			)

		center, npts = entry["center"], entry["npts"]
		if not (isinstance(center, (list, tuple)) and len(center) == 3):
			raise ValueError(f"Box config entry '{receptor_name}': 'center' must have 3 values")
		if not (isinstance(npts, (list, tuple)) and len(npts) == 3):
			raise ValueError(f"Box config entry '{receptor_name}': 'npts' must have 3 values")

		try:
			center_t = (float(center[0]), float(center[1]), float(center[2]))
			npts_t = (int(npts[0]), int(npts[1]), int(npts[2]))
		except (TypeError, ValueError) as exc:
			raise ValueError(
				f"Box config entry '{receptor_name}' has non-numeric center/npts values: {exc}"
			) from exc

		if any(value <= 0 for value in npts_t):
			raise ValueError(f"Box config entry '{receptor_name}': npts values must be > 0")

		boxes[receptor_name] = {"center": center_t, "npts": npts_t}

	return boxes


class ActiveSiteDetection:
	"""Detect binding pockets and generate GPF files for docking.

	Workflow:
	1. Run fpocket for each receptor
	2. Collect pocket1 atoms file
	3. Compute box center and size
	4. Generate receptor-specific .gpf with Meeko (mk_prepare_receptor.py)
	"""

	DEFAULT_FPOCKET_PATH = "/usr/bin/fpocket"
	FPOCKET_CANDIDATES = (
		"/usr/local/bin/fpocket",
		"/usr/bin/fpocket",
	)
	# Spacing por defecto de AutoGrid4 (A), usado para convertir npts a un
	# box_size real en Angstrom que Meeko pueda recibir (ver Fase 0 del plan).
	GRID_SPACING = 0.375
	LIGAND_BOX_PADDING = 30.0

	def __init__(
		self,
		receptor_path: str,
		ligand_path: str,
		output_path: str,
		fpocket_path: Optional[str] = None,
		box_padding: float = 5.0,
		autodock_margin: int = 55,
		npts_min: int = 20,
		npts_max: int = 126,
		manual_center: Optional[Tuple[float, float, float]] = None,
		manual_npts: Optional[Tuple[int, int, int]] = None,
		manual_boxes: Optional[Dict[str, Dict[str, Any]]] = None,
	):
		"""Initialize active site detection step.

		Args:
			receptor_path: Directory containing prepared receptor .pdbqt files
			ligand_path: Directory containing prepared ligand .pdbqt files
			output_path: Base directory for generated outputs/reports
			fpocket_path: Path to fpocket binary
			box_padding: Padding added to raw pocket dimensions (angstrom)
			autodock_margin: Margin added before converting to npts
			npts_min: Minimum npts dimension for AutoDock grid
			npts_max: Maximum npts dimension for AutoDock grid
			manual_center: Single box center applied to every receptor that
				has no entry in ``manual_boxes``. When manual_npts is omitted,
				the grid is sized from the largest prepared ligands.
			manual_npts: Single box npts applied to every receptor that has
				no entry in ``manual_boxes`` (requires manual_center)
			manual_boxes: Per-receptor box override, keyed by receptor name
				(see ``load_box_config``). Takes priority over
				manual_center/manual_npts and over fpocket auto-detection
				for the receptors it covers; receptors not listed fall back
				to manual_center/manual_npts, then to fpocket.
		"""
		self.receptor_path = receptor_path
		self.ligand_path = ligand_path
		self.output_path = output_path

		self.fpocket = FpocketAdapter(fpocket_path)
		self.fpocket_path = self.fpocket.fpocket_path

		self.box_padding = box_padding
		self.autodock_margin = autodock_margin
		self.npts_min = npts_min
		self.npts_max = npts_max
		self.manual_center = manual_center
		self.manual_npts = manual_npts
		self.manual_boxes = manual_boxes or {}
		self._ligand_sized_box: Optional[Dict[str, Any]] = None

		if self.manual_npts is not None and self.manual_center is None:
			raise ValueError("manual_npts requires manual_center")

		if self.manual_npts is not None:
			if any(value <= 0 for value in self.manual_npts):
				raise ValueError("manual_npts values must be > 0")

		self.pocket_folder = os.path.join(self.output_path, "pdb_boxes")
		self.dimensions_folder = os.path.join(self.output_path, "dimensions_boxes")

	@property
	def is_manual_mode(self) -> bool:
		"""Return True when a global manual grid center is configured."""
		return self.manual_center is not None

	def _box_for_receptor(self, receptor_name: str) -> Optional[Dict[str, Any]]:
		"""Return the manual box for a receptor, if any.

		Per-receptor entries in manual_boxes take priority over the global
		manual_center/manual_npts fallback. Returns None when the receptor
		should use automatic fpocket detection instead.
		"""
		if receptor_name in self.manual_boxes:
			return self.manual_boxes[receptor_name]
		if self.manual_center is not None and self.manual_npts is not None:
			return {"center": self.manual_center, "npts": self.manual_npts, "mode": "manual"}
		if self.manual_center is not None and self._ligand_sized_box is not None:
			return {
				**self._ligand_sized_box,
				"center": self.manual_center,
				"mode": "ligand-sized",
			}
		return None

	@staticmethod
	def _ligand_dimensions(ligand_file: str) -> Tuple[float, float, float]:
		"""Return the X/Y/Z spans of a prepared PDBQT ligand in angstroms."""
		coords: List[Tuple[float, float, float]] = []
		with open(ligand_file, encoding="utf-8", errors="ignore") as handle:
			for line in handle:
				if not line.startswith(("ATOM", "HETATM")):
					continue
				try:
					coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
				except ValueError:
					continue
		if not coords:
			raise RuntimeError(f"No valid atom coordinates found in ligand: {ligand_file}")
		xs, ys, zs = zip(*coords)
		return max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)

	def _size_grid_from_ligands(self) -> Dict[str, Any]:
		"""Calculate a grid that fits the maximum ligand span on each axis."""
		ligands = sorted(list_files_in_directory(self.ligand_path, ["*.pdbqt"]))
		if not ligands:
			raise RuntimeError(f"No ligand .pdbqt files found in: {self.ligand_path}")

		max_spans = [0.0, 0.0, 0.0]
		max_ligands = ["", "", ""]
		for ligand_file in ligands:
			spans = self._ligand_dimensions(ligand_file)
			for axis, span in enumerate(spans):
				if span > max_spans[axis]:
					max_spans[axis] = span
					max_ligands[axis] = os.path.basename(ligand_file)

		requested_size = tuple(span + self.LIGAND_BOX_PADDING for span in max_spans)
		npts_values = []
		for axis, size in zip("XYZ", requested_size):
			npts = math.ceil(size / self.GRID_SPACING)
			if npts % 2:
				npts += 1
			npts = max(self.npts_min, npts)
			if npts > self.npts_max:
				raise RuntimeError(
					f"Ligand-sized grid requires {npts} points on {axis}, exceeding "
					f"the configured AutoGrid maximum of {self.npts_max}."
				)
			npts_values.append(npts)

		npts_tuple = tuple(npts_values)
		return {
			"npts": npts_tuple,
			"size": tuple(n * self.GRID_SPACING for n in npts_tuple),
			"requested_size": requested_size,
			"ligand_max_spans": tuple(max_spans),
			"ligand_max_files": tuple(max_ligands),
			"ligand_count": len(ligands),
			"padding": self.LIGAND_BOX_PADDING,
		}

	def _validate_dependencies(self) -> None:
		"""Validate required external binaries and scripts."""
		self.fpocket.validate()
		if shutil.which("mk_prepare_receptor.py") is None:
			raise FileNotFoundError(
				"mk_prepare_receptor.py not found on PATH. Is meeko installed?"
			)

	def _collect_receptors(self) -> List[str]:
		"""Collect receptor files to process."""
		return sorted(list_files_in_directory(self.receptor_path, ["*.pdbqt"]))

	def _select_reference_ligand(self) -> str:
		"""Select first available ligand .pdbqt as reference for GPF generation."""
		ligands = sorted(list_files_in_directory(self.ligand_path, ["*.pdbqt"]))
		if not ligands:
			raise RuntimeError(f"No ligand .pdbqt files found in: {self.ligand_path}")
		return ligands[0]

	def _get_shard_config(self) -> Optional[Tuple[int, int]]:
		"""Return (shard_index, shard_count) from environment if configured."""
		shard_index_env = os.getenv("CHEMLINK_SHARD_INDEX")
		shard_count_env = os.getenv("CHEMLINK_SHARD_COUNT")

		if shard_index_env is not None or shard_count_env is not None:
			if shard_index_env is None or shard_count_env is None:
				raise RuntimeError(
					"Both CHEMLINK_SHARD_INDEX and CHEMLINK_SHARD_COUNT must be set"
				)

			try:
				shard_index = int(shard_index_env)
				shard_count = int(shard_count_env)
			except ValueError as exc:
				raise RuntimeError(
					"CHEMLINK_SHARD_INDEX and CHEMLINK_SHARD_COUNT must be integers"
				) from exc
		elif os.getenv("SLURM_ARRAY_TASK_ID") is not None:
			try:
				slurm_task_id = int(os.getenv("SLURM_ARRAY_TASK_ID", "0"))
			except ValueError as exc:
				raise RuntimeError("SLURM_ARRAY_TASK_ID must be an integer") from exc

			slurm_task_count = os.getenv("SLURM_ARRAY_TASK_COUNT")
			slurm_task_min = os.getenv("SLURM_ARRAY_TASK_MIN")
			slurm_task_max = os.getenv("SLURM_ARRAY_TASK_MAX")

			if slurm_task_count is not None:
				try:
					shard_count = int(slurm_task_count)
				except ValueError as exc:
					raise RuntimeError(
						"SLURM_ARRAY_TASK_COUNT must be an integer"
					) from exc

				if slurm_task_min is not None:
					try:
						task_min = int(slurm_task_min)
					except ValueError as exc:
						raise RuntimeError(
							"SLURM_ARRAY_TASK_MIN must be an integer"
						) from exc
					shard_index = slurm_task_id - task_min
				else:
					shard_index = slurm_task_id
			elif slurm_task_min is not None and slurm_task_max is not None:
				try:
					task_min = int(slurm_task_min)
					task_max = int(slurm_task_max)
				except ValueError as exc:
					raise RuntimeError(
						"SLURM_ARRAY_TASK_MIN and SLURM_ARRAY_TASK_MAX must be integers"
					) from exc

				shard_count = (task_max - task_min) + 1
				shard_index = slurm_task_id - task_min
			else:
				return None
		else:
			return None

		if shard_count <= 1:
			return None
		if shard_count <= 0:
			raise RuntimeError("Shard count must be > 0")
		if shard_index < 0 or shard_index >= shard_count:
			raise RuntimeError(
				f"Invalid shard index {shard_index} for shard count {shard_count}"
			)

		return shard_index, shard_count

	@staticmethod
	def _select_shard_files(files: List[str], shard_index: int, shard_count: int) -> List[str]:
		"""Return round-robin file subset assigned to a shard."""
		return files[shard_index::shard_count]

	@staticmethod
	def _strip_pdbqt_to_pdb(input_pdbqt: str, output_pdb: str) -> None:
		"""Create a PDB-like file from PDBQT for tools that expect PDB."""
		kept = 0
		with open(input_pdbqt, "r") as src, open(output_pdb, "w") as dst:
			for line in src:
				if line.startswith(("ATOM", "HETATM", "TER", "END")):
					dst.write(line[:66].rstrip() + "\n")
					kept += 1

		if kept == 0:
			raise RuntimeError(f"No ATOM/HETATM records found in {input_pdbqt}")

	@staticmethod
	def _parse_atom_xyz(line: str) -> Tuple[float, float, float]:
		"""Parse atom coordinates from a PDB atom line using fixed columns."""
		return (
			float(line[30:38].strip()),
			float(line[38:46].strip()),
			float(line[46:54].strip()),
		)

	def _calculate_box(self, pocket_file: str) -> Dict[str, Any]:
		"""Calculate center/size from pocket atom coordinates."""
		x_min, y_min, z_min = float("inf"), float("inf"), float("inf")
		x_max, y_max, z_max = float("-inf"), float("-inf"), float("-inf")
		atom_count = 0

		with open(pocket_file, "r") as handle:
			for line in handle:
				if not line.startswith(("ATOM", "HETATM")):
					continue
				x, y, z = self._parse_atom_xyz(line)
				x_min = min(x_min, x)
				y_min = min(y_min, y)
				z_min = min(z_min, z)
				x_max = max(x_max, x)
				y_max = max(y_max, y)
				z_max = max(z_max, z)
				atom_count += 1

		if atom_count == 0:
			raise RuntimeError(f"No atoms found in pocket file: {pocket_file}")

		center = (
			(x_min + x_max) / 2,
			(y_min + y_max) / 2,
			(z_min + z_max) / 2,
		)
		size = (
			(x_max - x_min) + self.box_padding,
			(y_max - y_min) + self.box_padding,
			(z_max - z_min) + self.box_padding,
		)

		return {
			"atom_count": atom_count,
			"limits": {
				"x": (x_min, x_max),
				"y": (y_min, y_max),
				"z": (z_min, z_max),
			},
			"center": center,
			"size": size,
		}

	def _calculate_npts(self, size: Tuple[float, float, float]) -> Tuple[int, int, int]:
		"""Convert floating size to AutoDock npts with safeguards."""
		nx = int(size[0]) + self.autodock_margin
		ny = int(size[1]) + self.autodock_margin
		nz = int(size[2]) + self.autodock_margin

		nx = max(self.npts_min, min(nx, self.npts_max))
		ny = max(self.npts_min, min(ny, self.npts_max))
		nz = max(self.npts_min, min(nz, self.npts_max))

		return nx, ny, nz

	def _protonated_pdb_for(self, receptor_file: str) -> str:
		"""Locate the protonated PDB that ReceptorPreparation kept for this receptor.

		Meeko's mk_prepare_receptor.py needs a PDB/PQR/mmCIF input, not a .pdbqt,
		so GPF generation reuses the "_h.pdb" intermediate written next to the
		receptor's clean PDB in Fase 2 (utils/receptor_processor.py).
		"""
		stem = os.path.splitext(os.path.basename(receptor_file))[0]
		protonated_pdb = os.path.join(self.output_path, "clean_receptors", f"{stem}_h.pdb")
		if not os.path.isfile(protonated_pdb):
			raise RuntimeError(
				f"Protonated PDB not found: {protonated_pdb}. GPF generation requires "
				"receptors prepared by this pipeline's ReceptorPreparation step "
				"(the 'bring your own .pdbqt' mode is not supported for GPF generation)."
			)
		return protonated_pdb

	def _run_prepare_gpf(
		self,
		receptor_file: str,
		ligand_file: str,
		center: Tuple[float, float, float],
		npts: Tuple[int, int, int],
		output_gpf: str,
	) -> None:
		"""Generate GPF file with Meeko (mk_prepare_receptor.py --write_gpf).

		ligand_file is unused: unlike prepare_gpf4.py, Meeko derives a
		comprehensive superset of AutoDock ligand types automatically instead
		of needing an explicit reference ligand.
		"""
		protonated_pdb = self._protonated_pdb_for(receptor_file)
		box_size = tuple(n * self.GRID_SPACING for n in npts)

		out_prefix = output_gpf[:-len(".gpf")] if output_gpf.endswith(".gpf") else output_gpf
		cmd = [
			"mk_prepare_receptor.py",
			"--read_pdb", os.path.abspath(protonated_pdb),
			"-o", out_prefix,
			"-p", "-v", "-g",
			"--box_center", f"{center[0]:.3f}", f"{center[1]:.3f}", f"{center[2]:.3f}",
			"--box_size", f"{box_size[0]:.3f}", f"{box_size[1]:.3f}", f"{box_size[2]:.3f}",
		]

		result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
		if result.returncode != 0:
			receptor_stem = os.path.splitext(os.path.basename(receptor_file))[0]
			raise RuntimeError(
				f"mk_prepare_receptor.py failed to generate GPF for "
				f"{receptor_stem}: {result.stderr.strip()}"
			)

		if not os.path.isfile(output_gpf) or os.path.getsize(output_gpf) == 0:
			raise RuntimeError(f"GPF output missing or empty: {output_gpf}")

	def _write_dimensions_report(self, rows: List[Dict[str, Any]], timestamp: str) -> str:
		"""Write computed center/size report for all processed receptors."""
		report_path = os.path.join(
			self.dimensions_folder,
			f"dimensions_results_{timestamp}.txt",
		)

		with open(report_path, "w") as handle:
			handle.write("Box Dimensions Report\n")
			handle.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
			handle.write("=============================================\n\n")

			for row in rows:
				limits = row.get("limits")
				center = row["center"]
				size = row.get("size")
				npts = row["npts"]
				mode = row.get("mode", "automatic")

				handle.write(f"File: {row['pocket_file']}\n")
				handle.write(f"Mode: {mode}\n")
				handle.write(f"Processed Atoms: {row['atom_count']}\n")
				if mode == "ligand-sized":
					spans = row["ligand_max_spans"]
					files = row["ligand_max_files"]
					requested = row["requested_size"]
					handle.write(f"Ligands measured: {row['ligand_count']}\n")
					handle.write(f"Ligand padding per axis: +{row['padding']:.3f} A\n")
					handle.write(
						f"Maximum ligand spans (X, Y, Z): "
						f"({spans[0]:.3f}, {spans[1]:.3f}, {spans[2]:.3f}) A\n"
					)
					handle.write(
						f"Requested box size (X, Y, Z): "
						f"({requested[0]:.3f}, {requested[1]:.3f}, {requested[2]:.3f}) A\n"
					)
					handle.write(
						f"Dimension source ligands (X, Y, Z): "
						f"({files[0]}, {files[1]}, {files[2]})\n"
					)
				if limits is not None:
					handle.write(f"X Limits: {limits['x'][0]:.3f} to {limits['x'][1]:.3f}\n")
					handle.write(f"Y Limits: {limits['y'][0]:.3f} to {limits['y'][1]:.3f}\n")
					handle.write(f"Z Limits: {limits['z'][0]:.3f} to {limits['z'][1]:.3f}\n")
				else:
					handle.write("X Limits: N/A (manual)\n")
					handle.write("Y Limits: N/A (manual)\n")
					handle.write("Z Limits: N/A (manual)\n")
				handle.write(
					f"Center of box: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})\n"
				)
				if size is not None:
					handle.write(
						f"Size of box (X, Y, Z): ({size[0]:.3f}, {size[1]:.3f}, {size[2]:.3f})\n"
					)
				else:
					handle.write("Size of box (X, Y, Z): N/A (manual)\n")
				handle.write(f"NPTS (X, Y, Z): ({npts[0]}, {npts[1]}, {npts[2]})\n")
				handle.write("---------------------------------------------\n\n")

		return report_path

	def _write_error_report(self, failed: List[Dict[str, Any]], timestamp: str) -> str:
		"""Write a detailed error report file."""
		log_path = os.path.join(
			self.output_path,
			f"active_site_detection_errors_{timestamp}.txt",
		)

		with open(log_path, "w") as handle:
			handle.write("Active Site Detection Error Report\n")
			handle.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
			handle.write(f"{'='*80}\n\n")
			handle.write(f"Summary: {len(failed)} receptor(s) failed\n\n")
			handle.write(f"{'='*80}\n\n")

			for i, error in enumerate(failed, 1):
				handle.write(f"Error #{i}\n")
				handle.write(f"{'-'*80}\n")
				handle.write(f"Receptor: {error['receptor']}\n")
				handle.write(f"File: {error['file']}\n")
				handle.write(f"Error: {error['error']}\n")
				handle.write(f"\nTraceback:\n{error['traceback']}\n")
				handle.write(f"{'='*80}\n\n")

		return log_path

	def _process_single(self, receptor_file: str, ligand_file: str) -> Dict[str, Any]:
		"""Process one receptor end-to-end."""
		receptor_name = find_compound_name(receptor_file)
		manual_box = self._box_for_receptor(receptor_name)

		if manual_box is not None:
			mode = manual_box.get("mode", "manual")
			pocket_file = mode
			center = manual_box["center"]
			npts = manual_box["npts"]
			box_data = {
				"atom_count": 0,
				"limits": None,
				"center": center,
				"size": manual_box.get("size"),
				"requested_size": manual_box.get("requested_size"),
				"ligand_max_spans": manual_box.get("ligand_max_spans"),
				"ligand_max_files": manual_box.get("ligand_max_files"),
				"ligand_count": manual_box.get("ligand_count"),
				"padding": manual_box.get("padding"),
			}
		else:
			mode = "automatic"
			pocket_file = self.fpocket.detect_pocket(receptor_file, self.pocket_folder)
			box_data = self._calculate_box(pocket_file)
			npts = self._calculate_npts(box_data["size"])

		out_gpf = create_out_file(self.receptor_path, f"{receptor_name}.gpf")
		self._run_prepare_gpf(
			receptor_file=receptor_file,
			ligand_file=ligand_file,
			center=box_data["center"],
			npts=npts,
			output_gpf=out_gpf,
		)

		return {
			"receptor": receptor_name,
			"receptor_file": receptor_file,
			"pocket_file": os.path.basename(pocket_file),
			"gpf_file": out_gpf,
			"atom_count": box_data["atom_count"],
			"limits": box_data["limits"],
			"center": box_data["center"],
			"size": box_data["size"],
			"npts": npts,
			"mode": mode,
			"requested_size": box_data.get("requested_size"),
			"ligand_max_spans": box_data.get("ligand_max_spans"),
			"ligand_max_files": box_data.get("ligand_max_files"),
			"ligand_count": box_data.get("ligand_count"),
			"padding": box_data.get("padding"),
		}

	def prepare(self, n_workers: Optional[int] = None) -> Dict[str, int]:
		"""Run active site detection and GPF generation for all receptors.

		Args:
			n_workers: Local worker count (None = auto)

		Returns:
			Dictionary with statistics: {'successful': int, 'failed': int}
		"""
		self._validate_dependencies()

		create_folder(self.dimensions_folder)

		receptors = self._collect_receptors()
		if not receptors:
			logger.warning("No receptor .pdbqt files found in %s", self.receptor_path)
			return {"successful": 0, "failed": 0}

		if self.manual_boxes:
			receptor_names = {find_compound_name(r) for r in receptors}
			unmatched = sorted(set(self.manual_boxes) - receptor_names)
			if unmatched:
				logger.warning(
					"Box config has %d entrie(s) with no matching receptor: %s",
					len(unmatched), ", ".join(unmatched),
				)

		ligand_file = self._select_reference_ligand()
		if self.manual_center is not None and self.manual_npts is None:
			self._ligand_sized_box = self._size_grid_from_ligands()
			logger.info(
				"Ligand-sized grid from %d ligand(s): spans=%s A, padding=+%.1f A, "
				"final size=%s A, npts=%s",
				self._ligand_sized_box["ligand_count"],
				self._ligand_sized_box["ligand_max_spans"],
				self._ligand_sized_box["padding"],
				self._ligand_sized_box["size"],
				self._ligand_sized_box["npts"],
			)

		shard_cfg = self._get_shard_config()
		total_files = len(receptors)
		if shard_cfg:
			shard_index, shard_count = shard_cfg
			receptors = self._select_shard_files(receptors, shard_index, shard_count)
			logger.info(
				"SLURM sharding enabled: shard %d/%d processing %d/%d receptor(s)",
				shard_index + 1, shard_count, len(receptors), total_files,
			)
			if not receptors:
				logger.info("No receptors assigned to this shard.")
				return {"successful": 0, "failed": 0}

		needs_fpocket = any(
			self._box_for_receptor(find_compound_name(r)) is None for r in receptors
		)
		if needs_fpocket:
			create_folder(self.pocket_folder)

		if n_workers is None:
			slurm_cpus = os.getenv("SLURM_CPUS_PER_TASK")
			if slurm_cpus and slurm_cpus.isdigit():
				n_workers = int(slurm_cpus)
			else:
				n_workers = cpu_count()

		n_workers = max(1, min(n_workers, len(receptors)))

		successful_rows: List[Dict[str, Any]] = []
		failed: List[Dict[str, Any]] = []
		timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

		logger.info(
			"Detecting active sites for %d receptor(s) using %d worker(s).",
			len(receptors), n_workers,
		)
		if self.manual_boxes:
			logger.info(
				"Per-receptor box config loaded for %d receptor(s): %s",
				len(self.manual_boxes), ", ".join(sorted(self.manual_boxes)),
			)
		if self.is_manual_mode:
			if self.manual_npts is not None:
				logger.info(
					"Global manual grid mode: center=%s npts=%s (receptors without a "
					"specific box entry)", self.manual_center, self.manual_npts,
				)
			else:
				logger.info(
					"Manual center with ligand-sized grid: center=%s npts=%s",
					self.manual_center, self._ligand_sized_box["npts"],
				)

		if n_workers == 1:
			for receptor_file in step_bar_iter(
				receptors, "Active Site Detection", unit="receptor", colour="yellow3"
			):
				receptor_name = find_compound_name(receptor_file)
				try:
					result = self._process_single(receptor_file, ligand_file)
					successful_rows.append(result)
					logger.info(f"✓ Active site prepared for {receptor_name}")
				except Exception as e:
					failed.append(
						{
							"receptor": receptor_name,
							"file": receptor_file,
							"error": str(e),
							"traceback": traceback.format_exc(),
						}
					)
					logger.error(f"Failed {receptor_name}: {e}")
		else:
			with ThreadPoolExecutor(max_workers=n_workers) as executor:
				future_map = {
					executor.submit(self._process_single, receptor_file, ligand_file): receptor_file
					for receptor_file in receptors
				}

				for future in step_bar_iter(
					as_completed(future_map), "Active Site Detection",
					total=len(future_map), unit="receptor", colour="yellow3",
				):
					receptor_file = future_map[future]
					receptor_name = find_compound_name(receptor_file)
					try:
						result = future.result()
						successful_rows.append(result)
						logger.info(f"✓ Active site prepared for {receptor_name}")
					except Exception as e:
						failed.append(
							{
								"receptor": receptor_name,
								"file": receptor_file,
								"error": str(e),
								"traceback": traceback.format_exc(),
							}
						)
						logger.error(f"Failed {receptor_name}: {e}")

		logger.info(
			"Successfully processed: %d/%d receptors", len(successful_rows), len(receptors)
		)

		if successful_rows:
			dimensions_report = self._write_dimensions_report(successful_rows, timestamp)
			logger.info("Dimensions report: %s", dimensions_report)

		if failed:
			error_report = self._write_error_report(failed, timestamp)
			logger.warning("Failed: %d receptor(s) — see %s", len(failed), error_report)

		return {"successful": len(successful_rows), "failed": len(failed)}
