"""Docking analysis step for molecular docking."""

import csv
import glob
import os
import re
import shutil
import statistics
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from ....storage.file_manager import create_folder
from ....scripts.export_docking_complexes import export as export_docking_complexes
from ....utils.logger import setup_logger

logger = setup_logger(__name__)

# AutoDock atom type -> element symbol, for writing real elements into exported
# pose PDBs instead of the ligand's PDBQT type string (e.g. "OA", "HD", "NA").
AUTODOCK_TYPE_TO_ELEMENT = {
	"A": "C", "C": "C", "N": "N", "NA": "N", "NS": "N",
	"O": "O", "OA": "O", "OS": "O", "S": "S", "SA": "S",
	"P": "P", "H": "H", "HD": "H", "HS": "H", "F": "F",
	"Cl": "Cl", "CL": "Cl", "Br": "Br", "BR": "Br", "I": "I",
	"Mg": "Mg", "MG": "Mg", "Mn": "Mn", "MN": "Mn",
	"Zn": "Zn", "ZN": "Zn", "Ca": "Ca", "CA": "Ca",
	"Fe": "Fe", "FE": "Fe", "B": "B", "Si": "Si", "SI": "Si",
}


@dataclass(frozen=True)
class DockingPose:
	"""Single docking pose from a DLG file."""

	protein: str
	ligand: str
	run: int
	affinity: float
	dlg_file: str


@dataclass(frozen=True)
class DockingLigandStats:
	"""Aggregated statistics for a ligand across all poses."""

	protein: str
	ligand: str
	num_poses: int
	best_run: int
	best_affinity: float  # Most negative (best binding)
	worst_affinity: float  # Least negative (worst binding)
	mean_affinity: float
	std_affinity: float
	dlg_file: str
	mean_rmsd: Optional[float] = None
	std_rmsd: Optional[float] = None
	qc_flag: Optional[str] = None
	best_pose_pdb: Optional[str] = None


class DockingAnalysis:
	"""Analyze AutoDock-GPU or Vina outputs with a shared statistics model."""

	def __init__(
		self,
		output_path: str,
		pdb_export_limit: int = 10,
		max_workers: int = 4,
		engine: str = "auto",
		analysis_dir: Optional[str] = None,
	):
		self.output_path = output_path
		self.results_dir = os.path.join(self.output_path, "docking_results")
		self.analysis_dir = analysis_dir or os.path.join(self.output_path, "docking_analysis")
		self.engine = engine
		self.summary_txt = os.path.join(self.analysis_dir, "docking_analysis_summary.txt")
		self.summary_csv = os.path.join(self.analysis_dir, "docking_statistics.csv")
		self.legacy_csv = os.path.join(self.analysis_dir, "statistics.csv")
		self.summary_md = os.path.join(self.analysis_dir, "docking_analysis_report.md")
		self.ranking_csv = os.path.join(self.analysis_dir, "ligand_ranking.csv")
		self.dynamics_candidates_csv = os.path.join(self.analysis_dir, "dynamics_candidates.csv")
		self.pdb_export_limit = max(0, pdb_export_limit)
		self.max_workers = max(1, max_workers)
		self._pose_coords_cache: Dict[Tuple[str, int], Optional[np.ndarray]] = {}
		self._ligand_pose_groups: Dict[Tuple[str, str, str], List[DockingPose]] = {}

	@staticmethod
	def _parse_float(value: str) -> Optional[float]:
		try:
			return float(value)
		except (TypeError, ValueError):
			return None

	@staticmethod
	def _parse_int(value: str) -> Optional[int]:
		try:
			return int(value)
		except (TypeError, ValueError):
			return None

	def _parse_dlg(self, dlg_path: str) -> List[DockingPose]:
		"""
		Parse ALL poses from a DLG file.
		
		AutoDock-GPU DLG files report each pose as a block:
		Run:   N / total
		Estimated Free Energy of Binding =  -8.10 kcal/mol
		"""
		poses: List[DockingPose] = []
		protein = Path(dlg_path).parent.parent.name
		base = Path(dlg_path).stem
		ligand = base.replace(f"{protein}_", "", 1)
		current_run: Optional[int] = None

		try:
			with open(dlg_path, "r", encoding="utf-8", errors="ignore") as handle:
				for line in handle:
					if "Run:" in line:
						run_match = re.search(r"Run:\s*(\d+)\s*/", line)
						if run_match:
							current_run = self._parse_int(run_match.group(1))

					if "Estimated Free Energy of Binding" in line and current_run is not None:
						affinity_match = re.search(
							r"Estimated Free Energy of Binding\s*=\s*([-+]?[0-9]*\.?[0-9]+)",
							line,
						)
						if affinity_match:
							affinity = self._parse_float(affinity_match.group(1))
							if affinity is not None:
								poses.append(DockingPose(
									protein=protein,
									ligand=ligand,
									run=current_run,
									affinity=affinity,
									dlg_file=dlg_path,
								))
		except Exception as exc:
			logger.warning("Error parsing DLG %s: %s", dlg_path, exc)

		return poses

	def _parse_vina_pdbqt(self, pdbqt_path: str) -> List[DockingPose]:
		"""Parse all MODEL/REMARK VINA RESULT poses from a Vina PDBQT."""
		poses: List[DockingPose] = []
		protein = Path(pdbqt_path).parent.parent.name
		stem = Path(pdbqt_path).stem.removesuffix("_out")
		prefix = f"{protein}_"
		ligand = stem[len(prefix):] if stem.startswith(prefix) else stem
		current_model: Optional[int] = None
		with open(pdbqt_path, "r", encoding="utf-8", errors="ignore") as handle:
			for line in handle:
				if line.startswith("MODEL"):
					parts = line.split()
					current_model = self._parse_int(parts[1]) if len(parts) > 1 else None
				elif line.startswith("REMARK VINA RESULT:") and current_model is not None:
					parts = line.split()
					affinity = self._parse_float(parts[3]) if len(parts) > 3 else None
					if affinity is not None:
						poses.append(DockingPose(
							protein=protein,
							ligand=ligand,
							run=current_model,
							affinity=affinity,
							dlg_file=pdbqt_path,
						))
		return poses

	def _extract_pose_coords(self, dlg_path: str, run_number: int) -> Optional[np.ndarray]:
		"""Extract 3D coordinates for a given run from a DLG file as an (N,3) array.

		Returns None if coordinates cannot be extracted.
		"""
		atoms = self._extract_pose_atoms(dlg_path, run_number)
		if atoms is None:
			return None
		return np.array([[x, y, z] for _name, _elem, x, y, z in atoms])

	def _extract_pose_atoms(
		self, dlg_path: str, run_number: int
	) -> Optional[List[Tuple[str, str, float, float, float]]]:
		"""Extract (atom_name, element, x, y, z) for a given run from a DLG file."""
		if dlg_path.lower().endswith(".pdbqt"):
			return self._extract_vina_pose_atoms(dlg_path, run_number)

		atoms: List[Tuple[str, str, float, float, float]] = []
		in_target = False
		try:
			with open(dlg_path, "r", encoding="utf-8", errors="ignore") as handle:
				for line in handle:
					run_match = re.match(r"\s*Run:\s*(\d+)\s*/", line)
					if run_match:
						current_run = int(run_match.group(1))
						if in_target and current_run != run_number:
							break
						in_target = current_run == run_number
						continue
					if not in_target:
						continue
					# DLG records contain a PDBQT record after the ``DOCKED:`` prefix.
					# Parse its fixed-width coordinate columns instead of splitting fields:
					# AutoDock-GPU may omit the optional chain identifier, which changes
					# the number of whitespace-separated fields but not the PDBQT columns.
					m = re.match(r"\s*DOCKED:\s?(ATOM  |HETATM)(.*)$", line)
					if m:
						pdbqt_record = m.group(1) + m.group(2)
						try:
							name = pdbqt_record[12:16].strip()
							ad_type = pdbqt_record.split()[-1]
							atoms.append((
								name or "C",
								AUTODOCK_TYPE_TO_ELEMENT.get(ad_type, "C"),
								float(pdbqt_record[30:38]),
								float(pdbqt_record[38:46]),
								float(pdbqt_record[46:54]),
							))
						except (ValueError, IndexError):
							logger.debug("Malformed DOCKED coordinate record in %s: %s", dlg_path, line.rstrip())
					# End of model block
					if in_target and line.strip().startswith("DOCKED: ENDMDL"):
						break
			return atoms or None
		except Exception as exc:
			logger.debug("Failed to extract coords from %s run %s: %s", dlg_path, run_number, exc)
			return None

	def _extract_vina_pose_coords(self, pdbqt_path: str, model_number: int) -> Optional[np.ndarray]:
		"""Extract coordinates from one MODEL block in a Vina PDBQT."""
		atoms = self._extract_vina_pose_atoms(pdbqt_path, model_number)
		if atoms is None:
			return None
		return np.array([[x, y, z] for _name, _elem, x, y, z in atoms])

	def _extract_vina_pose_atoms(
		self, pdbqt_path: str, model_number: int
	) -> Optional[List[Tuple[str, str, float, float, float]]]:
		"""Extract (atom_name, element, x, y, z) from one MODEL block in a Vina PDBQT."""
		atoms: List[Tuple[str, str, float, float, float]] = []
		in_target = False
		try:
			with open(pdbqt_path, "r", encoding="utf-8", errors="ignore") as handle:
				for line in handle:
					if line.startswith("MODEL"):
						parts = line.split()
						current = self._parse_int(parts[1]) if len(parts) > 1 else None
						in_target = current == model_number
						continue
					if in_target and line.startswith("ENDMDL"):
						break
					if in_target and line.startswith(("ATOM", "HETATM")):
						name = line[12:16].strip()
						ad_type = line.split()[-1]
						atoms.append((
							name or "C",
							AUTODOCK_TYPE_TO_ELEMENT.get(ad_type, "C"),
							float(line[30:38]), float(line[38:46]), float(line[46:54]),
						))
			return atoms or None
		except (OSError, ValueError, IndexError) as exc:
			logger.debug("Failed to extract Vina model %s from %s: %s", model_number, pdbqt_path, exc)
			return None

	def _get_pose_coords(self, dlg_path: str, run_number: int) -> Optional[np.ndarray]:
		"""Return cached pose coordinates when available."""
		cache_key = (dlg_path, run_number)
		if cache_key not in self._pose_coords_cache:
			self._pose_coords_cache[cache_key] = self._extract_pose_coords(dlg_path, run_number)
		return self._pose_coords_cache[cache_key]

	def _write_pose_pdb(
		self,
		atoms: List[Tuple[str, str, float, float, float]],
		out_path: str,
		ligand_name: str,
	) -> None:
		"""Write a simple PDB file for the ligand pose (no connectivity), one line per atom.

		Each atom keeps its real element and atom name (from the AutoDock type in the
		source DLG/PDBQT record) instead of being written as generic carbon.
		"""
		create_folder(os.path.dirname(out_path))
		with open(out_path, "w", encoding="utf-8") as handle:
			for i, (name, elem, x, y, z) in enumerate(atoms, 1):
				atom_line = (
					"HETATM%5d %-4s %3s     1    %8.3f%8.3f%8.3f  1.00  0.00          %2s\n"
					% (i, name[:4], ligand_name[:3].upper(), x, y, z, elem)
				)
				handle.write(atom_line)

	@staticmethod
	def _rmsd_kabsch(P: np.ndarray, Q: np.ndarray) -> Optional[float]:
		"""Compute RMSD between two coordinate sets using the Kabsch algorithm.

		P and Q must have shape (N,3) and the same ordering.
		Returns None if computation cannot be performed.
		"""
		try:
			if P.shape != Q.shape or P.size == 0:
				return None
			P_cent = P - P.mean(axis=0)
			Q_cent = Q - Q.mean(axis=0)
			C = P_cent.T @ Q_cent
			V, S, Wt = np.linalg.svd(C)
			d = np.sign(np.linalg.det(V @ Wt))
			D = np.diag([1.0, 1.0, d])
			U = V @ D @ Wt
			P_rot = P_cent @ U
			diff = P_rot - Q_cent
			rmsd = float(np.sqrt((diff * diff).sum() / P.shape[0]))
			return rmsd
		except Exception:
			return None

	def _collect_all_poses(self) -> List[DockingPose]:
		"""Collect every pose for the selected engine."""
		all_poses: List[DockingPose] = []
		search_roots = []
		for candidate_root in (self.output_path, self.results_dir):
			if candidate_root and os.path.isdir(candidate_root) and candidate_root not in search_roots:
				search_roots.append(candidate_root)

		selected_engine = self.engine
		if selected_engine == "auto":
			selected_engine = "gpu" if any(
				glob.glob(os.path.join(root, "**", "*.dlg"), recursive=True)
				for root in search_roots
			) else "vina"
		self.engine = selected_engine

		pattern = "*.dlg" if selected_engine == "gpu" else "*_out.pdbqt"
		pose_files: List[str] = []
		for root in search_roots:
			pose_files.extend(glob.glob(os.path.join(root, "**", pattern), recursive=True))
		pose_files = list(dict.fromkeys(pose_files))
		logger.info("Found %d %s result file(s) to analyze", len(pose_files), selected_engine)

		for pose_path in pose_files:
			poses = self._parse_dlg(pose_path) if selected_engine == "gpu" else self._parse_vina_pdbqt(pose_path)
			all_poses.extend(poses)
			if poses:
				logger.debug("Parsed %d poses from %s", len(poses), Path(pose_path).name)

		return all_poses

	def _calculate_ligand_statistics(self, poses: List[DockingPose]) -> List[DockingLigandStats]:
		"""
		Group poses by ligand and calculate statistics.
		
		For scientists:
		- Best affinity: most negative (strongest binding)
		- Worst affinity: least negative (weakest binding)
		- Mean affinity: average energy across all poses
		- Std affinity: variability in docking solutions
		"""
		ligand_groups: Dict[tuple, List[DockingPose]] = defaultdict(list)

		# Group by protein-ligand pair
		for pose in poses:
			key = (pose.protein, pose.ligand, pose.dlg_file)
			ligand_groups[key].append(pose)

		self._ligand_pose_groups = dict(ligand_groups)

		stats_list: List[DockingLigandStats] = []

		for (protein, ligand, dlg_file), group in ligand_groups.items():
			if not group:
				continue

			affinities = [p.affinity for p in group]
			best_pose = min(group, key=lambda p: p.affinity)

			best_affinity = min(affinities)  # Most negative
			worst_affinity = max(affinities)  # Least negative
			mean_affinity = statistics.mean(affinities)
			std_affinity = statistics.pstdev(affinities) if len(affinities) > 1 else 0.0

			# Build base stats
			best_pose_obj = best_pose
			best_run_num = best_pose_obj.run if best_pose_obj is not None else None

			# QC flags (simple heuristics)
			qc_flag = "PASS"
			if len(affinities) < 10:
				qc_flag = "LOW_POSES"
			if mean_affinity > -6.0:
				qc_flag = "WEAK_BINDER"
			if std_affinity > 3.0:
				qc_flag = "HIGH_VARIABILITY"

			stats_list.append(DockingLigandStats(
				protein=protein,
				ligand=ligand,
				num_poses=len(group),
				best_run=best_pose.run,
				best_affinity=best_affinity,
				worst_affinity=worst_affinity,
				mean_affinity=mean_affinity,
				std_affinity=std_affinity,
				dlg_file=dlg_file,
				mean_rmsd=None,
				std_rmsd=None,
				qc_flag=qc_flag,
				best_pose_pdb=None,
			))

		return sorted(
			stats_list,
			key=lambda s: (s.protein, s.mean_affinity)  # Sort by protein, then by affinity
		)

	def _write_statistics_csv(self, stats: List[DockingLigandStats]) -> None:
		"""Write aggregated statistics to CSV (scientific format)."""
		with open(self.summary_csv, "w", newline="", encoding="utf-8") as handle:
			writer = csv.DictWriter(
				handle,
				fieldnames=[
					"protein", "ligand", "num_poses", "best_run", "best_affinity_kcal_mol",
					"mean_affinity_kcal_mol", "std_affinity_kcal_mol",
					"worst_affinity_kcal_mol", "dlg_file",
					"mean_rmsd", "std_rmsd", "qc_flag", "best_pose_pdb",
				],
			)
			writer.writeheader()
			for stat in stats:
				writer.writerow({
					"protein": stat.protein,
					"ligand": stat.ligand,
					"num_poses": stat.num_poses,
					"best_run": stat.best_run,
					"best_affinity_kcal_mol": f"{stat.best_affinity:.2f}",
					"mean_affinity_kcal_mol": f"{stat.mean_affinity:.2f}",
					"std_affinity_kcal_mol": f"{stat.std_affinity:.2f}",
					"worst_affinity_kcal_mol": f"{stat.worst_affinity:.2f}",
					"dlg_file": stat.dlg_file,
					"mean_rmsd": f"{stat.mean_rmsd:.2f}" if stat.mean_rmsd is not None else "",
					"std_rmsd": f"{stat.std_rmsd:.2f}" if stat.std_rmsd is not None else "",
					"qc_flag": stat.qc_flag or "",
					"best_pose_pdb": stat.best_pose_pdb or "",
				})

	def _write_legacy_statistics_csv(self, stats: List[DockingLigandStats]) -> None:
		"""Write a compatibility CSV using the schema from the reference script."""
		with open(self.legacy_csv, "w", newline="", encoding="utf-8") as handle:
			writer = csv.DictWriter(
				handle,
				fieldnames=["Molécula", "MaxScore", "MeanScore", "StanDesv"],
			)
			writer.writeheader()
			for stat in stats:
				writer.writerow({
					"Molécula": stat.ligand,
					"MaxScore": f"{stat.best_affinity:.2f}",
					"MeanScore": f"{stat.mean_affinity:.2f}",
					"StanDesv": f"{stat.std_affinity:.2f}",
				})

	def _write_ranking_csv(self, stats: List[DockingLigandStats]) -> None:
		"""Write ligand ranking by binding affinity (most negative = best)."""
		ranked = sorted(stats, key=lambda s: s.mean_affinity)

		with open(self.ranking_csv, "w", newline="", encoding="utf-8") as handle:
			writer = csv.DictWriter(
				handle,
				fieldnames=[
					"rank", "ligand", "protein", "mean_affinity_kcal_mol",
					"std_affinity_kcal_mol", "best_affinity_kcal_mol", "num_poses",
					"mean_rmsd", "std_rmsd", "qc_flag", "best_pose_pdb",
				],
			)
			writer.writeheader()
			for rank, stat in enumerate(ranked, 1):
				writer.writerow({
					"rank": rank,
					"ligand": stat.ligand,
					"protein": stat.protein,
					"mean_affinity_kcal_mol": f"{stat.mean_affinity:.2f}",
					"std_affinity_kcal_mol": f"{stat.std_affinity:.2f}",
					"best_affinity_kcal_mol": f"{stat.best_affinity:.2f}",
					"num_poses": stat.num_poses,
					"mean_rmsd": f"{stat.mean_rmsd:.2f}" if stat.mean_rmsd is not None else "",
					"std_rmsd": f"{stat.std_rmsd:.2f}" if stat.std_rmsd is not None else "",
					"qc_flag": stat.qc_flag or "",
					"best_pose_pdb": stat.best_pose_pdb or "",
				})

	def _write_dynamics_candidates_csv(self, stats: List[DockingLigandStats], top_n: int = 10) -> None:
		"""Write the ligands that should be considered next for dynamics.

		This step does not start dynamics; it only exports ranked candidates so
		that a downstream dynamics step can consume a clean, isolated input.
		"""
		ranked = sorted(stats, key=lambda s: (s.mean_affinity, s.best_affinity))[:top_n]

		with open(self.dynamics_candidates_csv, "w", newline="", encoding="utf-8") as handle:
			writer = csv.DictWriter(
				handle,
				fieldnames=[
					"rank", "protein", "ligand", "mean_affinity_kcal_mol",
					"std_affinity_kcal_mol", "best_affinity_kcal_mol",
					"best_run", "num_poses", "dlg_file", "qc_flag", "best_pose_pdb",
				],
			)
			writer.writeheader()
			for rank, stat in enumerate(ranked, 1):
				writer.writerow({
					"rank": rank,
					"protein": stat.protein,
					"ligand": stat.ligand,
					"mean_affinity_kcal_mol": f"{stat.mean_affinity:.2f}",
					"std_affinity_kcal_mol": f"{stat.std_affinity:.2f}",
					"best_affinity_kcal_mol": f"{stat.best_affinity:.2f}",
					"best_run": stat.best_run,
					"num_poses": stat.num_poses,
					"dlg_file": stat.dlg_file,
					"qc_flag": stat.qc_flag or "",
					"best_pose_pdb": stat.best_pose_pdb or "",
				})

	def _export_best_candidate_pdbs(self, stats: List[DockingLigandStats], top_n: int = 10) -> None:
		"""Compute RMSD and export PDBs only for the top ranked candidates.

		This keeps the analysis fast by limiting the expensive pose reconstruction to
		the ligands that actually matter for the next dynamics step.
		"""
		ranked = sorted(stats, key=lambda s: (s.mean_affinity, s.best_affinity))[:top_n]
		pdb_dir = os.path.join(self.analysis_dir, "poses_for_md")
		create_folder(pdb_dir)
		stats_by_key = {(s.protein, s.ligand, s.dlg_file): s for s in stats}

		# Parallel export of PDBs using ThreadPoolExecutor (I/O-bound task)
		def _export_single_pdb_task(stat: DockingLigandStats) -> Tuple[Tuple[str, str, str], Optional[str]]:
			"""Helper: compute RMSD and export PDB for a single ligand."""
			best_coords = self._get_pose_coords(stat.dlg_file, stat.best_run)
			if best_coords is None:
				return (stat.protein, stat.ligand, stat.dlg_file), (None, None, None)

			rmsds: List[float] = []
			group_key = (stat.protein, stat.ligand, stat.dlg_file)
			for candidate in self._ligand_pose_groups.get(group_key, []):
				coords = self._get_pose_coords(candidate.dlg_file, candidate.run)
				if coords is None or coords.shape != best_coords.shape:
					continue
				rmsd_value = self._rmsd_kabsch(best_coords, coords)
				if rmsd_value is not None:
					rmsds.append(rmsd_value)

			pdb_name = f"{stat.protein}_{stat.ligand}_best_run_{stat.best_run}.pdb"
			pdb_path = os.path.join(pdb_dir, pdb_name)
			best_atoms = self._extract_pose_atoms(stat.dlg_file, stat.best_run)
			self._write_pose_pdb(best_atoms or [], pdb_path, stat.ligand)
			mean_rmsd = float(statistics.mean(rmsds)) if rmsds else None
			std_rmsd = float(statistics.pstdev(rmsds)) if len(rmsds) > 1 else 0.0 if rmsds else None
			return (stat.protein, stat.ligand, stat.dlg_file), (pdb_path, mean_rmsd, std_rmsd)

		with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
			futures = [executor.submit(_export_single_pdb_task, stat) for stat in ranked]
			for future in futures:
				key, payload = future.result()
				if key in stats_by_key and payload is not None:
					pdb_path, mean_rmsd, std_rmsd = payload
					updated = stats_by_key[key]
					stats_by_key[key] = DockingLigandStats(
						protein=updated.protein,
						ligand=updated.ligand,
						num_poses=updated.num_poses,
						best_run=updated.best_run,
						best_affinity=updated.best_affinity,
						worst_affinity=updated.worst_affinity,
						mean_affinity=updated.mean_affinity,
						std_affinity=updated.std_affinity,
						dlg_file=updated.dlg_file,
						mean_rmsd=mean_rmsd,
						std_rmsd=std_rmsd,
						qc_flag=updated.qc_flag,
						best_pose_pdb=pdb_path,
					)

		# Persist the updated best_pose_pdb values back into the original list order.
		for index, stat in enumerate(stats):
			key = (stat.protein, stat.ligand, stat.dlg_file)
			if key in stats_by_key:
				stats[index] = stats_by_key[key]

	def _write_text_summary(self, stats: List[DockingLigandStats], poses: List[DockingPose]) -> None:
		"""Write human-readable analysis summary for scientists."""
		with open(self.summary_txt, "w", encoding="utf-8") as handle:
			# Header
			handle.write("=" * 80 + "\n")
			handle.write("DOCKING ANALYSIS SUMMARY - ENERGY STATISTICS\n")
			handle.write(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
			handle.write("=" * 80 + "\n\n")

			# Global statistics
			if poses:
				all_affinities = [p.affinity for p in poses]
				global_mean = statistics.mean(all_affinities)
				global_std = statistics.stdev(all_affinities) if len(all_affinities) > 1 else 0.0
				global_best = min(all_affinities)
				global_worst = max(all_affinities)

				handle.write("GLOBAL STATISTICS (All Poses)\n")
				handle.write("-" * 80 + "\n")
				handle.write(f"Total poses analyzed: {len(all_affinities)}\n")
				handle.write(f"Total docking runs: {len(stats)}\n")
				handle.write(f"Mean binding affinity: {global_mean:>8.2f} kcal/mol\n")
				handle.write(f"Std deviation: {global_std:>8.2f} kcal/mol\n")
				handle.write(f"Best (most negative): {global_best:>8.2f} kcal/mol\n")
				handle.write(f"Worst (least negative): {global_worst:>8.2f} kcal/mol\n")
				handle.write(f"Energy range: {(global_worst - global_best):>8.2f} kcal/mol\n")
				handle.write("\n")

			# Per-ligand analysis
			handle.write("PER-LIGAND ANALYSIS\n")
			handle.write("-" * 80 + "\n\n")

			for i, stat in enumerate(stats, 1):
				handle.write(f"{i}. {stat.ligand}\n")
				handle.write(f"   Protein: {stat.protein}\n")
				handle.write(f"   Poses: {stat.num_poses}\n")
				handle.write(f"   Best affinity (strongest binding): {stat.best_affinity:>8.2f} kcal/mol\n")
				handle.write(f"   Mean affinity: {stat.mean_affinity:>8.2f} kcal/mol ± {stat.std_affinity:.2f}\n")
				handle.write(f"   Worst affinity (weakest binding): {stat.worst_affinity:>8.2f} kcal/mol\n")
				handle.write("\n")

			# Ranking table
			handle.write("\n" + "=" * 80 + "\n")
			handle.write("LIGAND RANKING BY MEAN BINDING AFFINITY\n")
			handle.write("=" * 80 + "\n")
			handle.write(f"{'Rank':<6} {'Ligand':<30} {'Mean Affinity':<18} {'±':<8} {'Poses':<8}\n")
			handle.write("-" * 80 + "\n")

			ranked = sorted(stats, key=lambda s: s.mean_affinity)
			for rank, stat in enumerate(ranked, 1):
				ligand_short = stat.ligand[:28]
				handle.write(
					f"{rank:<6} {ligand_short:<30} {stat.mean_affinity:>8.2f} kcal/mol  "
					f"± {stat.std_affinity:<6.2f}  {stat.num_poses:<8}\n"
				)

			# Interpretation
			handle.write("\n" + "=" * 80 + "\n")
			handle.write("INTERPRETATION GUIDE\n")
			handle.write("=" * 80 + "\n")
			handle.write("""
• AFFINITY (kcal/mol): Binding free energy. More negative = more favorable.
  - < -10.0: Strong binding (potentially good lead)
  - -8 to -10: Moderate binding (promising)
  - -6 to -8: Weak binding
  - > -6: Very weak binding

• STD AFFINITY: Variability across poses. Lower is better (more consistent).
  - < 2.0: Consistent docking solutions
  - > 3.0: Variable results (may need re-docking with different parameters)

• RANKING: Based on mean affinity. Best leads listed first.
""")

	def _write_markdown_summary(self, stats: List[DockingLigandStats]) -> None:
		"""Write markdown report for quick inspection and documentation."""
		with open(self.summary_md, "w", encoding="utf-8") as handle:
			handle.write("# Docking Analysis Report\n\n")
			handle.write(f"**Generated:** {datetime.now():%Y-%m-%d %H:%M:%S}\n\n")

			handle.write("## Summary Statistics\n\n")
			handle.write(f"**Total ligands analyzed:** {len(stats)}\n")
			handle.write(f"**Total docking poses:** {sum(s.num_poses for s in stats)}\n\n")

			handle.write("## Top 10 Ligands by Binding Affinity\n\n")
			handle.write("| Rank | Ligand | Mean Affinity | Std Dev | Best Pose | Poses |\n")
			handle.write("|---:|---|---:|---:|---:|---:|\n")

			ranked = sorted(stats, key=lambda s: s.mean_affinity)[:10]
			for rank, stat in enumerate(ranked, 1):
					handle.write(
						f"| {rank} | {stat.ligand} | {stat.mean_affinity:.2f} | "
						f"{stat.std_affinity:.2f} | {stat.best_affinity:.2f} | {stat.num_poses} |\n"
					)

			handle.write("\n## Output Files\n\n")
			handle.write("- **docking_statistics.csv**: Aggregated statistics per ligand\n")
			handle.write("- **ligand_ranking.csv**: Ranked ligands by binding affinity\n")
			handle.write("- **docking_analysis_summary.txt**: Detailed text report\n")

	def _export_vina_dynamics_inputs(self, stats: List[DockingLigandStats]) -> Optional[str]:
		"""Collect Vina's already-exported best-pose complexes for dynamics."""
		output = Path(self.analysis_dir) / "dynamics_inputs"
		receptor_out = output / "receptors"
		ligand_out = output / "ligands"
		complex_out = output / "complexes"
		for directory in (receptor_out, ligand_out, complex_out):
			directory.mkdir(parents=True, exist_ok=True)

		manifest_rows = []
		for rank, stat in enumerate(
			sorted(stats, key=lambda item: (item.mean_affinity, item.best_affinity))[:self.pdb_export_limit],
			1,
		):
			stem = f"{stat.protein}_{stat.ligand}"
			source_dir = Path(self.results_dir) / stat.protein / "complex"
			ligand_source = source_dir / f"{stem}_ligand.pdb"
			complex_source = source_dir / f"{stem}_complex.pdb"
			clean_roots = [Path(self.output_path) / "clean_receptors", Path(self.output_path).parent / "clean_receptors"]
			receptor_source = next((
				candidate
				for root in clean_roots
				for candidate in (root / f"{stat.protein}_h.pdb", root / f"{stat.protein}.pdb")
				if candidate.is_file()
			), None)
			if not ligand_source.is_file() or not complex_source.is_file() or receptor_source is None:
				continue
			receptor_target = receptor_out / f"{stat.protein}.pdb"
			ligand_target = ligand_out / f"rank_{rank:02d}_{stem}_ligand.pdb"
			complex_target = complex_out / f"rank_{rank:02d}_{stem}_complex.pdb"
			if not receptor_target.exists():
				shutil.copy2(receptor_source, receptor_target)
			shutil.copy2(ligand_source, ligand_target)
			shutil.copy2(complex_source, complex_target)
			manifest_rows.append({
				"rank": rank, "protein": stat.protein, "ligand": stat.ligand,
				"best_affinity_kcal_mol": f"{stat.best_affinity:.3f}",
				"receptor_pdb": str(receptor_target), "ligand_pdb": str(ligand_target),
				"complex_pdb": str(complex_target),
			})

		if not manifest_rows:
			return None
		with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
			writer = csv.DictWriter(handle, fieldnames=manifest_rows[0].keys())
			writer.writeheader()
			writer.writerows(manifest_rows)
		return str(output)

	def run(self) -> Dict[str, Any]:
		"""Execute full docking analysis pipeline."""
		logger.info("Starting docking analysis in %s", self.results_dir)
		create_folder(self.analysis_dir)

		# Collect and parse all poses
		all_poses = self._collect_all_poses()
		if not all_poses:
			label = "AutoDock-GPU DLG" if self.engine == "gpu" else "Vina PDBQT"
			logger.warning("No %s files or poses found for analysis under %s", label, self.output_path)
			return {"parsed_poses": 0, "analyzed_ligands": 0}

		logger.info("Parsed %d %s poses", len(all_poses), self.engine)

		# Calculate statistics
		ligand_stats = self._calculate_ligand_statistics(all_poses)
		logger.info("Calculated statistics for %d ligand docking runs", len(ligand_stats))

		# Export PDBs only for the best-ranked candidates before writing CSVs
		self._export_best_candidate_pdbs(ligand_stats, top_n=self.pdb_export_limit)

		# Generate outputs
		self._write_statistics_csv(ligand_stats)
		self._write_legacy_statistics_csv(ligand_stats)
		self._write_ranking_csv(ligand_stats)
		self._write_dynamics_candidates_csv(ligand_stats)
		self._write_text_summary(ligand_stats, all_poses)
		self._write_markdown_summary(ligand_stats)

		# Prepare ready-to-consume receptor, ligand-pose, and complex PDB files for
		# the downstream molecular-dynamics workflow.  Analysis can also be run on
		# imported/legacy DLG directories that do not contain ChemLink's prepared
		# receptor, so keep the statistical outputs usable in that case.
		dynamics_inputs_dir = None
		if self.engine == "vina":
			dynamics_inputs_dir = self._export_vina_dynamics_inputs(ligand_stats)
		else:
			dynamics_inputs_dir = os.path.join(self.analysis_dir, "dynamics_inputs")
			try:
				export_docking_complexes(
					Path(self.output_path),
					Path(dynamics_inputs_dir),
					self.pdb_export_limit,
					candidates_csv=Path(self.dynamics_candidates_csv),
				)
			except (FileNotFoundError, RuntimeError) as exc:
				logger.warning("Could not export docking complexes for dynamics: %s", exc)
				dynamics_inputs_dir = None

		logger.info("Docking analysis completed successfully")
		logger.info("  - Statistics: %s", self.summary_csv)
		logger.info("  - Legacy statistics: %s", self.legacy_csv)
		logger.info("  - Ranking: %s", self.ranking_csv)
		logger.info("  - Dynamics candidates: %s", self.dynamics_candidates_csv)
		logger.info("  - Report: %s", self.summary_txt)

		return {
			"parsed_poses": len(all_poses),
			"analyzed_ligands": len(ligand_stats),
			"outputs": {
				"statistics_csv": self.summary_csv,
				"legacy_statistics_csv": self.legacy_csv,
				"ranking_csv": self.ranking_csv,
				"dynamics_candidates_csv": self.dynamics_candidates_csv,
				"poses_for_md_dir": os.path.join(self.analysis_dir, "poses_for_md"),
				"dynamics_inputs_dir": dynamics_inputs_dir,
				"text_report": self.summary_txt,
				"markdown_report": self.summary_md,
			}
		}
