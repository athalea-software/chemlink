"""Adapter for AutoDock Vina command-line execution (CPU docking).
"""

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class VinaPose:
	"""One docked pose (one MODEL block) from the Vina output PDBQT."""

	mode: int
	affinity: float  
	rmsd_lb: float
	rmsd_ub: float


@dataclass(frozen=True)
class VinaDockingResult:
	"""Structured docking output metadata for a single receptor/ligand run."""

	receptor: str
	ligand: str
	out_file: str
	log_file: Optional[str]
	poses: List[VinaPose]

	@property
	def best_affinity(self) -> Optional[float]:
		"""Affinity of the top-ranked pose (Vina sorts best-first)."""
		return self.poses[0].affinity if self.poses else None


class AutoDockVinaAdapter:
	"""Wrapper around AutoDock Vina binary invocation."""

	
	DEFAULT_EXECUTABLE = os.environ.get("VINA_BIN", "vina")
	DEFAULT_FALLBACKS = (
		os.environ.get("VINA_BIN", "vina"),
		"/usr/local/bin/vina",
	)

	def __init__(self, executable: Optional[str] = None):
		self.executable = executable or self.DEFAULT_EXECUTABLE
		self.executable = self._resolve_executable(self.executable, self.DEFAULT_FALLBACKS)

	@staticmethod
	def _resolve_executable(executable: str, fallbacks: Sequence[str] = ()) -> str:
		"""Resolve executable from explicit path, PATH, or fallback locations.
		An absolute path wins if it exists, otherwise look it up on PATH
		(this is what finds /usr/local/bin/vina in the container), otherwise
		fall back to known install locations.
		"""
		if os.path.isabs(executable) and os.path.isfile(executable):
			return executable

		resolved = shutil.which(executable)
		if resolved:
			return resolved

		for candidate in fallbacks:
			if os.path.isfile(candidate):
				return candidate

		return executable

	def validate(self) -> None:
		"""Ensure the Vina binary exists and is executable."""
		if not os.path.isfile(self.executable):
			raise FileNotFoundError(f"AutoDock Vina not found: {self.executable}")
		if not os.access(self.executable, os.X_OK):
			raise RuntimeError(f"AutoDock Vina is not executable: {self.executable}")

	@staticmethod
	def _parse_out_pdbqt(out_path: str) -> List[VinaPose]:
		"""Extract poses from a Vina output PDBQT.

		The file interleaves MODEL headers and REMARK VINA RESULT lines:

		    MODEL 1
		    REMARK VINA RESULT:    -7.523    0.000    0.000
		    ...atoms...
		    ENDMDL

		We track the current MODEL number and read the three floats that
		follow the "REMARK VINA RESULT:" marker on the next matching line.
		"""
		poses: List[VinaPose] = []
		current_mode = 0

		with open(out_path, "r", encoding="utf-8", errors="ignore") as handle:
			for line in handle:
				if line.startswith("MODEL"):
					parts = line.split()
					
					current_mode = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else current_mode + 1
				elif line.startswith("REMARK VINA RESULT:"):
					
					numbers = line.split("REMARK VINA RESULT:")[1].split()
					if len(numbers) >= 3:
						poses.append(
							VinaPose(
								mode=current_mode or (len(poses) + 1),
								affinity=float(numbers[0]),
								rmsd_lb=float(numbers[1]),
								rmsd_ub=float(numbers[2]),
							)
						)

		return poses

	def dock(
		self,
		receptor_pdbqt: str,
		ligand_pdbqt: str,
		output_prefix: str,
		center: Tuple[float, float, float],
		box_size: Tuple[float, float, float],
		exhaustiveness: int = 8,
		num_modes: int = 9,
		cpu: Optional[int] = None,
		seed: Optional[int] = None,
		timeout_seconds: int = 600,
	) -> VinaDockingResult:
		"""Execute one Vina docking job and return structured metadata.

		center/box_size are (x, y, z) triples in angstroms describing the
		search box -- the same values you'd otherwise put in a Vina config
		file. `exhaustiveness` trades runtime for thoroughness of the search.
		"""

		output_prefix = os.path.abspath(output_prefix)
		output_dir = os.path.dirname(output_prefix) or os.getcwd()
		output_name = os.path.basename(output_prefix)
		os.makedirs(output_dir, exist_ok=True)

		out_file = os.path.join(output_dir, f"{output_name}_out.pdbqt")
		log_file = os.path.join(output_dir, f"{output_name}.log")

		for path in (out_file, log_file):
			if os.path.exists(path):
				os.remove(path)
    
		cx, cy, cz = center
		sx, sy, sz = box_size
		cmd: List[str] = [
			self.executable,
			"--receptor", receptor_pdbqt,
			"--ligand", ligand_pdbqt,
			"--center_x", str(cx),
			"--center_y", str(cy),
			"--center_z", str(cz),
			"--size_x", str(sx),
			"--size_y", str(sy),
			"--size_z", str(sz),
			"--exhaustiveness", str(exhaustiveness),
			"--num_modes", str(num_modes),
			"--out", out_file,
		]
		if cpu is not None:
			cmd += ["--cpu", str(cpu)]
		if seed is not None:
			cmd += ["--seed", str(seed)]

		result = subprocess.run(
			cmd,
			capture_output=True,
			text=True,
			timeout=timeout_seconds,
			cwd=output_dir,
		)

		if result.stdout:
			with open(log_file, "w", encoding="utf-8") as handle:
				handle.write(result.stdout)

		if result.returncode != 0:
			raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "AutoDock Vina failed")

		if not os.path.isfile(out_file):
			stdout_tail = (result.stdout or "").strip()[-600:]
			stderr_tail = (result.stderr or "").strip()[-600:]
			raise RuntimeError(
				f"Output PDBQT was not generated: {out_file}. "
				f"stdout_tail={stdout_tail!r} stderr_tail={stderr_tail!r}"
			)

		poses = self._parse_out_pdbqt(out_file)
		return VinaDockingResult(
			receptor=Path(receptor_pdbqt).stem,
			ligand=Path(ligand_pdbqt).stem,
			out_file=out_file,
			log_file=log_file if os.path.isfile(log_file) else None,
			poses=poses,
		)
