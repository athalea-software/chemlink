"""Adapter for AutoDock-GPU command-line execution."""

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


@dataclass(frozen=True)
class DockingResult:
	"""Structured docking output metadata."""

	protein: str
	ligand: str
	dlg_file: str
	xml_file: Optional[str]
	affinity: str
	run: str
	rmsd: str


class AutoDockGPUAdapter:
	"""Wrapper around AutoDock-GPU binary invocation."""

	DEFAULT_EXECUTABLE = os.environ.get("AUTODOCK_GPU_BIN", "/usr/local/bin/autodock-gpu")
	DEFAULT_FALLBACKS = (
		os.environ.get("AUTODOCK_GPU_BIN", "/usr/local/bin/autodock-gpu"),
		"/nfs/chemlink/software/autodock-gpu/bin/autodock-gpu",
		"/usr/local/bin/autodock-gpu",
		"/usr/local/bin/adgpu-v1.6_linux_x64_cuda12_128wi",
	)

	def __init__(self, executable: Optional[str] = None):
		self.executable = executable or self.DEFAULT_EXECUTABLE
		self.executable = self._resolve_executable(self.executable, self.DEFAULT_FALLBACKS)

	@staticmethod
	def _resolve_executable(executable: str, fallbacks: Sequence[str] = ()) -> str:
		"""Resolve executable from explicit path, PATH, or fallback locations."""
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
		"""Ensure AutoDock-GPU exists and is executable."""
		if not os.path.isfile(self.executable):
			raise FileNotFoundError(f"AutoDock-GPU not found: {self.executable}")
		if not os.access(self.executable, os.X_OK):
			raise RuntimeError(f"AutoDock-GPU is not executable: {self.executable}")

	@staticmethod
	def _parse_rmsd_table(dlg_path: str) -> Dict[str, str]:
		"""Extract a simple summary row from an AutoDock DLG file."""
		affinity = "N/A"
		run = "N/A"
		rmsd = "N/A"

		with open(dlg_path, "r", encoding="utf-8", errors="ignore") as handle:
			content = handle.read().splitlines()

		for index, line in enumerate(content):
			if "RMSD TABLE" not in line:
				continue
			for candidate in content[index + 1 : index + 25]:
				parts = candidate.split()
				if len(parts) >= 5 and parts[0].isdigit():
					run = parts[0]
					affinity = parts[3]
					rmsd = parts[4]
					return {"run": run, "affinity": affinity, "rmsd": rmsd}

		return {"run": run, "affinity": affinity, "rmsd": rmsd}

	def dock(
		self,
		protein_map: str,
		ligand_file: str,
		output_prefix: str,
		run_count: int = 1000,
		lsmet: str = "ad",
		nev: int = 2_500_000,
		timeout_seconds: int = 600,
		gpu_id: Optional[int] = None,
		flexres_file: Optional[str] = None,
	) -> DockingResult:
		"""Execute one docking job and return structured metadata."""
		output_prefix = os.path.abspath(output_prefix)
		output_dir = os.path.dirname(output_prefix) or os.getcwd()
		output_name = os.path.basename(output_prefix)
		os.makedirs(output_dir, exist_ok=True)

		dlg_file = os.path.join(output_dir, f"{output_name}.dlg")
		xml_file = os.path.join(output_dir, f"{output_name}.xml")

		for path in (dlg_file, xml_file):
			if os.path.exists(path):
				os.remove(path)

		cmd: List[str] = [
			self.executable,
			"-ffile",
			protein_map,
			"-lfile",
			ligand_file,
			"-nrun",
			str(run_count),
			"-lsmet",
			lsmet,
			"-resnam",
			output_name,
			"-nev",
			str(nev),
		]
		if flexres_file is not None:
			cmd += ["--flexres", flexres_file]

		env = None
		if gpu_id is not None:
			env = os.environ.copy()
			env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

		result = subprocess.run(
			cmd,
			capture_output=True,
			text=True,
			timeout=timeout_seconds,
			env=env,
			cwd=output_dir,
		)

		if result.returncode != 0:
			raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "AutoDock-GPU failed")

		if not os.path.isfile(dlg_file):
			stdout_tail = (result.stdout or "").strip()[-600:]
			stderr_tail = (result.stderr or "").strip()[-600:]
			raise RuntimeError(
				f"DLG file was not generated: {dlg_file}. "
				f"stdout_tail={stdout_tail!r} stderr_tail={stderr_tail!r}"
			)

		summary = self._parse_rmsd_table(dlg_file)
		return DockingResult(
			protein=Path(protein_map).stem,
			ligand=Path(ligand_file).stem,
			dlg_file=dlg_file,
			xml_file=xml_file if os.path.isfile(xml_file) else None,
			affinity=summary["affinity"],
			run=summary["run"],
			rmsd=summary["rmsd"],
		)
