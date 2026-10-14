"""Adapter for fpocket command-line execution."""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional


class FpocketAdapter:
	"""Wrapper around fpocket for pocket detection."""

	DEFAULT_FPOCKET_PATH = "/usr/bin/fpocket"
	FPOCKET_CANDIDATES = (
		"/usr/local/bin/fpocket",
		"/usr/bin/fpocket",
	)

	def __init__(self, fpocket_path: Optional[str] = None):
		self.fpocket_path = self._resolve_fpocket_path(fpocket_path)

	@classmethod
	def _resolve_fpocket_path(cls, fpocket_path: Optional[str]) -> str:
		"""Resolve fpocket executable from explicit path, $FPOCKET_BIN, PATH, or known locations."""
		fpocket_path = fpocket_path or os.environ.get("FPOCKET_BIN")
		if fpocket_path:
			if os.path.isabs(fpocket_path):
				return fpocket_path
			resolved = shutil.which(fpocket_path)
			if resolved:
				return resolved
			return fpocket_path

		resolved = shutil.which("fpocket")
		if resolved:
			return resolved

		for candidate in cls.FPOCKET_CANDIDATES:
			if os.path.isfile(candidate):
				return candidate

		return cls.DEFAULT_FPOCKET_PATH

	def validate(self) -> None:
		"""Ensure fpocket is available and executable."""
		if not os.path.isfile(self.fpocket_path):
			raise FileNotFoundError(f"fpocket not found: {self.fpocket_path}")
		if not os.access(self.fpocket_path, os.X_OK):
			raise RuntimeError(f"fpocket is not executable: {self.fpocket_path}")

	@staticmethod
	def strip_pdbqt_to_pdb(input_pdbqt: str, output_pdb: str) -> None:
		"""Create a PDB-like file from PDBQT for fpocket."""
		kept = 0
		with open(input_pdbqt, "r", encoding="utf-8", errors="ignore") as src, open(
			output_pdb, "w", encoding="utf-8"
		) as dst:
			for line in src:
				if line.startswith(("ATOM", "HETATM", "TER", "END")):
					dst.write(line[:66].rstrip() + "\n")
					kept += 1

		if kept == 0:
			raise RuntimeError(f"No ATOM/HETATM records found in {input_pdbqt}")

	def detect_pocket(self, receptor_file: str, output_dir: str) -> str:
		"""Run fpocket on one receptor and persist the pocket1 atoms file."""
		receptor_name = Path(receptor_file).stem
		os.makedirs(output_dir, exist_ok=True)
		with tempfile.TemporaryDirectory(prefix=f"fpocket_{receptor_name}_") as tmpdir:
			pdb_input = os.path.join(tmpdir, f"{receptor_name}.pdb")
			self.strip_pdbqt_to_pdb(receptor_file, pdb_input)

			result = subprocess.run(
				[self.fpocket_path, "-f", pdb_input],
				capture_output=True,
				text=True,
				cwd=tmpdir,
			)

			if result.returncode != 0:
				raise RuntimeError(f"fpocket failed for {receptor_name}: {result.stderr.strip()}")

			pocket1 = os.path.join(tmpdir, f"{receptor_name}_out", "pockets", "pocket1_atm.pdb")
			if not os.path.isfile(pocket1):
				raise RuntimeError(f"fpocket output missing pocket1_atm.pdb for {receptor_name}")

			output_pocket = os.path.join(output_dir, f"{receptor_name}_pocket_atm1.pdb")
			shutil.copy2(pocket1, output_pocket)
			return output_pocket
