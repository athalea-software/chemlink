import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from pipelines.dynamics.utils import find_gmx_path, resolve_gmx_binary


class GromacsBinaryResolutionTests(TestCase):
    def _executable(self, directory: str, name: str) -> str:
        path = Path(directory) / name
        path.write_text("#!/bin/sh\n", encoding="utf-8")
        path.chmod(0o755)
        return str(path)

    def test_finds_serial_gmx_from_override(self):
        with TemporaryDirectory() as directory:
            expected = self._executable(directory, "gmx")
            with patch.dict(os.environ, {"CHEMLINK_GMX_BIN": expected}, clear=False):
                self.assertEqual(find_gmx_path(), expected)

    def test_finds_serial_gmx_next_to_mpi_binary(self):
        with TemporaryDirectory() as directory:
            expected = self._executable(directory, "gmx")
            mpi = self._executable(directory, "gmx_mpi")
            environment = {
                "CHEMLINK_GMX_MPI_BIN": mpi,
                "CHEMLINK_GMX_BIN": "",
                "PATH": "",
                "GMXBIN": "",
                "CONDA_PREFIX": "",
            }
            def isolated_executable_path(candidate):
                candidate = str(candidate) if candidate else ""
                if candidate in {mpi, expected, "gmx_mpi"}:
                    return mpi if candidate in {mpi, "gmx_mpi"} else expected
                return None

            with patch.dict(os.environ, environment, clear=False), patch(
                "pipelines.dynamics.utils._executable_path",
                side_effect=isolated_executable_path,
            ):
                self.assertEqual(find_gmx_path(), expected)

    def test_require_serial_does_not_return_gmx_mpi(self):
        with TemporaryDirectory() as directory:
            mpi = self._executable(directory, "gmx_mpi")
            environment = {
                "CHEMLINK_GMX_MPI_BIN": mpi,
                "CHEMLINK_GMX_BIN": "",
                "PATH": "",
                "GMXBIN": "",
                "CONDA_PREFIX": "",
            }
            with patch.dict(os.environ, environment, clear=False), patch(
                "pipelines.dynamics.utils._gmx_candidates", return_value=[]
            ):
                self.assertIsNone(resolve_gmx_binary(require_serial=True))
