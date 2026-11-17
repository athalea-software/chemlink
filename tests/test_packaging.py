"""The wheel must ship every module and the data files read at run time."""

import re
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_NOT_PACKAGED = {"tests", "docs", "data", "paper", "build", "dist"}


def _declared_packages() -> set:
    text = (_REPO_ROOT / "pyproject.toml").read_text()
    block = re.search(r"^packages = \[(.*?)^\]", text, re.S | re.M)
    return set(re.findall(r'"(chemlink[\w.]*)"', block.group(1)))


def _package_dirs() -> set:
    found = set()
    for py in _REPO_ROOT.rglob("*.py"):
        rel = py.relative_to(_REPO_ROOT).parent
        if rel.parts and (rel.parts[0] in _NOT_PACKAGED or rel.parts[0].startswith(".")):
            continue
        if "__pycache__" in rel.parts or any(p.endswith(".egg-info") for p in rel.parts):
            continue
        found.add(".".join(("chemlink",) + rel.parts))
    return found


class PackagingTest(unittest.TestCase):
    def test_every_module_directory_is_declared(self):
        missing = _package_dirs() - _declared_packages()
        self.assertFalse(missing, f"add to [tool.setuptools] packages in pyproject.toml: {sorted(missing)}")

    def test_declared_packages_exist(self):
        stale = [name for name in _declared_packages()
                 if not (_REPO_ROOT.joinpath(*name.split(".")[1:])).is_dir()]
        self.assertFalse(stale, f"remove from pyproject.toml packages: {sorted(stale)}")

    def test_runtime_data_files_exist(self):
        for rel in ("hpc/slurm/native/run_multinode_pipeline.sh",
                    "hpc/slurm/native/run_dynamics_pipeline.sh",
                    "hpc/slurm/container/run_multinode_pipeline.sh",
                    "config/environments/gmx_mmpbsa.yml"):
            self.assertTrue((_REPO_ROOT / rel).is_file(), rel)


if __name__ == "__main__":
    unittest.main()
