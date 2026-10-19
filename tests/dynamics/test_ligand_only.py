from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from cli.main import _DYN_SPEC, _resolve_dynamics_input_sets
from pipelines.dynamics.steps.ligand_topology import LigandTopologyStep

ACPYPE_ITP = (
    "[ atomtypes ]\n"
    "; name at.num mass charge ptype sigma epsilon\n"
    "c3 6 12.01 0.0 A 0.33 0.45\n"
    "\n[ moleculetype ]\nLIG 3\n\n[ atoms ]\n1 c3 1 LIG C1 1 0.0 12.01\n"
)
ACPYPE_GRO = (
    "LIG\n    2\n"
    "    1LIG     C1    1   0.100   0.100   0.100\n"
    "    1LIG     C2    2   0.200   0.100   0.100\n"
    "   3.00000   3.00000   3.00000\n"
)


class LigandOnlyTests(TestCase):
    def test_spec_registered(self):
        self.assertEqual(_DYN_SPEC["oligand"][0], "7")

    def test_inputs_accept_single_ligand_only(self):
        sets, _ = _resolve_dynamics_input_sets("oligand", ligand_inputs=["lig.sdf"])
        self.assertEqual(len(sets[0]), 1)
        for kwargs in (
            dict(receptor_inputs=["p.pdb"], ligand_inputs=["lig.sdf"]),
            dict(ligand_inputs=["a.sdf", "b.sdf"]),
            dict(ligand_inputs=["lig.sdf"], cofactor_inputs=["zn.pdb"]),
        ):
            with self.assertRaises(ValueError):
                _resolve_dynamics_input_sets("oligand", **kwargs)

    def test_standalone_topology_has_no_protein(self):
        with TemporaryDirectory() as directory:
            work = Path(directory)
            (work / "LIG_GMX.itp").write_text(ACPYPE_ITP)
            (work / "LIG_GMX.gro").write_text(ACPYPE_GRO)
            (work / "lig.pdb").write_text("")
            step = LigandTopologyStep(
                {"work_dir": str(work), "ligand_pdb": str(work / "lig.pdb"), "ligand_charge": 0},
                "gmx", standalone=True,
            )
            step._write_standalone_system(str(work / "LIG_GMX.gro"), str(work / "LIG_GMX.itp"))

            top = (work / "topol.top").read_text()
            self.assertIn('#include "ligand.itp"', top)
            self.assertIn("c3 6 12.01", top)
            self.assertIn("[ molecules ]\nLIG                1", top)
            self.assertLess(top.index("forcefield.itp"), top.index("c3 6"))
            gro = (work / "processed.gro").read_text().splitlines()
            self.assertEqual(gro[1].strip(), "2")
            self.assertEqual(len(gro), 5)
