from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from chemlink.pipelines.dynamics.steps.cofactor_topology import CofactorTopologyStep
from chemlink.utils.cofactor_classifier import format_gro_atom_line

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FORCEFIELDS_DIR = _REPO_ROOT / "forcefields"


def _pdb_ion_line(name: str, resname: str, x: float, y: float, z: float, element: str) -> str:
    return (
        "HETATM"
        f"{1:>5}"
        " "
        f"{name:<4}"
        " "
        f"{resname:>3}"
        " A"
        f"{1:>4}"
        "    "
        f"{x:>8.3f}{y:>8.3f}{z:>8.3f}"
        f"{1.00:>6.2f}{0.00:>6.2f}"
        "          "
        f"{element:>2}"
        "\n"
    )


class CofactorTopologyIonBranchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.work_dir = Path(self.tmp.name)

        (self.work_dir / "complex.gro").write_text(
            "Complex System\n"
            "2\n"
            + format_gro_atom_line(1, "SOL", "OW", 1, 0.0, 0.0, 0.0)
            + format_gro_atom_line(1, "SOL", "HW1", 2, 0.1, 0.0, 0.0)
            + "   5.00000   5.00000   5.00000\n"
        )
        (self.work_dir / "topol.top").write_text(
            "; example topology\n"
            '#include "charmm36-jul2022.ff/forcefield.itp"\n'
            "\n"
            "[ molecules ]\n"
            "; Compound        #mols\n"
            "Protein_chain_A    1\n"
            "LIG                1\n"
        )
        (self.work_dir / "zn.pdb").write_text(
            _pdb_ion_line("ZN", "ZN", 10.0, 20.0, 30.0, "ZN")
        )

        self.config = {
            "work_dir": str(self.work_dir),
            "cofactor_pdb": str(self.work_dir / "zn.pdb"),
            "sim_type": "2",
            "force_field": "chemfusion",
            "resolved_force_field": "charmm36-jul2022",
            "resolved_force_field_dir": str(_FORCEFIELDS_DIR),
        }

    def tearDown(self):
        self.tmp.cleanup()

    @unittest.skipUnless(
        (_FORCEFIELDS_DIR / "charmm36-jul2022.ff" / "ions.itp").is_file(),
        "vendored CHARMM36 force field not present",
    )
    def test_ion_is_appended_to_gro_and_topology(self):
        step = CofactorTopologyStep(self.config, gmx_bin="gmx")

        step.run()

        gro_lines = (self.work_dir / "complex.gro").read_text().splitlines()
        self.assertEqual(gro_lines[1].strip(), "3")
        self.assertIn("ZN", gro_lines[-2])

        top_text = (self.work_dir / "topol.top").read_text()
        self.assertIn('#include "charmm36-jul2022.ff/ions.itp"', top_text)
        self.assertTrue(top_text.rstrip().splitlines()[-1].startswith("ZN"))

        self.assertEqual(self.config["cofactor_kind"], "ion")
        self.assertEqual(self.config["cofactor_resname"], "ZN")


class CofactorTopologyGaffMergeHelpersTests(unittest.TestCase):
    """Exercises the GAFF2/ACPYPE merge helpers directly, without invoking ACPYPE."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.work_dir = Path(self.tmp.name)

        (self.work_dir / "complex.gro").write_text(
            "Complex System\n"
            "1\n"
            + format_gro_atom_line(1, "SOL", "OW", 1, 0.0, 0.0, 0.0)
            + "   5.00000   5.00000   5.00000\n"
        )
        (self.work_dir / "topol.top").write_text(
            "; example topology\n"
            '#include "amber03.ff/forcefield.itp"\n'
            "\n"
            "; Ligand-specific atomtypes (GAFF2)\n"
            "[ atomtypes ]\n"
            "c3   c3   6   12.01   0.0000  A   3.39967e-01  4.57730e-01\n"
            "\n"
            '#include "ligand.itp"\n'
            "\n"
            "[ moleculetype ]\n"
            "; a moleculetype must precede [ molecules ] in any real topol.top,\n"
            "; and GROMACS requires every [ atomtypes ] entry to precede it too\n"
            "Protein_chain_A   3\n"
            "\n"
            "[ molecules ]\n"
            "; Compound        #mols\n"
            "Protein_chain_A    1\n"
            "LIG                1\n"
        )
        self.config = {
            "work_dir": str(self.work_dir),
            "cofactor_pdb": str(self.work_dir / "cofactor.pdb"),
            "cofactor_charge": -2,
        }
        (self.work_dir / "cofactor.pdb").write_text("placeholder")
        self.step = CofactorTopologyStep(self.config, gmx_bin="gmx")

    def tearDown(self):
        self.tmp.cleanup()

    def test_existing_atomtype_names_are_read_from_topology(self):
        self.assertEqual(self.step._existing_atomtype_names(), {"c3"})

    def test_clean_cofactor_itp_strips_atomtypes_section(self):
        raw_itp = self.work_dir / "COF_GMX.itp"
        raw_itp.write_text(
            "[ atomtypes ]\n"
            "c3   c3   6   12.01   0.0000  A   3.39967e-01  4.57730e-01\n"
            "os   os   8   16.00   0.0000  A   3.00000e-01  8.80314e-01\n"
            "\n"
            "[ moleculetype ]\n"
            "COF   3\n"
        )

        atomtype_lines = self.step._clean_cofactor_itp(str(raw_itp))

        self.assertEqual(len(atomtype_lines), 2)
        cleaned = (self.work_dir / "cofactor.itp").read_text()
        self.assertNotIn("[ atomtypes ]", cleaned)
        self.assertIn("[ moleculetype ]", cleaned)

    def test_patch_topology_dedups_atomtypes_and_appends_molecule(self):
        atomtype_lines = [
            "c3   c3   6   12.01   0.0000  A   3.39967e-01  4.57730e-01\n",
            "os   os   8   16.00   0.0000  A   3.00000e-01  8.80314e-01\n",
        ]

        self.step._patch_topology(atomtype_lines)

        top_text = (self.work_dir / "topol.top").read_text()
        self.assertEqual(top_text.count("c3   c3"), 1)  # not re-declared
        self.assertIn("os   os", top_text)
        self.assertIn('#include "cofactor.itp"', top_text)
        self.assertTrue(top_text.rstrip().splitlines()[-1].startswith("COF"))
        # GROMACS requires every [ atomtypes ] entry to precede the first
        # [ moleculetype ] in the file -- the new "os" type must land inside
        # the existing atomtypes block, not next to "#include cofactor.itp"
        # further down (that ordering previously produced grompp's
        # "Invalid order for directive atomtypes" fatal error).
        self.assertLess(top_text.index("os   os"), top_text.index("[ moleculetype ]"))
        self.assertLess(top_text.index("[ moleculetype ]"), top_text.index('#include "cofactor.itp"'))

    def test_merge_gro_appends_cofactor_atoms(self):
        cofactor_gro = self.work_dir / "COF_GMX.gro"
        cofactor_gro.write_text(
            "COF\n"
            "2\n"
            + format_gro_atom_line(1, "COF", "S1", 1, 1.0, 1.0, 1.0)
            + format_gro_atom_line(1, "COF", "O1", 2, 1.1, 1.0, 1.0)
            + "   0.00000   0.00000   0.00000\n"
        )

        self.step._merge_gro(str(cofactor_gro))

        gro_lines = (self.work_dir / "complex.gro").read_text().splitlines()
        self.assertEqual(gro_lines[1].strip(), "3")
        self.assertIn("COF", gro_lines[-2])
        self.assertIn("5.00000", gro_lines[-1])  # kept the original box vector


class CofactorTopologyGaffFormatHandlingTests(unittest.TestCase):
    """Regression test: ACPYPE must see the prepared cofactor under its real
    extension, not always as "cofactor.pdb" -- a pH-prepared cofactor is an
    SDF (see ligand_charge.prepare_protonated_sdf), and ACPYPE picks its
    parser from the file extension, not the content."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.work_dir = Path(self.tmp.name)
        (self.work_dir / "complex.gro").write_text(
            "Complex System\n1\n"
            + format_gro_atom_line(1, "SOL", "OW", 1, 0.0, 0.0, 0.0)
            + "   5.00000   5.00000   5.00000\n"
        )
        (self.work_dir / "topol.top").write_text(
            "; example topology\n"
            '#include "amber03.ff/forcefield.itp"\n\n'
            "[ molecules ]\n"
            "Protein_chain_A    1\n"
            "LIG                1\n"
        )
        prepared_sdf = self.work_dir / "cofactor_pH74.sdf"
        prepared_sdf.write_text("fake SDF content used as the prepared cofactor\n")
        self.config = {
            "work_dir": str(self.work_dir),
            "cofactor_pdb": str(self.work_dir / "cofactor.pdb"),
            "cofactor_prepared_file": str(prepared_sdf),
            "cofactor_charge": -2,
        }
        (self.work_dir / "cofactor.pdb").write_text("original pdb placeholder")

    def tearDown(self):
        self.tmp.cleanup()

    def test_prepared_sdf_cofactor_is_copied_with_sdf_extension(self):
        step = CofactorTopologyStep(self.config, gmx_bin="gmx")
        acpype_out = Path(step.acpype_workdir) / "cofactor.acpype"

        def fake_acpype(cmd, **kwargs):
            acpype_out.mkdir(parents=True, exist_ok=True)
            (acpype_out / "COF_GMX.gro").write_text(
                "COF\n1\n" + format_gro_atom_line(1, "COF", "C1", 1, 0.0, 0.0, 0.0)
                + "   0.00000   0.00000   0.00000\n"
            )
            (acpype_out / "COF_GMX.itp").write_text("[ dihedrals ]\n")
            return None

        with patch(
            "chemlink.pipelines.dynamics.steps.cofactor_topology.run_subprocess",
            side_effect=fake_acpype,
        ) as mocked:
            step._merge_gaff_cofactor(None)

        called_cmd = mocked.call_args.args[0]
        self.assertIn("-i", called_cmd)
        self.assertEqual(called_cmd[called_cmd.index("-i") + 1], "cofactor.sdf")

        copied = Path(step.acpype_workdir) / "cofactor.sdf"
        self.assertTrue(copied.is_file())
        self.assertEqual(copied.read_text(), (self.work_dir / "cofactor_pH74.sdf").read_text())
        self.assertFalse((Path(step.acpype_workdir) / "cofactor.pdb").exists())

    def test_charge_method_and_timeout_are_configurable(self):
        self.config["cofactor_charge_method"] = "gas"
        self.config["cofactor_charge_timeout"] = 60
        step = CofactorTopologyStep(self.config, gmx_bin="gmx")
        acpype_out = Path(step.acpype_workdir) / "cofactor.acpype"

        def fake_acpype(cmd, **kwargs):
            acpype_out.mkdir(parents=True, exist_ok=True)
            (acpype_out / "COF_GMX.gro").write_text(
                "COF\n1\n" + format_gro_atom_line(1, "COF", "C1", 1, 0.0, 0.0, 0.0)
                + "   0.00000   0.00000   0.00000\n"
            )
            (acpype_out / "COF_GMX.itp").write_text("[ dihedrals ]\n")
            return None

        with patch(
            "chemlink.pipelines.dynamics.steps.cofactor_topology.run_subprocess",
            side_effect=fake_acpype,
        ) as mocked:
            step._merge_gaff_cofactor(None)

        called_cmd = mocked.call_args.args[0]
        self.assertEqual(called_cmd[called_cmd.index("-c") + 1], "gas")
        self.assertEqual(mocked.call_args.kwargs["timeout"], 60)

    def test_charge_method_and_timeout_default_to_bcc_and_generous_timeout(self):
        step = CofactorTopologyStep(self.config, gmx_bin="gmx")
        acpype_out = Path(step.acpype_workdir) / "cofactor.acpype"

        def fake_acpype(cmd, **kwargs):
            acpype_out.mkdir(parents=True, exist_ok=True)
            (acpype_out / "COF_GMX.gro").write_text(
                "COF\n1\n" + format_gro_atom_line(1, "COF", "C1", 1, 0.0, 0.0, 0.0)
                + "   0.00000   0.00000   0.00000\n"
            )
            (acpype_out / "COF_GMX.itp").write_text("[ dihedrals ]\n")
            return None

        with patch(
            "chemlink.pipelines.dynamics.steps.cofactor_topology.run_subprocess",
            side_effect=fake_acpype,
        ) as mocked:
            step._merge_gaff_cofactor(None)

        called_cmd = mocked.call_args.args[0]
        self.assertEqual(called_cmd[called_cmd.index("-c") + 1], "bcc")
        self.assertGreaterEqual(mocked.call_args.kwargs["timeout"], 3600)
