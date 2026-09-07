"""Subprocess entrypoint (runs under the pyrosetta conda env): builds .params
files for a candidate's non-canonical side chain(s) and assembles the real-
chemistry complex pose on top of LightDock's already-docked backbone.

Generalizes this project's proven ad-hoc scripts (build_all_params.py +
assemble_noncanonical.py from this session's scratch work) into one
reusable entrypoint. Two-step init is intentional and required: the first
bare pyrosetta.init() loads enough of Rosetta core for rdkit_to_params to
build a valid .params file; only the SECOND init(), with `-extra_res_fa`
pointing at those freshly-written files, registers the new residue types
so pose_from_sequence() can actually place them.

Usage: <pyrosetta_python> -m chemlink.pipelines.peptide_docking.steps._flexpepdock_ncaa_assemble_worker \
    --job-dir DIR --complex-pdb PATH --seq-json PATH
"""

import argparse
import json
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--complex-pdb", required=True)
    parser.add_argument("--seq-json", required=True)
    args = parser.parse_args()

    try:
        import pyrosetta
        pyrosetta.init("-mute all")

        from rdkit import Chem
        from rdkit_to_params import Params

        with open(args.seq_json) as fh:
            info = json.load(fh)
        cid = info["cid"]
        seq = info["seq"]

        params_files = []
        for pos, name in info["residue_names"].items():
            frag_sdf = os.path.join(args.job_dir, f"frag_{cid}_{pos}.sdf")
            out_params = os.path.join(args.job_dir, f"{name}.params")
            mol = Chem.MolFromMolFile(frag_sdf, sanitize=True, removeHs=False)
            if mol is None:
                raise RuntimeError(f"could not read fragment {frag_sdf}")
            Params.from_mol(mol, name=name).dump(out_params)
            params_files.append(out_params)

        extra_res_flags = " ".join(f"-extra_res_fa {pf}" for pf in params_files)
        pyrosetta.init(f"-mute all {extra_res_flags} -ignore_unrecognized_res true")

        from pyrosetta import pose_from_pdb, pose_from_sequence
        from pyrosetta.rosetta.core.pose import append_pose_to_pose
        from pyrosetta.rosetta.core.scoring import calpha_superimpose_pose

        full = pose_from_pdb(args.complex_pdb)
        n_receptor = full.chain_end(1)

        receptor = pyrosetta.Pose()
        receptor.assign(full)
        for i in range(full.total_residue(), n_receptor, -1):
            receptor.delete_residue_slow(i)

        std_peptide = pyrosetta.Pose()
        std_peptide.assign(full)
        for i in range(n_receptor, 0, -1):
            std_peptide.delete_residue_slow(i)

        real_peptide = pose_from_sequence(seq, "fa_standard")
        if real_peptide.total_residue() != std_peptide.total_residue():
            raise RuntimeError(
                f"real sequence has {real_peptide.total_residue()} residues, "
                f"docked peptide has {std_peptide.total_residue()}"
            )

        for i in range(1, std_peptide.total_residue() + 1):
            real_peptide.set_phi(i, std_peptide.phi(i))
            real_peptide.set_psi(i, std_peptide.psi(i))
            real_peptide.set_omega(i, std_peptide.omega(i))

        rmsd = calpha_superimpose_pose(real_peptide, std_peptide)

        complex_out = pyrosetta.Pose()
        complex_out.assign(receptor)
        append_pose_to_pose(complex_out, real_peptide, new_chain=True)

        out_pdb = os.path.join(args.job_dir, "real_complex.pdb")
        complex_out.dump_pdb(out_pdb)

        print(json.dumps({
            "success": True, "cid": cid, "rmsd": rmsd,
            "params_files": params_files, "out_pdb": out_pdb,
        }))
    except Exception as exc:  # noqa: BLE001 - reported to the parent process, not raised here
        print(json.dumps({"success": False, "error": f"{type(exc).__name__}: {exc}"}))
        sys.exit(1)


if __name__ == "__main__":
    main()
