"""Subprocess entrypoint (runs under the pyrosetta conda env): one
independent FlexPepDock refinement trial for one candidate complex.

Generalizes this project's proven flexpepdock_trial.py scratch script.
Each trial gets its own OS process (PyRosetta is not thread-safe and not
meant to be re-initialized in-process) and its own seed. FastRelax always
runs before FlexPepDockingProtocol -- without it, trials produce
catastrophic clash scores (~90% of trials, observed this project).
InterfaceAnalyzerMover.get_interface_dG() is the reported metric, not
pose.energies().total_energy() (that mixes in the whole receptor's
internal energy, not just the interface).

Usage: <pyrosetta_python> -m chemlink.pipelines.docking.peptide_docking.steps._flexpepdock_trial_worker \
    --complex-pdb PATH --seed N [--params PATH,PATH,...]
"""

import argparse
import json
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--complex-pdb", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--params", default="")
    parser.add_argument("--out-pdb", default="",
                         help="If given, dump the post-FastRelax/FlexPepDocking pose here -- "
                              "the caller keeps only the best-scoring trial's file (see "
                              "FlexPepDockRefinement._refine_one), since this is the only "
                              "point in the pipeline where the relaxed pose (clash-free,"
                              " unlike the raw assemble-worker scaffold) is ever available.")
    args = parser.parse_args()

    try:
        import pyrosetta
        from pyrosetta import get_fa_scorefxn, pose_from_pdb
        from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
        from pyrosetta.rosetta.protocols.flexpep_docking import FlexPepDockingProtocol
        from pyrosetta.rosetta.protocols.relax import FastRelax

        params_files = [p for p in args.params.split(",") if p]
        extra_res_flags = " ".join(f"-extra_res_fa {pf}" for pf in params_files)
        pyrosetta.init(
            "-mute all -flexPepDocking:pep_refine -ex1 -ex2aro -use_input_sc "
            f"{extra_res_flags} -run:constant_seed -run:jran {args.seed}"
        )

        pose = pose_from_pdb(args.complex_pdb)
        scorefxn = get_fa_scorefxn()

        relax = FastRelax(scorefxn, 1)  # 1 cycle: resolve clashes, don't fully reoptimize
        relax.apply(pose)

        FlexPepDockingProtocol().apply(pose)

        ia = InterfaceAnalyzerMover()
        ia.set_interface_jump(1)  # the pose has exactly one jump: receptor <-> peptide chain
        ia.apply(pose)

        if args.out_pdb:
            pose.dump_pdb(args.out_pdb)

        print(json.dumps({
            "seed": args.seed,
            "interface_dG": ia.get_interface_dG(),
            "total_score": pose.energies().total_energy(),
        }))
    except Exception as exc:  # noqa: BLE001 - reported to the parent process, not raised here
        print(json.dumps({"seed": args.seed, "error": f"{type(exc).__name__}: {exc}"}))
        sys.exit(1)


if __name__ == "__main__":
    main()
