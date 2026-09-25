import os
from tqdm import tqdm

from ..logger import get_step_logger
from utils.retry import SubprocessError, run_subprocess

_TRJCONV_TIMEOUT  = 600
_MAKE_NDX_TIMEOUT = 120
_SEGMENT_TIMEOUT  = 300


class PostProcessingStep:
    def __init__(self, config, gmx_bin):
        self.config      = config
        self.gmx_bin     = gmx_bin
        work             = self.config["work_dir"]
        self.work_dir    = work
        self.rel_seg_dir = "segmented_trajectories"
        self.segment_dir = os.path.join(work, self.rel_seg_dir)
        self.logger      = get_step_logger(__name__, os.path.join(work, "simulation.log"))

    def run(self) -> None:
        os.makedirs(self.segment_dir, exist_ok=True)

        with tqdm(total=4, desc="  └─ Post-Processing & Cleanup", leave=False) as pbar:
            try:
                # Step 1: PBC correction + centering
                run_subprocess(
                    [
                        self.gmx_bin, "trjconv",
                        "-s", "md.tpr", "-f", "md.xtc",
                        "-o", "md_center.xtc",
                        "-center", "-pbc", "mol", "-ur", "compact",
                    ],
                    input_data=(
                        "Other\nSystem\n"
                        if self.config.get("sim_type") == "7" else "1\n0\n"
                    ),
                    timeout=_TRJCONV_TIMEOUT,
                    retries=1,
                    cwd=self.work_dir,
                    logger=self.logger,
                )
                pbar.update(1)

                # Step 2: Final structure extraction (PDB)
                sim_time_ps = int(float(self.config["ns_time"]) * 1000)
                self._dump_final_pdb(sim_time_ps)
                pbar.update(1)

                self._generate_smart_index()
                pbar.update(1)

                self._extract_segments(sim_time_ps)
                pbar.update(1)

            except SubprocessError as exc:
                self.logger.error("Post-processing failed:\n%s", exc.stderr)
                raise RuntimeError(str(exc)) from exc
            except Exception:
                self.logger.exception("Unexpected error in PostProcessingStep")
                raise

    def _dump_final_pdb(self, sim_time_ps: int) -> None:
        dump_cmd = [
            self.gmx_bin, "trjconv",
            "-s", "md.tpr", "-f", "md_center.xtc",
            "-o", "md.pdb", "-dump", str(sim_time_ps),
        ]
        try:
            run_subprocess(
                dump_cmd,
                input_data="0\n",
                timeout=_TRJCONV_TIMEOUT,
                retries=1,
                cwd=self.work_dir,
                logger=self.logger,
            )
        except SubprocessError:
            # Fall back to last frame if the requested time is out of range.
            run_subprocess(
                [
                    self.gmx_bin, "trjconv",
                    "-s", "md.tpr", "-f", "md_center.xtc",
                    "-o", "md.pdb", "-dump", "-1",
                ],
                input_data="0\n",
                timeout=_TRJCONV_TIMEOUT,
                retries=1,
                cwd=self.work_dir,
                logger=self.logger,
            )

    def _generate_smart_index(self) -> None:
        sim_type = self.config.get("sim_type")
        if sim_type == "2":
            ndx_input = "1 | 13\nname 22 Protein_Ligand\nq\n"
        elif sim_type == "4":
            ndx_input = "r DNA RNA DR DNA5 DNA3\nname 22 Nucleic_Acid\n1 | 22\nname 23 Complex\nq\n"
        elif sim_type in ["3", "5"]:
            ndx_input = "splitres 0\nq\n"
        elif sim_type == "6":
            ndx_input = "splitres 0\n1 | 13\nname 22 Complex_System\nq\n"
        else:
            ndx_input = "q\n"

        run_subprocess(
            [self.gmx_bin, "make_ndx", "-f", "md.tpr", "-o", "index.ndx"],
            input_data=ndx_input,
            timeout=_MAKE_NDX_TIMEOUT,
            retries=1,
            cwd=self.work_dir,
            logger=self.logger,
        )

    def _extract_segments(self, sim_time_ps: int) -> None:
        last_10_ps = sim_time_ps * 0.9
        segments = [
            (os.path.join(self.rel_seg_dir, "md_first_100ps.xtc"), "0", "100"),
            (os.path.join(self.rel_seg_dir, "md_last_10percent.xtc"), str(last_10_ps), str(sim_time_ps)),
        ]
        for out_file, start, end in segments:
            if sim_time_ps > float(start):
                try:
                    run_subprocess(
                        [
                            self.gmx_bin, "trjconv",
                            "-s", "md.tpr", "-f", "md_center.xtc",
                            "-o", out_file,
                            "-b", start, "-e", end, "-tu", "ps",
                        ],
                        input_data="0\n",
                        timeout=_SEGMENT_TIMEOUT,
                        retries=1,
                        cwd=self.work_dir,
                        logger=self.logger,
                    )
                except SubprocessError as exc:
                    # Segment extraction is non-critical — log and continue.
                    self.logger.warning("Segment extraction skipped (%s): %s", out_file, exc)
