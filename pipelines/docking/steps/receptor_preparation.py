"""Receptor preparation pipeline step for molecular docking."""

import logging
import os
import traceback
from datetime import datetime
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional, Dict, List, Any
from ....utils.progress import step_bar_iter

from ....storage.file_manager import create_folder, list_files_in_directory, find_compound_name
from ....utils.logger import setup_logger
from ....utils.receptor_processor import ReceptorProcessor

logger = setup_logger(__name__, level=logging.INFO)


def _prepare_receptor_worker(task: Dict[str, Any]) -> Dict[str, Any]:
    """Worker for process-based receptor preparation."""
    receptor_file = task["receptor_file"]
    receptor_name = find_compound_name(receptor_file)

    prep = ReceptorPreparation(
        input_path=task["input_path"],
        output_path=task["output_path"],
        remove_water=task["remove_water"],
        remove_ligands=task["remove_ligands"],
        remove_ions=task["remove_ions"],
        keep_clean_pdb=task["keep_clean_pdb"],
    )

    try:
        out_pdbqt = prep._prepare_single(receptor_file)
        return {
            "receptor": receptor_name,
            "file": receptor_file,
            "success": True,
            "output": out_pdbqt,
            "error": None,
            "traceback": None,
        }
    except Exception as exc:
        return {
            "receptor": receptor_name,
            "file": receptor_file,
            "success": False,
            "output": None,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


class ReceptorPreparation:
    """Batch receptor preparation for molecular docking."""

    def __init__(
        self,
        input_path: str,
        output_path: str,
        remove_water: bool = True,
        remove_ligands: bool = True,
        remove_ions: bool = True,
        keep_clean_pdb: bool = True,
    ):
        self.input_path = input_path
        self.output_path = output_path

        self.remove_water = remove_water
        self.remove_ligands = remove_ligands
        self.remove_ions = remove_ions
        self.keep_clean_pdb = keep_clean_pdb

    def _create_processor(self) -> ReceptorProcessor:
        """Build a receptor processor with current options."""
        return ReceptorProcessor(
            remove_water=self.remove_water,
            remove_ligands=self.remove_ligands,
            remove_ions=self.remove_ions,
            keep_clean_pdb=self.keep_clean_pdb,
        )

    def _prepare_single(self, receptor_file: str) -> str:
        """Run full pipeline (clean + PDBFixer + Meeko) for a single receptor."""
        processor = self._create_processor()
        return processor.prepare_receptor(receptor_file, self.output_path)

    def _write_error_report(self, failed: List[dict], timestamp: str) -> str:
        """Write detailed error report to file."""
        log_path = os.path.join(
            self.output_path,
            f"receptor_preparation_errors_{timestamp}.txt",
        )

        with open(log_path, "w") as handle:
            handle.write("Receptor Preparation Error Report\n")
            handle.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            handle.write(f"{'=' * 80}\n\n")
            handle.write(f"Summary: {len(failed)} receptor(s) failed to prepare\n\n")
            handle.write(f"{'=' * 80}\n\n")

            for i, error in enumerate(failed, 1):
                handle.write(f"Error #{i}\n")
                handle.write(f"{'-' * 80}\n")
                handle.write(f"Receptor: {error['receptor']}\n")
                handle.write(f"File: {error['file']}\n")
                handle.write(f"Error: {error['error']}\n")
                handle.write(f"\nTraceback:\n{error['traceback']}\n")
                handle.write(f"{'=' * 80}\n\n")

        return log_path

    def _get_shard_config(self) -> Optional[tuple]:
        """Return (shard_index, shard_count) from environment if configured."""
        shard_index_env = os.getenv("CHEMLINK_SHARD_INDEX")
        shard_count_env = os.getenv("CHEMLINK_SHARD_COUNT")

        if shard_index_env is not None or shard_count_env is not None:
            if shard_index_env is None or shard_count_env is None:
                raise RuntimeError(
                    "Both CHEMLINK_SHARD_INDEX and CHEMLINK_SHARD_COUNT must be set"
                )

            try:
                shard_index = int(shard_index_env)
                shard_count = int(shard_count_env)
            except ValueError as exc:
                raise RuntimeError(
                    "CHEMLINK_SHARD_INDEX and CHEMLINK_SHARD_COUNT must be integers"
                ) from exc
        elif os.getenv("SLURM_ARRAY_TASK_ID") is not None:
            try:
                slurm_task_id = int(os.getenv("SLURM_ARRAY_TASK_ID", "0"))
            except ValueError as exc:
                raise RuntimeError("SLURM_ARRAY_TASK_ID must be an integer") from exc

            slurm_task_count = os.getenv("SLURM_ARRAY_TASK_COUNT")
            slurm_task_min = os.getenv("SLURM_ARRAY_TASK_MIN")
            slurm_task_max = os.getenv("SLURM_ARRAY_TASK_MAX")

            if slurm_task_count is not None:
                try:
                    shard_count = int(slurm_task_count)
                except ValueError as exc:
                    raise RuntimeError("SLURM_ARRAY_TASK_COUNT must be an integer") from exc

                if slurm_task_min is not None:
                    try:
                        task_min = int(slurm_task_min)
                    except ValueError as exc:
                        raise RuntimeError("SLURM_ARRAY_TASK_MIN must be an integer") from exc
                    shard_index = slurm_task_id - task_min
                else:
                    shard_index = slurm_task_id
            elif slurm_task_min is not None and slurm_task_max is not None:
                try:
                    task_min = int(slurm_task_min)
                    task_max = int(slurm_task_max)
                except ValueError as exc:
                    raise RuntimeError(
                        "SLURM_ARRAY_TASK_MIN and SLURM_ARRAY_TASK_MAX must be integers"
                    ) from exc

                shard_count = (task_max - task_min) + 1
                shard_index = slurm_task_id - task_min
            else:
                return None
        else:
            return None

        if shard_count <= 1:
            return None
        if shard_count <= 0:
            raise RuntimeError("Shard count must be > 0")
        if shard_index < 0 or shard_index >= shard_count:
            raise RuntimeError(
                f"Invalid shard index {shard_index} for shard count {shard_count}"
            )

        return shard_index, shard_count

    @staticmethod
    def _select_shard_files(files: List[str], shard_index: int, shard_count: int) -> List[str]:
        """Return the subset of files assigned to a shard."""
        return files[shard_index::shard_count]

    def _build_worker_task(self, receptor_file: str) -> Dict[str, Any]:
        """Build worker task payload for process-based execution."""
        return {
            "receptor_file": receptor_file,
            "input_path": self.input_path,
            "output_path": self.output_path,
            "remove_water": self.remove_water,
            "remove_ligands": self.remove_ligands,
            "remove_ions": self.remove_ions,
            "keep_clean_pdb": self.keep_clean_pdb,
        }

    def prepare(self, n_workers: Optional[int] = None) -> Dict[str, int]:
        """Prepare all receptors found in input directory."""
        if self.keep_clean_pdb:
            create_folder(f"{self.output_path}/clean_receptors")
        create_folder(f"{self.output_path}/prepared_receptors_pdbqt")

        files = sorted(list_files_in_directory(self.input_path, ["*.pdb"]))
        if not files:
            logger.warning(f"No receptor files found in {self.input_path}")
            return {"successful": 0, "failed": 0}

        shard_cfg = self._get_shard_config()
        total_files = len(files)
        if shard_cfg:
            shard_index, shard_count = shard_cfg
            files = self._select_shard_files(files, shard_index, shard_count)

            logger.info(
                f"SLURM sharding enabled: shard {shard_index + 1}/{shard_count} "
                f"processing {len(files)}/{total_files} receptor(s)"
            )

            if not files:
                logger.warning("No receptors assigned to this shard.")
                return {"successful": 0, "failed": 0}

        if n_workers is None:
            slurm_cpus = os.getenv("SLURM_CPUS_PER_TASK")
            if slurm_cpus and slurm_cpus.isdigit():
                n_workers = int(slurm_cpus)
            else:
                n_workers = cpu_count()

        n_workers = max(1, min(n_workers, len(files)))

        successful: List[str] = []
        failed: List[Dict[str, Any]] = []

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info(
            f"Preparing {len(files)} receptor(s) using {n_workers} worker(s)..."
        )

        if n_workers == 1:
            for receptor_file in step_bar_iter(
                files, "Receptor Preparation", unit="receptor", colour="blue"
            ):
                receptor_name = find_compound_name(receptor_file)
                try:
                    self._prepare_single(receptor_file)
                    successful.append(receptor_name)
                    logger.info(f"Prepared {receptor_name}")
                except Exception as exc:
                    error_details = {
                        "receptor": receptor_name,
                        "file": receptor_file,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                    failed.append(error_details)
                    error_msg = str(exc)
                    if "not found" in error_msg.lower() or "File found at:" in error_msg:
                        logger.error(
                            f"Failed {receptor_name}: {error_msg}"
                        )
                    else:
                        logger.error(f"Failed {receptor_name}: {exc}")
        else:
            tasks = [self._build_worker_task(receptor_file) for receptor_file in files]

            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = [executor.submit(_prepare_receptor_worker, task) for task in tasks]
                for future in step_bar_iter(
                    as_completed(futures), "Receptor Preparation",
                    total=len(futures), unit="receptor", colour="blue",
                ):
                    result = future.result()
                    receptor_name = result["receptor"]
                    if result["success"]:
                        successful.append(receptor_name)
                        logger.info(f"Prepared {receptor_name}")
                    else:
                        error_details = {
                            "receptor": result["receptor"],
                            "file": result["file"],
                            "error": result["error"],
                            "traceback": result["traceback"],
                        }
                        failed.append(error_details)
                        error_msg = result["error"] or ""
                        if "not found" in error_msg.lower() or "File found at:" in error_msg:
                            logger.error(
                                f"Failed {receptor_name}: {error_msg}"
                            )
                        else:
                            logger.error(f"Failed {receptor_name}: {result['error']}")

        logger.info(f"Successfully prepared: {len(successful)}/{len(files)} receptors")

        if failed:
            error_log = self._write_error_report(failed, timestamp)
            logger.error(f"Failed: {len(failed)} - See: {error_log}")
        return {"successful": len(successful), "failed": len(failed)}

