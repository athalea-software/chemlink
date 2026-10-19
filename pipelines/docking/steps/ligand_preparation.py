"""Ligand preparation pipeline step for molecular docking."""

import logging
import os
import traceback
from datetime import datetime
from multiprocessing import Pool, cpu_count
from functools import partial
from typing import Optional, List, Dict, Tuple
from ....utils.progress import step_bar_iter


from ....storage.file_manager import (
    create_folder, list_files_in_directory, 
    verify_only_file, split_multi_molecule_sdf
)
from ....utils.molecule_processor import process_ligand
from ....utils.logger import setup_logger

logger = setup_logger(__name__, level=logging.INFO)


class LigandPreparation:
    """Batch ligand preparation for molecular docking.
    
    This class orchestrates the preparation of multiple ligand files,
    handling format conversion, conformer generation, and optimization.
    """
    
    def __init__(self, input_path: str, output_path: str):
        """Initialize ligand preparation pipeline.
        
        Args:
            input_path: Directory containing input ligand files
            output_path: Base directory for output files
        """
        self.input_path = input_path
        self.output_path = output_path
    
    def _collect_ligand_files(self) -> list:
        """Collect and split multi-molecule files if needed.
        
        Returns:
            List of ligand file paths ready for processing
        """
        extensions = ['*.sdf', '*.mol2', '*.pdb', '*.mol', '*.pdbqt']
        start_files = list_files_in_directory(self.input_path, extensions)
        
        files = []
        for f in start_files:
            if not verify_only_file(f):
                # Split multi-molecule SDF files
                temp_files = split_multi_molecule_sdf(f, self.input_path)
                files.extend(temp_files)
                logger.info(f"Split {os.path.basename(f)} into {len(temp_files)} molecules")
            else:
                files.append(f)
        
        return sorted(files)

    def _get_shard_config(self) -> Optional[Tuple[int, int]]:
        """Return (shard_index, shard_count) from environment if configured.

        Supports both generic variables and SLURM array variables:
        - CHEMLINK_SHARD_INDEX / CHEMLINK_SHARD_COUNT
        - SLURM_ARRAY_TASK_ID / SLURM_ARRAY_TASK_COUNT
        - SLURM_ARRAY_TASK_MIN / SLURM_ARRAY_TASK_MAX (fallback for count)

        Returns:
            Tuple (index, count) using zero-based index, or None when disabled.

        Raises:
            RuntimeError: If shard values are invalid
        """
        shard_index_env = os.getenv('CHEMLINK_SHARD_INDEX')
        shard_count_env = os.getenv('CHEMLINK_SHARD_COUNT')

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
        elif os.getenv('SLURM_ARRAY_TASK_ID') is not None:
            try:
                slurm_task_id = int(os.getenv('SLURM_ARRAY_TASK_ID', '0'))
            except ValueError as exc:
                raise RuntimeError("SLURM_ARRAY_TASK_ID must be an integer") from exc

            slurm_task_count = os.getenv('SLURM_ARRAY_TASK_COUNT')
            slurm_task_min = os.getenv('SLURM_ARRAY_TASK_MIN')
            slurm_task_max = os.getenv('SLURM_ARRAY_TASK_MAX')

            if slurm_task_count is not None:
                try:
                    shard_count = int(slurm_task_count)
                except ValueError as exc:
                    raise RuntimeError(
                        "SLURM_ARRAY_TASK_COUNT must be an integer"
                    ) from exc

                if slurm_task_min is not None:
                    try:
                        task_min = int(slurm_task_min)
                    except ValueError as exc:
                        raise RuntimeError(
                            "SLURM_ARRAY_TASK_MIN must be an integer"
                        ) from exc
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
        """Return the subset of files assigned to a shard.

        Uses round-robin assignment for balanced distribution.
        """
        return files[shard_index::shard_count]
    
    def _write_error_report(self, failed: list, timestamp: str) -> str:
        """Write detailed error report to file.
        
        Args:
            failed: List of error dictionaries
            timestamp: Timestamp string for filename
            
        Returns:
            Path to error log file
        """
        log_path = os.path.join(
            self.output_path,
            f'ligand_preparation_errors_{timestamp}.txt'
        )
        
        with open(log_path, 'w') as f:
            f.write("Ligand Preparation Error Report\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*80}\n\n")
            f.write(f"Summary: {len(failed)} ligand(s) failed to prepare\n\n")
            f.write(f"{'='*80}\n\n")
            
            for i, error in enumerate(failed, 1):
                f.write(f"Error #{i}\n")
                f.write(f"{'-'*80}\n")
                f.write(f"Ligand: {error['ligand']}\n")
                f.write(f"File: {error['file']}\n")
                f.write(f"Error: {error['error']}\n")
                
                if error.get('warnings'):
                    f.write("\nWarnings before failure:\n")
                    for w in error['warnings']:
                        f.write(f"  - {w}\n")
                
                f.write(f"\nTraceback:\n{error['traceback']}\n")
                f.write(f"{'='*80}\n\n")
        
        return log_path
    
    def _write_warning_report(self, all_warnings: list, timestamp: str) -> str:
        """Write warning report to file.
        
        Args:
            all_warnings: List of (ligand_name, warning) tuples
            timestamp: Timestamp string for filename
            
        Returns:
            Path to warning log file
        """
        log_path = os.path.join(
            self.output_path,
            f'ligand_preparation_warnings_{timestamp}.txt'
        )
        
        with open(log_path, 'w') as f:
            f.write("Ligand Preparation Warning Report\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*80}\n\n")
            f.write(f"Summary: {len(all_warnings)} warning(s)\n\n")
            f.write(f"{'='*80}\n\n")
            
            for i, (ligand_name, warning) in enumerate(all_warnings, 1):
                f.write(f"Warning #{i}\n")
                f.write(f"{'-'*80}\n")
                f.write(f"Ligand: {ligand_name}\n")
                f.write(f"Issue: {warning}\n\n")
        
        return log_path
    
    def prepare(self, n_workers: Optional[int] = None) -> dict:
        """Prepare all ligands found in input directory.
        
        Args:
            n_workers: Number of parallel workers (None = all CPUs, 1 = sequential)
            
        Returns:
            Dictionary with statistics: {'successful': int, 'failed': int, 'warnings': int}
        """
        # Setup output directory
        create_folder(f'{self.output_path}/prepared_ligands_pdbqt')
        
        # Collect files
        files = self._collect_ligand_files()
        
        if not files:
            logger.warning(f'No ligand files found in {self.input_path}')
            return {'successful': 0, 'failed': 0, 'warnings': 0}

        shard_cfg = self._get_shard_config()
        total_files = len(files)
        if shard_cfg:
            shard_index, shard_count = shard_cfg
            files = self._select_shard_files(files, shard_index, shard_count)

            logger.info(
                f"SLURM sharding enabled: shard {shard_index + 1}/{shard_count} "
                f"processing {len(files)}/{total_files} ligand(s)"
            )

            if not files:
                logger.warning("No ligands assigned to this shard.")
                return {'successful': 0, 'failed': 0, 'warnings': 0}
        
        # Determine worker count
        if n_workers is None:
            n_workers = cpu_count()
        n_workers = max(1, min(n_workers, len(files)))
        
        # Track results
        successful = []
        failed = []
        all_warnings = []
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        logger.info(
            f"Preparing {len(files)} ligand(s) using {n_workers} worker(s)..."
        )
        
        # Process ligands
        if n_workers == 1:
            results = []
            for f in step_bar_iter(
                files, "Ligand Preparation", unit="ligand", colour="magenta"
            ):
                results.append(process_ligand(f, self.output_path))
        else:
            worker_func = partial(process_ligand, output_path=self.output_path)
            with Pool(processes=n_workers) as pool:
                results = list(step_bar_iter(
                    pool.imap(worker_func, files), "Ligand Preparation",
                    total=len(files), unit="ligand", colour="magenta",
                ))
        
        # Collect results
        for result in results:
            try:
                ligand_name, success, error_details, warnings = result
            except Exception:
                ligand_name = "unknown_ligand"
                success = False
                error_details = {
                    'ligand': ligand_name,
                    'file': 'unknown',
                    'error': f'Unexpected result format: {result}',
                    'traceback': traceback.format_exc()
                }
                warnings = []

            if success:
                successful.append(ligand_name)
            else:
                failed.append(error_details)

            if warnings:
                all_warnings.extend([(ligand_name, w) for w in warnings])
        
        # Print summary
        logger.info(f"Successfully prepared: {len(successful)}/{len(files)} ligands")
        
        # Write reports
        if all_warnings:
            warn_log = self._write_warning_report(all_warnings, timestamp)
        else:
            warn_log = None
        
        if failed:
            error_log = self._write_error_report(failed, timestamp)
            logger.error(f"Failed: {len(failed)} - See: {error_log}")
        
        return {
            'successful': len(successful),
            'failed': len(failed),
            'warnings': len(all_warnings),
            'warning_report': warn_log,
        }
