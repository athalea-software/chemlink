import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from utils.progress import step_bar
from utils.style import log_line
from utils.theme import get_console

class DCCMAnalyzer:
    def __init__(self, results_dir: str, gmx_bin: str = "gmx"):
        self.results_dir = Path(results_dir)
        self.gmx_bin = gmx_bin
        self.input_dir = self.results_dir.parent 
        self.dccm_dir = self.results_dir 
        self.dccm_dir.mkdir(exist_ok=True, parents=True)

    def _run_gmx_command(self, cmd: list, stdin_input: str = "") -> tuple:
        try:
            process = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, cwd=str(self.input_dir.absolute())
            )
            stdout, stderr = process.communicate(input=stdin_input)
            if process.returncode != 0:
                return False, stderr
            return True, stdout
        except Exception as e:
            return False, str(e)

    def extract_calpha_trajectory(self, selection: str = "C-alpha"):      
        selection_map = {"C-alpha": "3\n", "Backbone": "4\n", "Protein": "1\n"}
        sel_input = selection_map.get(selection, "3\n")
        traj_pdb = self.dccm_dir / 'calpha_traj.pdb'
        cmd = [
            self.gmx_bin, 'trjconv',
            '-f', 'md_center.xtc', 
            '-s', str((self.input_dir / 'md.tpr').absolute()),
            '-n', str((self.input_dir / 'index.ndx').absolute()),
            '-o', str(traj_pdb.absolute())
        ]
        
        success, error_msg = self._run_gmx_command(cmd, sel_input)
        if success and traj_pdb.exists():
            return traj_pdb
        get_console().print(log_line("ERROR", f"GROMACS trjconv failed:\n{error_msg}", phase="dccm"))
        return None

    def parse_pdb_trajectory(self, pdb_file: Path, max_frames: int = None):
        frames, current_frame = [], []

        total_lines = sum(1 for _ in open(pdb_file, 'r'))

        with open(pdb_file, 'r') as f:
            with step_bar("Parsing PDB Coordinates", total_lines, unit="line") as pbar:
                for line in f:
                    if line.startswith(('ATOM', 'HETATM')):
                        current_frame.append([
                            float(line[30:38].strip()),
                            float(line[38:46].strip()),
                            float(line[46:54].strip())
                        ])
                    elif line.startswith(('ENDMDL', 'END')):
                        if current_frame:
                            frames.append(current_frame)
                            current_frame = []
                            if max_frames and len(frames) >= max_frames: break
                    pbar.update(1)
            coords = np.array(frames)
            return coords

    def compute_dccm_from_trajectory(self, coords: np.ndarray, max_residues: int = 2000):
        n_frames, n_atoms, _ = coords.shape
        if n_atoms > max_residues:
            coords = coords[:, :max_residues, :]
            n_atoms = max_residues

        avg_pos = np.mean(coords, axis=0)
        displacements = coords - avg_pos
        dccm_matrix = np.zeros((n_atoms, n_atoms))
        
        with step_bar("Computing DCCM Matrix", n_atoms, unit="atom") as pbar:
            for i in range(n_atoms):
                d_i = displacements[:, i, :]
                var_i = np.mean(np.sum(d_i**2, axis=1))
                for j in range(i, n_atoms):
                    d_j = displacements[:, j, :]
                    var_j = np.mean(np.sum(d_j**2, axis=1))
                    cov_ij = np.mean(np.sum(d_i * d_j, axis=1))
                    if var_i > 1e-10 and var_j > 1e-10:
                        val = cov_ij / np.sqrt(var_i * var_j)
                        dccm_matrix[i, j] = dccm_matrix[j, i] = val
                pbar.update(1)
        np.savetxt(self.dccm_dir / 'dccm_matrix.dat', dccm_matrix, fmt='%.6f')
        return dccm_matrix

    def visualize_dccm(self, dccm_matrix: np.ndarray):
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(dccm_matrix, cmap='RdBu_r', vmin=-1.0, vmax=1.0, origin='lower')
        plt.colorbar(im, ax=ax, label='Correlation')
        ax.set_title('Dynamic Cross-Correlation Matrix')
        
        output_file = self.dccm_dir / 'dccm_matrix.png'
        plt.savefig(output_file, dpi=300)
        plt.close()

    def _create_dccm_regions(self, dccm_matrix: np.ndarray):
        n_res = dccm_matrix.shape[0]
        if n_res > 150:
            mid = n_res // 2
            regions = [
                (0, mid, 0, mid, "N-terminal_Zone"),
                (mid, n_res, mid, n_res, "C-terminal_Zone")
            ]
            for i_s, i_e, j_s, j_e, label in regions:
                fig, ax = plt.subplots()
                region = dccm_matrix[i_s:i_e, j_s:j_e]
                im = ax.imshow(region, cmap='RdBu_r', vmin=-1.0, vmax=1.0, origin='lower')
                plt.colorbar(im)
                ax.set_title(f'DCCM Region: {label}')
                plt.savefig(self.dccm_dir / f'dccm_region_{label}.png', dpi=200)
                plt.close()

    def _print_dccm_statistics(self, dccm_matrix: np.ndarray):
        upper_tri = np.triu(dccm_matrix, k=1)
        correlations = upper_tri[upper_tri != 0] 
        if len(correlations) > 0:
            stats_file = self.dccm_dir / 'dccm_statistics.txt'
            with open(stats_file, 'w') as f:
                f.write(f"DCCM Statistics Report\n{'='*30}\n")
                f.write(f"Analyzed pairs: {len(correlations):,}\n")
                f.write(f"Mean Correlation: {np.mean(correlations):.4f}\n")
                f.write(f"Max Positive: {np.max(correlations):.4f}\n")
                f.write(f"Max Negative: {np.min(correlations):.4f}\n")

    def run_pipeline_analysis(self, selection="C-alpha", max_residues=2000):
        traj_pdb = None
        try:
            # 1. Coordinate Extraction
            traj_pdb = self.extract_calpha_trajectory(selection)
            if not traj_pdb or not traj_pdb.exists():
                return False
            
            # 2. Coordinate Parsing
            coords = self.parse_pdb_trajectory(traj_pdb)
            if coords is None or len(coords) == 0:
                return False

            # 3. Mathematical Calculation
            dccm_matrix = self.compute_dccm_from_trajectory(coords, max_residues)
            if dccm_matrix is None:
                return False
            
            # 4. Visualization & Reporting
            self.visualize_dccm(dccm_matrix)
            self._create_dccm_regions(dccm_matrix)
            self._print_dccm_statistics(dccm_matrix)
            
            return True
        except Exception as e:
            get_console().print(log_line("ERROR", f"Unexpected error in DCCM analysis: {e}", phase="dccm"))
            return False
        finally:
            if traj_pdb and traj_pdb.exists():
                traj_pdb.unlink()