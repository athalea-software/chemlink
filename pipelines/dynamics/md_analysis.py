import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import subprocess
import logging
from utils.progress import step_bar

from .dccm_analysis import DCCMAnalyzer

matplotlib.use('Agg')
plt.style.use('seaborn-v0_8-darkgrid')
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

class GromacsAnalyzer:
    
    def __init__(
        self,
        results_dir: str,
        sim_type: str,
        gmx_bin: str = "gmx",
        logger: Optional[logging.Logger] = None,
    ):
        self.results_dir = Path(results_dir)
        self.gmx_bin = gmx_bin
        self.sim_type = sim_type
        self.logger = logger
        
        self.has_ligand = sim_type in ["2", "6"]
        self.is_protein_only = sim_type == "1"
        self.is_ligand_only = sim_type == "7"
        self.is_complex = sim_type in ["3", "5"]
        
        self.analysis_dirs = {
            'energy': self.results_dir / 'energy_analysis',
            'rmsd': self.results_dir / 'rmsd_analysis',
            'rmsf': self.results_dir / 'rmsf_analysis',
            'structural': self.results_dir / 'structural_analysis',
            'sasa': self.results_dir / 'sasa_analysis',
            'hbonds': self.results_dir / 'hbonds_analysis',
            'plots': self.results_dir / 'plots'
        }
        for folder in self.analysis_dirs.values():
            folder.mkdir(parents=True, exist_ok=True)
    
    def _read_xvg(self, filename: str) -> Optional[np.ndarray]:
        try:
            data = []
            with open(filename, 'r') as f:
                for line in f:
                    if not line.startswith(('#', '@')):
                        try:
                            data.append([float(x) for x in line.strip().split()])
                        except ValueError:
                            continue
            return np.array(data) if data else None
        except Exception as e:
            return None
    
    def _run_gmx_command(self, cmd: List[str], stdin_input: str = "") -> Tuple[bool, str]:
        try:
            process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, cwd=str(self.results_dir))
            stdout, stderr = process.communicate(input=stdin_input)
            
            if process.returncode != 0:
                with open(self.results_dir / "gmx_analysis_errors.log", "a") as f:
                    f.write(f"Command: {' '.join(cmd)}\nError: {stderr}\n")
                return False, stderr
                
            return True, stdout
        except Exception as e:
            return False, str(e)
    
    def analyze_energy(self):
        analyses = [
            ('em.edr',  'potential.xvg',        '10\n0\n'), # Potential
            ('nvt.edr', 'temperature.xvg',      '14\n0\n'), # Temperature
            ('npt.edr', 'pressure.xvg',         '15\n0\n'), # Pressure
            ('md.edr',  'energy_total.xvg',     '12\n0\n'), # Total Energy
            ('md.edr',  'energy_potential.xvg', '10\n0\n'), # Potential
            ('md.edr',  'energy_kinetic.xvg',   '11\n0\n'), # Kinetic En.
            ('md.edr',  'temperature_md.xvg',   '14\n0\n'), # Temperature
            ('md.edr',  'volume.xvg',           '20\n0\n'), # Volume
            ('md.edr',  'density.xvg',          '21\n0\n'), # Density
        ]
        
        # Internal progress bar
        with step_bar("Energy Analysis", len(analyses)) as pbar:
            for edr, out, sel in analyses:
                edr_path = self.results_dir / edr
                if edr_path.exists():
                    rel_out_path = self.analysis_dirs['energy'].relative_to(self.results_dir) / out
                    cmd = [self.gmx_bin, 'energy', '-f', edr, '-o', str(rel_out_path)]
                    self._run_gmx_command(cmd, sel)
                pbar.update(1)
    
    def analyze_rmsd(self):
        out_dir = self.analysis_dirs['rmsd'].relative_to(self.results_dir)
        
        with step_bar("RMSD Analysis", 6) as pbar:
            cmd_bb = [self.gmx_bin, 'rms', '-s', 'em.tpr', '-f', 'md_center.xtc', 
                    '-n', 'index.ndx', '-tu', 'ns', '-o', str(out_dir / 'rmsd_backbone.xvg')]
            self._run_gmx_command(cmd_bb, "4\n4\n")
            pbar.update(1)

            
            cmd_ca = [self.gmx_bin, 'rms', '-s', 'em.tpr', '-f', 'md_center.xtc', 
                    '-n', 'index.ndx', '-tu', 'ns', '-o', str(out_dir / 'rmsd_calpha.xvg')]
            self._run_gmx_command(cmd_ca, "3\n3\n")
            pbar.update(1)

            cmd_prot = [self.gmx_bin, 'rms', '-s', 'em.tpr', '-f', 'md_center.xtc', 
                        '-n', 'index.ndx', '-tu', 'ns', '-o', str(out_dir / 'rmsd_protein.xvg')]
            self._run_gmx_command(cmd_prot, "1\n1\n")
            pbar.update(1)

            if str(self.sim_type) == "4":
                cmd_dna = [self.gmx_bin, 'rms', '-s', 'em.tpr', '-f', 'md_center.xtc', 
                        '-n', 'index.ndx', '-tu', 'ns', '-o', str(out_dir / 'rmsd_dna.xvg')]
                self._run_gmx_command(cmd_dna, "12\n12\n")
            pbar.update(1)

            cmd_dna_fit = [self.gmx_bin, 'rms', '-s', 'em.tpr', '-f', 'md_center.xtc', 
                        '-n', 'index.ndx', '-tu', 'ns', '-o', str(out_dir / 'rmsd_dna_fit_prot.xvg')]
            self._run_gmx_command(cmd_dna_fit, "1\n12\n")
            pbar.update(1)

            if self.has_ligand:
                cmd_lig = [self.gmx_bin, 'rms', '-s', 'em.tpr', '-f', 'md_center.xtc', 
                        '-n', 'index.ndx', '-tu', 'ns', '-o', str(out_dir / 'rmsd_ligand_fit_protein.xvg')]
                self._run_gmx_command(cmd_lig, "1\nOther\n")
            pbar.update(1)
        
    def analyze_rmsf(self):
        out_dir = self.analysis_dirs['rmsf'].relative_to(self.results_dir)
        with step_bar("RMSF Fluctuations", 4) as pbar:
            for sel, out in [('3\n', 'rmsf_calpha'), ('4\n', 'rmsf_backbone'), ('1\n', 'rmsf_protein')]:
                cmd = [self.gmx_bin, 'rmsf', 
                    '-s', 'md.tpr', 
                    '-f', 'md_center.xtc',
                    '-n', 'index.ndx',
                    '-o', str(out_dir / f'{out}.xvg'),
                    '-ox', str(out_dir / f'{out}_avg.pdb'),
                    '-oq', str(out_dir / f'bfactors_{out.split("_")[1]}.pdb'),
                    '-res']
                self._run_gmx_command(cmd, sel)
                pbar.update(1)
            
            if self.has_ligand:
                cmd = [self.gmx_bin, 'rmsf', 
                    '-s', 'md.tpr', 
                    '-f', 'md_center.xtc',
                    '-n', 'index.ndx',
                    '-o', str(out_dir / 'rmsf_ligand.xvg'),
                    '-ox', str(out_dir / 'rmsf_ligand_avg.pdb'),
                    '-oq', str(out_dir / 'bfactors_ligand.pdb'), 
                    '-res']
                self._run_gmx_command(cmd, '13\n')
            pbar.update(1)
    
    def analyze_gyration(self):
        out_dir = self.analysis_dirs['structural'].relative_to(self.results_dir)
        with step_bar("Gyration Analysis", 3) as pbar:
            for sel, out in [('1\n', 'gyrate_protein.xvg'), ('4\n', 'gyrate_backbone.xvg'), ('3\n', 'gyrate_calpha.xvg')]:
                cmd = [self.gmx_bin, 'gyrate', 
                   '-f', 'md_center.xtc',
                   '-s', 'md.tpr',
                   '-n', 'index.ndx',
                   '-o', str(out_dir / out)]
                self._run_gmx_command(cmd, sel)
                pbar.update(1)
    
    def analyze_sasa(self):
        out_dir = self.analysis_dirs['sasa'].relative_to(self.results_dir)
        with step_bar("SASA Analysis", 3) as pbar:
            cmd_prot = [self.gmx_bin, 'sasa', 
                    '-f', 'md_center.xtc',
                    '-s', 'md.tpr',
                    '-n', 'index.ndx',
                    '-o', str(out_dir / 'sasa_protein.xvg'),
                    '-or', str(out_dir / 'sasa_residue.xvg'),
                    '-oa', str(out_dir / 'sasa_atom.xvg')]
            self._run_gmx_command(cmd_prot, "1\n") 
            pbar.update(1)

            if str(self.sim_type) == "4":
                cmd_dna = [self.gmx_bin, 'sasa', 
                        '-f', 'md_center.xtc',
                        '-s', 'md.tpr',
                        '-n', 'index.ndx',
                        '-o', str(out_dir / 'sasa_dna.xvg')]
                self._run_gmx_command(cmd_dna, "DNA\n")
            pbar.update(1)

            if self.has_ligand:
                for group_name, out in [('Other', 'sasa_ligand.xvg'), ('System', 'sasa_complex.xvg')]:
                    cmd = [self.gmx_bin, 'sasa', 
                        '-f', 'md_center.xtc',
                        '-s', 'md.tpr',
                        '-n', 'index.ndx',
                        '-o', str(out_dir / out)]
                    self._run_gmx_command(cmd, f'{group_name}\n')
            pbar.update(1)

    def analyze_hbonds_detailed(self):
        out_dir = self.analysis_dirs['hbonds'].relative_to(self.results_dir)
        with step_bar("H-Bond Analysis", 2) as pbar:
            cmd_intra = [self.gmx_bin, 'hbond', 
                        '-f', 'md_center.xtc',
                        '-s', 'md.tpr',
                        '-n', 'index.ndx',
                        '-num', str(out_dir / 'hbond_protein_intra.xvg')]
            self._run_gmx_command(cmd_intra, '1\n1\n')
            pbar.update(1)
            if self.has_ligand:
                cmd_pair = [self.gmx_bin, 'pairdist',
                            '-f', 'md_center.xtc',
                            '-s', 'md.tpr',
                            '-n', 'index.ndx',
                            '-cutoff', '0.35',
                            '-o', str(out_dir / 'contacts_prot_lig.xvg')]
                self._run_gmx_command(cmd_pair, '1\n13\n')

                cmd_hbond = [self.gmx_bin, 'hbond', 
                            '-f', 'md_center.xtc',
                            '-s', 'md.tpr',
                            '-n', 'index.ndx',
                            '-num', str(out_dir / 'hbond_prot_lig.xvg'),
                            '-dist', str(out_dir / 'hbond_prot_lig_dist.xvg'),
                            '-ang', str(out_dir / 'hbond_prot_lig_angle.xvg')]
                
                success, _ = self._run_gmx_command(cmd_hbond, '1\n13\n')
                
                if success:
                    self._extract_hbond_residues(self.analysis_dirs['hbonds'])
                    self._convert_hbond_matrix(self.analysis_dirs['hbonds'])
                else:
                    messages = (
                        "No chemical H-bonds were detected (lack of donors/acceptors).",
                        "'contacts_prot_lig.xvg' has been generated as an alternative interaction metric.",
                    )
                    if self.logger:
                        for message in messages:
                            self.logger.info(message, extra={"file_only": True})
            pbar.update(1)
    
    def _extract_hbond_residues(self, hbonds_dir: Path):
        ndx_file = hbonds_dir / 'hbond_prot_lig.ndx'
        if not ndx_file.exists():
            return
        
        try:
            with open(ndx_file, 'r') as f:
                lines = f.readlines()
            
            hbond_groups = {}
            current_group = None
            
            for line in lines:
                if line.startswith('[') and line.endswith(']'):
                    current_group = line[1:-1].strip()
                    hbond_groups[current_group] = []
                elif current_group:
                    hbond_groups[current_group].extend([int(x) for x in line.split() if x.isdigit()])
            
            summary = ["="*80, "RESIDUES FORMING PROTEIN-LIGAND H-BONDS", "="*80,
                      f"\nTotal groups found: {len(hbond_groups)}\n"]
            
            for group, atoms in hbond_groups.items():
                summary.extend([f"{group}:", f"  Atoms: {len(atoms)}",
                              f"  IDs: {atoms[:10]}..." if len(atoms) > 10 else f"  IDs: {atoms}", ""])
            
            with open(hbonds_dir / 'hbond_residues_summary.txt', 'w') as f:
                f.write('\n'.join(summary))
            
        except Exception as e:
            print(f"Error extracting residues: {e}")
    
    def _convert_hbond_matrix(self, hbonds_dir: Path):
        xpm_file = hbonds_dir / 'hbond_matrix.xpm'
        eps_file = hbonds_dir / 'hbond_matrix.eps'

        if not xpm_file.exists():
            print(f"Matrix {xpm_file.name} not found. Cannot convert to EPS.")
            return
        
        try:
            rel_xpm = xpm_file.relative_to(self.results_dir)
            rel_eps = eps_file.relative_to(self.results_dir)
            
            cmd = [
                self.gmx_bin, 'xpm2ps', 
                '-f', str(rel_xpm),
                '-o', str(rel_eps)
            ]
            success, stderr = self._run_gmx_command(cmd, '\n')
            
            if not success or not eps_file.exists():
                print("Failed to convert the XPM matrix. Check GROMACS logs.") 
        except ValueError:
            print(f"Path error: {xpm_file} is not relative to {self.results_dir}")
    
    def plot_time_series(self):
        plots_dir = self.analysis_dirs['plots']
        if self.is_ligand_only:
            for method in (self._plot_energy_analysis, self._plot_ligand_only):
                try:
                    method(plots_dir)
                except Exception as e:
                    with open(self.results_dir / "gmx_analysis_errors.log", "a") as f:
                        f.write(f"Plotting Error in {method.__name__}: {e}\n")
            return
        plot_methods = [
            self._plot_energy_analysis,
            self._plot_rmsd_analysis,
            self._plot_rmsf_analysis,
            self._plot_structural_properties,
            self._plot_sasa_analysis,
            self._plot_hbonds_analysis,
            self._plot_hbonds_detailed,
            self._plot_ligand_contacts,
            self._create_dashboard
        ]
        with step_bar("Visualization Rendering", len(plot_methods)) as pbar:
            for method in plot_methods:
                try:
                    method(plots_dir)
                except Exception as e:
                    with open(self.results_dir / "gmx_analysis_errors.log", "a") as f:
                        f.write(f"Plotting Error in {method.__name__}: {e}\n")
                pbar.update(1)
    
    def _plot_hbonds_detailed(self, output_dir: Path):
        if not self.has_ligand:
            return
        
        hbonds_dir = self.analysis_dirs['hbonds']
        hbond_file = hbonds_dir / 'hbond_prot_lig.xvg'
        
        if not hbond_file.exists():
            return
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Detailed Protein-Ligand Hydrogen Bond Analysis', fontsize=16, fontweight='bold')
            
            # 1. H-bonds vs Time (Temporal Evolution)
            data = self._read_xvg(str(hbond_file))
            if data is not None:
                axes[0, 0].plot(data[:, 0], data[:, 1], color=COLORS[0], linewidth=1)
                mean_val = np.mean(data[:, 1])
                axes[0, 0].axhline(y=mean_val, color='r', linestyle='--', label=f'Mean: {mean_val:.2f}')
                axes[0, 0].set_xlabel('Time (ps)', fontweight='bold')
                axes[0, 0].set_ylabel('Number of H-bonds', fontweight='bold')
                axes[0, 0].set_title('Temporal Evolution', fontweight='bold')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)

            # 2. Distance Distribution
            dist_file = hbonds_dir / 'hbond_prot_lig_dist.xvg'
            if dist_file.exists():
                data_dist = self._read_xvg(str(dist_file))
                if data_dist is not None and data_dist.shape[1] > 1:
                    distances = data_dist[:, 1:].flatten()
                    distances = distances[distances > 0] # Filter non-existent bonds
                    if len(distances) > 0:
                        axes[0, 1].hist(distances, bins=40, color=COLORS[1], alpha=0.7, edgecolor='black')
                        axes[0, 1].axvline(x=np.mean(distances), color='r', linestyle='--', label=f'Mean: {np.mean(distances):.3f} nm')
                        axes[0, 1].set_xlabel('Distance (nm)', fontweight='bold')
                        axes[0, 1].set_ylabel('Frequency', fontweight='bold')
                        axes[0, 1].set_title('Distance Distribution', fontweight='bold')
                        axes[0, 1].legend()

            # 3. Angle Distribution
            angle_file = hbonds_dir / 'hbond_prot_lig_angle.xvg'
            if angle_file.exists():
                data_angle = self._read_xvg(str(angle_file))
                if data_angle is not None and data_angle.shape[1] > 1:
                    angles = data_angle[:, 1:].flatten()
                    angles = angles[angles > 0]
                    if len(angles) > 0:
                        axes[1, 0].hist(angles, bins=40, color=COLORS[2], alpha=0.7, edgecolor='black')
                        axes[1, 0].axvline(x=np.mean(angles), color='r', linestyle='--', label=f'Mean: {np.mean(angles):.1f}°')
                        axes[1, 0].set_xlabel('Angle (degrees)', fontweight='bold')
                        axes[1, 0].set_ylabel('Frequency', fontweight='bold')
                        axes[1, 0].set_title('Angle Distribution', fontweight='bold')
                        axes[1, 0].legend()

            # 4. Inter vs Intra Comparison
            found_comp = False
            for filename, label, color in [('hbond_prot_lig.xvg', 'Inter (Prot-Lig)', COLORS[0]),
                                          ('hbond_protein_intra.xvg', 'Intra (Protein)', COLORS[3])]:
                filepath = hbonds_dir / filename
                if filepath.exists():
                    data_comp = self._read_xvg(str(filepath))
                    if data_comp is not None:
                        axes[1, 1].plot(data_comp[:, 0], data_comp[:, 1], label=label, color=color, alpha=0.6)
                        found_comp = True
            
            if found_comp:
                axes[1, 1].set_xlabel('Time (ps)', fontweight='bold')
                axes[1, 1].set_ylabel('Number of H-bonds', fontweight='bold')
                axes[1, 1].set_title('Inter vs Intra Comparison', fontweight='bold')
                axes[1, 1].legend()
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(output_dir / 'hbonds_detailed.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            plt.close()
    
    def _plot_energy_analysis(self, output_dir: Path):
        try:
            fig, axes = plt.subplots(3, 2, figsize=(15, 12))
            fig.suptitle('MD Simulation Energy Analysis', fontsize=16, fontweight='bold')
            
            plots = [('energy_potential.xvg', 'Potential Energy', 'kJ/mol', axes[0, 0]),
                    ('energy_kinetic.xvg', 'Kinetic Energy', 'kJ/mol', axes[0, 1]),
                    ('energy_total.xvg', 'Total Energy', 'kJ/mol', axes[1, 0]),
                    ('temperature_md.xvg', 'Temperature', 'K', axes[1, 1]),
                    ('density.xvg', 'Density', 'kg/m³', axes[2, 0]),
                    ('volume.xvg', 'Volume', 'nm³', axes[2, 1])]
            
            for filename, title, ylabel, ax in plots:
                filepath = self.analysis_dirs['energy'] / filename
                if filepath.exists():
                    data = self._read_xvg(str(filepath))
                    if data is not None:
                        ax.plot(data[:, 0], data[:, 1], linewidth=1.5, color=COLORS[0])
                        ax.set_xlabel('Time (ps)', fontweight='bold')
                        ax.set_ylabel(ylabel, fontweight='bold')
                        ax.set_title(title, fontweight='bold')
                        ax.grid(True, alpha=0.3)
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(output_dir / 'energy_time.png', dpi=300, bbox_inches='tight')
            plt.close()
        except Exception:
            plt.close()
    
    def _plot_rmsd_analysis(self, output_dir: Path):
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            files = [('rmsd_backbone.xvg', 'Backbone', COLORS[0]),
                    ('rmsd_calpha.xvg', 'C-alpha', COLORS[1]),
                    ('rmsd_protein.xvg', 'Full Protein', COLORS[2])]
            
            if self.sim_type == "4":
                files.extend([('rmsd_dna.xvg', 'DNA (Structure)', COLORS[3]),
                             ('rmsd_dna_fit_prot.xvg', 'DNA (Protein Fit)', COLORS[5])])
            
            if self.has_ligand:
                files.extend([('rmsd_ligand_fit_protein.xvg', 'Ligand (Protein Fit)', COLORS[3])])

            found = False
            for filename, label, color in files:
                filepath = self.analysis_dirs['rmsd'] / filename
                if filepath.exists():
                    data = self._read_xvg(str(filepath))
                    if data is not None:
                        ax.plot(data[:, 0], data[:, 1], label=label, linewidth=1.5, color=color)
                        found = True
            
            if found:
                ax.set_xlabel('Time (ns)', fontsize=12, fontweight='bold')
                ax.set_ylabel('RMSD (nm)', fontsize=12, fontweight='bold')
                ax.set_title('Root Mean Square Deviation (RMSD)', fontsize=14, fontweight='bold')
                ax.legend(loc='best', frameon=True, shadow=True)
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(output_dir / 'rmsd_time.png', dpi=300)
            plt.close()
        except Exception:
            plt.close()
    
    def _plot_rmsf_analysis(self, output_dir: Path):
        try:
            fig, ax = plt.subplots(figsize=(14, 6))
            found = False
            tasks = [
                ('rmsf_calpha.xvg', 'C-alpha', COLORS[0]),
                ('rmsf_backbone.xvg', 'Backbone', COLORS[1])
            ]
            
            for filename, label, color in tasks:
                filepath = self.analysis_dirs['rmsf'] / filename
                if filepath.exists():
                    data = self._read_xvg(str(filepath))
                    if data is not None:
                        ax.plot(data[:, 0], data[:, 1], label=label, linewidth=1.5, color=color)
                        found = True
                        
            if found:
                ax.set_xlabel('Residue Number', fontsize=12, fontweight='bold')
                ax.set_ylabel('RMSF (nm)', fontsize=12, fontweight='bold')
                ax.set_title('Root Mean Square Fluctuation per Residue', fontsize=14, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.savefig(output_dir / 'rmsf_residues.png', dpi=300)
            plt.close()
        except Exception as e:
            # Errors are logged to the central log file instead of terminal
            with open(self.results_dir / "gmx_analysis_errors.log", "a") as f:
                f.write(f"RMSF Plotting Error: {e}\n")
            plt.close()
    
    def _plot_structural_properties(self, output_dir: Path):
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            found = False
            tasks = [
                ('gyrate_protein.xvg', 'Full Protein', COLORS[0]),
                ('gyrate_backbone.xvg', 'Backbone', COLORS[1]),
                ('gyrate_calpha.xvg', 'C-alpha', COLORS[2])
            ]
            
            for filename, label, color in tasks:
                filepath = self.analysis_dirs['structural'] / filename
                if filepath.exists():
                    data = self._read_xvg(str(filepath))
                    if data is not None:
                        ax.plot(data[:, 0], data[:, 1], label=label, linewidth=1.5, color=color)
                        found = True
                        
            if found:
                ax.set_xlabel('Time (ps)', fontweight='bold')
                ax.set_ylabel('Radius of Gyration (nm)', fontweight='bold')
                ax.set_title('System Compactness (Radius of Gyration)', fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.savefig(output_dir / 'gyrate_time.png', dpi=300)
            plt.close()
        except Exception as e:
            with open(self.results_dir / "gmx_analysis_errors.log", "a") as f:
                f.write(f"Radius of Gyration Plotting Error: {e}\n")
            plt.close()
    
    def _plot_sasa_analysis(self, output_dir: Path):
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            found = False
            files = [('sasa_protein.xvg', 'Protein', COLORS[0])]
            
            if self.sim_type == "4":
                files.append(('sasa_dna.xvg', 'DNA/RNA', COLORS[4]))
                
            if self.has_ligand: 
                files.extend([
                    ('sasa_ligand.xvg', 'Ligand', COLORS[1]), 
                    ('sasa_complex.xvg', 'Complex', COLORS[2])
                ])
                
            for filename, label, color in files:
                filepath = self.analysis_dirs['sasa'] / filename
                if filepath.exists():
                    data = self._read_xvg(str(filepath))
                    if data is not None:
                        ax.plot(data[:, 0], data[:, 1], label=label, color=color)
                        found = True
                        
            if found:
                ax.set_xlabel('Time (ps)', fontsize=12, fontweight='bold')
                ax.set_ylabel('SASA (nm²)', fontsize=12, fontweight='bold')
                ax.set_title('Solvent Accessible Surface Area (SASA)', fontsize=14, fontweight='bold')
                ax.legend(loc='best', frameon=True)
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(output_dir / 'sasa_time.png', dpi=300)
            plt.close()
        except Exception as e:
            with open(self.results_dir / "gmx_analysis_errors.log", "a") as f:
                f.write(f"SASA Plotting Error: {e}\n")
            plt.close()
    
    def _plot_hbonds_analysis(self, output_dir: Path):
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            # Define files to plot
            files = [('hbond_protein_intra.xvg', 'Protein (intra)', COLORS[0])]
            if self.has_ligand: 
                files.append(('hbond_prot_lig.xvg', 'Protein-Ligand', COLORS[1]))
            
            found = False
            for filename, label, color in files:
                filepath = self.analysis_dirs['hbonds'] / filename
                if filepath.exists():
                    data = self._read_xvg(str(filepath))
                    if data is not None:
                        ax.plot(data[:, 0], data[:, 1], label=label, color=color)
                        found = True
            
            if found:
                ax.set_xlabel('Time (ps)', fontweight='bold')
                ax.set_ylabel('Number of H-bonds', fontweight='bold')
                ax.set_title('Hydrogen Bond Evolution', fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.savefig(output_dir / 'hbonds_time.png', dpi=300)
            plt.close()
        except Exception as e: 
            with open(self.results_dir / "gmx_analysis_errors.log", "a") as f:
                f.write(f"H-bonds Plotting Error: {e}\n")
            plt.close()
    
    def _plot_ligand_contacts(self, output_dir: Path):
        if not self.has_ligand:
            return

        hbonds_dir = self.analysis_dirs['hbonds']
        contacts_file = hbonds_dir / 'contacts_prot_lig.xvg'
        
        if not contacts_file.exists():
            return

        try:
            data = self._read_xvg(str(contacts_file))
            if data is not None and data.shape[1] >= 2:
                plt.figure(figsize=(12, 6))
                # Plot primary data
                plt.plot(data[:, 0], data[:, 1], color='#2ca02c', linewidth=1.5, alpha=0.8)
                
                # Stability mean line
                mean_val = np.mean(data[:, 1])
                plt.axhline(y=mean_val, color='r', linestyle='--', 
                            label=f'Mean: {mean_val:.2f}')
                
                plt.title('Protein-Ligand Short Range Contacts (<0.35 nm)', fontweight='bold')
                plt.xlabel('Time (ps)', fontweight='bold')
                plt.ylabel('Number of Contacts', fontweight='bold')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(output_dir / 'ligand_contacts_time.png', dpi=300)
                plt.close()
        except Exception as e:
            with open(self.results_dir / "gmx_analysis_errors.log", "a") as f:
                f.write(f"Ligand Contacts Plotting Error: {e}\n")
            plt.close()

    def _create_dashboard(self, output_dir: Path):
        try:
            fig = plt.figure(figsize=(20, 12))
            gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
            
            # Map of (GridPosition, DirectoryKey, Filename, Title, X-Label, Y-Label)
            plots = [
                (gs[0, 0], 'rmsd', 'rmsd_backbone.xvg', 'Backbone RMSD', 'Time (ns)', 'RMSD (nm)'),
                (gs[0, 1], 'energy', 'potential.xvg', 'Potential Energy', 'Time (ps)', 'kJ/mol'),
                (gs[0, 2], 'energy', 'temperature_md.xvg', 'Temperature', 'Time (ps)', 'Temp (K)'),
                (gs[1, 0], 'structural', 'gyrate_protein.xvg', 'Radius of Gyration', 'Time (ps)', 'Rg (nm)'),
                (gs[1, 1], 'sasa', 'sasa_protein.xvg', 'SASA (Protein)', 'Time (ps)', 'SASA (nm²)'),
                (gs[1, 2], 'hbonds', 'hbond_protein_intra.xvg', 'Intra Protein H-bonds', 'Time (ps)', 'Count'),
                (gs[2, 2], 'energy', 'density.xvg', 'Density', 'Time (ps)', 'kg/m³')
            ]
            
            for pos, dir_key, filename, title, xlabel, ylabel in plots:
                ax = fig.add_subplot(pos)
                filepath = self.analysis_dirs[dir_key] / filename
                
                if filepath.exists():
                    data = self._read_xvg(str(filepath))
                    if data is not None and data.size > 0:
                        ax.plot(data[:, 0], data[:, 1], color=COLORS[0], linewidth=1.5)
                        ax.set_title(title, fontweight='bold', fontsize=11)
                        ax.set_xlabel(xlabel, fontsize=9)
                        ax.set_ylabel(ylabel, fontsize=9)
                        ax.grid(True, alpha=0.3)
                else:
                    ax.text(0.5, 0.5, f'Data Not Found:\n{filename}', ha='center', va='center', fontsize=9, color='gray')
            
            # Specialized RMSF wide plot at the bottom
            ax_rmsf = fig.add_subplot(gs[2, :2])
            rmsf_path = self.analysis_dirs['rmsf'] / 'rmsf_calpha.xvg'
            if rmsf_path.exists():
                data = self._read_xvg(str(rmsf_path))
                if data is not None:
                    ax_rmsf.plot(data[:, 0], data[:, 1], color=COLORS[0], linewidth=1.5)
                    ax_rmsf.set_title('C-alpha RMSF per Residue', fontweight='bold', fontsize=11)
                    ax_rmsf.set_xlabel('Residue Number', fontsize=9)
                    ax_rmsf.set_ylabel('RMSF (nm)', fontsize=9)
                    ax_rmsf.grid(True, alpha=0.3)
            
            fig.suptitle('Comprehensive MD Analysis Dashboard', fontsize=18, fontweight='bold', y=0.99)
            plt.savefig(output_dir / 'summary_dashboard.png', dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            with open(self.results_dir / "gmx_analysis_errors.log", "a") as f:
                f.write(f"Dashboard Generation Error: {e}\n")
            plt.close()
    
    def generate_summary_report(self):
            """Generates a comprehensive statistical report from all analysis files."""
            report = ["="*80, "COMPREHENSIVE MOLECULAR DYNAMICS ANALYSIS SUMMARY", "="*80, "",
                    f"System Type: {'Protein-Ligand' if self.has_ligand else 'Protein Only'}",
                    f"Working Directory: {self.results_dir}", ""]
            
            def add_stats(title, key, filename, unit=''):
                """Helper to extract mean, std, and range from XVG data."""
                filepath = self.analysis_dirs[key] / filename
                if filepath.exists():
                    data = self._read_xvg(str(filepath))
                    if data is not None and data.ndim > 1 and data.shape[0] > 0:
                        values = data[:, 1]
                        report.extend([f"{title}:",
                                    f"  Average: {np.mean(values):.3f} ± {np.std(values):.3f} {unit}",
                                    f"  Range: {np.min(values):.3f} - {np.max(values):.3f} {unit}", ""])
                    else:
                        report.extend([f"{title}: No data available", ""])
                else:
                    report.extend([f"{title}: Analysis file not found", ""])
            
            # Define the structure of the report
            sections = [
                ("1. STABILITY ANALYSIS (RMSD)", [
                    ("Backbone RMSD", 'rmsd', 'rmsd_backbone.xvg', 'nm'),
                    ("C-alpha RMSD", 'rmsd', 'rmsd_calpha.xvg', 'nm'),
                    ("Full Protein RMSD", 'rmsd', 'rmsd_protein.xvg', 'nm')]),
                ("2. FLUCTUATION ANALYSIS (RMSF)", [("C-alpha RMSF", 'rmsf', 'rmsf_calpha.xvg', 'nm')]),
                ("3. STRUCTURAL PROPERTIES", [("Radius of Gyration", 'structural', 'gyrate_protein.xvg', 'nm')]),
                ("4. SURFACE ACCESSIBILITY (SASA)", [("Protein SASA", 'sasa', 'sasa_protein.xvg', 'nm²')]),
                ("5. THERMODYNAMIC PROPERTIES", [
                    ("Potential Energy", 'energy', 'energy_potential.xvg', 'kJ/mol'),
                    ("Kinetic Energy", 'energy', 'energy_kinetic.xvg', 'kJ/mol'),
                    ("Total Energy", 'energy', 'energy_total.xvg', 'kJ/mol'),
                    ("Temperature", 'energy', 'temperature_md.xvg', 'K'),
                    ("Density", 'energy', 'density.xvg', 'kg/m³'),
                    ("Volume", 'energy', 'volume.xvg', 'nm³')]),
                ("6. INTERACTION ANALYSIS (H-BONDS)", [("Intramolecular H-bonds", 'hbonds', 'hbond_protein_intra.xvg', 'counts')])
            ]
            
            # Iterate through sections and append to report list
            for section_title, stats_list in sections:
                report.extend([section_title, "-" * 80])
                for stat_title, key, fname, unit in stats_list:
                    add_stats(stat_title, key, fname, unit)
            
            # Append Ligand-specific interactions if applicable
            if self.has_ligand:
                report.extend(["7. DETAILED PROTEIN-LIGAND INTERACTIONS", "-" * 80])
                hb_dir = self.analysis_dirs['hbonds']
                
                # H-bond Distance logic
                dist_file = hb_dir / 'hbond_prot_lig_dist.xvg'
                if dist_file.exists():
                    data = self._read_xvg(str(dist_file))
                    if data is not None and data.ndim > 1:
                        distances = data[:, 1:].flatten()
                        distances = distances[distances > 0]
                        if len(distances) > 0:
                            report.append(f"Average H-bond Distance: {np.mean(distances):.3f} ± {np.std(distances):.3f} nm")
                
                # H-bond Angle logic
                angle_file = hb_dir / 'hbond_prot_lig_angle.xvg'
                if angle_file.exists():
                    data = self._read_xvg(str(angle_file))
                    if data is not None and data.ndim > 1:
                        angles = data[:, 1:].flatten()
                        angles = angles[angles > 0]
                        if len(angles) > 0:
                            report.append(f"Average H-bond Angle: {np.mean(angles):.1f} ± {np.std(angles):.1f}°")
                report.append("")
            
            report.extend(["="*80, "END OF SUMMARY REPORT", "="*80])
            
            # Save and export
            report_file = self.results_dir / 'ANALYSIS_SUMMARY.txt'
            try:
                report_file.write_text('\n'.join(report), encoding='utf-8')
            except Exception as e:
                with open(self.results_dir / "gmx_analysis_errors.log", "a") as f:
                    f.write(f"Report Generation Error: {e}\n")

    def detect_protein_protein(self):
        if self.has_ligand:
            return False, []

        pdb_file = self.results_dir / 'md.pdb'
        if not pdb_file.exists():
            return False, []

        chains = set()
        try:
            with open(pdb_file, 'r') as f:
                for line in f:
                    # ATOM records are standard for protein residues
                    if line.startswith('ATOM  '):
                        if len(line) >= 22:
                            chain_id = line[21].strip()
                            if chain_id: 
                                chains.add(chain_id)
                    
                    # Stop after the first frame to optimize performance on large trajectories
                    if line.startswith('ENDMDL'):
                        break
        except Exception as e:
            print(f"PDB parsing error for chain detection: {e}")
            return False, []

        sorted_chains = sorted(list(chains))
        if len(sorted_chains) > 1:
            # Report detected chains to the user
            return True, sorted_chains
        
        return False, []
    
    def analyze_ligand_only(self):
        """Structural analysis of a ligand-only system (no protein groups)."""
        rmsd = self.analysis_dirs['rmsd'].relative_to(self.results_dir)
        rmsf = self.analysis_dirs['rmsf'].relative_to(self.results_dir)
        struct = self.analysis_dirs['structural'].relative_to(self.results_dir)
        sasa = self.analysis_dirs['sasa'].relative_to(self.results_dir)
        hb = self.analysis_dirs['hbonds'].relative_to(self.results_dir)
        common = ['-f', 'md_center.xtc', '-n', 'index.ndx']
        jobs = [
            ('RMSD', [self.gmx_bin, 'rms', '-s', 'em.tpr', *common, '-tu', 'ns',
                      '-o', str(rmsd / 'rmsd_ligand.xvg')], "Other\nOther\n"),
            ('RMSF', [self.gmx_bin, 'rmsf', '-s', 'md.tpr', *common,
                      '-o', str(rmsf / 'rmsf_ligand.xvg'),
                      '-ox', str(rmsf / 'rmsf_ligand_avg.pdb')], "Other\n"),
            ('Gyration', [self.gmx_bin, 'gyrate', '-s', 'md.tpr', *common,
                          '-o', str(struct / 'gyrate_ligand.xvg')], "Other\n"),
            ('SASA', [self.gmx_bin, 'sasa', '-s', 'md.tpr', *common,
                      '-o', str(sasa / 'sasa_ligand.xvg')], "Other\n"),
            # gmx hbond cannot classify GAFF atom types (no atomic numbers in the
            # ACPYPE atomtypes), so count ligand-water contacts instead.
            ('Contacts', [self.gmx_bin, 'mindist', '-s', 'md.tpr', *common, '-d', '0.35',
                          '-on', str(hb / 'contacts_ligand_water.xvg')], "Other\nSOL\n"),
        ]
        with step_bar("Ligand Analysis", len(jobs)) as pbar:
            for _label, cmd, sel in jobs:
                self._run_gmx_command(cmd, sel)
                pbar.update(1)

    def _plot_ligand_only(self, output_dir: Path):
        panels = [
            (self.analysis_dirs['rmsd'] / 'rmsd_ligand.xvg', 'Ligand RMSD', 'Time (ns)', 'RMSD (nm)'),
            (self.analysis_dirs['structural'] / 'gyrate_ligand.xvg', 'Radius of gyration', 'Time (ps)', 'Rg (nm)'),
            (self.analysis_dirs['sasa'] / 'sasa_ligand.xvg', 'SASA', 'Time (ps)', 'SASA (nm²)'),
            (self.analysis_dirs['hbonds'] / 'contacts_ligand_water.xvg', 'Ligand-water contacts (<0.35 nm)', 'Time (ps)', 'Count'),
            (self.analysis_dirs['rmsf'] / 'rmsf_ligand.xvg', 'RMSF per atom', 'Atom', 'RMSF (nm)'),
        ]
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        fig.suptitle('Ligand-only Molecular Dynamics', fontsize=16, fontweight='bold')
        flat = axes.flatten()
        for ax, (path, title, xl, yl) in zip(flat, panels):
            data = self._read_xvg(str(path)) if path.exists() else None
            if data is not None and data.ndim > 1:
                ax.plot(data[:, 0], data[:, 1], linewidth=1.5, color=COLORS[0])
            ax.set_title(title, fontweight='bold')
            ax.set_xlabel(xl)
            ax.set_ylabel(yl)
            ax.grid(True, alpha=0.3)
        for ax in flat[len(panels):]:
            ax.axis('off')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(output_dir / 'summary_dashboard.png', dpi=200, bbox_inches='tight')
        plt.savefig(output_dir / 'rmsd_time.png', dpi=200, bbox_inches='tight')
        plt.close()

    def _write_ligand_only_summary(self):
        lines = ["=" * 60, "LIGAND-ONLY MD ANALYSIS SUMMARY", "=" * 60, ""]
        entries = [
            ('Ligand RMSD (nm)', 'rmsd', 'rmsd_ligand.xvg'),
            ('Radius of gyration (nm)', 'structural', 'gyrate_ligand.xvg'),
            ('SASA (nm²)', 'sasa', 'sasa_ligand.xvg'),
            ('Ligand-water contacts (<0.35 nm)', 'hbonds', 'contacts_ligand_water.xvg'),
        ]
        for title, key, name in entries:
            path = self.analysis_dirs[key] / name
            data = self._read_xvg(str(path)) if path.exists() else None
            if data is not None and data.ndim > 1 and data.shape[0] > 0:
                v = data[:, 1]
                lines.append(f"{title}: mean {v.mean():.4f}  std {v.std():.4f}  "
                             f"min {v.min():.4f}  max {v.max():.4f}")
            else:
                lines.append(f"{title}: not available")
        (self.results_dir / 'ANALYSIS_SUMMARY.txt').write_text("\n".join(lines) + "\n")

    def run_full_analysis(self):
        if self.is_ligand_only:
            self.analyze_energy()
            self.analyze_ligand_only()
            self.plot_time_series()
            self._write_ligand_only_summary()
            return
        self.analyze_energy()
        self.analyze_rmsd()
        self.analyze_rmsf()
        self.analyze_gyration()
        self.analyze_sasa()
        self.analyze_hbonds_detailed()
        
        # 2. Rendering & Reporting
        self.plot_time_series()
        self.generate_summary_report()
        
        # 3. Advanced Correlation Analysis (DCCM)
        self.run_dccm_analysis_automatically()

        # 4. Automated System Identification (Protein-Protein)
        is_pp, chains = self.detect_protein_protein()
        if is_pp:
            try:
                # Flag file for downstream MM-PBSA modules
                sim_file = self.results_dir / 'SIMULATION_TYPE.txt'
                sim_file.write_text('Protein-Protein\n')
                
                # Append findings to the summary report without printing to console
                report_file = self.results_dir / 'ANALYSIS_SUMMARY.txt'
                if report_file.exists():
                    with open(report_file, 'a', encoding='utf-8') as f:
                        f.write('\nAUTOMATIC DETECTION: System identified as Protein-Protein Complex\n')
                        f.write(f'  Detected Chains: {", ".join(chains)}\n')
            except Exception:
                # Silent failure for non-critical meta-data
                pass
    
    def run_dccm_analysis_automatically(self):
        dccm_parent = self.results_dir / "dccm_analysis_pipeline"
        dccm_parent.mkdir(exist_ok=True)
        
        # Check required files silently
        required = [self.results_dir / 'md_center.xtc', self.results_dir / 'md.tpr', self.results_dir / 'index.ndx']
        if not all(f.exists() for f in required):
            return

        try:
            dccm_analyzer = DCCMAnalyzer(results_dir=str(dccm_parent), gmx_bin=self.gmx_bin)
            # This method has its own internal progress bars
            dccm_analyzer.run_pipeline_analysis(selection="C-alpha", max_residues=2000)
        except Exception as e:
            # Log technical errors to a file, keep console clean
            with open(self.results_dir / "analysis_errors.log", "a") as f:
                import traceback
                f.write(f"\nDCCM Orchestration Error:\n{traceback.format_exc()}") 

    def run_pipeline_analysis(self, plot_only=False):
        if plot_only:
            self.plot_time_series()
            if self.is_ligand_only:
                self._write_ligand_only_summary()
            else:
                self.generate_summary_report()
        else:
            self.run_full_analysis()
