from pathlib import Path
import json
import subprocess
import os
import datetime
from tqdm import tqdm

from ..logger import get_step_logger

from pipelines.dynamics.md_analysis import GromacsAnalyzer
from pipelines.dynamics.mmpbsa_analysis import GMX_MMPBSA_Analyzer
from pipelines.dynamics.run_mmpbsa_for_pp_with_ligand import MMPBSAPPLigandAnalyzer

class AnalysisStep:
    def __init__(self, config, gmx_bin):
        self.config = config
        self.gmx_bin = gmx_bin
        self.results_dir = Path(config["work_dir"])
        self.threads = config.get("threads", 8)
        self.logger = get_step_logger(__name__, str(self.results_dir / "simulation.log"))

        self.base_path = "pipeline/dynamics"
        self.mmpbsa_peptide = os.path.join(self.base_path, "run_mmpbsa_for_peptide.py")
        self.mmpbsa_pp_ligand = os.path.join(self.base_path, "run_mmpbsa_for_pp_with_ligand.py")
        
    def run(self):
        with tqdm(total=3, desc="  └─ Comprehensive System Analysis", leave=False) as pbar:
            sim_type = self.config["sim_type"]
            try:
                # 1. GENERAL STRUCTURAL ANALYSIS (RMSD, RMSF, Rg, SASA)
                self._run_general_analysis()
                pbar.update(1)
                # 2. BINDING ENERGY ANALYSIS (MM-PBSA/MM-GBSA)
                self._run_binding_energy_analysis()
                pbar.update(1)
                # 3. FINAL REPORT GENERATION
                self._generate_final_report()
                pbar.update(1)
            except Exception as e:
                self.logger.exception("Critical Error during Analysis Phase")
                raise e

    def _run_general_analysis(self):
        try:
            analyzer = GromacsAnalyzer(
                results_dir=self.results_dir, 
                sim_type=str(self.config.get("sim_type")), 
                gmx_bin=self.gmx_bin,
                logger=self.logger,
            )
            for key in analyzer.analysis_dirs:
                os.makedirs(analyzer.analysis_dirs[key], exist_ok=True)
            # Run analysis
            analyzer.run_pipeline_analysis(plot_only=self.config.get("plot_only", False))
            return True
        except Exception as e:
            self.logger.exception("Critical Error in General Analysis")
            return False

    def _execute_mmpbsa(self, analyzer, label="no specified"):
        required_files = ['md_center.xtc', 'md.tpr', 'index.ndx']
        missing = [f for f in required_files if not (Path(self.results_dir) / f).exists()]
        
        if missing:
            self.logger.warning("MM-PBSA skipped. Missing required files: %s", ", ".join(missing))
            return False
        try:
            success = analyzer.run_analysis(use_pb=True)
            if not success:
                self.logger.warning("MM-PBSA analysis for '%s' completed with warnings.", label)
            return bool(success)
        except Exception as e:
            self.logger.exception("MM-PBSA analysis failed for '%s'", label)
            return False

    def _run_binding_energy_analysis(self):
        sim_type = self.config["sim_type"]
        if not self.config.get("binding_energy", True):
            self.logger.info(
                "Binding-energy analysis disabled by configuration.",
                extra={"file_only": True},
            )
            return True
        analyzer = None
        if sim_type == "2":
            analyzer = GMX_MMPBSA_Analyzer(
                str(self.results_dir), self.gmx_bin,
                method=self.config.get("mmpbsa_method", "both"),
                target_frames=self.config.get("mmpbsa_frames", 1000),
                discard_percent=self.config.get("mmpbsa_discard", 20.0),
                n_cores=self.config.get("mmpbsa_cores", self.threads),
                salt_concentration=self.config.get("mmpbsa_salt", 0.150),
                timeout=self.config.get("mmpbsa_timeout"),
            )
            label = "Protein-Ligand"
        elif sim_type in ["3", "5"]:
            analyzer = GMX_MMPBSA_Analyzer(
                str(self.results_dir), self.gmx_bin,
                method="gb",
                binding_system="protein-protein",
                target_frames=self.config.get("mmpbsa_frames", 1000),
                discard_percent=self.config.get("mmpbsa_discard", 20.0),
                n_cores=self.config.get("mmpbsa_cores", self.threads),
                salt_concentration=self.config.get("mmpbsa_salt", 0.150),
                timeout=self.config.get("mmpbsa_timeout"),
            )
            label = "Protein-Protein/Peptide"
        elif sim_type == "6":
            analyzer = MMPBSAPPLigandAnalyzer(
                results_dir=str(self.results_dir), 
                gmx_bin=self.gmx_bin,
                n_cores=self.config.get("mmpbsa_cores", self.threads),
            )
            label = "P-P + Cofactor"

        if analyzer is not None:
            return self._execute_mmpbsa(analyzer, label)
        self.logger.info("Binding-energy analysis is not available for simulation type %s.", sim_type)
        return False

    def _verify_mmpbsa_results(self):
        output_dir = os.path.join(self.results_dir, "analisis_binding_energy/gmx_MMPBSA")
        if os.path.exists(output_dir):
            files = ["SUMMARY_REPORT.txt", "FINAL_RESULTS_MMPBSA.dat", "binding_energy.png"]
            found = [f for f in files if os.path.exists(os.path.join(output_dir, f))]
            if found:
                self.logger.info("Resultados verificados: %s archivos generados.", len(found))

    def _generate_final_report(self):
        report_path = self.results_dir / "SIMULATION_SUMMARY.txt"
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        sim_label = self.config.get("sim_type_label", "Standard MD")
        ns = self.config.get("ns_time", "0")
        force_field = self.config.get("resolved_force_field")
        if not force_field:
            requested = self.config.get("force_field", "default")
            if requested != "default":
                force_field = requested
            elif self.config.get("sim_type") in {"3", "4", "5", "6"}:
                force_field = "amber99sb-ildn"
            else:
                force_field = "amber03"
        water_model = self.config.get("water_model", "tip3p")
        
        binding_status = "not requested"
        binding_results = "N/A"
        metadata_path = self.results_dir / "binding_energy_analysis/gmx_MMPBSA/analysis_metadata.json"
        if metadata_path.is_file():
            try:
                metadata = json.loads(metadata_path.read_text())
                binding_status = metadata.get("status", "unknown")
                values = []
                for method, result in metadata.get("energies", {}).items():
                    if "total_mean" in result:
                        values.append(
                            f"MM-{method}SA {result['total_mean']:.2f} ± "
                            f"{result['total_std']:.2f} kcal/mol"
                        )
                binding_results = "; ".join(values) or "N/A"
            except (OSError, ValueError, TypeError):
                binding_status = "metadata unreadable"

        report_content = f"""
═══════════════════════════════════════════════════════════
                MD SIMULATION SUMMARY REPORT
═══════════════════════════════════════════════════════════
CONFIGURATION:
-------------
   System Type:       {sim_label}
   Simulation Time:   {ns} ns
   Working Directory: {self.results_dir}
   Force Field:       {force_field}
   Ligand Charge:     {self.config.get('ligand_charge', 'N/A')}
   Charge Method:     {self.config.get('ligand_charge_method', 'N/A')}
   Charge Confidence: {self.config.get('ligand_charge_confidence', 'N/A')}
   Binding Analysis:  {binding_status}
   Binding Energy:    {binding_results}
   Water Model:       {water_model}
   Date:              {timestamp}

KEY OUTPUT FILES:
----------------
   Plots/Dashboard:   plots/summary_dashboard.png
   Statistical Data:  ANALYSIS_SUMMARY.txt
   Final Trajectory:  md_center.xtc
   Final Structure:   md.pdb
   Index Groups:      index.ndx

NEXT STEPS:
-----------
  1. Review the dashboard in 'plots/' directory.
  2. Check RMSD convergence in 'plots/rmsd_time.png'.
  3. Visualize the corrected trajectory using:
     vmd md.pdb md_center.xtc

═══════════════════════════════════════════════════════════
"""
        with report_path.open("w", encoding="utf-8") as f:
            f.write(report_content)
