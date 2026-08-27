import os
import logging
from pathlib import Path
from utils.progress import step_bar
from .mmpbsa_analysis import GMX_MMPBSA_Analyzer

logger = logging.getLogger(__name__)

class MMPBSAPeptideAnalyzer:
    def __init__(self, results_dir: str, charge: int = 0, gmx_bin: str = None):
        self.results_dir = Path(results_dir)
        self.charge = charge
        self.gmx_bin = gmx_bin

    def _read_sim_type(self):
        summary = self.results_dir / 'SIMULATION_SUMMARY.txt'
        if not summary.exists():
            return 'Unknown'
            
        try:
            txt = summary.read_text(encoding='utf-8', errors='ignore')
            for line in txt.splitlines():
                # Check for English or Legacy Spanish labels
                if 'System Type:' in line or 'Sistema:' in line:
                    parts = line.split(':', 1)
                    if len(parts) > 1:
                        return parts[1].strip()
        except Exception as e:
            logger.debug(f"Error reading simulation summary: {e}")
        return 'Unknown'

    def _get_chain_information(self, pdb_file: Path):
        chains_residues = {}
        chains_atoms = {}
        
        with open(pdb_file, 'r') as f:
            for line in f:
                if line.startswith(('ATOM', 'HETATM')) and len(line) >= 22:
                    chain = line[21].strip()
                    if not chain:
                        continue
                        
                    res_seq = line[22:26].strip()
                    try:
                        atom_idx = int(line[6:11].strip())
                    except ValueError:
                        continue
                        
                    chains_residues.setdefault(chain, set()).add(res_seq)
                    chains_atoms.setdefault(chain, []).append(atom_idx)
                    
        return chains_residues, chains_atoms

    def _is_existing_index_valid(self, idx_path: Path) -> bool:
        if not idx_path.exists():
            return False
        try:
            content = idx_path.read_text()
            has_protein = '[ Protein ]' in content
            has_other = '[ Other ]' in content
            
            # gmx_MMPBSA expects exactly 2 groups for Receptor-Ligand
            if has_protein and has_other and content.count('[') == 2:
                return True
        except Exception:
            pass
        return False

    def _ensure_index_file(self, pdb_file: Path) -> bool:
        idx_path = self.results_dir / 'index.ndx'
        
        # 1. Respect manual index.ndx if already valid
        if self._is_existing_index_valid(idx_path):
            logger.debug("Valid index.ndx detected. Skipping generation.")
            return True

        # 2. Parse PDB for chain detection
        chains_residues, chains_atoms = self._get_chain_information(pdb_file)
        
        if not chains_atoms:
            logger.error("No valid chains found in the PDB structure.")
            return False

        # 3. Main logic (Residue-based) and Fallback (Atom-count based)
        sorted_chains = sorted(chains_residues.items(), key=lambda kv: len(kv[1]), reverse=True)
        
        if len(sorted_chains) < 2:
            logger.warning("Residue-based detection failed. Trying atom-count fallback...")
            # Plan B: Sort by atom count instead of unique residues
            sorted_chains = sorted(chains_atoms.items(), key=lambda kv: len(kv[1]), reverse=True)
            
            if len(sorted_chains) < 2:
                logger.error("Critical failure: Could not detect at least 2 chains using fallback logic.")
                return False 
        chain_a, chain_b = sorted_chains[0][0], sorted_chains[1][0]
        protein_atoms = chains_atoms[chain_a]
        other_atoms = chains_atoms[chain_b]

        # 4. Write the index.ndx file ensuring exactly 2 groups
        try:
            with open(idx_path, 'w') as f:
                f.write('[ Protein ]\n')
                for i in range(0, len(protein_atoms), 15):
                    f.write(' '.join(str(x) for x in protein_atoms[i:i+15]) + '\n')
                
                f.write('\n[ Other ]\n')
                for i in range(0, len(other_atoms), 15):
                    f.write(' '.join(str(x) for x in other_atoms[i:i+15]) + '\n')
            return True
        except Exception as e:
            logger.error(f"Error writing index.ndx manually: {e}")
            return False

    def run_analysis(self, use_pb: bool = False) -> bool:
        if not self.results_dir.exists():
            logger.error(f"Directory not found: {self.results_dir}")
            return False

        pdb_file = self.results_dir / 'md.pdb'
        tpr_file = self.results_dir / 'md.tpr'

        if not pdb_file.exists() or not tpr_file.exists():
            logger.error("Required simulation files (md.pdb or md.tpr) are missing.")
            return False

        with step_bar("Protein-Peptide Interaction Analysis", 3) as pbar:
            try:
                # Step 1: System type detection
                self._read_sim_type()
                pbar.update(1)

                # Step 2: Index validation/generation with fallback support
                if not self._ensure_index_file(pdb_file):
                    return False
                pbar.update(1)

                # Step 3: Secure Execution
                analyzer = GMX_MMPBSA_Analyzer(str(self.results_dir), self.gmx_bin)
                success = analyzer.run_analysis(use_pb=use_pb)
                pbar.update(1)

                return success

            except Exception as e:
                logger.error(f"Analysis failed due to an unexpected error: {e}")
                return False