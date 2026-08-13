import os
import subprocess
import logging
from pathlib import Path
from .mmpbsa_analysis import GMX_MMPBSA_Analyzer

logger = logging.getLogger(__name__)
class MMPBSAPPLigandAnalyzer:
    def __init__(self, results_dir: str, gmx_bin: str = None, n_cores: int = 1):
        self.results_dir = Path(results_dir)
        self.gmx_bin = gmx_bin
        self.n_cores = n_cores
        self.output_dir = self.results_dir / 'binding_energy_analysis' / 'gmx_MMPBSA'

    def _detect_protein_chains(self, pdb_file: Path):
        chains_info = {}
        with open(pdb_file, 'r') as f:
            for line in f:
                if line.startswith('ATOM'):
                    chain = line[21] if len(line) > 21 else ' '
                    if chain.strip():
                        if chain not in chains_info:
                            chains_info[chain] = {'atoms': [], 'residues': set()}
                        atom_num = int(line[6:11].strip())
                        res_seq = line[22:26].strip()
                        chains_info[chain]['atoms'].append(atom_num)
                        chains_info[chain]['residues'].add(res_seq)

        return sorted(chains_info.items(), key=lambda x: len(x[1]['residues']), reverse=True)

    def _get_atom_indices_for_chain(self, pdb_file: Path, chain_id: str):
        indices = []
        with open(pdb_file, 'r') as f:
            for line in f:
                if line.startswith('ATOM') and len(line) > 21 and line[21] == chain_id:
                    indices.append(int(line[6:11].strip()))
        return sorted(indices)

    def _create_index_file(self, pdb_file: Path, output_ndx: Path) -> bool:
        sorted_chains = self._detect_protein_chains(pdb_file)
        if len(sorted_chains) < 2:
            return False
            
        chain_a, chain_b = sorted_chains[0][0], sorted_chains[1][0]
        atoms_a = self._get_atom_indices_for_chain(pdb_file, chain_a)
        atoms_b = self._get_atom_indices_for_chain(pdb_file, chain_b)
        
        with open(output_ndx, 'w') as f:
            f.write("[ Protein ]\n")
            for i, atom in enumerate(atoms_a):
                f.write(f"{atom} " + ("\n" if (i + 1) % 15 == 0 else ""))
            f.write("\n\n[ Other ]\n")
            for i, atom in enumerate(atoms_b):
                f.write(f"{atom} " + ("\n" if (i + 1) % 15 == 0 else ""))
            f.write("\n")
        return True

    def run_analysis(self, use_pb: bool = True) -> bool:
        pdb_file = self.results_dir / 'md.pdb'
        index_file = self.results_dir / 'index.ndx'
        
        if not self._create_index_file(pdb_file, index_file):
            return False

        analyzer = GMX_MMPBSA_Analyzer(
            results_dir=str(self.results_dir),
            gmx_bin=self.gmx_bin,
            n_cores=self.n_cores,
        )
        return analyzer.run_analysis(use_pb=use_pb)
