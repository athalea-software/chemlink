import os
from pathlib import Path
from utils.progress import step_bar

from ..logger import get_step_logger

class ComplexBuilderStep:
    def __init__(self, config):
        self.config = config
        self.prot_path = os.path.abspath(self.config["pdb_protein"])
        self.partner_path = os.path.abspath(self.config["pdb_partner"])
        self.output_path = os.path.abspath(self.config["pdb_input"])
        self.logger = get_step_logger(__name__, os.path.join(self.config["work_dir"], "simulation.log"))

    def run(self):
        with step_bar("Building Molecular Complex", 2) as pbar:
            if not os.path.exists(self.config["work_dir"]):
                os.makedirs(self.config["work_dir"])

            try:
                with open(self.output_path, 'w') as f_out:
                    # Write one unambiguous chain per protein. Source TER/END records
                    # are deliberately discarded so pdb2gmx sees exactly two chains.
                    self._write_chain(self.prot_path, f_out, "A")
                    f_out.write("TER\n")
                    pbar.update(1)
                    self._write_chain(self.partner_path, f_out, "B")
                    f_out.write("TER\n")
                    f_out.write("END\n")
                    pbar.update(1) 
            except Exception as e:
                self.logger.exception("Error during complex assembly")
                raise e

    @staticmethod
    def _write_chain(source, output, chain_id):
        atom_count = 0
        with open(source, "r") as handle:
            for line in handle:
                if line.startswith(("ATOM  ", "HETATM")) and len(line) >= 22:
                    output.write(line[:21] + chain_id + line[22:])
                    atom_count += 1
                elif line.startswith("ANISOU") and len(line) >= 22:
                    output.write(line[:21] + chain_id + line[22:])
        if not atom_count:
            raise ValueError(f"No atoms were found in protein input: {Path(source)}")
