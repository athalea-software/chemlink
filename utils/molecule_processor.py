"""Molecular processing utilities using RDKit and Meeko."""

import io
import logging
import math
import os
import re
import shutil
import subprocess
import tempfile
import traceback
from pathlib import Path
from typing import Any, Callable, Tuple, List, Optional
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem
from meeko import MoleculePreparation, PDBQTWriterLegacy

from ..storage.file_manager import find_extension, find_compound_name, create_out_file
from .logger import setup_logger
from ..pipelines.ligand_validation import require_valid_ligand, perceive_pdb_bond_orders

logger = setup_logger(__name__)


class MoleculeProcessor:
    """Handles molecular structure processing and format conversion."""
    
    def __init__(self, num_conformers: int = 10, random_seed: int = 42):
        """Initialize processor with conformer generation parameters.
        
        Args:
            num_conformers: Number of conformers to generate
            random_seed: Random seed for reproducibility
        """
        self.num_conformers = num_conformers
        self.random_seed = random_seed
        self.rdkit_warnings: List[str] = []

    def _capture_rdkit_messages(self, operation: Callable[[], Any]) -> Any:
        """Run an RDKit operation without leaking native diagnostics to stderr."""
        rdkit_logger = logging.getLogger("rdkit")
        previous_handlers = list(rdkit_logger.handlers)
        previous_level = rdkit_logger.level
        previous_propagate = rdkit_logger.propagate
        capture = io.StringIO()
        handler = logging.StreamHandler(capture)

        rdBase.LogToPythonLogger()
        rdkit_logger.handlers = [handler]
        rdkit_logger.setLevel(logging.WARNING)
        rdkit_logger.propagate = False
        try:
            return operation()
        finally:
            rdkit_logger.handlers = previous_handlers
            rdkit_logger.setLevel(previous_level)
            rdkit_logger.propagate = previous_propagate
            rdBase.LogToCppStreams()

            for line in capture.getvalue().splitlines():
                message = re.sub(r"^\[\d{2}:\d{2}:\d{2}\]\s*", "", line).strip()
                if message and message not in self.rdkit_warnings:
                    self.rdkit_warnings.append(message)
    
    def read_molecule(self, filepath: str) -> Optional[Chem.Mol]:
        """Read molecule from file using RDKit.
        
        Args:
            filepath: Path to molecule file
            
        Returns:
            RDKit molecule object or None if read fails
        """
        ext = find_extension(filepath).lower()
        
        try:
            if ext == ".sdf":
                suppl = Chem.SDMolSupplier(filepath, removeHs=False)
                mol = suppl[0] if suppl and len(suppl) > 0 else None
            elif ext == ".mol2":
                mol = Chem.MolFromMol2File(filepath, removeHs=False)
            elif ext == ".mol":
                mol = Chem.MolFromMolFile(filepath, removeHs=False)
            elif ext == ".pdb":
                # PDB CONECT records carry no bond-order info, so a direct
                # parse collapses double/triple bonds and strands formal
                # charges (e.g. a nitrile reads as strained sp3; a pyrylium
                # O+ reads as neutral). Recover the real bond orders via
                # Open Babel's geometry-based perception before falling back.
                mol = perceive_pdb_bond_orders(Path(filepath), sanitize=True)
                if mol is None:
                    mol = Chem.MolFromPDBFile(filepath, removeHs=False)
            else:
                raise ValueError(f"Unsupported format: {ext}")
            
            return mol
        except Exception as e:
            logger.error(f"Failed to read molecule from {filepath}: {e}")
            return None
    
    def add_hydrogens(self, mol: Chem.Mol) -> Chem.Mol:
        """Add hydrogens to molecule.
        
        Args:
            mol: RDKit molecule
            
        Returns:
            Molecule with hydrogens added
        """
        return Chem.AddHs(mol)

    @staticmethod
    def _canonical_identity(mol: Chem.Mol) -> Tuple[str, int]:
        """Return stereochemistry-aware connectivity and total formal charge."""
        heavy = Chem.RemoveHs(Chem.Mol(mol))
        return (
            Chem.MolToSmiles(heavy, isomericSmiles=True, canonical=True),
            Chem.GetFormalCharge(heavy),
        )

    @staticmethod
    def _has_charged_phosphorus(mol: Chem.Mol) -> bool:
        return any(
            atom.GetAtomicNum() == 15 and atom.GetFormalCharge() != 0
            for atom in mol.GetAtoms()
        )

    def _generate_openbabel_conformer(
        self, source_file: str, reference_mol: Chem.Mol
    ) -> Chem.Mol:
        """Generate 3D coordinates for UFF-problematic chemistry with Open Babel."""
        executable = shutil.which("obabel")
        if not executable:
            raise RuntimeError(
                "Charged phosphorus requires the Open Babel 3D fallback, but obabel was not found"
            )

        with tempfile.TemporaryDirectory(prefix="chemlink-obabel-") as temp_dir:
            output_file = os.path.join(temp_dir, "conformer.sdf")
            result = subprocess.run(
                [executable, source_file, "-O", output_file, "--gen3d"],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode != 0 or not os.path.isfile(output_file):
                detail = (result.stderr or result.stdout or "no diagnostic output").strip()
                raise RuntimeError(f"Open Babel 3D generation failed: {detail}")
            generated = Chem.MolFromMolFile(output_file, removeHs=False)

        if generated is None or generated.GetNumConformers() == 0:
            raise RuntimeError("Open Babel did not produce a readable 3D conformer")
        if self._canonical_identity(generated) != self._canonical_identity(reference_mol):
            raise RuntimeError(
                "Open Babel 3D generation changed ligand connectivity, stereochemistry, or charge"
            )
        if not generated.GetConformer().Is3D():
            raise RuntimeError("Open Babel fallback returned coordinates not marked as 3D")
        return Chem.AddHs(generated, addCoords=True)

    def prepare_conformers(
        self, mol: Chem.Mol, source_file: str
    ) -> Tuple[Chem.Mol, List[int], List[str]]:
        """Select a safe 3D backend and return molecule, conformer IDs, diagnostics."""
        if self._has_charged_phosphorus(mol):
            generated = self._generate_openbabel_conformer(source_file, mol)
            return generated, [generated.GetConformer().GetId()], [
                "charged phosphorus detected; generated 3D geometry with Open Babel "
                "instead of UFF/ETKDG, preserving connectivity, stereochemistry, and charge"
            ]
        return mol, self.generate_conformers(mol), []
    
    def generate_conformers(self, mol: Chem.Mol) -> List[int]:
        """Generate 3D conformers using ETKDG.
        
        Args:
            mol: RDKit molecule
            
        Returns:
            List of conformer IDs
            
        Raises:
            RuntimeError: If conformer generation fails
        """
        params = AllChem.ETKDGv3()
        params.numThreads = 1
        params.pruneRmsThresh = 0.5
        params.randomSeed = self.random_seed
        params.useRandomCoords = True
        params.maxIterations = 1000
        
        conf_ids = self._capture_rdkit_messages(
            lambda: AllChem.EmbedMultipleConfs(
                mol,
                numConfs=self.num_conformers,
                params=params,
            )
        )
        
        if len(conf_ids) == 0:
            raise RuntimeError("Conformer embedding failed")
        
        return list(conf_ids)
    
    def optimize_conformers(self, mol: Chem.Mol, conf_ids: List[int]) -> Tuple[int, List[str]]:
        """Optimize conformers with MMFF and return best conformer ID.
        
        Args:
            mol: RDKit molecule with conformers
            conf_ids: List of conformer IDs to optimize
            
        Returns:
            Tuple of (best_conformer_id, warnings)
        """
        warnings = []
        
        if not AllChem.MMFFHasAllMoleculeParams(mol):
            warnings.append("MMFF parameters missing, skipping optimization")
            return conf_ids[0], warnings
        
        results = AllChem.MMFFOptimizeMoleculeConfs(
            mol,
            numThreads=1,
            maxIters=1000,
        )
        
        valid_ids = []
        valid_energies = []
        
        for cid, (status, energy) in zip(conf_ids, results):
            if status == 0:  # Converged
                valid_ids.append(cid)
                valid_energies.append(energy)
        
        if valid_ids:
            best_idx = valid_energies.index(min(valid_energies))
            return valid_ids[best_idx], warnings
        else:
            raise RuntimeError(
                "No conformers converged in MMFF optimization after 1000 iterations"
            )
    
    def extract_best_conformer(self, mol: Chem.Mol, conf_id: int) -> Chem.Mol:
        """Create new molecule with only the best conformer.
        
        Args:
            mol: Original molecule with multiple conformers
            conf_id: ID of the conformer to extract
            
        Returns:
            New molecule with single conformer
        """
        best_mol = Chem.Mol(mol)
        best_conf = mol.GetConformer(conf_id)
        
        best_mol.RemoveAllConformers()
        best_mol.AddConformer(best_conf, assignId=True)
        
        return best_mol

    def validate_optimized_geometry(self, reference: Chem.Mol, optimized: Chem.Mol) -> None:
        """Reject optimized structures that changed identity or have invalid P geometry."""
        if self._canonical_identity(reference) != self._canonical_identity(optimized):
            raise RuntimeError(
                "Optimized ligand changed connectivity, stereochemistry, or formal charge"
            )
        if optimized.GetNumConformers() != 1 or not optimized.GetConformer().Is3D():
            raise RuntimeError("Optimized ligand does not contain one valid 3D conformer")

        conformer = optimized.GetConformer()
        for position in conformer.GetPositions():
            if not all(math.isfinite(float(value)) for value in position):
                raise RuntimeError("Optimized ligand contains non-finite coordinates")

        for atom in optimized.GetAtoms():
            if atom.GetAtomicNum() != 15:
                continue
            phosphorus = conformer.GetAtomPosition(atom.GetIdx())
            for bond in atom.GetBonds():
                other = bond.GetOtherAtom(atom)
                if other.GetAtomicNum() != 8:
                    continue
                oxygen = conformer.GetAtomPosition(other.GetIdx())
                distance = phosphorus.Distance(oxygen)
                if not 1.20 <= distance <= 2.00:
                    raise RuntimeError(
                        f"Invalid optimized P-O distance {distance:.2f} A at atom {atom.GetIdx()}"
                    )


def process_ligand(ligand_file: str, output_path: str) -> Tuple[str, bool, Optional[dict], List[str]]:
    """Process a single ligand file: prepare, optimize, and convert formats.
    
    This is a standalone function suitable for multiprocessing.
    
    Args:
        ligand_file: Path to input ligand file
        output_path: Base output directory
        
    Returns:
        Tuple of (ligand_name, success, error_details, warnings)
    """
    ligand_name = find_compound_name(ligand_file)
    warnings = []
    
    try:
        processor = MoleculeProcessor()

        # Fail before conformer generation/Meeko when the source graph or
        # supplied geometry is chemically broken.
        validation = require_valid_ligand(ligand_file)
        warnings.extend(validation.warnings)
        
        # Read molecule
        mol = processor.read_molecule(ligand_file)
        if mol is None:
            raise RuntimeError("Failed to read molecule")
        
        reference_mol = Chem.Mol(mol)

        # Add hydrogens
        mol = processor.add_hydrogens(mol)
        
        # Generate conformers using an engine compatible with the chemistry.
        mol, conf_ids, backend_warnings = processor.prepare_conformers(mol, ligand_file)
        warnings.extend(backend_warnings)
        warnings.extend(processor.rdkit_warnings)
        
        # Optimize and get best conformer
        best_conf_id, opt_warnings = processor.optimize_conformers(mol, conf_ids)
        warnings.extend(opt_warnings)
        
        # Extract best conformer
        mol = processor.extract_best_conformer(mol, best_conf_id)
        processor.validate_optimized_geometry(reference_mol, mol)

        # Parameterize and write PDBQT directly with Meeko (no MOL2 intermediate)
        mk_prep = MoleculePreparation()
        molsetup = mk_prep(mol)[0]
        pdbqt_string, is_ok, err_msg = PDBQTWriterLegacy.write_string(molsetup)
        if not is_ok:
            raise RuntimeError(f"Meeko failed to write PDBQT: {err_msg}")

        pdbqt_output = create_out_file(
            f'{output_path}/prepared_ligands_pdbqt',
            f'{ligand_name}.pdbqt'
        )
        with open(pdbqt_output, 'w') as handle:
            handle.write(pdbqt_string)

        return (ligand_name, True, None, warnings)
        
    except Exception as e:
        error_details = {
            'ligand': ligand_name,
            'file': ligand_file,
            'error': str(e),
            'traceback': traceback.format_exc(),
            'warnings': warnings
        }
        logger.error(f"{ligand_name}: Failed to prepare - {str(e)}")
        return (ligand_name, False, error_details, warnings)
