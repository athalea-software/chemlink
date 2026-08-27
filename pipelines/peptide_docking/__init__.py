"""Peptide-protein docking pipeline (LightDock), separate from the
small-molecule AutoDock-GPU pipeline in pipelines.docking."""

from .peptide_docking_pipeline import PeptideDockingPipeline, PeptideDockingResult

__all__ = ["PeptideDockingPipeline", "PeptideDockingResult"]
