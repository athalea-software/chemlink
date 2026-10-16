"""Docking pipelines and orchestration."""

__all__ = ["DockingPipeline", "DockingPipelineResult"]


def __getattr__(name):
    # Lazy so importing the peptide_docking subpackage doesn't pull in the
    # small-molecule docking stack (openbabel, meeko, ...) it doesn't need.
    if name in __all__:
        from . import docking_pipeline
        return getattr(docking_pipeline, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
