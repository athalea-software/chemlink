"""Docking pipelines and orchestration."""

__all__ = ["DockingPipeline", "DockingPipelineResult"]


def __getattr__(name):
    # Lazy so importing pipelines.docking.steps doesn't pull in the whole
    # docking stack (openbabel, meeko, ...) up front.
    if name in __all__:
        from . import docking_pipeline
        return getattr(docking_pipeline, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
