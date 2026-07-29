"""Deterministic dataset-scale analysis for the ship-detection paper."""

from .core import (
    INSTANCE_COLUMNS,
    analyze_dataset,
    box_geometry,
    dilution_metrics,
    quantile_linear,
)
__all__ = [
    "INSTANCE_COLUMNS",
    "analyze_dataset",
    "box_geometry",
    "dilution_metrics",
    "quantile_linear",
    "run_analysis",
]


def run_analysis(*args, **kwargs):
    """Import the pipeline lazily so module CLI entry points remain warning-free."""
    from .pipeline import run_analysis as _run_analysis

    return _run_analysis(*args, **kwargs)
