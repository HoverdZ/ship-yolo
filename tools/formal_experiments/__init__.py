"""Formal Ocean Engineering experiment orchestration."""

from tools.formal_experiments.protocol import (
    FormalRunConfig,
    build_and_initialize,
    finalize_run,
    prepare_experiment,
    train_foreground,
)
from tools.formal_experiments.registry import (
    load_registry,
    validate_registry,
)

__all__ = [
    "FormalRunConfig",
    "build_and_initialize",
    "finalize_run",
    "load_registry",
    "prepare_experiment",
    "train_foreground",
    "validate_registry",
]
