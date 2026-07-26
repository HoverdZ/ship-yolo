"""Colab data helpers for ERUP/VGUP experiments.

The already-audited cumulative copy implementation is reused verbatim:
16-thread ``shutil.copyfile`` plus live file and byte progress. It deliberately
does not compare fixed local/Drive image or label counts.
"""

from colab.cumulative_training import (
    COPY_WORKERS,
    DRIVE_AUDIT_ROOT,
    DRIVE_RUNS_ROOT,
    LOCAL_DATA_YAML,
    copy_dataset_to_local,
    create_local_data_yaml,
)

__all__ = [
    "COPY_WORKERS",
    "DRIVE_AUDIT_ROOT",
    "DRIVE_RUNS_ROOT",
    "LOCAL_DATA_YAML",
    "copy_dataset_to_local",
    "create_local_data_yaml",
]
