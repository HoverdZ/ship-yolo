"""Definitions of formal paper tables and one generic builder."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tools.paper_artifacts.results.common import table_rows, write_rows

TABLES: dict[str, dict[str, Any]] = {
    "cumulative_ablation": {
        "aliases": [("A0", "R00"), ("A1", "R02"), ("A2", "R04"), ("A3", "R10")],
        "title": "Cumulative ablation study",
        "extra": {
            "A0": {"DPLS": "no", "CA-SCAM": "no", "VGUP": "no"},
            "A1": {"DPLS": "yes", "CA-SCAM": "no", "VGUP": "no"},
            "A2": {"DPLS": "yes", "CA-SCAM": "yes", "VGUP": "no"},
            "A3": {"DPLS": "yes", "CA-SCAM": "yes", "VGUP": "yes"},
        },
    },
    "dpls_ablation": {
        "aliases": [("D0", "R00"), ("D1", "R01"), ("D2", "R02")],
        "title": "DPLS controlled ablation",
        "extra": {
            "D0": {"pyramid": "P3-P5", "upsampling": "Nearest"},
            "D1": {"pyramid": "P2-P4", "upsampling": "Nearest"},
            "D2": {"pyramid": "P2-P4", "upsampling": "DySample"},
        },
    },
    "ca_scam": {
        "aliases": [
            ("C0", "R02"),
            ("C1/CI0", "R03"),
            ("CI1", "R05A"),
            ("CI2", "R05B"),
            ("C2/CI3", "R04"),
        ],
        "title": "SCAM and CA-SCAM ablations",
        "extra": {
            "C0": {"calibration": "none"},
            "C1/CI0": {"calibration": "original SCAM"},
            "CI1": {"calibration": "contrast map + fixed beta"},
            "CI2": {"calibration": "contrast map + learnable unbounded beta"},
            "C2/CI3": {"calibration": "contrast map + learnable bounded beta"},
        },
    },
    "vgup": {
        "aliases": [
            ("V0", "R04"),
            ("V1", "R06"),
            ("VG0", "R07"),
            ("VG1", "R08"),
            ("VG2", "R09"),
            ("V2/VG3", "R10"),
        ],
        "title": "ERUP/VGUP and gate ablations",
        "extra": {
            "V0": {"preprocessor": "none", "global_gate": "no", "spatial_gate": "no"},
            "V1": {"preprocessor": "ERUP", "global_gate": "n/a", "spatial_gate": "n/a"},
            "VG0": {"preprocessor": "VGUP", "global_gate": "no", "spatial_gate": "no"},
            "VG1": {"preprocessor": "VGUP", "global_gate": "yes", "spatial_gate": "no"},
            "VG2": {"preprocessor": "VGUP", "global_gate": "no", "spatial_gate": "yes"},
            "V2/VG3": {"preprocessor": "VGUP", "global_gate": "yes", "spatial_gate": "yes"},
        },
    },
    "cross_dataset": {
        "aliases": [("S0", "S00"), ("S1", "S01")],
        "title": "Independent second-dataset evaluation",
        "extra": {},
    },
    "cross_model": {
        "aliases": [
            ("YOLO11n baseline", "R00"),
            ("YOLO11n final", "R10"),
            ("YOLOv8n baseline", "R11"),
            ("YOLOv8n final", "R12"),
        ],
        "title": "Cross-model generalization",
        "extra": {},
    },
    "complexity": {
        "aliases": [
            ("R00", "R00"),
            ("R01", "R01"),
            ("R02", "R02"),
            ("R03", "R03"),
            ("R04", "R04"),
            ("R05A", "R05A"),
            ("R05B", "R05B"),
            ("R06", "R06"),
            ("R07", "R07"),
            ("R08", "R08"),
            ("R09", "R09"),
            ("R10", "R10"),
            ("R11", "R11"),
            ("R12", "R12"),
        ],
        "title": "Model complexity and efficiency",
        "extra": {},
    },
}


def build(name: str, run_root: str | Path, output: str | Path):
    spec = TABLES[name]
    rows = table_rows(
        run_root,
        spec["aliases"],
        extra=spec["extra"],
    )
    preferred = [
        "paper_id",
        *sorted(
            {
                key
                for values in spec["extra"].values()
                for key in values
            }
        ),
        "precision",
        "recall",
        "map50",
        "map75",
        "map50_95",
        "params",
        "gflops",
        "latency_ms",
        "fps",
        "run_id",
        "seed",
        "status",
    ]
    return write_rows(
        rows,
        output,
        columns=preferred,
        title=spec["title"],
    )


__all__ = ["TABLES", "build"]
