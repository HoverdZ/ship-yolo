"""Generate a dedicated Colab notebook for the AMP-stable P2 auxiliary experiment."""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import build_spddown_p2aux_colab as base


OUTPUT = ROOT / "notebooks" / "YOLO11n_InceptionDW_P2GaussianAux_AMPStable.ipynb"
RUN_NAME = "yolo11n_inceptiondw_p2_gaussian_aux_ampstable_640"
INVALID_RUN_NAME = "yolo11n_inceptiondw_p2_gaussian_aux_640"


cells = [
    base.markdown(
        """# YOLO11n-InceptionDW: AMP-stable training-only P2 Gaussian supervision

This notebook runs only the corrected P2 auxiliary experiment. The earlier
`yolo11n_inceptiondw_p2_gaussian_aux_640` run is invalid because its auxiliary
loss became `inf`/`nan`; this notebook never resumes that checkpoint.

The corrected focal calculation is evaluated in FP32 with stable
`logsigmoid` identities. Model forward and native YOLO losses still use AMP.
Run cells in order. Formal training runs directly in the notebook kernel.
"""
    ),
    *deepcopy(base.notebook["cells"][1:5]),
    base.code(
        """# Cell 5 — verify the AMP-stable loss and run the P2-only 640 preflight
import inspect
import torch

from custom_modules.p2_gaussian_aux import dense_gaussian_focal_loss
from tools.check_spddown_p2aux import main as run_preflight

loss_source = inspect.getsource(dense_gaussian_focal_loss)
assert "logsigmoid" in loss_source
assert "logits.float()" in loss_source

extreme_logits = torch.tensor(
    [[[[-100.0, -20.0, -10.0, 0.0, 10.0, 20.0, 100.0]]]],
    dtype=torch.float16,
    requires_grad=True,
)
extreme_target = torch.tensor(
    [[[[1.0, 0.75, 0.25, 0.0, 0.0, 0.0, 0.0]]]],
    dtype=torch.float16,
)
stability_loss = dense_gaussian_focal_loss(extreme_logits, extreme_target)
stability_loss.backward()
assert stability_loss.dtype == torch.float32
assert torch.isfinite(stability_loss)
assert extreme_logits.grad is not None and torch.isfinite(extreme_logits.grad).all()
print("AMP stability check passed; loss:", float(stability_loss.detach()))

reports = run_preflight(
    [
        "--variant", "p2_gaussian_aux",
        "--weights", str(REPO / "yolo11n.pt"),
        "--imgsz", "640",
        "--output-dir", str(DRIVE_REPORTS / PINNED_COMMIT / "ampstable"),
    ]
)
assert reports["p2_gaussian_aux"]["all_checks_passed"]
print("P2 Gaussian auxiliary 640 preflight passed.")
"""
    ),
    base.markdown(
        f"""## Formal corrected experiment

This starts a clean run named `{RUN_NAME}`. It does not read the invalid
`{INVALID_RUN_NAME}/weights/last.pt` checkpoint. Do not change the new name
back to the invalid run name.
"""
    ),
    base.code(
        f"""# Cell 6 — START CLEAN FORMAL P2 GAUSSIAN AUXILIARY TRAINING
from tools.train_spddown_p2aux import TrainingRequest, run_training

RUN_NAME = "{RUN_NAME}"
INVALID_RUN_NAME = "{INVALID_RUN_NAME}"
invalid_checkpoint = DRIVE_RUNS / INVALID_RUN_NAME / "weights" / "last.pt"
print("Invalid checkpoint intentionally ignored:", invalid_checkpoint)

p2_request = TrainingRequest(
    variant="p2_gaussian_aux",
    data=str(LOCAL_DATA_YAML),
    project=str(DRIVE_RUNS),
    weights=str(REPO / "yolo11n.pt"),
    name=RUN_NAME,
    epochs=150,
    imgsz=640,
    batch=8,
    workers=2,
    device="0",
    seed=0,
    optimizer="auto",
    resume=False,
)
p2_result = run_training(p2_request)
"""
    ),
    base.code(
        """# Cell 7 — resume only the corrected run after an interruption
from tools.train_spddown_p2aux import TrainingRequest, run_training

RUN_NAME = "yolo11n_inceptiondw_p2_gaussian_aux_ampstable_640"
last_pt = DRIVE_RUNS / RUN_NAME / "weights" / "last.pt"
if not last_pt.is_file():
    raise FileNotFoundError(
        f"No corrected checkpoint found: {last_pt}. "
        "Never substitute the invalid p2_gaussian_aux_640 checkpoint."
    )
print("Resuming corrected checkpoint:", last_pt)

resume_request = TrainingRequest(
    variant="p2_gaussian_aux",
    data=str(LOCAL_DATA_YAML),
    project=str(DRIVE_RUNS),
    weights=str(REPO / "yolo11n.pt"),
    name=RUN_NAME,
    resume=True,
)
resume_result = run_training(resume_request)
"""
    ),
    base.code(
        """# Cell 8 — inspect metrics and reject any non-finite auxiliary loss
import numpy as np
import pandas as pd

RUN_NAME = "yolo11n_inceptiondw_p2_gaussian_aux_ampstable_640"
csv_path = DRIVE_RUNS / RUN_NAME / "results.csv"
if not csv_path.is_file():
    raise FileNotFoundError(csv_path)

frame = pd.read_csv(csv_path)
aux_column = "train/p2_aux_loss"
if aux_column not in frame:
    raise KeyError(f"Missing {aux_column} in {csv_path}")
if not np.isfinite(frame[aux_column].to_numpy(dtype=float)).all():
    raise RuntimeError("Non-finite P2 auxiliary loss detected; stop this run.")

print("Recorded epochs:", len(frame))
print("All recorded P2 auxiliary losses are finite.")
display(
    frame[
        [
            "epoch",
            aux_column,
            "metrics/precision(B)",
            "metrics/recall(B)",
            "metrics/mAP50(B)",
            "metrics/mAP50-95(B)",
        ]
    ].tail(20)
)
"""
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {
            "name": OUTPUT.name,
            "provenance": [],
        },
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(OUTPUT)
