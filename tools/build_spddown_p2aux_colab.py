"""Generate the cell-by-cell Colab notebook for the two prepared experiments."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "YOLO11n_InceptionDW_SPDDown_P2GaussianAux.ipynb"


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


cells = [
    markdown(
        """# YOLO11n-InceptionDW: SPDDown and training-only P2 Gaussian supervision

This notebook prepares and runs two **independent** formal experiments:

1. `spddown`: replace only C2→C3 downsampling.
2. `p2_gaussian_aux`: add P2 Gaussian supervision only during training.

Run cells in order. Dataset copying and preflight cells do not start formal
training. The two formal training cells are clearly marked.
"""
    ),
    code(
        """# Cell 1 — mount Drive and declare immutable experiment settings
from google.colab import drive
drive.mount("/content/drive")

from pathlib import Path

DRIVE_ROOT = Path("/content/drive/MyDrive")
SOURCE_DATA_ROOT = DRIVE_ROOT / "ship_detection" / "data"
DRIVE_RUNS = DRIVE_ROOT / "ship_detection" / "runs_spddown_p2aux"
DRIVE_REPORTS = DRIVE_ROOT / "ship_detection" / "preflight_spddown_p2aux"

REPO_URL = "https://github.com/HoverdZ/ship-yolo.git"
BRANCH = "feature/spddown-p2-gaussian-aux"
REPO = Path("/content/ship-yolo")
LOCAL_DATA_ROOT = Path("/content/datasets/ship_detection")

for path in (DRIVE_RUNS, DRIVE_REPORTS):
    path.mkdir(parents=True, exist_ok=True)
print("Drive dataset:", SOURCE_DATA_ROOT)
print("Drive runs:", DRIVE_RUNS)
"""
    ),
    code(
        """# Cell 2 — securely clone/update the private branch, then pin its commit
import getpass
import os
import stat
import subprocess

token = getpass.getpass("GitHub token (input hidden; never written to notebook): ")
askpass = Path("/content/.ship_yolo_askpass.py")
askpass.write_text(
    "import os, sys\\n"
    "print('x-access-token' if 'Username' in sys.argv[1] else os.environ['SHIP_GITHUB_TOKEN'])\\n",
    encoding="utf-8",
)
askpass.chmod(askpass.stat().st_mode | stat.S_IXUSR)
env = {
    **os.environ,
    "GIT_ASKPASS": str(askpass),
    "GIT_TERMINAL_PROMPT": "0",
    "SHIP_GITHUB_TOKEN": token,
}
try:
    if REPO.exists():
        remote = subprocess.check_output(
            ["git", "-C", str(REPO), "remote", "get-url", "origin"], text=True
        ).strip()
        if remote.rstrip("/") not in {REPO_URL.rstrip("/"), REPO_URL.removesuffix(".git")}:
            raise RuntimeError(f"Refusing unrelated existing repository: {remote}")
        subprocess.run(["git", "-C", str(REPO), "fetch", "origin", BRANCH], check=True, env=env)
        subprocess.run(
            ["git", "-C", str(REPO), "checkout", "-B", BRANCH, f"origin/{BRANCH}"],
            check=True,
        )
    else:
        subprocess.run(
            [
                "git", "-c", "credential.helper=", "clone", "--branch", BRANCH,
                "--single-branch", REPO_URL, str(REPO),
            ],
            check=True,
            env=env,
        )
finally:
    token = ""
    env.pop("SHIP_GITHUB_TOKEN", None)
    os.environ.pop("SHIP_GITHUB_TOKEN", None)
    askpass.unlink(missing_ok=True)

PINNED_COMMIT = subprocess.check_output(
    ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
).strip()
subprocess.run(["git", "-C", str(REPO), "checkout", "--detach", PINNED_COMMIT], check=True)
print("Pinned commit:", PINNED_COMMIT)
"""
    ),
    code(
        """# Cell 3 — install and verify the pinned runtime
%pip install -q ultralytics==8.4.92

import sys
sys.path.insert(0, str(REPO))

import torch
import ultralytics
from custom_modules.register import register_custom_modules

assert ultralytics.__version__ == "8.4.92", ultralytics.__version__
register_custom_modules()
print("Python:", sys.version.split()[0])
print("Torch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
"""
    ),
    code(
        """# Cell 4 — copy Drive data to local disk with threaded shutil.copyfile
from concurrent.futures import ThreadPoolExecutor
import shutil
import yaml

if not SOURCE_DATA_ROOT.is_dir():
    raise FileNotFoundError(f"Drive dataset directory not found: {SOURCE_DATA_ROOT}")

if LOCAL_DATA_ROOT.exists():
    shutil.rmtree(LOCAL_DATA_ROOT)
LOCAL_DATA_ROOT.mkdir(parents=True)

source_files = [
    path for path in SOURCE_DATA_ROOT.rglob("*")
    if path.is_file() and path.suffix.lower() != ".cache"
]
if not source_files:
    raise RuntimeError(f"No files found under {SOURCE_DATA_ROOT}")

def copy_one(source: Path):
    relative = source.relative_to(SOURCE_DATA_ROOT)
    destination = LOCAL_DATA_ROOT / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return relative

with ThreadPoolExecutor(max_workers=min(32, (os.cpu_count() or 4) * 2)) as pool:
    copied = list(pool.map(copy_one, source_files))
print(f"Copied {len(copied)} files with shutil.copyfile")

image_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
if (LOCAL_DATA_ROOT / "train" / "images").is_dir():
    split_paths = {split: f"{split}/images" for split in ("train", "val", "test")}
elif (LOCAL_DATA_ROOT / "images" / "train").is_dir():
    split_paths = {split: f"images/{split}" for split in ("train", "val", "test")}
else:
    raise RuntimeError("Expected train/images or images/train dataset layout.")

counts = {}
for split, relative in split_paths.items():
    folder = LOCAL_DATA_ROOT / relative
    counts[split] = sum(
        path.is_file() and path.suffix.lower() in image_suffixes for path in folder.rglob("*")
    )
print("Cloud dataset copied successfully; local image counts:", counts)

LOCAL_DATA_YAML = LOCAL_DATA_ROOT / "data.yaml"
LOCAL_DATA_YAML.write_text(
    yaml.safe_dump(
        {
            "path": str(LOCAL_DATA_ROOT),
            "train": split_paths["train"],
            "val": split_paths["val"],
            "test": split_paths["test"],
            "nc": 1,
            "names": {0: "ship"},
        },
        sort_keys=False,
    ),
    encoding="utf-8",
)
print(LOCAL_DATA_YAML.read_text())
"""
    ),
    code(
        """# Cell 5 — run both 640 preflight audits; still no formal training
from tools.check_spddown_p2aux import main as run_preflight

reports = run_preflight(
    [
        "--variant", "all",
        "--weights", str(REPO / "yolo11n.pt"),
        "--imgsz", "640",
        "--output-dir", str(DRIVE_REPORTS / PINNED_COMMIT),
    ]
)
assert all(report["all_checks_passed"] for report in reports.values())
print("Both preflight audits passed.")
"""
    ),
    markdown(
        """## Formal experiment A — SPDDown

The next cell starts formal training in the current notebook kernel. It does
not use `subprocess`, `Popen`, `!python`, or a background process.
"""
    ),
    code(
        """# Cell 6 — START FORMAL SPDDown TRAINING
from tools.train_spddown_p2aux import TrainingRequest, run_training

spddown_request = TrainingRequest(
    variant="spddown",
    data=str(LOCAL_DATA_YAML),
    project=str(DRIVE_RUNS),
    weights=str(REPO / "yolo11n.pt"),
    name="yolo11n_inceptiondw_spddown_p3_640",
    epochs=150,
    imgsz=640,
    batch=8,
    workers=2,
    device="0",
    seed=0,
    optimizer="auto",
    resume=False,
)
spddown_result = run_training(spddown_request)
"""
    ),
    markdown(
        """## Formal experiment B — training-only P2 Gaussian supervision

Run this only as an independent experiment, not immediately after an
unfinished SPDDown run.
"""
    ),
    code(
        """# Cell 7 — START FORMAL P2 GAUSSIAN AUXILIARY TRAINING
p2_aux_request = TrainingRequest(
    variant="p2_gaussian_aux",
    data=str(LOCAL_DATA_YAML),
    project=str(DRIVE_RUNS),
    weights=str(REPO / "yolo11n.pt"),
    name="yolo11n_inceptiondw_p2_gaussian_aux_640",
    epochs=150,
    imgsz=640,
    batch=8,
    workers=2,
    device="0",
    seed=0,
    optimizer="auto",
    resume=False,
)
p2_aux_result = run_training(p2_aux_request)
"""
    ),
    code(
        """# Cell 8 — resume exactly one interrupted run in the current process
# Set one variant/name pair, then run this cell only after verifying last.pt exists.
RESUME_VARIANT = "spddown"  # or "p2_gaussian_aux"
RESUME_NAME = (
    "yolo11n_inceptiondw_spddown_p3_640"
    if RESUME_VARIANT == "spddown"
    else "yolo11n_inceptiondw_p2_gaussian_aux_640"
)
last_pt = DRIVE_RUNS / RESUME_NAME / "weights" / "last.pt"
if not last_pt.is_file():
    raise FileNotFoundError(last_pt)

resume_request = TrainingRequest(
    variant=RESUME_VARIANT,
    data=str(LOCAL_DATA_YAML),
    project=str(DRIVE_RUNS),
    weights=str(REPO / "yolo11n.pt"),
    name=RESUME_NAME,
    resume=True,
)
resume_result = run_training(resume_request)
"""
    ),
    code(
        """# Cell 9 — inspect saved metrics without changing training state
import pandas as pd

for run_name in (
    "yolo11n_inceptiondw_spddown_p3_640",
    "yolo11n_inceptiondw_p2_gaussian_aux_640",
):
    csv_path = DRIVE_RUNS / run_name / "results.csv"
    if csv_path.is_file():
        frame = pd.read_csv(csv_path)
        print("\\n", run_name, "epochs recorded:", len(frame))
        display(frame.tail())
    else:
        print(run_name, "has no results.csv yet")
"""
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"name": OUTPUT.name, "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(OUTPUT)
