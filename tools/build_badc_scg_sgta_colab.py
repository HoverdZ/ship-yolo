"""Generate the staged BADC/SCG/SGTA Colab screening notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "YOLO11n_BADC_SCG_SGTA_Screening.ipynb"


def code(source: str):
    return nbf.v4.new_code_cell(source.strip() + "\n")


def markdown(source: str):
    return nbf.v4.new_markdown_cell(source.strip() + "\n")


notebook = nbf.v4.new_notebook()
notebook["metadata"] = {
    "accelerator": "GPU",
    "colab": {"name": OUTPUT.name, "provenance": []},
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python"},
}
notebook["cells"] = [
    markdown(
        """# YOLO11n BADC / SCG / SGTA screening

This notebook runs one independent screening experiment at a time. The existing
150-epoch InceptionDW run is the baseline; use its first 80 epochs as the
comparison, so no baseline rerun is required.

The learning-rate scheduler is configured for 150 epochs from the beginning.
Training pauses after epoch 80 and preserves `stage80_resume.pt`. Only after
reviewing the 80-epoch result should the continuation cell be enabled; it
resumes optimizer, scheduler, AMP scaler, and EMA state through epoch 150.

Run cells in order. Formal training stays in the notebook kernel so live output
is visible."""
    ),
    code(
        """# Cell 1 - mount Drive and select exactly one screening variant
from google.colab import drive
drive.mount("/content/drive")

from pathlib import Path

# Choose one of: "badc", "scg", "sgta".
# "full" is implemented but intentionally blocked until the three screens are reviewed.
VARIANT = "badc"
SCREENING_VARIANTS = {"badc", "scg", "sgta"}
ALLOW_FULL_MODEL = False
if VARIANT not in SCREENING_VARIANTS and not (VARIANT == "full" and ALLOW_FULL_MODEL):
    raise ValueError(
        f"VARIANT={VARIANT!r} is not enabled. Use badc/scg/sgta for the current screening stage."
    )

DRIVE_ROOT = Path("/content/drive/MyDrive")
SOURCE_DATA_ROOT = DRIVE_ROOT / "ship_detection" / "data"
DRIVE_RUNS = DRIVE_ROOT / "ship_detection" / "runs_badc_scg_sgta"
DRIVE_REPORTS = DRIVE_ROOT / "ship_detection" / "preflight_badc_scg_sgta"

REPO_URL = "https://github.com/HoverdZ/ship-yolo.git"
BRANCH = "experiment/badc-scg-sgta"
REPO = Path("/content/ship-yolo")
LOCAL_DATA_ROOT = Path("/content/datasets/ship_detection")

RUN_NAMES = {
    "badc": "yolo11n_badc_p23_640",
    "scg": "yolo11n_inceptiondw_scg_p3_640",
    "sgta": "yolo11n_inceptiondw_sgta_640",
    "full": "yolo11n_badc_scg_sgta_full_640",
}
RUN_NAME = RUN_NAMES[VARIANT]
for path in (DRIVE_RUNS, DRIVE_REPORTS):
    path.mkdir(parents=True, exist_ok=True)
print("Variant:", VARIANT)
print("Run:", RUN_NAME)
print("Drive dataset:", SOURCE_DATA_ROOT)
print("Drive output:", DRIVE_RUNS / RUN_NAME)"""
    ),
    code(
        """# Cell 2 - securely clone/update the private branch and pin its commit
import getpass
import os
import stat
import subprocess

token = getpass.getpass(
    "GitHub token (input hidden; used only by git and never saved): "
).strip()
if not token:
    raise ValueError("GitHub token cannot be empty.")

askpass = Path("/content/.ship_yolo_askpass.py")
askpass.write_text(
    "#!/usr/bin/env python3\\n"
    "import os, sys\\n"
    "prompt = sys.argv[1] if len(sys.argv) > 1 else ''\\n"
    "print('x-access-token' if 'Username' in prompt else os.environ['SHIP_GITHUB_TOKEN'])\\n",
    encoding="utf-8",
)
askpass.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
git_env = {
    **os.environ,
    "GIT_ASKPASS": str(askpass),
    "GIT_TERMINAL_PROMPT": "0",
    "SHIP_GITHUB_TOKEN": token,
}

def run_git(arguments, *, authenticated=False):
    completed = subprocess.run(
        ["git", *arguments],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=git_env if authenticated else None,
    )
    output = (completed.stdout + completed.stderr).replace(token, "[REDACTED]").strip()
    if completed.returncode:
        raise RuntimeError(
            f"git {' '.join(map(str, arguments))} failed with exit "
            f"{completed.returncode}:\\n{output or 'No Git diagnostic output.'}"
        )
    if output:
        print(output)
    return completed.stdout.strip()

try:
    if REPO.exists() and not (REPO / ".git").is_dir():
        if any(REPO.iterdir()):
            raise RuntimeError(
                f"{REPO} exists but is not the expected Git clone. "
                "Move that unrelated directory and rerun this cell."
            )
        REPO.rmdir()

    if REPO.exists():
        remote = run_git(["-C", str(REPO), "remote", "get-url", "origin"])
        accepted = {REPO_URL.rstrip("/"), REPO_URL.removesuffix(".git").rstrip("/")}
        if remote.rstrip("/") not in accepted:
            raise RuntimeError(f"Refusing unrelated repository remote: {remote}")
        run_git(
            ["-C", str(REPO), "fetch", "origin", BRANCH],
            authenticated=True,
        )
        run_git(
            ["-C", str(REPO), "checkout", "-B", BRANCH, f"origin/{BRANCH}"]
        )
    else:
        run_git(
            [
                "-c", "credential.helper=", "clone", "--branch", BRANCH,
                "--single-branch", REPO_URL, str(REPO),
            ],
            authenticated=True,
        )
finally:
    token = ""
    git_env.pop("SHIP_GITHUB_TOKEN", None)
    os.environ.pop("SHIP_GITHUB_TOKEN", None)
    askpass.unlink(missing_ok=True)

PINNED_COMMIT = run_git(["-C", str(REPO), "rev-parse", "HEAD"])
run_git(["-C", str(REPO), "checkout", "--detach", PINNED_COMMIT])
print("Pinned commit:", PINNED_COMMIT)"""
    ),
    code(
        """# Cell 3 - install and verify the pinned runtime
%pip install -q ultralytics==8.4.92

import sys
sys.path.insert(0, str(REPO))

import torch
import ultralytics
from custom_modules.register import register_custom_modules

assert ultralytics.__version__ == "8.4.92", ultralytics.__version__
register_custom_modules()
if not torch.cuda.is_available():
    raise RuntimeError("A Colab GPU runtime is required for formal training.")
print("Python:", sys.version.split()[0])
print("Torch:", torch.__version__)
print("Ultralytics:", ultralytics.__version__)
print("GPU:", torch.cuda.get_device_name(0))"""
    ),
    code(
        """# Cell 4 - copy Drive data locally with threads and live progress
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import shutil
import time
from tqdm.auto import tqdm
import yaml

if not SOURCE_DATA_ROOT.is_dir():
    raise FileNotFoundError(f"Drive dataset directory not found: {SOURCE_DATA_ROOT}")

source_files = sorted(
    path for path in SOURCE_DATA_ROOT.rglob("*")
    if path.is_file() and path.suffix.lower() != ".cache"
)
if not source_files:
    raise RuntimeError(f"No files found under {SOURCE_DATA_ROOT}")
source_sizes = {path: path.stat().st_size for path in source_files}
total_bytes = sum(source_sizes.values())

# Local Colab storage only; Drive source is never removed or modified.
if LOCAL_DATA_ROOT.exists():
    shutil.rmtree(LOCAL_DATA_ROOT)
LOCAL_DATA_ROOT.mkdir(parents=True)

def copy_one(source: Path):
    relative = source.relative_to(SOURCE_DATA_ROOT)
    destination = LOCAL_DATA_ROOT / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return relative, source_sizes[source]

workers = min(32, max(4, (os.cpu_count() or 4) * 2))
started = time.perf_counter()
copied_files = 0
copied_bytes = 0
with (
    tqdm(
        total=len(source_files), desc="Files", unit="file",
        dynamic_ncols=True, mininterval=0.2, position=0,
    ) as file_bar,
    tqdm(
        total=total_bytes, desc="Bytes", unit="B", unit_scale=True,
        unit_divisor=1024, dynamic_ncols=True, mininterval=0.2, position=1,
    ) as byte_bar,
    ThreadPoolExecutor(max_workers=workers) as pool,
):
    futures = [pool.submit(copy_one, path) for path in source_files]
    for future in as_completed(futures):
        _, size = future.result()
        copied_files += 1
        copied_bytes += size
        file_bar.update(1)
        byte_bar.update(size)
        elapsed = max(time.perf_counter() - started, 1e-6)
        file_bar.set_postfix(
            workers=workers,
            speed=f"{copied_bytes / elapsed / (1024**2):.1f} MiB/s",
            refresh=False,
        )

local_files = [
    path for path in LOCAL_DATA_ROOT.rglob("*")
    if path.is_file() and path.suffix.lower() != ".cache"
]
local_bytes = sum(path.stat().st_size for path in local_files)
if len(local_files) != copied_files or local_bytes != copied_bytes:
    raise RuntimeError(
        "Dynamic source/local copy verification failed: "
        f"source={copied_files} files/{copied_bytes} bytes, "
        f"local={len(local_files)} files/{local_bytes} bytes."
    )
print(
    f"Copied {copied_files} files ({copied_bytes / 1024**3:.2f} GiB) "
    f"with {workers} shutil.copyfile workers."
)

if (LOCAL_DATA_ROOT / "train" / "images").is_dir():
    split_paths = {split: f"{split}/images" for split in ("train", "val", "test")}
elif (LOCAL_DATA_ROOT / "images" / "train").is_dir():
    split_paths = {split: f"images/{split}" for split in ("train", "val", "test")}
else:
    raise RuntimeError("Expected train/images or images/train dataset layout.")

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
print(LOCAL_DATA_YAML.read_text())"""
    ),
    code(
        """# Cell 5 - run the selected 640 preflight before any training
from tools.check_badc_scg_sgta import main as run_preflight

reports = run_preflight(
    [
        "--variant", VARIANT,
        "--weights", str(REPO / "yolo11n.pt"),
        "--imgsz", "640",
        "--output-dir", str(DRIVE_REPORTS / PINNED_COMMIT / VARIANT),
    ]
)
report = reports[VARIANT]
assert report["all_checks_passed"]
print("Parameters:", report["statistics"]["parameters"])
print("GFLOPs:", report["statistics"]["gflops_at_imgsz"])
transfer = report["weight_transfer"]
print(
    "Loaded state tensors:",
    f"{transfer['loaded_state_tensors']}/{transfer['total_state_tensors']}",
)
print("Selected experiment passed all preflight checks.")"""
    ),
    markdown(
        """## Stage 1: pause after epoch 80

This cell declares a 150-epoch scheduler but stops after epoch 80. That makes
the first 80 epochs directly comparable to the first 80 epochs of the existing
150-epoch InceptionDW run. A resumable `stage80_resume.pt` is preserved before
Ultralytics strips optimizer state from its normal final checkpoint."""
    ),
    code(
        """# Cell 6 - START THE SELECTED 80-EPOCH SCREENING STAGE
from tools.train_badc_scg_sgta import TrainingRequest, run_training

stage_request = TrainingRequest(
    variant=VARIANT,
    data=str(LOCAL_DATA_YAML),
    project=str(DRIVE_RUNS),
    weights=str(REPO / "yolo11n.pt"),
    name=RUN_NAME,
    total_epochs=150,
    stage_epochs=80,
    imgsz=640,
    batch=8,
    workers=2,
    device="0",
    seed=0,
    optimizer="auto",
    resume=False,
)
stage_result = run_training(stage_request)
stage_checkpoint = DRIVE_RUNS / RUN_NAME / "weights" / "stage80_resume.pt"
if not stage_checkpoint.is_file():
    raise FileNotFoundError(
        f"Screening ended without the required resumable checkpoint: {stage_checkpoint}"
    )
print("80-epoch screening complete:", stage_checkpoint)"""
    ),
    code(
        """# Cell 7 - use only if Colab was interrupted before reaching epoch 80
from tools.train_badc_scg_sgta import TrainingRequest, run_training

last_checkpoint = DRIVE_RUNS / RUN_NAME / "weights" / "last.pt"
stage_checkpoint = DRIVE_RUNS / RUN_NAME / "weights" / "stage80_resume.pt"
if stage_checkpoint.is_file():
    raise RuntimeError(
        "The formal 80-epoch stage checkpoint already exists. "
        "Do not use the interruption-resume cell."
    )
if not last_checkpoint.is_file():
    raise FileNotFoundError(f"No interrupted checkpoint found: {last_checkpoint}")

interrupted_request = TrainingRequest(
    variant=VARIANT,
    data=str(LOCAL_DATA_YAML),
    project=str(DRIVE_RUNS),
    name=RUN_NAME,
    total_epochs=150,
    stage_epochs=80,
    resume=True,
    resume_checkpoint=str(last_checkpoint),
)
interrupted_result = run_training(interrupted_request)"""
    ),
    code(
        """# Cell 8 - inspect the 80-epoch metrics before deciding on continuation
import pandas as pd

results_csv = DRIVE_RUNS / RUN_NAME / "results.csv"
if not results_csv.is_file():
    raise FileNotFoundError(results_csv)
frame = pd.read_csv(results_csv)
if len(frame) < 80:
    print(f"Run is incomplete: {len(frame)}/80 recorded epochs.")
else:
    stage = frame.iloc[:80].copy()
    metric = "metrics/mAP50-95(B)"
    best_index = stage[metric].astype(float).idxmax()
    columns = [
        "epoch",
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        metric,
    ]
    print("Best row in epochs 1-80:")
    display(stage.loc[[best_index], columns])
    print("Last 10 screening rows:")
    display(stage[columns].tail(10))"""
    ),
    markdown(
        """## Stage 2: optional continuation from epoch 81 to 150

Do not run the next cell immediately. First compare the 80-epoch curve with the
existing InceptionDW curve and return for a go/no-go decision. If approved, set
`CONFIRM_CONTINUE_TO_150 = True`. This is a real resume, not a new 70-epoch
fine-tune."""
    ),
    code(
        """# Cell 9 - OPTIONAL: resume the approved experiment to 150 total epochs
CONFIRM_CONTINUE_TO_150 = False
if not CONFIRM_CONTINUE_TO_150:
    raise RuntimeError(
        "Continuation is intentionally locked. Review the 80-epoch result first, "
        "then set CONFIRM_CONTINUE_TO_150=True only for an approved variant."
    )

from tools.train_badc_scg_sgta import TrainingRequest, run_training

stage_checkpoint = DRIVE_RUNS / RUN_NAME / "weights" / "stage80_resume.pt"
if not stage_checkpoint.is_file():
    raise FileNotFoundError(stage_checkpoint)
continuation_request = TrainingRequest(
    variant=VARIANT,
    data=str(LOCAL_DATA_YAML),
    project=str(DRIVE_RUNS),
    name=RUN_NAME,
    total_epochs=150,
    stage_epochs=80,
    resume=True,
    resume_checkpoint=str(stage_checkpoint),
)
continuation_result = run_training(continuation_request)"""
    ),
    code(
        """# Cell 10 - final metrics preview (works after 80 or 150 epochs)
import pandas as pd

results_csv = DRIVE_RUNS / RUN_NAME / "results.csv"
frame = pd.read_csv(results_csv)
columns = [
    "epoch",
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
]
print("Recorded epochs:", len(frame))
display(frame[columns].tail(20))
best_index = frame["metrics/mAP50-95(B)"].astype(float).idxmax()
print("Best recorded mAP50-95 row:")
display(frame.loc[[best_index], columns])"""
    ),
]

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
nbf.validate(notebook)
nbf.write(notebook, OUTPUT)
print(OUTPUT)
