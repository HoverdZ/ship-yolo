"""Generate the six thin, independent formal-ablation Colab notebooks."""

from __future__ import annotations

from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "colab" / "formal_ablation_v1"
PINNED_COMMIT = "ece6844085f687ee37e38a9b057a825c57d1eaaf"

NOTEBOOKS = {
    "A0_yolo11n": "A0_YOLO11n_Baseline.ipynb",
    "A1_inceptiondw": "A1_InceptionDW.ipynb",
    "A2_inceptiondw_dpls": "A2_InceptionDW_DPLS.ipynb",
    "A3_inceptiondw_dpls_scam": "A3_InceptionDW_DPLS_SCAM.ipynb",
    "A4_inceptiondw_dpls_scam_vgup": "A4_InceptionDW_DPLS_SCAM_VGUP.ipynb",
    "A5_inceptiondw_dpls_ca_scam_vgup": "A5_InceptionDW_DPLS_CA_SCAM_VGUP.ipynb",
}


def markdown(text: str):
    return nbformat.v4.new_markdown_cell(text.strip() + "\n")


def code(text: str, tags: list[str] | None = None):
    cell = nbformat.v4.new_code_cell(text.strip() + "\n")
    if tags:
        cell.metadata["tags"] = tags
    return cell


def build(experiment_id: str):
    filename = NOTEBOOKS[experiment_id]
    notebook = nbformat.v4.new_notebook()
    notebook.metadata = {
        "accelerator": "GPU",
        "colab": {"name": filename, "provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    }
    notebook.cells = [
        markdown(
            f"""# Formal cumulative ablation v1 — {experiment_id}

This notebook is independent and uses the same dataset, official
`yolo11n.pt` initialization policy, and fixed 150-epoch protocol as the other
five runs. Training calls the official `YOLO.train(...)` API in this notebook
kernel; it is never placed in a subprocess.

The validation split selects `best.pt`. Test evaluation is off by default."""
        ),
        markdown("## 1. Central configuration — edit only this cell"),
        code(
            f"""
EXPERIMENT_ID = "{experiment_id}"
FORMAL_CODE_COMMIT = "{PINNED_COMMIT}"

DRIVE_DATA_YAML = "/content/drive/MyDrive/ship_detection/data/data.yaml"
DRIVE_DATA_ROOT = None
LOCAL_DATA_ROOT = "/content/datasets/ship_clean_v1"
DRIVE_EXPERIMENT_ROOT = "/content/drive/MyDrive/ShipPaper/formal_ablation_v1"

RUN_TRAINING = True
RUN_TEST_EVALUATION = False

# Fixed comparison settings. Do not change for the six formal runs.
EPOCHS = 150
IMGSZ = 640
BATCH = 8
WORKERS = 2
SEED = 0
CACHE = "disk"
DETERMINISTIC = False
SAVE_PERIOD = 10
"""
        ),
        markdown("## 2. Mount Drive and install the fixed environment"),
        code(
            """
import platform
import subprocess
import sys

from google.colab import drive

drive.mount("/content/drive")
assert sys.version_info[:2] == (3, 12), f"Python 3.12.x required, found {platform.python_version()}"
subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--quiet",
        "ultralytics==8.4.92",
        "nbformat",
        "pandas",
        "openpyxl",
        "matplotlib",
        "tabulate",
        "tqdm",
    ],
    check=True,
)
import ultralytics
print("Ultralytics:", ultralytics.__version__)
assert ultralytics.__version__ == "8.4.92"
""",
            ["setup"],
        ),
        markdown("## 3. Clone the private repository and check out the pinned code commit"),
        code(
            """
import base64
import os
import shutil
import subprocess
from pathlib import Path

from google.colab import userdata

REPO_URL = "https://github.com/HoverdZ/ship-yolo.git"
REPO_DIR = Path("/content/ship-yolo")
token = userdata.get("GITHUB_TOKEN")
if not token:
    raise RuntimeError(
        "Colab secret GITHUB_TOKEN is not configured. Add a read-only GitHub "
        "token in Colab Secrets, enable notebook access, then rerun this cell."
    )
git_env = os.environ.copy()
git_env["GIT_TERMINAL_PROMPT"] = "0"
credential = base64.b64encode(f"x-access-token:{token}".encode()).decode()
header = f"AUTHORIZATION: basic {credential}"

if REPO_DIR.exists():
    if not (REPO_DIR / ".git").is_dir():
        raise FileExistsError(f"{REPO_DIR} exists but is not the expected Git repository.")
    status = subprocess.run(
        ["git", "-C", str(REPO_DIR), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
        env=git_env,
    ).stdout
    if status.strip():
        raise RuntimeError("Existing /content/ship-yolo is dirty; start a fresh runtime instead of deleting blindly.")
else:
    subprocess.run(
        ["git", "-c", f"http.extraHeader={header}", "clone", "--no-checkout", REPO_URL, str(REPO_DIR)],
        check=True,
        env=git_env,
    )
subprocess.run(
    ["git", "-c", f"http.extraHeader={header}", "-C", str(REPO_DIR), "fetch", "origin", FORMAL_CODE_COMMIT],
    check=True,
    env=git_env,
)
subprocess.run(["git", "-C", str(REPO_DIR), "checkout", "--detach", FORMAL_CODE_COMMIT], check=True, env=git_env)
actual_commit = subprocess.run(
    ["git", "-C", str(REPO_DIR), "rev-parse", "HEAD"],
    check=True,
    capture_output=True,
    text=True,
    env=git_env,
).stdout.strip()
assert actual_commit == FORMAL_CODE_COMMIT
del token, credential, header
os.chdir(REPO_DIR)
print("Pinned commit:", actual_commit)
""",
            ["setup"],
        ),
        markdown("## 4. Resolve run state and prepare the read-only dataset"),
        code(
            """
from tools.paper_artifacts.formal_protocol import (
    FormalConfig,
    prepare_experiment,
    restore_or_guard_run,
)

config = FormalConfig(
    experiment_id=EXPERIMENT_ID,
    drive_data_yaml=DRIVE_DATA_YAML,
    drive_data_root=DRIVE_DATA_ROOT,
    local_data_root=LOCAL_DATA_ROOT,
    drive_experiment_root=DRIVE_EXPERIMENT_ROOT,
    epochs=EPOCHS,
    imgsz=IMGSZ,
    batch=BATCH,
    workers=WORKERS,
    seed=SEED,
    cache=CACHE,
    deterministic=DETERMINISTIC,
    save_period=SAVE_PERIOD,
    run_training=RUN_TRAINING,
    run_test_evaluation=RUN_TEST_EVALUATION,
)
run_mode = restore_or_guard_run(config)
print("Run mode:", run_mode)
prepared = prepare_experiment(config)
print("Dataset audit:", prepared["audit"]["splits"])
print(
    "Loaded/Total tensors:",
    f'{prepared["transfer"]["loaded_tensors"]}/{prepared["transfer"]["target_state_tensors"]}',
)
print("Structure audit passed:", prepared["structure"]["passed"])
"""
        ),
        markdown(
            """## 5. FORMAL TRAINING — foreground official Ultralytics API

This cell emits the normal Ultralytics epoch progress in real time. If a
verified `last.pt` for this exact experiment exists in Drive, it resumes that
checkpoint; otherwise it starts an independent run."""
        ),
        code(
            """
from tools.paper_artifacts.formal_protocol import train_foreground

trained_model = None
train_results = None
drive_mirror = None
if RUN_TRAINING:
    trained_model, train_results, drive_mirror = train_foreground(
        config,
        initialized_model=prepared["model"],
    )
else:
    print("RUN_TRAINING=False: formal training was intentionally skipped.")
""",
            ["formal-training"],
        ),
        markdown("## 6. Validation, per-image records, explanations, manifest, and ZIP export"),
        code(
            """
from tools.paper_artifacts.formal_protocol import finalize_run

run_manifest = None
if RUN_TRAINING:
    run_manifest = finalize_run(config, trained_model, drive_mirror)
    print("Completed:", run_manifest["experiment_id"])
    print("Best validation metrics:", run_manifest["best_metrics"])
else:
    print("Post-training export skipped because RUN_TRAINING=False.")
""",
            ["post-training"],
        ),
        markdown("## 7. Optional browser download of the completed artifact bundle"),
        code(
            """
from pathlib import Path

bundle = Path(DRIVE_EXPERIMENT_ROOT) / EXPERIMENT_ID / f"{EXPERIMENT_ID}_paper_artifacts.zip"
print("Drive bundle:", bundle)
# Uncomment only when a browser download is desired:
# from google.colab import files
# files.download(str(bundle))
"""
        ),
    ]
    return notebook


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for experiment_id, filename in NOTEBOOKS.items():
        nbformat.write(build(experiment_id), OUTPUT / filename)
        print(OUTPUT / filename)


if __name__ == "__main__":
    main()
