"""Generate the Colab notebook for the CA-SCAM experiment."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "colab/YOLO11n_CA_SCAM_VGUP_Training.ipynb"


def code(source: str):
    return nbf.v4.new_code_cell(source.strip())


def markdown(source: str):
    return nbf.v4.new_markdown_cell(source.strip())


def build_notebook():
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
            """
# YOLO11n CA-SCAM + VGUP training

## Goal

Train the single controlled CA-SCAM experiment derived from the successful
InceptionDW + DySample + PLS + 3×SCAM + VGUP topology. Only the three SCAM
blocks immediately before Detect become CA-SCAM. PC-SCAM is deliberately
deferred.

The model uses the same official `yolo11n.pt` initialization route and the
same training recipe as the successful comparison; it does **not** load that
experiment's `best.pt`.
"""
        ),
        markdown("## 1. Install the pinned official Ultralytics release"),
        code(
            """
%pip install -q ultralytics==8.4.92 tqdm pyyaml

import ultralytics
assert ultralytics.__version__ == "8.4.92", ultralytics.__version__
print("Ultralytics:", ultralytics.__version__)
"""
        ),
        markdown("## 2. Mount Drive and check out the private CA-SCAM branch"),
        code(
            """
from google.colab import drive, userdata
drive.mount("/content/drive")

import base64
import os
import subprocess
from pathlib import Path

REPO_URL = "https://github.com/HoverdZ/ship-yolo.git"
BRANCH = "feature/ca-scam"
REPO_ROOT = Path("/content/ship-yolo")
token = userdata.get("GITHUB_TOKEN")
if not token:
    raise RuntimeError("Create a private Colab secret named GITHUB_TOKEN.")

basic = base64.b64encode(f"x-access-token:{token}".encode()).decode()
git = ["git", "-c", f"http.extraHeader=AUTHORIZATION: basic {basic}"]
if not (REPO_ROOT / ".git").is_dir():
    subprocess.run(
        [*git, "clone", "--branch", BRANCH, "--single-branch",
         REPO_URL, str(REPO_ROOT)],
        check=True,
    )
else:
    subprocess.run([*git, "-C", str(REPO_ROOT), "fetch", "origin", BRANCH], check=True)
    subprocess.run(["git", "-C", str(REPO_ROOT), "switch", BRANCH], check=True)
    subprocess.run(["git", "-C", str(REPO_ROOT), "pull", "--ff-only", "origin", BRANCH], check=True)

os.chdir(REPO_ROOT)
commit = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], text=True
).strip()
print("Repository:", REPO_ROOT)
print("Branch:", BRANCH)
print("Commit:", commit)
"""
        ),
        markdown("## 3. Fast-copy Drive data to local Colab storage"),
        code(
            """
import sys
sys.path.insert(0, str(REPO_ROOT))

from colab.erup_vgup_training import (
    COPY_WORKERS,
    copy_dataset_to_local,
    create_local_data_yaml,
)

# 16-thread shutil.copyfile. The two live tqdm bars display completed file
# count and transferred bytes; no fixed local/Drive count equality is assumed.
copy_report = copy_dataset_to_local(workers=COPY_WORKERS)
DATA_YAML = create_local_data_yaml("data.yaml")
print(copy_report)
"""
        ),
        markdown("## 4. Configuration and direct CPU preflight"),
        code(
            """
RUN_NAME = "yolo11n_incdw_dysample_pls_ca_scam_vgup_640"
AUDIT_PATH = Path(
    f"/content/drive/MyDrive/ship_detection/audits/{RUN_NAME}_preflight.json"
)

from tools.check_calibrated_scam_models import run_checks
from tools.calibrated_scam_utils import write_json

preflight = run_checks(weights="yolo11n.pt", full_imgsz=640)
write_json(AUDIT_PATH, preflight)
assert preflight["passed"], preflight
inheritance = preflight["inheritance"]
print("Loaded/Total tensors:", inheritance["loaded_total"])
print("Only new CA tensors:", inheritance["missing_new_ca_tensors"])
print("Parameter delta:", preflight["statistics"]["parameter_delta"])
print("Preflight report:", AUDIT_PATH)
"""
        ),
        markdown("## 5. Official in-process training Cell"),
        code(
            """
from tools.train_ca_scam import train_ca_scam

RUN_TRAINING = True
if RUN_TRAINING:
    # Direct call in this notebook kernel: the official Ultralytics epoch table,
    # tqdm progress, validation output, and warnings remain visible in real time.
    train_ca_scam(
        data=DATA_YAML,
        weights="yolo11n.pt",
        project="/content/drive/MyDrive/ship_detection/runs",
        name=RUN_NAME,
        device="0",
        epochs=150,
        imgsz=640,
        batch=8,
        workers=2,
        cache="disk",
        deterministic=False,
    )
else:
    print("Training disabled.")
"""
        ),
        markdown(
            """
## Checks and next steps

- The run directory must be new; existing training artifacts are rejected.
- `cache="disk"` avoids the non-deterministic RAM-cache warning.
- Review `inheritance_report.json`: all shared tensors load and only nine new
  CA tensors remain zero-initialized.
- Review `trainer_weight_verification.json`: it must report a full reload
  before epoch 1.
- Validation/test data are never augmented or copied into the train split.
- Keep the test set sealed until the final selected model.
"""
        ),
    ]
    return notebook


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(build_notebook(), OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
