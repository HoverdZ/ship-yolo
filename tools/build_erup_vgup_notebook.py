"""Generate the Colab notebook for the four ERUP/VGUP experiments."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "colab/YOLO11n_ERUP_VGUP_Final_Models.ipynb"


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
# YOLO11n ERUP / VGUP final-model experiments

## Goal

Prepare one of four controlled comparisons derived from the existing
InceptionDW + DySample + PLS/SFL topology. The detector inherits compatible
tensors from official `yolo11n.pt`; ERUP/VGUP starts independently.

This committed notebook keeps `RUN_TRAINING=False`, because this preparation
round must not start formal training. Set it to `True` only after the preflight
and inheritance report are reviewed.
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
        markdown("## 2. Mount Drive and clone the private branch"),
        code(
            """
from google.colab import drive, userdata
drive.mount("/content/drive")

import base64
import os
import subprocess
from pathlib import Path

REPO_URL = "https://github.com/HoverdZ/ship-yolo.git"
BRANCH = "feature/erup-vgup-final-models"
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

# 16-thread shutil.copyfile with two live tqdm bars:
# one for completed files and one for transferred bytes.
copy_report = copy_dataset_to_local(workers=COPY_WORKERS)
DATA_YAML = create_local_data_yaml("data.yaml")
print(copy_report)
"""
        ),
        markdown("## 4. Select exactly one independent experiment"),
        code(
            """
EXPERIMENT = "incdw_dysample_sfl_vgup"
# Alternatives:
# EXPERIMENT = "incdw_dysample_sfl_scam_vgup"
# EXPERIMENT = "incdw_dysample_sfl_erup"
# EXPERIMENT = "incdw_dysample_sfl_scam_erup"

RUN_NAMES = {
    "incdw_dysample_sfl_vgup": "yolo11n_incdw_dysample_sfl_vgup_640",
    "incdw_dysample_sfl_scam_vgup": "yolo11n_incdw_dysample_sfl_scam_vgup_640",
    "incdw_dysample_sfl_erup": "yolo11n_incdw_dysample_sfl_erup_640",
    "incdw_dysample_sfl_scam_erup": "yolo11n_incdw_dysample_sfl_scam_erup_640",
}
if EXPERIMENT not in RUN_NAMES:
    raise ValueError(f"Unknown experiment: {EXPERIMENT}")
RUN_NAME = RUN_NAMES[EXPERIMENT]
print("Selected:", EXPERIMENT, "->", RUN_NAME)
"""
        ),
        markdown("## 5. Direct Python preflight and inheritance audit"),
        code(
            """
from tools.check_erup_vgup_models import check_one
from tools.erup_vgup_utils import write_json

AUDIT_PATH = Path(
    f"/content/drive/MyDrive/ship_detection/audits/{RUN_NAME}_preflight.json"
)
preflight = check_one(
    EXPERIMENT,
    weights="yolo11n.pt",
    imgsz=640,
    print_network=True,
)
write_json(AUDIT_PATH, preflight)
assert preflight["passed"], preflight
inheritance = preflight["inheritance"]
print(
    "Detector Loaded/Total tensors:",
    f"{inheritance['detector_loaded_tensors']}/"
    f"{inheritance['detector_state_tensors']}",
)
print("Preflight report:", AUDIT_PATH)
"""
        ),
        markdown("## 6. Official in-process training Cell"),
        code(
            """
from tools.train_erup_vgup_models import train_experiment

RUN_TRAINING = False
if not RUN_TRAINING:
    print("Training is disabled for this preparation round.")
else:
    # Direct call in this notebook kernel. Ultralytics' official epoch table,
    # tqdm progress, validation output, and warnings remain visible here.
    # Never launch this call in a child process or with a shell command.
    train_experiment(
        experiment=EXPERIMENT,
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
        # DySample's CUDA grid_sample backward is non-deterministic. Keeping
        # this False avoids warning tracebacks corrupting the live tqdm table.
        deterministic=False,
    )
"""
        ),
        markdown(
            """
## Checks and next steps

- Use a fresh `RUN_NAME`; the entrypoint rejects directories with old training
  artifacts.
- Confirm `trainer_weight_verification.json` reports a full checkpoint reload.
- All four experiments use identical data split, 640 resolution, 150 epochs,
  batch 8, seed 0, optimizer/learning schedule and augmentation.
- `cache="disk"` avoids the `cache='ram' may produce non-deterministic
  training results` warning.
- Validation and inference pass through the same layer-0 preprocessor.
- Keep the test split sealed until the final selected model.
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
