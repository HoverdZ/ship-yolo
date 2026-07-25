"""Generate the Colab training notebook for cumulative model experiments."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "colab/YOLO11n_Cumulative_DySample_PLS_SCAM.ipynb"


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
# YOLO11n cumulative DySample / PLS / SCAM experiments

This notebook runs exactly one independently initialized experiment at a time:

1. `incdw_dysample`
2. `incdw_dysample_pls`
3. `incdw_dysample_pls_scam`

Every run inherits compatible tensors from the same official `yolo11n.pt`.
It never loads another experiment's `best.pt`. Formal training is disabled
until the preflight report has passed and `RUN_TRAINING` is explicitly enabled.
"""
        ),
        markdown("## 1. Install the pinned environment"),
        code(
            """
import subprocess
import sys

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q",
     "ultralytics==8.4.92", "tqdm", "pyyaml"],
    check=True,
)
import ultralytics
assert ultralytics.__version__ == "8.4.92", ultralytics.__version__
print("Ultralytics:", ultralytics.__version__)
"""
        ),
        markdown("## 2. Mount Drive and clone the private experiment branch"),
        code(
            """
from google.colab import drive, userdata
drive.mount("/content/drive")

import base64
import os
import subprocess
from pathlib import Path

REPO_URL = "https://github.com/HoverdZ/ship-yolo.git"
BRANCH = "feature/dysample-pls-scam-cumulative"
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
        markdown("## 3. Copy the Drive dataset to local Colab storage"),
        code(
            """
import sys
sys.path.insert(0, str(REPO_ROOT))

from colab.cumulative_training import (
    COPY_WORKERS,
    copy_dataset_to_local,
    create_local_data_yaml,
)

copy_report = copy_dataset_to_local(workers=COPY_WORKERS)
DATA_YAML = create_local_data_yaml("data.yaml")
"""
        ),
        markdown("## 4. Choose one model and run the CPU preflight"),
        code(
            """
EXPERIMENT = "incdw_dysample"
# Alternatives:
# EXPERIMENT = "incdw_dysample_pls"
# EXPERIMENT = "incdw_dysample_pls_scam"

RUN_NAMES = {
    "incdw_dysample": "yolo11n_incdw_dysample_640",
    "incdw_dysample_pls": "yolo11n_incdw_dysample_pls_640",
    "incdw_dysample_pls_scam": "yolo11n_incdw_dysample_pls_scam_640",
}
if EXPERIMENT not in RUN_NAMES:
    raise ValueError(f"Unknown experiment: {EXPERIMENT}")
RUN_NAME = RUN_NAMES[EXPERIMENT]
print("Selected:", EXPERIMENT, "->", RUN_NAME)
"""
        ),
        code(
            """
from pathlib import Path

AUDIT_PATH = Path(
    f"/content/drive/MyDrive/ship_detection/audits/{RUN_NAME}_preflight.json"
)
subprocess.run(
    [
        sys.executable,
        "tools/check_cumulative_models.py",
        "--model", EXPERIMENT,
        "--weights", "yolo11n.pt",
        "--imgsz", "640",
        "--output", str(AUDIT_PATH),
    ],
    check=True,
)
print("Preflight report:", AUDIT_PATH)
"""
        ),
        markdown("## 5. Train only after reviewing the preflight"),
        code(
            """
RUN_TRAINING = False
if not RUN_TRAINING:
    print("Training is disabled. Review the preflight, then set RUN_TRAINING=True.")
else:
    subprocess.run(
        [
            sys.executable,
            "tools/train_cumulative_models.py",
            "--experiment", EXPERIMENT,
            "--data", str(DATA_YAML),
            "--weights", "yolo11n.pt",
            "--project", "/content/drive/MyDrive/ship_detection/runs",
            "--name", RUN_NAME,
            "--device", "0",
            "--epochs", "150",
            "--imgsz", "640",
            "--batch", "8",
            "--workers", "2",
        ],
        check=True,
    )
"""
        ),
        markdown(
            """
## Checks and next steps

- Confirm the inheritance report says `passed: true` and record
  `inherited_tensors / target_state_tensors`.
- Each run name must be unused. The training entrypoint rejects a directory
  containing prior training artifacts.
- Run the next experiment by changing only `EXPERIMENT`; every model still
  starts from official `yolo11n.pt`.
- Keep the test set sealed. Ultralytics trains against the unchanged training
  split and validates on the unchanged validation split from `DATA_YAML`.
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
