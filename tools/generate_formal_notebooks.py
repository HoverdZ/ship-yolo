"""Generate one independent Colab Notebook per canonical formal run."""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.formal_experiments.registry import ROOT, load_registry

TEMPLATE_PATH = (
    ROOT / "notebooks" / "templates" / "formal_experiment_template.ipynb"
)


def _lines(value: str) -> list[str]:
    return value.splitlines(keepends=True)


def _markdown(value: str) -> dict[str, Any]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": _lines(value),
    }


def _code(value: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _lines(value),
    }


def build_template() -> dict[str, Any]:
    return {
        "cells": [
            _markdown(
                """# {{RUN_ID}} — {{MODEL_NAME}}

Canonical formal experiment for the Ocean Engineering paper.

- Paper aliases: `{{PAPER_ALIASES}}`
- Model YAML: `{{MODEL_YAML}}`
- Detect strides: `{{DETECT_STRIDES}}`
- This Notebook is generated from the canonical registry.
- Training defaults to **disabled** and runs only in the current kernel.
"""
            ),
            _code(
                """from google.colab import drive

drive.mount("/content/drive", force_remount=False)
print("Google Drive mounted.")
"""
            ),
            _code(
                """RUN_ID = "{{RUN_ID}}"
SEED = 0
FORMAL_CODE_COMMIT = "{{FORMAL_CODE_COMMIT}}"
REPOSITORY_URL = "https://github.com/HoverdZ/ship-yolo.git"
REPOSITORY_DIR = "/content/ship-yolo-formal"

# Safety defaults. Change RUN_TRAINING only after reviewing the printed banner.
RUN_TRAINING = False
RUN_TEST_EVALUATION = False
RUN_VISUALIZATIONS = False
REPRESENTATIVE_IMAGE = None

# S00/S01 must set both values after the second-dataset audit.
DATA_YAML_OVERRIDE = None
DRIVE_DATA_ROOT_OVERRIDE = None
"""
            ),
            _markdown(
                """## Fixed code checkout and dependency installation

The token itself is never stored in this Notebook, a URL, or the repository.
For this private repository, optionally expose a Colab Secret named
`SHIP_YOLO_GITHUB_TOKEN`. If authentication is unavailable, this cell stops
immediately. It never deletes an existing checkout or disables SSL.
"""
            ),
            _code(
                """import base64
import os
import subprocess
import sys
from pathlib import Path

subprocess.run(
    [sys.executable, "-m", "pip", "install", "--quiet", "ultralytics==8.4.92"],
    check=True,
)

try:
    from google.colab import userdata
    _github_token = userdata.get("SHIP_YOLO_GITHUB_TOKEN")
except Exception:
    _github_token = None

def git_run(arguments, cwd=None):
    command = ["git"]
    if _github_token:
        basic = base64.b64encode(
            f"x-access-token:{_github_token}".encode("utf-8")
        ).decode("ascii")
        command += ["-c", f"http.extraHeader=AUTHORIZATION: basic {basic}"]
    result = subprocess.run(
        command + list(arguments),
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        stderr = result.stderr.replace(_github_token or "", "***")
        raise RuntimeError(
            "Git operation failed. Stop here and fix GitHub authentication; "
            "do not retry other experiment cells.\\n" + stderr[-2000:]
        )
    return result.stdout.strip()

repo = Path(REPOSITORY_DIR)
if repo.exists():
    if not (repo / ".git").is_dir():
        raise FileExistsError(
            f"{repo} exists but is not the expected Git checkout. "
            "No directory was deleted."
        )
    dirty = git_run(["status", "--porcelain", "--untracked-files=no"], cwd=repo)
    if dirty:
        raise RuntimeError("Existing checkout has tracked changes; refusing checkout.")
else:
    git_run(
        ["clone", "--filter=blob:none", "--no-checkout", REPOSITORY_URL, str(repo)]
    )

git_run(["fetch", "--depth=1", "origin", FORMAL_CODE_COMMIT], cwd=repo)
git_run(["checkout", "--detach", FORMAL_CODE_COMMIT], cwd=repo)
actual_commit = git_run(["rev-parse", "HEAD"], cwd=repo)
assert actual_commit == FORMAL_CODE_COMMIT, (actual_commit, FORMAL_CODE_COMMIT)
os.chdir(repo)
if str(repo) not in sys.path:
    sys.path.insert(0, str(repo))
print("Fixed repository commit:", actual_commit)
"""
            ),
            _code(
                """import platform
from pathlib import Path

import torch
import ultralytics

assert ultralytics.__version__ == "8.4.92", ultralytics.__version__
print("Python:", platform.python_version())
print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("cuDNN:", torch.backends.cudnn.version())
print("Ultralytics:", ultralytics.__version__)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)

DRIVE_PROJECT_ROOT = Path(
    "/content/drive/MyDrive/ship_detection/paper_project"
)
for relative in (
    "datasets",
    "repository_snapshots",
    "formal_experiments",
    "paper_artifacts/tables",
    "paper_artifacts/figures",
    "paper_artifacts/visualizations",
    "paper_artifacts/manifests",
    "exports",
):
    (DRIVE_PROJECT_ROOT / relative).mkdir(parents=True, exist_ok=True)
print("Drive project root:", DRIVE_PROJECT_ROOT)
"""
            ),
            _code(
                """from tools.formal_experiments.protocol import (
    FormalRunConfig,
    print_run_banner,
    resolve_run_state,
)

if RUN_ID.startswith("S") and not DATA_YAML_OVERRIDE:
    raise RuntimeError(
        "Second-dataset run is intentionally blocked. Complete the dataset "
        "descriptor/audit, then set DATA_YAML_OVERRIDE and DRIVE_DATA_ROOT_OVERRIDE."
    )

config = FormalRunConfig.from_registry(
    RUN_ID,
    seed=SEED,
    run_training=RUN_TRAINING,
    run_test_evaluation=RUN_TEST_EVALUATION,
    data_yaml_override=DATA_YAML_OVERRIDE,
    drive_data_root_override=DRIVE_DATA_ROOT_OVERRIDE,
)
run_mode = resolve_run_state(config)
print("Resolved run mode:", run_mode)
print_run_banner(config)
"""
            ),
            _markdown(
                """## Read-only dataset copy and preflight

The source Drive dataset is not cleaned, repartitioned, augmented, or edited.
Copy progress is printed in real time for both files and processed bytes.
The verified local copy is used to generate the runtime data YAML and to run
the image/label audit, model build, GFLOPs/parameter count, stride check,
dummy forward/backward, and official-weight inheritance audit.
"""
            ),
            _code(
                """from tools.formal_experiments.protocol import prepare_experiment

prepared = prepare_experiment(config)
print("Dataset splits:", prepared["dataset_audit"]["splits"])
print(
    "Loaded/Total tensors:",
    prepared["transfer"]["loaded_total"],
)
print("Structure audit passed:", prepared["structure"]["passed"])
print("Model info:", prepared["model_info"])
"""
            ),
            _markdown(
                """## Formal training — foreground only

Review the banner immediately above. Set `RUN_TRAINING=True` in the
configuration cell only when the RUN_ID, YAML, data, official initialization,
commit, parameters, and output directory are correct.

The call below executes the official Ultralytics API directly in this kernel.
It is not sent to a subprocess. Official epoch progress therefore remains
visible. Important state and `best.pt`/`last.pt` are atomically mirrored to
Drive after every epoch/checkpoint.
"""
            ),
            _code(
                """from tools.formal_experiments.protocol import (
    print_run_banner,
    train_foreground,
)

trained_model = None
train_results = None
drive_mirror = None
print_run_banner(config)
if RUN_TRAINING:
    trained_model, train_results, drive_mirror = train_foreground(
        config,
        initialized_model=prepared["model"],
    )
else:
    print("RUN_TRAINING=False: no GPU training was started.")
"""
            ),
            _markdown(
                """## Interruption and resume

Rerun the Notebook from the top with the same RUN_ID and seed. If Drive
contains a matching `weights/last.pt` plus `experiment_state.json`, the local
run is restored and the same training cell calls `model.train(resume=True)`.
Cross-run/cross-seed resume and non-resumable residual directories are
rejected. A completed run is never overwritten.
"""
            ),
            _code(
                """from tools.formal_experiments.protocol import finalize_run

manifest = None
if RUN_TRAINING:
    manifest = finalize_run(config, mirror=drive_mirror)
    print("Final validation metrics:", manifest["validation_metrics"])
    print("Completed Drive run:", config.drive_dir)
else:
    print("Finalization skipped because training was not run in this session.")
"""
            ),
            _code(
                """from tools.paper_artifacts.results.builders import TABLES, build

table_root = DRIVE_PROJECT_ROOT / "paper_artifacts" / "tables"
run_root = DRIVE_PROJECT_ROOT / "formal_experiments"
if (config.run_dir / "run_manifest.json").is_file():
    for table_name in TABLES:
        paths = build(
            table_name,
            run_root,
            table_root / table_name,
        )
        print(table_name, paths)
else:
    print("No completed manifest in this session; table generation skipped.")
"""
            ),
            _code(
                """import subprocess

best = config.run_dir / "weights" / "best.pt"
visual_root = (
    DRIVE_PROJECT_ROOT / "paper_artifacts" / "visualizations" / RUN_ID
)
if RUN_VISUALIZATIONS:
    if not best.is_file() or not REPRESENTATIVE_IMAGE:
        raise FileNotFoundError(
            "Set REPRESENTATIVE_IMAGE and ensure this run has weights/best.pt."
        )
    commands = [
        [
            sys.executable,
            "tools/paper_artifacts/visualize_pyramid_features.py",
            "--weights", str(best),
            "--image", REPRESENTATIVE_IMAGE,
            "--output", str(visual_root / "pyramid"),
        ]
    ]
    if RUN_ID in {"R04", "R05A", "R05B", "R10", "R12"}:
        commands.append(
            [
                sys.executable,
                "tools/paper_artifacts/visualize_ca_scam_forward.py",
                "--weights", str(best),
                "--image", REPRESENTATIVE_IMAGE,
                "--output", str(visual_root / "ca_scam"),
            ]
        )
    if RUN_ID in {"R07", "R08", "R09", "R10", "R12"}:
        commands.append(
            [
                sys.executable,
                "tools/paper_artifacts/visualize_vgup_forward.py",
                "--weights", str(best),
                "--image", REPRESENTATIVE_IMAGE,
                "--output", str(visual_root / "vgup"),
            ]
        )
    for command in commands:
        subprocess.run(command, check=True)
else:
    print("RUN_VISUALIZATIONS=False: real-hook visualizations were not generated.")
"""
            ),
            _code(
                """from tools.windows_collection import verify_checksum_manifest

checksum_file = config.run_dir / "artifact_checksums.sha256"
if checksum_file.is_file():
    checks = verify_checksum_manifest(checksum_file)
    failures = [row for row in checks if not row["passed"]]
    assert not failures, failures[:10]
    print(f"Verified {len(checks)} local artifact checksums.")
    export_zip = (
        DRIVE_PROJECT_ROOT / "exports" / f"{RUN_ID}_seed_{SEED}.zip"
    )
    print("ZIP export:", export_zip, "exists:", export_zip.is_file())
    print("Run manifest:", config.drive_dir / "run_manifest.json")
else:
    print("Checksum/ZIP verification waits for a completed run.")
"""
            ),
        ],
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": []},
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


def _replace(value: Any, replacements: dict[str, str]) -> Any:
    if isinstance(value, str):
        for key, replacement in replacements.items():
            value = value.replace("{{" + key + "}}", replacement)
        return value
    if isinstance(value, list):
        return [_replace(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: _replace(item, replacements) for key, item in value.items()}
    return value


def write_template(path: Path = TEMPLATE_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(build_template(), ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    return path


def generate(commit: str) -> list[Path]:
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("--commit must be a full 40-character lowercase SHA.")
    if not TEMPLATE_PATH.is_file():
        write_template()
    template = json.loads(TEMPLATE_PATH.read_text(encoding="utf-8"))
    registry = load_registry()
    outputs = []
    for run_id, run in registry["canonical_runs"].items():
        replacements = {
            "RUN_ID": run_id,
            "MODEL_NAME": run["base_model"] + " formal variant",
            "PAPER_ALIASES": ", ".join(run["paper_aliases"]),
            "MODEL_YAML": run["model_yaml"],
            "DETECT_STRIDES": json.dumps(run["expected_detect_strides"]),
            "FORMAL_CODE_COMMIT": commit,
        }
        notebook = _replace(copy.deepcopy(template), replacements)
        output = ROOT / run["notebook_path"]
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
            encoding="utf-8",
        )
        outputs.append(output)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit")
    parser.add_argument("--write-template", action="store_true")
    args = parser.parse_args()
    if args.write_template:
        print(write_template())
    if args.commit:
        for output in generate(args.commit):
            print(output.relative_to(ROOT))
    if not args.write_template and not args.commit:
        parser.error("Pass --write-template and/or --commit.")


if __name__ == "__main__":
    main()
