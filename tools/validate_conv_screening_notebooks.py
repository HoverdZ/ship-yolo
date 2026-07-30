"""Static validation for the controlled convolution-screening notebooks."""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "colab" / "conv_screening_v1"
EXPECTED = {
    "C0_YOLO11n_Official_Baseline_640.ipynb": (
        "C0_yolo11n_official",
        "c00ba9ef3df80e07da4bcc2b20c0496e2288fa02",
    ),
    "C1_YOLO11n_PConv_P23.ipynb": (
        "C1_pconv_p23",
        "d95a9b8adcad90d2d94bf771f34a121393ca609c",
    ),
    "C2_YOLO11n_LSKConv_P23.ipynb": (
        "C2_lskconv_p23",
        "d95a9b8adcad90d2d94bf771f34a121393ca609c",
    ),
    "C3_YOLO11n_PKIConv_P23.ipynb": (
        "C3_pkiconv_p23",
        "d95a9b8adcad90d2d94bf771f34a121393ca609c",
    ),
}


def _python_without_magics(source: str) -> str:
    return "\n".join(
        "" if line.lstrip().startswith(("%", "!")) else line
        for line in source.splitlines()
    )


def validate_one(
    path: Path,
    experiment_id: str,
    pinned_commit: str,
) -> list[str]:
    errors: list[str] = []
    notebook = nbformat.read(path, as_version=4)
    try:
        nbformat.validate(notebook)
    except Exception as error:
        errors.append(f"nbformat validation: {error}")
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    all_source = "\n".join(cell.source for cell in code_cells)
    for index, cell in enumerate(code_cells):
        try:
            ast.parse(_python_without_magics(cell.source))
        except SyntaxError as error:
            errors.append(f"code cell {index} syntax: {error}")
        if cell.execution_count is not None or cell.outputs:
            errors.append(f"code cell {index} contains execution state")

    required_literals = {
        f'EXPERIMENT_ID = "{experiment_id}"': "experiment id",
        f'CODE_COMMIT = "{pinned_commit}"': "pinned commit",
        "RUN_TRAINING = True": "foreground training enabled",
        "RUN_TEST_EVALUATION = False": "sealed test default",
        "EPOCHS = 150": "fixed epoch count",
        "IMGSZ = 640": "fixed image size",
        "BATCH = 8": "fixed batch size",
        "WORKERS = 2": "fixed dataloader workers",
        "SEED = 0": "fixed random seed",
        'CACHE = "disk"': "deterministic disk cache",
        "DETERMINISTIC = False": "C1-matched deterministic setting",
        "SAVE_PERIOD = 10": "fixed checkpoint interval",
        "COPY_WORKERS = 16": "concurrent copy workers",
        "ultralytics==8.4.92": "fixed Ultralytics",
        'DRIVE_DATA_ROOT = "/content/drive/MyDrive/ship_detection/data"': "Drive data root",
        'LOCAL_DATA_ROOT = "/content/ship_detection/data"': "local data root",
        "copy_dataset_to_local(config)": "dataset copy helper",
        "resolve_run_state(config)": "collision-safe run resolution",
        "install_trainer_handoff_guard(": "trainer handoff audit",
        "pretrained=True": "in-memory model handoff flag",
    }
    for literal, description in required_literals.items():
        if literal not in all_source:
            errors.append(f"missing {description}: {literal}")

    training_cells = [
        cell
        for cell in code_cells
        if "formal-training" in cell.metadata.get("tags", [])
    ]
    if len(training_cells) != 1:
        errors.append(
            f"expected one formal-training cell, found {len(training_cells)}"
        )
    else:
        source = training_cells[0].source
        if "train_model.train(" not in source:
            errors.append("training cell does not call train_model.train directly")
        forbidden_training = (
            "subprocess",
            "Popen",
            "multiprocessing",
            "os.system",
            "ThreadPoolExecutor",
            "ProcessPoolExecutor",
        )
        for token in forbidden_training:
            if token in source:
                errors.append(f"training cell contains forbidden token: {token}")

    secret_patterns = (
        r"https://[^/\s]*token[^@\s]*@github\.com",
        r"https://x-access-token:",
        r"git remote set-url.*token",
        r"sslVerify\s*=\s*false",
        r"GIT_SSL_NO_VERIFY",
    )
    for pattern in secret_patterns:
        if re.search(pattern, all_source, flags=re.IGNORECASE):
            errors.append(f"unsafe authentication/SSL pattern: {pattern}")
    if "shutil.rmtree" in all_source or "rm -rf" in all_source:
        errors.append("notebook contains destructive run/repository deletion")
    if "cache=\"ram\"" in all_source or "cache='ram'" in all_source:
        errors.append("notebook enables RAM cache")
    return errors


def main() -> None:
    failures: dict[str, list[str]] = {}
    actual = {path.name for path in NOTEBOOK_DIR.glob("*.ipynb")}
    if actual != set(EXPECTED):
        failures["directory"] = [
            f"expected {sorted(EXPECTED)}, found {sorted(actual)}"
        ]
    for filename, (experiment_id, pinned_commit) in EXPECTED.items():
        path = NOTEBOOK_DIR / filename
        if not path.is_file():
            continue
        errors = validate_one(path, experiment_id, pinned_commit)
        if errors:
            failures[filename] = errors
        else:
            print("PASS", filename)
    if failures:
        for filename, errors in failures.items():
            print("FAIL", filename, file=sys.stderr)
            for error in errors:
                print("  -", error, file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
