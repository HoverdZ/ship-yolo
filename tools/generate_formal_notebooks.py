"""为每个正式实验生成一份可独立运行的 Colab Notebook。"""

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
    """构建只有三个操作区的中文正式实验模板。"""

    return {
        "cells": [
            _markdown(
                """# {{RUN_ID}}：{{MODEL_NAME}}

这是用于 Ocean Engineering 论文的独立正式实验 Notebook。

- 论文别名：`{{PAPER_ALIASES}}`
- 模型 YAML：`{{MODEL_YAML}}`
- 检测步长：`{{DETECT_STRIDES}}`
- 随机种子由正式实验注册表固定为 `0`，不会循环运行多个随机种子。
- 运行到训练步骤时，全部检查通过后会直接开始训练，无需修改任何开关。
- {{DATASET_NOTE}}
"""
            ),
            _markdown(
                """## 1. 挂载云盘、安装固定环境并获取实验代码

下面的单元格一次完成 Google Drive 挂载、Ultralytics 8.4.92 安装、私有仓库认证、固定提交检出和运行环境核验。它只读取 Colab Secret 中已有的 `GITHUB_TOKEN`，不会把令牌写入 URL、Notebook 或仓库。若认证失败，程序会立即停止，不会继续执行训练。
"""
            ),
            _code(
                """from google.colab import drive, userdata

drive.mount("/content/drive", force_remount=False)

import base64
import os
import platform
import subprocess
import sys
from pathlib import Path

# 本实验的编号和代码版本已经固定，无需手工修改。
RUN_ID = "{{RUN_ID}}"
FORMAL_CODE_COMMIT = "{{FORMAL_CODE_COMMIT}}"
REPOSITORY_URL = "https://github.com/HoverdZ/ship-yolo.git"
REPOSITORY_DIR = "/content/ship-yolo"

# 安装此前 6 个 InceptionDW 正式实验使用的 Ultralytics 版本。
subprocess.run(
    [sys.executable, "-m", "pip", "install", "--quiet", "ultralytics==8.4.92"],
    check=True,
)

try:
    github_token = userdata.get("GITHUB_TOKEN")
except Exception as error:
    raise RuntimeError(
        "无法读取 Colab Secret：GITHUB_TOKEN。请先完成 GitHub 身份认证，"
        "然后从本单元格重新运行；当前程序已停止。"
    ) from error
if not github_token:
    raise RuntimeError(
        "Colab Secret 中没有可用的 GITHUB_TOKEN。请添加该 Secret 并允许"
        "当前 Notebook 访问；当前程序已停止。"
    )


def git_run(arguments, cwd=None):
    basic = base64.b64encode(
        f"x-access-token:{github_token}".encode("utf-8")
    ).decode("ascii")
    command = [
        "git",
        "-c",
        f"http.extraHeader=AUTHORIZATION: basic {basic}",
        *list(arguments),
    ]
    result = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        stderr = result.stderr.replace(github_token, "***")
        raise RuntimeError(
            "Git 操作失败。请先处理 GitHub 身份认证或仓库状态，"
            "不要继续运行后续单元格。\\n" + stderr[-2000:]
        )
    return result.stdout.strip()


repo = Path(REPOSITORY_DIR)
if repo.exists():
    if not (repo / ".git").is_dir():
        raise FileExistsError(
            f"{repo} 已存在，但不是预期的 Git 仓库。程序没有删除该目录。"
        )
    dirty = git_run(
        ["status", "--porcelain", "--untracked-files=no"],
        cwd=repo,
    )
    if dirty:
        raise RuntimeError("现有仓库包含未提交的受跟踪文件修改，已拒绝切换版本。")
else:
    git_run(
        [
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            REPOSITORY_URL,
            str(repo),
        ]
    )

git_run(["fetch", "--depth=1", "origin", FORMAL_CODE_COMMIT], cwd=repo)
git_run(["checkout", "--detach", FORMAL_CODE_COMMIT], cwd=repo)
actual_commit = git_run(["rev-parse", "HEAD"], cwd=repo)
assert actual_commit == FORMAL_CODE_COMMIT, (
    actual_commit,
    FORMAL_CODE_COMMIT,
)
os.chdir(repo)
if str(repo) not in sys.path:
    sys.path.insert(0, str(repo))
print("固定仓库提交：", actual_commit)

import torch
import ultralytics

assert ultralytics.__version__ == "8.4.92", ultralytics.__version__
print("Python 版本：", platform.python_version())
print("PyTorch 版本：", torch.__version__)
print("CUDA 版本：", torch.version.cuda)
print("cuDNN 版本：", torch.backends.cudnn.version())
print("Ultralytics 版本：", ultralytics.__version__)
print(
    "GPU：",
    torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
)

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
print("论文项目云盘目录：", DRIVE_PROJECT_ROOT)
"""
            ),
            _markdown(
                """## 2. 复制数据集、完成训练前审计并直接开始正式训练

下面的单元格把云盘数据集以只读方式多线程复制到 Colab 本地，并实时显示文件数和字节进度；随后自动生成本地 `data.yaml`，检查图像与标签、模型结构、检测步长、CPU 前向/反向、复杂度和官方预训练权重继承。任何检查失败都会在训练前停止；全部检查通过后，会在当前内核中直接调用官方 `YOLO.train(...)` 开始训练，完整 epoch 输出会实时显示。若同一实验存在匹配的 `last.pt` 和状态文件，则自动从最近一轮续训。
"""
            ),
            _code(
                """from tools.formal_experiments.protocol import (
    FormalRunConfig,
    prepare_experiment,
    print_run_banner,
    resolve_run_state,
    train_foreground,
)

# 随机种子直接使用正式注册表中的固定值 0，不执行多种子循环。
config = FormalRunConfig.from_registry(RUN_ID, run_training=True)
run_mode = resolve_run_state(config)
print("运行方式：", "从断点续训" if run_mode == "resume" else "全新训练")

prepared = prepare_experiment(config)
assert prepared["structure"]["passed"], prepared["structure"]
print("数据集划分审计：", prepared["dataset_audit"]["splits"])
print("预训练权重 Loaded/Total：", prepared["transfer"]["loaded_total"])
print("结构审计：通过")
print("模型规模：", prepared["model_info"])

print_run_banner(config)
print("全部训练前检查通过，开始正式训练。")
trained_model, train_results, drive_mirror = train_foreground(
    config,
    initialized_model=prepared["model"],
)

# 后处理会从 best.pt 单独加载模型，因此先释放训练器、优化器和旧模型显存。
import contextlib
import gc

prepared_model = prepared.pop("model", None)
del prepared_model, trained_model, train_results
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    with contextlib.suppress(Exception):
        torch.cuda.ipc_collect()
    print(
        "训练对象已释放，当前显存：",
        f"allocated={torch.cuda.memory_allocated() / 1024**3:.2f} GiB，",
        f"reserved={torch.cuda.memory_reserved() / 1024**3:.2f} GiB",
    )
"""
            ),
            _markdown(
                """## 3. 完成最终验证、更新论文结果表并核验备份

训练结束后运行下面的单元格。训练器显存已经在上一单元格末尾释放；本单元格使用 `best.pt` 完成固定验证，并把逐图统计限制在不超过 8 张的小批次，避免将整个验证集一次送入显存。所有状态文件稳定后才生成校验清单和 ZIP，随后原子同步到 Google Drive。测试集仍保持封存，不参与模型选择。
"""
            ),
            _code(
                """from tools.formal_experiments.protocol import finalize_run
from tools.paper_artifacts.results.builders import TABLES, build
from tools.windows_collection import verify_checksum_manifest

manifest = finalize_run(config, mirror=drive_mirror)
print("最终验证指标：", manifest["validation_metrics"])
print("已完成的云盘实验目录：", config.drive_dir)

table_root = DRIVE_PROJECT_ROOT / "paper_artifacts" / "tables"
run_root = DRIVE_PROJECT_ROOT / "formal_experiments"
for table_name in TABLES:
    paths = build(table_name, run_root, table_root / table_name)
    print("已更新结果表：", table_name, paths)

checksum_file = config.run_dir / "artifact_checksums.sha256"
assert checksum_file.is_file(), checksum_file
checks = verify_checksum_manifest(checksum_file)
failures = [row for row in checks if not row["passed"]]
assert not failures, failures[:10]
assert not (config.run_dir / "RUNNING.lock").exists()
assert not (config.drive_dir / "RUNNING.lock").exists()

export_zip = (
    DRIVE_PROJECT_ROOT
    / "exports"
    / f"{config.run_id}_{config.run_name}.zip"
)
assert export_zip.is_file(), export_zip
print(f"已通过 {len(checks)} 项本地文件校验。")
print("ZIP 备份：", export_zip)
print("运行清单：", config.drive_dir / "run_manifest.json")
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
        return {
            key: _replace(item, replacements)
            for key, item in value.items()
        }
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
        raise ValueError("--commit 必须是完整的 40 位小写提交 SHA。")
    if not TEMPLATE_PATH.is_file():
        write_template()
    template = json.loads(TEMPLATE_PATH.read_text(encoding="utf-8"))
    registry = load_registry()
    outputs = []
    for run_id, run in registry["canonical_runs"].items():
        replacements = {
            "RUN_ID": run_id,
            "MODEL_NAME": run["base_model"] + " 正式实验",
            "PAPER_ALIASES": ", ".join(run["paper_aliases"]),
            "MODEL_YAML": run["model_yaml"],
            "DETECT_STRIDES": json.dumps(run["expected_detect_strides"]),
            "FORMAL_CODE_COMMIT": commit,
            "DATASET_NOTE": (
                "本实验使用已冻结的主数据集，训练集、验证集和测试集划分保持不变。"
                if run["dataset_id"] != "external_dataset_pending"
                else (
                    "本实验等待第二数据集完成登记；在正式注册前会明确停止，"
                    "不会误用主数据集。"
                )
            ),
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
        parser.error("请传入 --write-template 和/或 --commit。")


if __name__ == "__main__":
    main()
