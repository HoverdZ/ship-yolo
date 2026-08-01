"""Generate four pinned, direct-training PLS-SCAM formal Colab notebooks."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.formal_experiments.registry import ROOT
from tools.pls_scam_experiments import RUN_IDS, load_pls_scam_registry

DISPLAY_NAMES = {
    "PLS_CA_SCAM_150ep": "YOLO11n + PLS + CA-SCAM",
    "PLS_SCAM_150ep": "YOLO11n + PLS + SCAM",
    "PLS_CA_SCAM_VGUP_150ep": "YOLO11n + PLS + CA-SCAM + VGUP",
    "PLS_CA_SCAM_ERUP_150ep": "YOLO11n + PLS + CA-SCAM + ERUP",
}


def _lines(source: str) -> list[str]:
    return source.splitlines(keepends=True)


def _markdown(source: str) -> dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(source)}


def _code(source: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _lines(source),
    }


SETUP_CODE = '''from google.colab import drive, userdata

drive.mount("/content/drive", force_remount=False)

# 与前面正式实验一致，固定 Ultralytics 版本并立即核验。
%pip install -q ultralytics==8.4.92

import base64
import os
import platform
import subprocess
import sys
from pathlib import Path

RUN_ID = "__RUN_ID__"
FORMAL_CODE_COMMIT = "__COMMIT__"
REPOSITORY_URL = "https://github.com/HoverdZ/ship-yolo.git"
REPOSITORY_DIR = Path("/content/ship-yolo")

try:
    github_token = userdata.get("GITHUB_TOKEN")
except Exception as error:
    raise RuntimeError(
        "无法读取 Colab Secret：GITHUB_TOKEN。请先完成身份认证；当前流程已停止。"
    ) from error
if not github_token:
    raise RuntimeError(
        "Colab Secret 中没有可用的 GITHUB_TOKEN；当前流程已停止。"
    )


def git_run(arguments, cwd=None):
    """使用临时请求头访问私有仓库，不把令牌写入远程 URL。"""
    authorization = base64.b64encode(
        f"x-access-token:{github_token}".encode("utf-8")
    ).decode("ascii")
    result = subprocess.run(
        [
            "git",
            "-c",
            f"http.extraHeader=AUTHORIZATION: basic {authorization}",
            *list(arguments),
        ],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        safe_error = result.stderr.replace(github_token, "***")
        raise RuntimeError(
            "GitHub 认证或 Git 操作失败；当前流程已停止，请不要运行后续单元格。\\n"
            + safe_error[-2000:]
        )
    return result.stdout.strip()


if REPOSITORY_DIR.exists():
    if not (REPOSITORY_DIR / ".git").is_dir():
        raise FileExistsError(
            f"{REPOSITORY_DIR} 已存在但不是 ship-yolo Git 仓库；当前流程已停止。"
        )
    tracked_changes = git_run(
        ["status", "--porcelain", "--untracked-files=no"],
        cwd=REPOSITORY_DIR,
    )
    if tracked_changes:
        raise RuntimeError(
            "现有 /content/ship-yolo 含受跟踪的未提交修改，已拒绝覆盖。"
        )
    origin = git_run(["remote", "get-url", "origin"], cwd=REPOSITORY_DIR)
    if "HoverdZ/ship-yolo" not in origin:
        raise RuntimeError(f"现有仓库 origin 不正确：{origin}；当前流程已停止。")
else:
    git_run(
        [
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            REPOSITORY_URL,
            str(REPOSITORY_DIR),
        ]
    )

git_run(
    ["fetch", "--depth=1", "origin", FORMAL_CODE_COMMIT],
    cwd=REPOSITORY_DIR,
)
git_run(["checkout", "--detach", FORMAL_CODE_COMMIT], cwd=REPOSITORY_DIR)
actual_commit = git_run(["rev-parse", "HEAD"], cwd=REPOSITORY_DIR)
assert actual_commit == FORMAL_CODE_COMMIT, (actual_commit, FORMAL_CODE_COMMIT)
os.chdir(REPOSITORY_DIR)
if str(REPOSITORY_DIR) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_DIR))

import torch
import ultralytics

assert ultralytics.__version__ == "8.4.92", ultralytics.__version__
print("固定仓库提交：", actual_commit)
print("Python：", platform.python_version())
print("PyTorch：", torch.__version__)
print("CUDA：", torch.version.cuda)
print("cuDNN：", torch.backends.cudnn.version())
print("Ultralytics：", ultralytics.__version__)
print("GPU：", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)

DRIVE_PROJECT_ROOT = Path("/content/drive/MyDrive/ship_detection/paper_project")
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
'''

TRAIN_CODE = '''from tools.formal_experiments.protocol import (
    print_run_banner,
    resolve_run_state,
    train_foreground,
)
from tools.pls_scam_experiments import (
    FROZEN_TRAINING,
    build_pls_scam_config,
    prepare_pls_scam_experiment,
)

# 不设置人工训练开关；所有检查通过后直接运行 seed=0 的150轮正式训练。
config = build_pls_scam_config(RUN_ID, run_training=True)
assert config.run_training is True
assert config.copy_workers == 32
for key, expected in FROZEN_TRAINING.items():
    assert config.training[key] == expected, (key, config.training[key], expected)
assert config.initialization_weight == "yolo11n.pt"
assert config.expected_detect_strides == (4.0, 8.0, 16.0)

run_mode = resolve_run_state(config)
print("运行方式：", "从最近一轮续训" if run_mode == "resume" else "全新训练")

prepared = prepare_pls_scam_experiment(config)
assert prepared["structure"]["passed"], prepared["structure"]
assert prepared["pls_scam_topology"]["passed"], prepared["pls_scam_topology"]
assert prepared["transfer"]["passed"], prepared["transfer"]
print("数据集划分审计：", prepared["dataset_audit"]["splits"])
print("PLS 上采样层：", prepared["pls_scam_topology"]["upsample_layers"])
print("注意力类型与层：", prepared["pls_scam_topology"]["attention_type"], prepared["pls_scam_topology"]["attention_layers"])
print("保持不变的 PAN Concat：", prepared["pls_scam_topology"]["pan_concat_layers"])
print("输入预处理器：", prepared["pls_scam_topology"]["input_preprocessor"])
print("官方预训练权重 Loaded/Total：", prepared["transfer"]["loaded_total"])
print("模型规模：", prepared["model_info"])

print_run_banner(config)
print("全部训练前检查通过，开始当前内核中的官方训练。")
trained_model, train_results, drive_mirror = train_foreground(
    config,
    initialized_model=prepared["model"],
)

# 后处理会从 best.pt 单独加载模型；先释放训练器、优化器和旧模型显存，避免 OOM。
import contextlib
import gc

prepared_model = prepared.pop("model", None)
del prepared_model, prepared, trained_model, train_results
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
'''

FINAL_CODE = '''from tools.formal_experiments.protocol import finalize_run
from tools.paper_artifacts.results.builders import TABLES, build
from tools.pls_scam_experiments import update_pls_scam_comparison
from tools.windows_collection import verify_checksum_manifest

manifest = finalize_run(
    config,
    mirror=globals().get("drive_mirror"),
)
print("最终验证指标：", manifest["validation_metrics"])
print("已完成的云盘实验目录：", config.drive_dir)

comparison = update_pls_scam_comparison(config)
print("PLS-SCAM 系列对比表：", comparison)

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
'''


def build_notebook(run_id: str, commit: str) -> dict[str, Any]:
    """Build one seven-cell formal training notebook."""

    if run_id not in RUN_IDS:
        raise KeyError(run_id)
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("commit 必须是完整的40位小写 Git SHA。")
    registry = load_pls_scam_registry()
    description = registry["experiments"][run_id]["description"]
    setup = SETUP_CODE.replace("__RUN_ID__", run_id).replace(
        "__COMMIT__",
        commit,
    )
    return {
        "cells": [
            _markdown(
                f"""# {DISPLAY_NAMES[run_id]}：150 epoch 正式实验

实验 ID：`{run_id}`

结构定义：{description}

本实验使用 PLS 的两个 nearest×2 上采样，PAN 路径不变；从官方 `yolo11n.pt` 独立初始化，只运行 `seed=0` 一次，不继承其他实验的 `best.pt`。
"""
            ),
            _markdown(
                """## 1. 挂载云盘、安装固定环境并检出固定代码

这个单元格读取 Colab Secret 中的 `GITHUB_TOKEN`，不会把令牌写入 URL、Notebook 或仓库。认证失败会立即停止。Git 只用于获取固定提交；训练不会放入子进程。
"""
            ),
            _code(setup),
            _markdown(
                """## 2. 快速复制数据、完成训练前审计并直接训练

这个单元格用32个线程把 `/content/drive/MyDrive/ship_detection/data` 只读复制到 `/content/ship_detection/data`，实时显示文件数和字节进度，并生成本地 `data.yaml`。随后自动检查数据、PLS、注意力模块、PAN、CPU 前向/反向、复杂度和权重继承；全部通过后在当前内核直接调用官方 `YOLO.train()`，实时显示完整 epoch 输出。存在匹配的 `last.pt` 时自动从最近一轮续训。
"""
            ),
            _code(TRAIN_CODE),
            _markdown(
                """## 3. 验证 best.pt、保存论文产物并核验备份

训练结束后运行本单元格。它先使用节省显存的固定验证和分批逐图统计，再写入指标、复杂度、校验和与 ZIP，并原子同步到 Drive。测试集保持封存，不参与模型选择。
"""
            ),
            _code(FINAL_CODE),
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


def generate(commit: str, *, output_root: Path = ROOT) -> list[Path]:
    """Generate all four notebooks below ``output_root``."""

    registry = load_pls_scam_registry()
    outputs = []
    for run_id in RUN_IDS:
        relative = Path(registry["experiments"][run_id]["notebook_path"])
        output = output_root / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                build_notebook(run_id, commit),
                ensure_ascii=False,
                indent=1,
            )
            + "\n",
            encoding="utf-8",
        )
        outputs.append(output)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", required=True)
    parser.add_argument("--output-root", type=Path, default=ROOT)
    args = parser.parse_args()
    for output in generate(args.commit, output_root=args.output_root):
        print(output)


if __name__ == "__main__":
    main()
