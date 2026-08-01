"""Generate pinned direct-training InceptionDW CA-SCAM Colab notebooks."""

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
from tools.generate_pls_scam_notebooks import SETUP_CODE, _code, _markdown
from tools.incdw_ca_scam_experiments import (
    MODEL_NAMES,
    RUN_IDS,
    load_incdw_ca_scam_registry,
)


TRAIN_CODE = '''from tools.formal_experiments.protocol import (
    print_run_banner,
    resolve_run_state,
    train_foreground,
)
from tools.incdw_ca_scam_experiments import (
    FROZEN_TRAINING,
    build_incdw_ca_scam_config,
    prepare_incdw_ca_scam_experiment,
)

# 不设置人工训练开关；所有检查通过后直接运行 seed=0 的150轮正式训练。
config = build_incdw_ca_scam_config(RUN_ID, run_training=True)
assert config.run_training is True
assert config.copy_workers == 32
for key, expected in FROZEN_TRAINING.items():
    assert config.training[key] == expected, (key, config.training[key], expected)
assert config.initialization_weight == "yolo11n.pt"
assert config.expected_detect_strides == (4.0, 8.0, 16.0)

run_mode = resolve_run_state(config)
print("运行方式：", "从最近一轮续训" if run_mode == "resume" else "全新训练")

prepared = prepare_incdw_ca_scam_experiment(config)
assert prepared["structure"]["passed"], prepared["structure"]
assert prepared["incdw_ca_scam_topology"]["passed"], prepared["incdw_ca_scam_topology"]
assert prepared["transfer"]["passed"], prepared["transfer"]
topology = prepared["incdw_ca_scam_topology"]
print("数据集划分审计：", prepared["dataset_audit"]["splits"])
print("InceptionDW 层与范围：", topology["inceptiondw_layers"], topology["inceptiondw_scope"])
print("上采样方式与层：", topology["upsampling"], topology["upsample_layers"])
print("CA-SCAM 层：", topology["attention_layers"])
print("保持不变的 PAN Concat：", topology["pan_concat_layers"])
print("输入预处理器：", topology["input_preprocessor"])
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
from tools.incdw_ca_scam_experiments import update_incdw_ca_scam_comparison
from tools.paper_artifacts.results.builders import TABLES, build
from tools.windows_collection import verify_checksum_manifest

manifest = finalize_run(
    config,
    mirror=globals().get("drive_mirror"),
)
print("最终验证指标：", manifest["validation_metrics"])
print("已完成的云盘实验目录：", config.drive_dir)

comparison = update_incdw_ca_scam_comparison(config)
print("InceptionDW-CA-SCAM 对比表：", comparison)

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
    """Build one seven-cell formal training Notebook."""

    if run_id not in RUN_IDS:
        raise KeyError(run_id)
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("commit 必须是完整的40位小写 Git SHA。")
    registry = load_incdw_ca_scam_registry()
    experiment = registry["experiments"][run_id]
    modules = experiment["modules"]
    upsampling = "PLS nearest×2" if modules["pls"] else "DPLS DySample"
    setup = SETUP_CODE.replace("__RUN_ID__", run_id).replace(
        "__COMMIT__",
        commit,
    )
    return {
        "cells": [
            _markdown(
                f"""# {MODEL_NAMES[run_id]}：150 epoch 正式实验

实验 ID：`{run_id}`

结构定义：{experiment["description"]}

InceptionDW 仅用于 P2/P3 的 `C3k2` Bottleneck 第二个空间卷积，第一个卷积与 P4 深层 `C3k2` 保留；上采样采用 {upsampling}，PAN 其余路径不变。本实验从官方 `yolo11n.pt` 独立初始化，只运行 `seed=0` 一次，不继承其他实验的 `best.pt`。
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

这个单元格用32个线程把 `/content/drive/MyDrive/ship_detection/data` 只读复制到 `/content/ship_detection/data`，实时显示文件数和字节进度，并生成本地 `data.yaml`。随后自动检查数据、InceptionDW 替换范围、PLS/DPLS、CA-SCAM、VGUP、PAN、CPU 前向/反向、复杂度和官方权重继承；全部通过后在当前内核直接调用官方 `YOLO.train()`，实时显示完整 epoch 输出。存在匹配的 `last.pt` 时自动从最近一轮续训。
"""
            ),
            _code(TRAIN_CODE),
            _markdown(
                """## 3. 验证 best.pt、保存论文产物并核验备份

训练结束后运行本单元格。它先释放训练对象显存，再使用节省显存的固定验证和分批逐图统计，随后写入指标、复杂度、校验和与 ZIP，并原子同步到 Drive。测试集保持封存，不参与模型选择。
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
    """Generate both Notebooks below ``output_root``."""

    registry = load_incdw_ca_scam_registry()
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
