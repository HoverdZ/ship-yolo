"""Build the formal foreground-training Colab notebook for D-FINE-N."""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "notebooks" / "formal" / "DFINE_N_Complexity_Tradeoff.ipynb"


def _source(value: str) -> str:
    return textwrap.dedent(value).strip() + "\n"


def build_notebook(ship_commit: str) -> nbformat.NotebookNode:
    notebook = nbformat.v4.new_notebook()
    notebook["metadata"] = {
        "accelerator": "GPU",
        "colab": {
            "name": "DFINE_N_Complexity_Tradeoff.ipynb",
            "provenance": [],
        },
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    }
    notebook["cells"] = [
        nbformat.v4.new_markdown_cell(
            _source(
                """
                # D-FINE-N：复杂度—精度权衡正式实验

                本 Notebook 使用官方 `Peterande/D-FINE` 的 Nano 配置与
                HGNetv2-B0，固定输入 640×640、150 epochs、总 batch=8、seed=0。
                训练直接调用官方 `train.main()`，始终在当前 Colab 内核前台
                输出；任何子进程只用于 Git 和依赖安装，绝不承载训练。

                测试集只做格式转换与冻结性审计，不参与训练、验证、选权重或
                本 Notebook 的最终指标计算。
                """
            )
        ),
        nbformat.v4.new_markdown_cell(
            _source(
                """
                ## 1. 固定代码版本并安装官方环境

                本单元挂载 Google Drive，直接从已经公开的
                `HoverdZ/ship-yolo` 拉取固定提交；随后拉取固定提交的 D-FINE
                官方仓库并安装其官方依赖。全程不读取 Colab Secrets，也不需要
                GitHub Token。
                """
            )
        ),
        nbformat.v4.new_code_cell(
            _source(
                f"""
                # 本单元只做环境和公开仓库准备；不启动训练。
                import gc
                import hashlib
                import importlib.metadata
                import json
                import os
                import platform
                import shutil
                import subprocess
                import sys
                from pathlib import Path

                from google.colab import drive

                drive.mount("/content/drive")

                SHIP_REPOSITORY = "https://github.com/HoverdZ/ship-yolo.git"
                SHIP_COMMIT = "{ship_commit}"
                DFINE_REPOSITORY = "https://github.com/Peterande/D-FINE.git"
                DFINE_COMMIT = "7fe2f8889f0b7b817f20c315b40fc15a4fb64ae6"

                SHIP_ROOT = Path("/content/ship-yolo")
                DFINE_ROOT = Path("/content/D-FINE")
                DRIVE_DATA_ROOT = Path("/content/drive/MyDrive/ship_detection/data")
                DRIVE_DATA_YAML = DRIVE_DATA_ROOT / "data.yaml"
                LOCAL_DATA_ROOT = Path("/content/ship_detection/data")
                LOCAL_DATA_YAML = LOCAL_DATA_ROOT / "data.yaml"
                COCO_ROOT = Path("/content/ship_detection/dfine_coco")
                LOCAL_RUN = Path("/content/dfine_runs/DFINE_N/seed_0")
                DRIVE_RUN = Path(
                    "/content/drive/MyDrive/ship_detection/paper_project/"
                    "formal_experiments/DFINE_N/seed_0"
                )
                OFFICIAL_WEIGHT = Path("/content/weights/dfine_n_coco.pth")
                RUNTIME_CONFIG = (
                    DFINE_ROOT / "configs/dfine/custom/dfine_hgnetv2_n_ship.yml"
                )

                def run_checked(command, *, cwd=None):
                    # 运行准备命令并在失败时立即停止。
                    return subprocess.run(
                        command,
                        cwd=cwd,
                        check=True,
                        text=True,
                    )

                # /content 是临时运行盘；每次按固定提交干净拉取，避免 pull 冲突。
                for exact_path in (SHIP_ROOT, DFINE_ROOT):
                    if exact_path.exists():
                        shutil.rmtree(exact_path)

                run_checked(
                    ["git", "clone", "--no-checkout", SHIP_REPOSITORY, str(SHIP_ROOT)]
                )
                run_checked(
                    ["git", "-C", str(SHIP_ROOT), "fetch", "--depth", "1", "origin", SHIP_COMMIT]
                )
                run_checked(["git", "-C", str(SHIP_ROOT), "checkout", "--detach", SHIP_COMMIT])

                run_checked(["git", "clone", "--no-checkout", DFINE_REPOSITORY, str(DFINE_ROOT)])
                run_checked(
                    ["git", "-C", str(DFINE_ROOT), "fetch", "--depth", "1", "origin", DFINE_COMMIT]
                )
                run_checked(["git", "-C", str(DFINE_ROOT), "checkout", "--detach", DFINE_COMMIT])
                assert subprocess.check_output(
                    ["git", "-C", str(SHIP_ROOT), "rev-parse", "HEAD"], text=True
                ).strip() == SHIP_COMMIT
                assert subprocess.check_output(
                    ["git", "-C", str(DFINE_ROOT), "rev-parse", "HEAD"], text=True
                ).strip() == DFINE_COMMIT

                run_checked(
                    [sys.executable, "-m", "pip", "install", "-r", str(DFINE_ROOT / "requirements.txt")]
                )
                sys.path.insert(0, str(SHIP_ROOT))
                sys.path.insert(0, str(DFINE_ROOT))
                os.chdir(DFINE_ROOT)
                print("✅ 两个公开仓库的固定提交与 D-FINE 官方环境准备完成。")
                """
            )
        ),
        nbformat.v4.new_markdown_cell(
            _source(
                """
                ## 2. 快速复制数据、转换 COCO、下载权重并做训练前审计

                本单元使用 32 线程 `shutil.copyfile` 将云盘数据集复制到 Colab
                本地盘，同时分别实时显示文件数和已处理字节。随后保持冻结的
                train/val/test 划分，将 YOLO-HBB 标签确定性转换为 D-FINE 官方
                自定义数据加载器需要的 COCO 格式；单类别原始 category id 固定
                为 0。最后下载并核验官方 D-FINE-N COCO 权重，检查结构、输入、
                训练轮数、batch、类别及前向输出。测试集不会进入运行配置。
                """
            )
        ),
        nbformat.v4.new_code_cell(
            _source(
                """
                # 本单元完成数据、配置、官方权重和模型结构的全部训练前检查。
                import urllib.request

                import torch
                import yaml
                from tqdm.auto import tqdm

                from tools.external_baselines.dfine_n import (
                    OFFICIAL_CHECKPOINT_BYTES,
                    OFFICIAL_CHECKPOINT_SHA256,
                    OFFICIAL_CHECKPOINT_URL,
                    convert_yolo_to_dfine_coco,
                    copy_yolo_dataset_to_local,
                    sha256_file,
                    write_json,
                )

                assert torch.cuda.is_available(), "请在 Colab 中选择 GPU 运行时后再继续。"
                if not DRIVE_DATA_YAML.is_file():
                    raise FileNotFoundError(f"找不到正式数据配置：{DRIVE_DATA_YAML}")

                copy_report = copy_yolo_dataset_to_local(
                    DRIVE_DATA_YAML,
                    drive_data_root=DRIVE_DATA_ROOT,
                    local_data_root=LOCAL_DATA_ROOT,
                    local_data_yaml=LOCAL_DATA_YAML,
                    workers=32,
                )
                conversion_report = convert_yolo_to_dfine_coco(LOCAL_DATA_YAML, COCO_ROOT)
                assert set(conversion_report["splits"]) == {"train", "val", "test"}

                RUNTIME_CONFIG.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(
                    SHIP_ROOT / "experiments/external_baselines/dfine_hgnetv2_n_ship_640.yml",
                    RUNTIME_CONFIG,
                )
                runtime_text = RUNTIME_CONFIG.read_text(encoding="utf-8")
                assert "instances_test.json" not in runtime_text

                def download_official_weight():
                    # 流式下载官方权重并实时显示字节进度，最后核验大小和哈希。
                    OFFICIAL_WEIGHT.parent.mkdir(parents=True, exist_ok=True)
                    if (
                        OFFICIAL_WEIGHT.is_file()
                        and OFFICIAL_WEIGHT.stat().st_size == OFFICIAL_CHECKPOINT_BYTES
                        and sha256_file(OFFICIAL_WEIGHT) == OFFICIAL_CHECKPOINT_SHA256
                    ):
                        print("✅ 已存在且校验通过的官方 D-FINE-N 权重。")
                        return
                    temporary = OFFICIAL_WEIGHT.with_suffix(".download")
                    temporary.unlink(missing_ok=True)
                    with urllib.request.urlopen(OFFICIAL_CHECKPOINT_URL) as response, temporary.open("wb") as stream:
                        total = int(response.headers.get("Content-Length", OFFICIAL_CHECKPOINT_BYTES))
                        with tqdm(total=total, unit="B", unit_scale=True, desc="官方权重") as bar:
                            while True:
                                block = response.read(1024 * 1024)
                                if not block:
                                    break
                                stream.write(block)
                                bar.update(len(block))
                    assert temporary.stat().st_size == OFFICIAL_CHECKPOINT_BYTES
                    assert sha256_file(temporary) == OFFICIAL_CHECKPOINT_SHA256
                    os.replace(temporary, OFFICIAL_WEIGHT)

                download_official_weight()

                from src.core import YAMLConfig

                audit_config = YAMLConfig(str(RUNTIME_CONFIG), tuning=str(OFFICIAL_WEIGHT))
                audit_config.yaml_cfg["HGNetv2"]["pretrained"] = False
                merged = audit_config.yaml_cfg
                assert merged["epochs"] == 150
                assert merged["eval_spatial_size"] == [640, 640]
                assert merged["num_classes"] == 1
                assert merged["remap_mscoco_category"] is False
                assert merged["HGNetv2"]["name"] == "B0"
                assert merged["train_dataloader"]["total_batch_size"] == 8
                assert merged["val_dataloader"]["total_batch_size"] == 8
                assert merged["train_dataloader"]["collate_fn"]["base_size"] == 640
                assert merged["train_dataloader"]["collate_fn"]["base_size_repeat"] is None

                # 仅做一次 CPU 推理形状检查；正式权重继承由训练入口逐张量审计。
                audit_model = audit_config.model.cpu().eval()
                with torch.no_grad():
                    audit_output = audit_model(torch.zeros(1, 3, 640, 640))
                assert audit_output["pred_boxes"].shape[0] == 1
                assert audit_output["pred_logits"].shape[-1] == 1
                structure_report = {
                    "model": "D-FINE-N",
                    "backbone": "HGNetv2-B0",
                    "parameters_before_training": sum(p.numel() for p in audit_model.parameters()),
                    "pred_boxes_shape": list(audit_output["pred_boxes"].shape),
                    "pred_logits_shape": list(audit_output["pred_logits"].shape),
                    "passed": True,
                }
                del audit_output, audit_model, audit_config
                gc.collect()

                LOCAL_RUN.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(RUNTIME_CONFIG, LOCAL_RUN / "runtime_config.yml")
                write_json(LOCAL_RUN / "dataset_copy_report.json", copy_report)
                write_json(LOCAL_RUN / "dataset_conversion_report.json", conversion_report)
                write_json(LOCAL_RUN / "structure_report.json", structure_report)
                print(json.dumps(structure_report, ensure_ascii=False, indent=2))
                print("✅ 数据、结构和官方权重均通过训练前审计。")
                """
            )
        ),
        nbformat.v4.new_markdown_cell(
            _source(
                """
                ## 3. 在当前内核前台运行官方 D-FINE 训练

                本单元自动判断是否已经完成、是否可从本地或云盘中最靠后的
                官方断点续训；不需要修改任何开关。首次训练使用官方 D-FINE-N
                COCO 权重，并输出精确的 Loaded/Total 张量审计。训练调用是当前
                内核中的 `dfine_train.main(train_args)`，每个 epoch 的官方日志
                会实时显示。后台线程只复制已经稳定的断点文件，不执行训练。
                """
            )
        ),
        nbformat.v4.new_code_cell(
            _source(
                """
                # 本单元直接运行官方训练；严禁改成 subprocess、Popen 或 torchrun。
                import importlib.util
                from types import SimpleNamespace

                from tools.external_baselines.dfine_n import (
                    StableDriveMirror,
                    install_dfine_tuning_audit,
                )

                def load_official_train_module():
                    spec = importlib.util.spec_from_file_location(
                        "dfine_official_train", DFINE_ROOT / "train.py"
                    )
                    module = importlib.util.module_from_spec(spec)
                    assert spec.loader is not None
                    spec.loader.exec_module(module)
                    return module

                dfine_train = load_official_train_module()
                DRIVE_RUN.mkdir(parents=True, exist_ok=True)

                # 云盘中有断点时先恢复到本地；只复制本实验的精确目录。
                drive_checkpoints = [
                    DRIVE_RUN / name
                    for name in ("last.pth", "best_stg1.pth", "best_stg2.pth")
                    if (DRIVE_RUN / name).is_file()
                ]
                if drive_checkpoints:
                    shutil.copytree(DRIVE_RUN, LOCAL_RUN, dirs_exist_ok=True)

                def completed_log(path):
                    if not path.is_file():
                        return False
                    epochs = []
                    for line in path.read_text(encoding="utf-8").splitlines():
                        if line.strip():
                            row = json.loads(line)
                            if "epoch" in row:
                                epochs.append(int(row["epoch"]))
                    return bool(epochs) and max(epochs) >= 149

                def newest_valid_checkpoint():
                    candidates = []
                    failures = {}
                    for name in ("last.pth", "best_stg1.pth", "best_stg2.pth"):
                        path = LOCAL_RUN / name
                        if not path.is_file():
                            continue
                        try:
                            try:
                                state = torch.load(path, map_location="cpu", weights_only=False)
                            except TypeError:
                                state = torch.load(path, map_location="cpu")
                            assert "model" in state and "last_epoch" in state
                            candidates.append((int(state["last_epoch"]), path))
                        except Exception as error:
                            failures[name] = repr(error)
                    if failures:
                        print("忽略损坏断点：", failures)
                    return max(candidates, default=(None, None), key=lambda item: item[0])

                already_complete = (DRIVE_RUN / "COMPLETED.ok").is_file() or completed_log(
                    LOCAL_RUN / "log.txt"
                )
                if already_complete:
                    print("✅ 已检测到完整的 150-epoch 日志；跳过重复训练，进入成果整理。")
                else:
                    resume_epoch, resume_checkpoint = newest_valid_checkpoint()
                    if resume_checkpoint is None:
                        tuning_checkpoint = str(OFFICIAL_WEIGHT)
                        print("开始独立训练：使用官方 D-FINE-N COCO 权重初始化。")
                    else:
                        tuning_checkpoint = None
                        print(
                            f"从官方断点续训：{resume_checkpoint.name}，"
                            f"已保存到 zero-based epoch {resume_epoch}。"
                        )

                    install_dfine_tuning_audit(
                        LOCAL_RUN / "pretrained_transfer_report.json"
                    )
                    train_args = SimpleNamespace(
                        config=str(RUNTIME_CONFIG),
                        resume=str(resume_checkpoint) if resume_checkpoint else None,
                        tuning=tuning_checkpoint,
                        device="cuda",
                        seed=0,
                        use_amp=True,
                        output_dir=str(LOCAL_RUN),
                        summary_dir=None,
                        test_only=False,
                        update=None,
                        print_method="builtin",
                        print_rank=0,
                        local_rank=None,
                    )
                    mirror = StableDriveMirror(LOCAL_RUN, DRIVE_RUN, interval_seconds=30.0)
                    mirror.start()
                    try:
                        dfine_train.main(train_args)
                    finally:
                        mirror.stop()

                assert completed_log(LOCAL_RUN / "log.txt"), (
                    "训练尚未达到第 150 轮；重新运行本单元会自动从最靠后的有效断点续训。"
                )
                print("✅ D-FINE-N 的 150-epoch 官方训练日志已完整生成。")
                """
            )
        ),
        nbformat.v4.new_markdown_cell(
            _source(
                """
                ## 4. 验证最佳权重并生成论文所需复杂度产物

                本单元按验证集 COCO AP50–95 选择最佳阶段权重，在当前内核中用
                官方验证入口重新验证；记录 AP50、AP75、AP50–95，以及官方
                Validator 在 conf=0.5、IoU=0.5 下的 P/R。随后在 640×640、batch=1
                下计算参数量、FLOPs/MACs，并在当前 GPU 上测量纯 PyTorch FP16
                前向延迟（不含预处理、后处理和 NMS）。最后才生成固定校验清单并
                一次性同步云盘，避免把仍会变化的文件写入校验清单。
                """
            )
        ),
        nbformat.v4.new_code_cell(
            _source(
                """
                # 本单元完成验证、复杂度测量和最终原子化保存，不访问测试集。
                import contextlib
                import copy
                import csv
                import io
                import re
                import statistics
                import time

                from calflops import calculate_flops

                from tools.external_baselines.dfine_n import (
                    parse_training_log,
                    parse_validator_metrics,
                    sha256_file,
                    write_checksum_manifest,
                    write_json,
                    write_metrics_csv,
                )

                training_metrics = parse_training_log(LOCAL_RUN / "log.txt", stop_epoch=140)
                selected_checkpoint = LOCAL_RUN / training_metrics["checkpoint_name"]
                if not selected_checkpoint.is_file():
                    raise FileNotFoundError(
                        f"日志选择了 {selected_checkpoint.name}，但该官方最佳权重不存在。"
                    )
                best_checkpoint = LOCAL_RUN / "best.pth"
                shutil.copyfile(selected_checkpoint, best_checkpoint)
                assert sha256_file(best_checkpoint) == sha256_file(selected_checkpoint)

                class LiveCapture(io.TextIOBase):
                    # 一边保留官方实时输出，一边捕获最终验证摘要。
                    def __init__(self, live_stream):
                        self.live_stream = live_stream
                        self.buffer = io.StringIO()

                    def write(self, value):
                        self.live_stream.write(value)
                        self.buffer.write(value)
                        return len(value)

                    def flush(self):
                        self.live_stream.flush()

                    def getvalue(self):
                        return self.buffer.getvalue()

                gc.collect()
                torch.cuda.empty_cache()
                validation_output = LOCAL_RUN / "validation"
                validation_args = SimpleNamespace(
                    config=str(RUNTIME_CONFIG),
                    resume=str(best_checkpoint),
                    tuning=None,
                    device="cuda",
                    seed=0,
                    use_amp=True,
                    output_dir=str(validation_output),
                    summary_dir=None,
                    test_only=True,
                    update=None,
                    print_method="builtin",
                    print_rank=0,
                    local_rank=None,
                )
                capture = LiveCapture(sys.stdout)
                with contextlib.redirect_stdout(capture):
                    dfine_train.main(validation_args)
                validation_text = capture.getvalue()
                (LOCAL_RUN / "best_validation_stdout.txt").write_text(
                    validation_text, encoding="utf-8"
                )
                pr_metrics = parse_validator_metrics(validation_text)
                printed_ap = [
                    float(value)
                    for value in re.findall(
                        r"Average Precision\\s+\\(AP\\).*?=\\s+(-?\\d+\\.\\d+)",
                        validation_text,
                    )
                ]
                assert len(printed_ap) >= 6, "没有捕获到完整的官方 COCO AP 摘要。"
                assert abs(printed_ap[0] - training_metrics["map50_95"]) <= 0.002
                assert abs(printed_ap[1] - training_metrics["map50"]) <= 0.002
                assert abs(printed_ap[2] - training_metrics["map75"]) <= 0.002

                # 重新构建并加载最佳权重，确保复杂度和延迟对应最终一类别模型。
                gc.collect()
                torch.cuda.empty_cache()
                complexity_config = YAMLConfig(str(RUNTIME_CONFIG), resume=str(best_checkpoint))
                complexity_config.yaml_cfg["HGNetv2"]["pretrained"] = False
                complexity_model = complexity_config.model
                try:
                    checkpoint_state = torch.load(
                        best_checkpoint, map_location="cpu", weights_only=False
                    )
                except TypeError:
                    checkpoint_state = torch.load(best_checkpoint, map_location="cpu")
                model_state = (
                    checkpoint_state["ema"]["module"]
                    if "ema" in checkpoint_state
                    else checkpoint_state["model"]
                )
                complexity_model.load_state_dict(model_state, strict=True)
                deployed = copy.deepcopy(complexity_model).deploy().cpu().eval()
                flops, macs, _ = calculate_flops(
                    model=deployed,
                    input_shape=(1, 3, 640, 640),
                    output_as_string=False,
                    print_detailed=False,
                )
                parameters = sum(parameter.numel() for parameter in deployed.parameters())

                # 同一运行时的 batch=1、640、FP16 纯模型前向延迟。
                latency_model = deployed.cuda().half().eval()
                latency_input = torch.zeros(1, 3, 640, 640, device="cuda", dtype=torch.float16)
                with torch.inference_mode():
                    for _ in range(50):
                        latency_model(latency_input)
                    torch.cuda.synchronize()
                    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(200)]
                    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(200)]
                    for start, end in zip(start_events, end_events):
                        start.record()
                        latency_model(latency_input)
                        end.record()
                    torch.cuda.synchronize()
                latency_ms = [start.elapsed_time(end) for start, end in zip(start_events, end_events)]
                mean_latency = statistics.fmean(latency_ms)
                sorted_latency = sorted(latency_ms)
                complexity = {
                    "model": "D-FINE-N",
                    "input_shape": [1, 3, 640, 640],
                    "parameters": int(parameters),
                    "flops": float(flops),
                    "gflops": float(flops) / 1e9,
                    "macs": float(macs),
                    "gmacs": float(macs) / 1e9,
                    "latency_mean_ms": mean_latency,
                    "latency_median_ms": statistics.median(latency_ms),
                    "latency_p95_ms": sorted_latency[int(0.95 * (len(sorted_latency) - 1))],
                    "fps_from_mean_latency": 1000.0 / mean_latency,
                    "latency_warmup": 50,
                    "latency_iterations": 200,
                    "framework": "PyTorch",
                    "precision": "FP16",
                    "batch": 1,
                    "includes_preprocess": False,
                    "includes_postprocess": False,
                    "includes_nms": False,
                    "gpu": torch.cuda.get_device_name(0),
                }

                metrics = {
                    "experiment": "DFINE_N",
                    "best_epoch": training_metrics["best_epoch"],
                    "precision": float(pr_metrics["precision"]),
                    "recall": float(pr_metrics["recall"]),
                    "map50": training_metrics["map50"],
                    "map75": training_metrics["map75"],
                    "map50_95": training_metrics["map50_95"],
                    "map_small": training_metrics["map_small"],
                    "map_medium": training_metrics["map_medium"],
                    "map_large": training_metrics["map_large"],
                    "imgsz": 640,
                    "epochs": 150,
                    "batch": 8,
                    "seed": 0,
                    "precision_recall_conf": 0.5,
                    "precision_recall_iou": 0.5,
                    "test_evaluated": False,
                }
                environment = {
                    "ship_yolo_commit": SHIP_COMMIT,
                    "dfine_repository": DFINE_REPOSITORY,
                    "dfine_commit": DFINE_COMMIT,
                    "official_checkpoint": str(OFFICIAL_WEIGHT),
                    "official_checkpoint_sha256": sha256_file(OFFICIAL_WEIGHT),
                    "best_checkpoint_sha256": sha256_file(best_checkpoint),
                    "python": platform.python_version(),
                    "torch": torch.__version__,
                    "torchvision": importlib.metadata.version("torchvision"),
                    "cuda_runtime": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(0),
                }
                run_manifest = {
                    "experiment": "DFINE_N",
                    "implementation": "official Peterande/D-FINE",
                    "initialization": "official D-FINE-N COCO checkpoint",
                    "loaded_total_report": "pretrained_transfer_report.json",
                    "selection": "validation COCO AP50-95",
                    "selected_checkpoint": selected_checkpoint.name,
                    "controls": {"imgsz": 640, "epochs": 150, "batch": 8, "seed": 0},
                    "dataset_splits_frozen": True,
                    "test_evaluated": False,
                }
                write_json(LOCAL_RUN / "metrics.json", metrics)
                write_metrics_csv(LOCAL_RUN / "metrics.csv", metrics)
                write_json(LOCAL_RUN / "complexity.json", complexity)
                write_json(LOCAL_RUN / "environment.json", environment)
                write_json(LOCAL_RUN / "run_manifest.json", run_manifest)
                (LOCAL_RUN / "COMPLETED.ok").write_text(
                    "D-FINE-N formal run completed and packaged.\\n", encoding="utf-8"
                )
                write_checksum_manifest(LOCAL_RUN)

                # 校验清单生成后不再修改本地正式文件，最后一次性同步整个目录。
                shutil.copytree(LOCAL_RUN, DRIVE_RUN, dirs_exist_ok=True)
                assert (DRIVE_RUN / "COMPLETED.ok").is_file()
                assert sha256_file(DRIVE_RUN / "best.pth") == environment["best_checkpoint_sha256"]
                print(json.dumps(metrics, ensure_ascii=False, indent=2))
                print(json.dumps(complexity, ensure_ascii=False, indent=2))
                print(f"✅ 正式产物已保存：{DRIVE_RUN}")
                """
            )
        ),
    ]
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile(cell["source"], "<dfine-n-colab-cell>", "exec")
            cell["execution_count"] = None
            cell["outputs"] = []
    return notebook


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ship-commit", required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    notebook = build_notebook(args.ship_commit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
