"""在当前 Python 进程内分阶段训练 InceptionDW-SCSharedHead。"""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.scshared_head_utils import (
    EXPERIMENT_NAME,
    git_commit,
    require_ultralytics_version,
    save_initialized_model,
    write_json,
)


METADATA_FILE = "experiment_metadata.json"
STAGE_CHECKPOINT = "stage80_resume.pt"


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"无法识别的布尔值：{value}")


@dataclass
class TrainingRequest:
    data: str
    project: str
    weights: str = "yolo11n.pt"
    name: str = EXPERIMENT_NAME
    total_epochs: int = 150
    stage_epochs: int = 80
    imgsz: int = 640
    batch: int = 8
    workers: int = 2
    device: str = "0"
    seed: int = 0
    optimizer: str = "auto"
    resume: bool = False
    resume_checkpoint: str | None = None


def validate_data_yaml(path: str | Path) -> Path:
    data = Path(path).expanduser().resolve()
    if not data.is_file():
        raise FileNotFoundError(f"找不到数据集 YAML：{data}；禁止回退到示例数据集。")
    return data


def validate_resume_checkpoint(path: str | Path, expected_total_epochs: int) -> Path:
    import torch

    checkpoint = Path(path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"找不到续训检查点：{checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError(f"检查点类型无效：{type(payload).__name__}")
    missing = [
        key
        for key in ("epoch", "optimizer", "train_args")
        if payload.get(key) is None
    ]
    if missing:
        raise RuntimeError(f"检查点缺少续训状态 {missing}：{checkpoint}")
    configured_epochs = int(payload["train_args"].get("epochs", -1))
    if configured_epochs != expected_total_epochs:
        raise RuntimeError(
            f"检查点调度总轮数为 {configured_epochs}，预期为 {expected_total_epochs}；"
            "禁止用不同的调度周期重新启动。"
        )
    return checkpoint


def prepare_new_run_directory(run_dir: Path) -> None:
    if not run_dir.exists():
        run_dir.mkdir(parents=True)
        return
    if not run_dir.is_dir():
        raise FileExistsError(f"输出路径不是文件夹：{run_dir}")
    entries = {entry.name for entry in run_dir.iterdir()}
    if entries and not entries.issubset({METADATA_FILE}):
        raise FileExistsError(
            f"输出文件夹已经包含训练产物：{run_dir}。"
            "请使用续训单元格，或更换实验名称。"
        )


def _attach_stage_callbacks(model, run_dir: Path, stage_epochs: int) -> None:
    """在筛选轮数停止，并保存包含优化器与调度器状态的原始检查点。"""

    stage_path = run_dir / "weights" / STAGE_CHECKPOINT

    def stop_at_stage(trainer):
        if trainer.epoch + 1 == stage_epochs:
            print(
                f"\n已完成第 {stage_epochs} 轮筛选；"
                "正在保留可继续训练的优化器、调度器、EMA 与 AMP 状态。"
            )
            trainer.stop = True

    def preserve_stage_checkpoint(trainer):
        if trainer.epoch + 1 == stage_epochs:
            if not trainer.last.is_file():
                raise FileNotFoundError(
                    f"Ultralytics 未生成原始续训检查点：{trainer.last}"
                )
            stage_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(trainer.last, stage_path)
            print(f"已保存第 {stage_epochs} 轮续训检查点：{stage_path}")

    model.add_callback("on_train_epoch_end", stop_at_stage)
    model.add_callback("on_model_save", preserve_stage_checkpoint)


def run_training(request: TrainingRequest):
    """同步执行训练，确保 Colab 可以实时显示每一轮输出。"""

    from custom_modules.register import register_custom_modules

    register_custom_modules()
    from ultralytics import YOLO

    data_yaml = validate_data_yaml(request.data)
    project = Path(request.project).expanduser().resolve()
    run_dir = project / request.name
    version = require_ultralytics_version()
    initialization_manifest = None

    if request.total_epochs <= request.stage_epochs:
        raise ValueError("total_epochs 必须大于 stage_epochs。")

    if request.resume:
        candidate = (
            Path(request.resume_checkpoint).expanduser()
            if request.resume_checkpoint
            else run_dir / "weights" / STAGE_CHECKPOINT
        )
        checkpoint = validate_resume_checkpoint(candidate, request.total_epochs)
        import torch

        checkpoint_payload = torch.load(
            checkpoint, map_location="cpu", weights_only=False
        )
        completed_epochs = int(checkpoint_payload["epoch"]) + 1
        model = YOLO(str(checkpoint), task="detect")
        resume_source = str(checkpoint)
        if completed_epochs < request.stage_epochs:
            _attach_stage_callbacks(model, run_dir, request.stage_epochs)
    else:
        prepare_new_run_directory(run_dir)
        initialization_path = (
            project / "_initialization" / f"{request.name}_init.pt"
        )
        initialization_manifest = save_initialized_model(
            weights=request.weights,
            output=initialization_path,
            seed=request.seed,
        )
        write_json(
            initialization_path.with_suffix(".json"),
            initialization_manifest,
        )
        model = YOLO(str(initialization_path), task="detect")
        resume_source = None
        _attach_stage_callbacks(model, run_dir, request.stage_epochs)

    metadata = {
        **asdict(request),
        "data": str(data_yaml),
        "project": str(project),
        "run_dir": str(run_dir),
        "resume_source": resume_source,
        "stage_checkpoint": str(run_dir / "weights" / STAGE_CHECKPOINT),
        "ultralytics_version": version,
        "git_commit": git_commit(),
        "initialization": initialization_manifest,
        "training_process": "direct current-process Python call",
        "scheduler_horizon": request.total_epochs,
    }
    write_json(run_dir / METADATA_FILE, metadata)

    print(f"实验名称：{request.name}")
    print(f"数据集：{data_yaml}")
    print(f"输出目录：{run_dir}")
    print(f"筛选轮数/调度总轮数：{request.stage_epochs}/{request.total_epochs}")
    print(f"是否续训：{request.resume}")
    if initialization_manifest is not None:
        transfer = initialization_manifest["weight_transfer"]
        print(
            "从外部预训练权重继承的状态张量："
            f"{transfer['loaded_state_tensors']}/{transfer['total_state_tensors']}"
        )
        print(
            "其中映射的原生 Detect 输出张量："
            f"{transfer['mapped_native_detect_output_tensors']}"
        )
        print(
            "继承的目标参数元素比例："
            f"{transfer['loaded_target_parameter_element_ratio']:.4%}"
        )
        print(
            "说明：随后 Ultralytics 显示的 Transferred x/x，"
            "表示从本初始化检查点载入同构训练模型，两者统计口径不同。"
        )

    if request.resume:
        train_args = {"resume": True}
    else:
        train_args = {
            "data": str(data_yaml),
            # 从第一轮就声明 150 轮调度，在第 80 轮由回调暂停。
            "epochs": request.total_epochs,
            "imgsz": request.imgsz,
            "batch": request.batch,
            "workers": request.workers,
            "device": request.device,
            "seed": request.seed,
            "deterministic": True,
            "optimizer": request.optimizer,
            "project": str(project),
            "name": request.name,
            "exist_ok": True,
        }
    result = model.train(**train_args)
    write_json(
        run_dir / "summary.json",
        {
            "experiment_name": request.name,
            "resume": request.resume,
            "resume_source": resume_source,
            "results": getattr(result, "results_dict", None) or {},
            "ultralytics_version": version,
            "git_commit": git_commit(),
        },
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--weights", default="yolo11n.pt")
    parser.add_argument("--name", default=EXPERIMENT_NAME)
    parser.add_argument("--total-epochs", type=int, default=150)
    parser.add_argument("--stage-epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--optimizer", default="auto")
    parser.add_argument("--resume", type=parse_bool, default=False)
    parser.add_argument("--resume-checkpoint")
    return parser


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    return run_training(
        TrainingRequest(
            data=args.data,
            project=args.project,
            weights=args.weights,
            name=args.name,
            total_epochs=args.total_epochs,
            stage_epochs=args.stage_epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            device=args.device,
            seed=args.seed,
            optimizer=args.optimizer,
            resume=args.resume,
            resume_checkpoint=args.resume_checkpoint,
        )
    )


if __name__ == "__main__":
    main()
