"""Recompute model complexity from the real checkpoints and YAML files."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _logical_trainable_parameters(
    network: torch.nn.Module,
    train_args: dict[str, Any],
) -> tuple[int, str]:
    """Return architecture-level trainable parameters, not stripped-PT flags."""

    freeze = train_args.get("freeze")
    total = sum(parameter.numel() for parameter in network.parameters())
    if freeze in (None, False, 0, [], ()):
        return total, "train_args.freeze is empty; stripped checkpoint flags ignored"

    if isinstance(freeze, int):
        prefixes = tuple(f"model.{index}." for index in range(freeze))
    elif isinstance(freeze, (list, tuple)):
        prefixes = tuple(f"model.{int(index)}." for index in freeze)
    else:
        return total, f"unsupported freeze specification {freeze!r}; reported architecture total"
    trainable = sum(
        parameter.numel()
        for name, parameter in network.named_parameters()
        if not name.startswith(prefixes)
    )
    return trainable, f"reconstructed from train_args.freeze={freeze!r}"


def _official_gflops(network: torch.nn.Module, imgsz: int) -> tuple[float | None, str, str | None]:
    from ultralytics.utils.torch_utils import get_flops

    try:
        value = float(get_flops(network, imgsz=imgsz))
        if value > 0:
            return value, "ultralytics.utils.torch_utils.get_flops", None
        raise RuntimeError(f"Ultralytics returned non-positive GFLOPs: {value}")
    except Exception as official_error:
        try:
            from thop import profile

            example = torch.empty(1, 3, imgsz, imgsz, device="cpu")
            macs, _parameters = profile(
                deepcopy(network).cpu(),
                inputs=(example,),
                verbose=False,
            )
            return (
                float(macs) * 2.0 / 1e9,
                "THOP profile fallback (2 MACs per FLOP convention)",
                f"Ultralytics failure: {type(official_error).__name__}: {official_error}",
            )
        except Exception as thop_error:
            return (
                None,
                "failed",
                (
                    f"Ultralytics: {type(official_error).__name__}: {official_error}; "
                    f"THOP: {type(thop_error).__name__}: {thop_error}"
                ),
            )


def _graph_rows(network: torch.nn.Module) -> list[dict[str, Any]]:
    sequence = getattr(network, "model", None)
    if sequence is None:
        return []
    rows = []
    for index, layer in enumerate(sequence):
        rows.append(
            {
                "index": int(getattr(layer, "i", index)),
                "from": getattr(layer, "f", None),
                "type": type(layer).__name__,
                "parameters": sum(parameter.numel() for parameter in layer.parameters()),
                "arguments": repr(getattr(layer, "np", None)),
            }
        )
    return rows


def _measure(
    label: str,
    source: Path,
    *,
    imgsz: int,
    train_args: dict[str, Any] | None,
    checkpoint_sha256: str | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from custom_modules.register import register_custom_modules
    from ultralytics import YOLO, __version__ as ultralytics_version

    register_custom_modules()
    wrapper = YOLO(str(source), verbose=False)
    network = wrapper.model.float().cpu().eval()
    total = sum(parameter.numel() for parameter in network.parameters())
    trainable, trainable_note = _logical_trainable_parameters(
        network,
        train_args or {},
    )
    gflops, method, error = _official_gflops(network, imgsz)
    rows = _graph_rows(network)
    record = {
        "模型": label,
        "来源": str(source.resolve()),
        "来源类型": "checkpoint" if source.suffix.lower() == ".pt" else "model_yaml",
        "Params": total,
        "trainable Params": trainable,
        "trainable统计说明": trainable_note,
        f"GFLOPs@{imgsz}": gflops,
        "GFLOPs方法": method,
        "GFLOPs错误": error,
        "checkpoint大小_字节": source.stat().st_size if source.suffix.lower() == ".pt" else None,
        "checkpoint_SHA256": checkpoint_sha256,
        "state tensors": len(network.state_dict()),
        "顶层层数": len(rows),
        "Detect strides": json.dumps(
            [float(value) for value in network.stride.detach().cpu()],
            ensure_ascii=False,
        ),
        "审计Ultralytics版本": ultralytics_version,
        "审计Torch版本": torch.__version__,
    }
    return record, rows


def compute_complexities(
    audit_json: str | Path,
    output_csv: str | Path,
    graph_summary: str | Path,
    *,
    imgsz: int = 640,
    extra_models: list[tuple[str, str | Path]] | None = None,
) -> list[dict[str, Any]]:
    audit = json.loads(Path(audit_json).read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    graphs: list[tuple[str, list[dict[str, Any]]]] = []
    for item in audit["records"]:
        label = item["identity"]
        print(f"[复杂度] {label}: {item['checkpoint_name']}", flush=True)
        try:
            record, graph = _measure(
                label,
                Path(item["checkpoint"]),
                imgsz=imgsz,
                train_args=item.get("train_args") or {},
                checkpoint_sha256=item.get("checkpoint_sha256"),
            )
        except Exception as error:
            record = {
                "模型": label,
                "来源": item["checkpoint"],
                "来源类型": "checkpoint",
                "Params": item.get("parameters"),
                "trainable Params": None,
                "trainable统计说明": "构建失败",
                f"GFLOPs@{imgsz}": None,
                "GFLOPs方法": "failed",
                "GFLOPs错误": f"{type(error).__name__}: {error}",
                "checkpoint大小_字节": item.get("checkpoint_size_bytes"),
                "checkpoint_SHA256": item.get("checkpoint_sha256"),
                "state tensors": item.get("state_tensors"),
                "顶层层数": None,
                "Detect strides": json.dumps(item.get("detect_strides"), ensure_ascii=False),
                "审计Ultralytics版本": None,
                "审计Torch版本": torch.__version__,
            }
            graph = []
        records.append(record)
        graphs.append((label, graph))

    for label, source in extra_models or []:
        path = Path(source)
        print(f"[复杂度] {label}: {path}", flush=True)
        try:
            record, graph = _measure(
                label,
                path,
                imgsz=imgsz,
                train_args={"freeze": None},
                checkpoint_sha256=None,
            )
        except Exception as error:
            record = {
                "模型": label,
                "来源": str(path.resolve()),
                "来源类型": "model_yaml",
                "Params": None,
                "trainable Params": None,
                "trainable统计说明": "构建失败",
                f"GFLOPs@{imgsz}": None,
                "GFLOPs方法": "failed",
                "GFLOPs错误": f"{type(error).__name__}: {error}",
                "checkpoint大小_字节": None,
                "checkpoint_SHA256": None,
                "state tensors": None,
                "顶层层数": None,
                "Detect strides": None,
                "审计Ultralytics版本": None,
                "审计Torch版本": torch.__version__,
            }
            graph = []
        records.append(record)
        graphs.append((label, graph))

    destination = Path(output_csv)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = list(records[0]) if records else ["模型"]
    with destination.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    graph_path = Path(graph_summary)
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 模型图摘要",
        "",
        f"统一输入尺寸：{imgsz}×{imgsz}。参数量与 GFLOPs 均在当前真实代码中重新计算。",
        "",
    ]
    for label, graph in graphs:
        lines.extend([f"## {label}", "", "index\tfrom\ttype\tparameters"])
        if not graph:
            lines.append("构建失败，详见复杂度CSV。")
        else:
            for row in graph:
                lines.append(
                    f"{row['index']}\t{row['from']}\t{row['type']}\t{row['parameters']}"
                )
        lines.append("")
    graph_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return records


def _parse_extra(values: list[str]) -> list[tuple[str, str]]:
    result = []
    for value in values:
        if "=" not in value:
            raise ValueError("--extra-model must use LABEL=PATH")
        label, path = value.split("=", 1)
        result.append((label, path))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-json", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--graph-summary", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--extra-model", action="append", default=[])
    args = parser.parse_args()
    rows = compute_complexities(
        args.audit_json,
        args.output_csv,
        args.graph_summary,
        imgsz=args.imgsz,
        extra_models=_parse_extra(args.extra_model),
    )
    print(json.dumps({"models": len(rows), "output": args.output_csv}, ensure_ascii=False))


if __name__ == "__main__":
    main()
