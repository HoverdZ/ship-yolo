"""Inspect SDWF gate parameters and optional spatial masks."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from custom_modules.register import register_sa_dwpn_modules
from tools.sa_dwpn_utils import write_json


def default_model_path() -> str:
    return str(ROOT / "experiments" / "yolo11n_sa_dwpn_c_lite.yaml")


def tensor_stats(tensor):
    import torch

    values = tensor.detach().float().cpu().flatten()
    q = torch.quantile(values, torch.tensor([0.01, 0.05, 0.5, 0.95, 0.99]))
    return {
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "min": float(values.min()),
        "max": float(values.max()),
        "p01": float(q[0]),
        "p05": float(q[1]),
        "p50": float(q[2]),
        "p95": float(q[3]),
        "p99": float(q[4]),
        "ratio_below_0_1": float((values < 0.1).float().mean()),
        "ratio_above_0_9": float((values > 0.9).float().mean()),
    }


def module_rows(model):
    import torch

    rows = []
    for order, module in enumerate([m for m in model.model.modules() if m.__class__.__name__ == "SDWF"], start=1):
        static = torch.relu(module.static_w.detach().float().cpu())
        normalized = static / (static.sum() + module.eps)
        spatial_params = list(module.spatial_gate.parameters()) if getattr(module, "spatial_gate", None) is not None else []
        spatial_values = torch.cat([p.detach().float().cpu().flatten() for p in spatial_params]) if spatial_params else None
        rows.append(
            {
                "order": order,
                "layer_index": getattr(module, "i", None),
                "use_spatial": bool(module.use_spatial),
                "eta_value": None if getattr(module, "eta", None) is None else float(module.eta.detach().cpu()),
                "eta_requires_grad": None if getattr(module, "eta", None) is None else bool(module.eta.requires_grad),
                "gamma_value": float(module.gamma.detach().cpu()),
                "static_weights": static.tolist(),
                "normalized_static_weights": normalized.tolist(),
                "spatial_gate_parameter_count": sum(p.numel() for p in spatial_params),
                "spatial_gate_parameter_stats": None if spatial_values is None else tensor_stats(spatial_values),
            }
        )
    return rows


def run_backward_probe(model):
    import torch

    model.model.train()
    x = torch.zeros(1, 3, 640, 640)
    output = model.model(x)
    tensors = output if isinstance(output, (list, tuple)) else [output]
    loss = None
    for item in tensors:
        if isinstance(item, torch.Tensor):
            value = item.float().sum()
            loss = value if loss is None else loss + value
    if loss is None:
        return {"ran": False, "reason": "no tensor output"}
    loss.backward()
    eta_grad = {}
    for order, module in enumerate([m for m in model.model.modules() if m.__class__.__name__ == "SDWF"], start=1):
        eta = getattr(module, "eta", None)
        if eta is not None:
            eta_grad[str(order)] = eta.grad is not None
    return {"ran": True, "eta_grad": eta_grad}


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect SA-DWPN SDWF gate parameters.")
    parser.add_argument("--weights", default="", help="Optional checkpoint path.")
    parser.add_argument("--model", default=default_model_path(), help="Model YAML when loading checkpoint into architecture.")
    parser.add_argument("--data", default="", help="Optional dataset YAML for future image sampling.")
    parser.add_argument("--output", required=True, help="Output directory.")
    parser.add_argument("--num-images", type=int, default=64)
    parser.add_argument("--skip-backward", action="store_true")
    args = parser.parse_args()

    register_sa_dwpn_modules()
    from ultralytics import YOLO

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.model)
    if args.weights:
        model.load(args.weights)

    rows = module_rows(model)
    write_json(output / "gate_parameters.json", {"modules": rows})
    with (output / "gate_parameters.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "order",
            "layer_index",
            "use_spatial",
            "eta_value",
            "eta_requires_grad",
            "gamma_value",
            "static_weights",
            "normalized_static_weights",
            "spatial_gate_parameter_count",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})

    backward = {"skipped": True}
    if not args.skip_backward:
        backward = run_backward_probe(model)
    write_json(output / "eta_backward_probe.json", backward)
    write_json(
        output / "mask_statistics.json",
        {
            "status": "not_collected",
            "reason": "mask collection requires dataset image loading; parameter inspection completed.",
            "requested_num_images": args.num_images,
            "data": args.data,
        },
    )
    print(f"Saved gate inspection to {output}")


if __name__ == "__main__":
    main()
