"""Forward checks for YOLO11n-SA-DWPN-C-lite."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from custom_modules.register import register_sa_dwpn_modules


EXPECTED_SPATIAL = {21: "T3", 22: "O3"}


def default_model_path() -> str:
    source_yaml = Path("ultralytics/cfg/models/11/yolo11n-sa-dwpn-c-lite.yaml")
    repo_yaml = Path("experiments/yolo11n_sa_dwpn_c_lite.yaml")
    return str(source_yaml if source_yaml.exists() else repo_yaml)


def flatten_tensors(output: Any) -> list[torch.Tensor]:
    import torch

    if isinstance(output, torch.Tensor):
        return [output]
    if isinstance(output, (list, tuple)):
        values: list[torch.Tensor] = []
        for item in output:
            values.extend(flatten_tensors(item))
        return values
    if isinstance(output, dict):
        values = []
        for item in output.values():
            values.extend(flatten_tensors(item))
        return values
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Forward-check YOLO11n-SA-DWPN-C-lite.")
    parser.add_argument("--model", default=default_model_path(), help="Path to C-lite YAML.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    import torch

    register_sa_dwpn_modules()
    from ultralytics import YOLO

    device = torch.device(args.device)
    model = YOLO(args.model)
    model.model.to(device).eval()

    sdwf = [m for m in model.model.modules() if m.__class__.__name__ == "SDWF"]
    spatial_modules = {int(m.i): m for m in sdwf if getattr(m, "use_spatial", False)}
    if set(spatial_modules) != set(EXPECTED_SPATIAL):
        raise AssertionError(f"Expected spatial gates at {sorted(EXPECTED_SPATIAL)}, got {sorted(spatial_modules)}")

    for idx, module in spatial_modules.items():
        eta = getattr(module, "eta", None)
        if eta is None:
            raise AssertionError(f"Spatial module {idx} has no eta parameter")
        if not torch.allclose(eta.detach().cpu(), torch.zeros_like(eta.detach().cpu())):
            raise AssertionError(f"Spatial module {idx} eta is not initialized to 0")
        probes = [torch.randn(1, module.c1, 16, 16, device=device) for _ in range(module.n_inputs)]
        recalibrated = module._apply_spatial_gate(probes)
        for before, after in zip(probes, recalibrated):
            if not torch.allclose(before, after, atol=1e-6, rtol=1e-6):
                raise AssertionError(f"Spatial module {idx} is not identity-like at eta=0")

    spatial_calls = {idx: 0 for idx in spatial_modules}

    def make_spatial_hook(idx):
        def hook(_module, _inputs, _output):
            spatial_calls[idx] += 1

        return hook

    hooks = []
    for idx, module in spatial_modules.items():
        hooks.append(module.spatial_gate.register_forward_hook(make_spatial_hook(idx)))

    detect_inputs: list[tuple[int, int]] = []

    def detect_pre_hook(_module, inputs):
        features = inputs[0]
        detect_inputs.extend([tuple(t.shape[-2:]) for t in features])

    detects = [m for m in model.model.modules() if m.__class__.__name__ == "Detect"]
    if len(detects) != 1:
        raise AssertionError(f"Expected one Detect module, got {len(detects)}")
    hooks.append(detects[0].register_forward_pre_hook(detect_pre_hook))

    x = torch.zeros(1, 3, args.imgsz, args.imgsz, device=device)
    with torch.no_grad():
        output = model.model(x)

    for hook in hooks:
        hook.remove()

    tensors = flatten_tensors(output)
    if not tensors:
        raise AssertionError("Forward produced no tensor outputs")
    for tensor in tensors:
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            raise AssertionError("Forward output contains NaN or Inf")

    expected_sizes = [(args.imgsz // 8, args.imgsz // 8), (args.imgsz // 16, args.imgsz // 16), (args.imgsz // 32, args.imgsz // 32)]
    if detect_inputs != expected_sizes:
        raise AssertionError(f"Expected Detect feature sizes {expected_sizes}, got {detect_inputs}")

    if any(count == 0 for count in spatial_calls.values()):
        raise AssertionError(f"Spatial gates did not all execute: {spatial_calls}")

    state_keys = set(model.model.state_dict())
    required_fragments = [
        "static_w",
        "gamma",
        "gate.0.weight",
        "gate.2.weight",
        "spatial_gate.weight",
        "eta",
    ]
    for fragment in required_fragments:
        if not any(fragment in key for key in state_keys):
            raise AssertionError(f"Missing state_dict key containing {fragment!r}")

    optimizer_param_ids = {id(p) for group in torch.optim.SGD(model.model.parameters(), lr=0.01).param_groups for p in group["params"]}
    trainable_param_ids = {id(p) for p in model.model.parameters() if p.requires_grad}
    if trainable_param_ids - optimizer_param_ids:
        raise AssertionError("Some trainable parameters were not collected by the optimizer")

    print("Forward OK")
    print(f"Detect feature sizes: {detect_inputs}")
    print(f"Spatial gate calls: {spatial_calls}")
    print("No NaN/Inf detected")


if __name__ == "__main__":
    main()
