"""Shared helpers for the YOLO11n-SCConv-C3k2-Full experiment."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
MODEL_YAML = ROOT / "experiments" / "yolo11n_scconv_c3k2_full.yaml"
BASELINE_MODEL = "yolo11n.yaml"
DEFAULT_WEIGHTS = "yolo11n.pt"
BACKBONE_SC_LAYERS = {2, 4, 6, 8}
BACKBONE_DOWNSAMPLE_LAYERS = {0, 1, 3, 5, 7}
HEAD_LAYERS = set(range(11, 24))


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    """Write a JSON report using stable UTF-8 formatting."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _inner_model(model: Any) -> Any:
    return getattr(model, "model", model)


def model_statistics(model: Any, imgsz: int = 640) -> dict[str, Any]:
    """Return comparable Ultralytics model size statistics."""

    from ultralytics.utils.torch_utils import get_flops

    inner = _inner_model(model)
    parameters = sum(parameter.numel() for parameter in inner.parameters())
    trainable = sum(
        parameter.numel() for parameter in inner.parameters() if parameter.requires_grad
    )
    layers = sum(1 for module in inner.modules() if not module._modules)
    gflops = float(get_flops(inner, imgsz=imgsz) or 0.0)
    return {
        "parameters": parameters,
        "trainable_parameters": trainable,
        "gflops": gflops,
        "layers": layers,
        "fp32_parameter_size_mib": parameters * 4 / (1024**2),
    }


def _layer_index(key: str) -> int | None:
    parts = key.split(".")
    if len(parts) >= 2 and parts[0] == "model" and parts[1].isdigit():
        return int(parts[1])
    return None


def _key_group(key: str) -> str:
    layer = _layer_index(key)
    if layer is None:
        return "other"
    if layer in BACKBONE_SC_LAYERS:
        return f"backbone.scconv_layer_{layer}"
    if layer <= 10:
        return f"backbone.layer_{layer}"
    if layer == 23:
        return "detect.layer_23"
    return f"neck_head.layer_{layer}"


def _summarize_groups(keys: list[str], state: dict[str, Any]) -> dict[str, dict[str, int]]:
    tensor_counts: Counter[str] = Counter()
    element_counts: Counter[str] = Counter()
    for key in keys:
        group = _key_group(key)
        tensor_counts[group] += 1
        element_counts[group] += int(state[key].numel())
    return {
        group: {
            "tensors": tensor_counts[group],
            "elements": element_counts[group],
        }
        for group in sorted(tensor_counts)
    }


def _section_report(
    target_state: dict[str, Any],
    matched_keys: set[str],
    selector: Callable[[str], bool],
) -> dict[str, Any]:
    keys = [key for key in target_state if selector(key)]
    matched = [key for key in keys if key in matched_keys]
    elements = sum(int(target_state[key].numel()) for key in keys)
    matched_elements = sum(int(target_state[key].numel()) for key in matched)
    return {
        "target_tensors": len(keys),
        "inherited_tensors": len(matched),
        "tensor_ratio": len(matched) / len(keys) if keys else 1.0,
        "target_elements": elements,
        "inherited_elements": matched_elements,
        "element_ratio": matched_elements / elements if elements else 1.0,
    }


def inspect_weight_transfer(
    target_model: Any,
    weights: str | Path = DEFAULT_WEIGHTS,
    apply: bool = True,
) -> dict[str, Any]:
    """Inspect and optionally apply name-and-shape-compatible pretrained weights."""

    from ultralytics import YOLO

    source = YOLO(str(weights))
    source_state = source.model.state_dict()
    target_state = target_model.model.state_dict()

    matched = {
        key
        for key, value in target_state.items()
        if key in source_state and tuple(value.shape) == tuple(source_state[key].shape)
    }
    shape_mismatches = [
        {
            "key": key,
            "source_shape": list(source_state[key].shape),
            "target_shape": list(value.shape),
        }
        for key, value in target_state.items()
        if key in source_state and tuple(value.shape) != tuple(source_state[key].shape)
    ]
    missing = [key for key in target_state if key not in matched]
    source_only = [key for key in source_state if key not in target_state]

    target_named_parameters = dict(target_model.model.named_parameters())
    matched_parameter_names = set(target_named_parameters) & matched
    target_parameter_elements = sum(
        int(parameter.numel()) for parameter in target_named_parameters.values()
    )
    inherited_parameter_elements = sum(
        int(target_named_parameters[key].numel()) for key in matched_parameter_names
    )
    target_state_elements = sum(int(value.numel()) for value in target_state.values())
    inherited_state_elements = sum(int(target_state[key].numel()) for key in matched)

    outer_prefixes = tuple(
        f"model.{layer}.{name}."
        for layer in sorted(BACKBONE_SC_LAYERS)
        for name in ("cv1", "cv2")
    )
    critical_sections = {
        "backbone_downsample_conv": _section_report(
            target_state,
            matched,
            lambda key: _layer_index(key) in BACKBONE_DOWNSAMPLE_LAYERS,
        ),
        "backbone_c3k2_outer_1x1": _section_report(
            target_state,
            matched,
            lambda key: key.startswith(outer_prefixes),
        ),
        "sppf": _section_report(
            target_state, matched, lambda key: _layer_index(key) == 9
        ),
        "c2psa": _section_report(
            target_state, matched, lambda key: _layer_index(key) == 10
        ),
        "neck": _section_report(
            target_state,
            matched,
            lambda key: (_layer_index(key) or -1) in set(range(11, 23)),
        ),
        "detect": _section_report(
            target_state, matched, lambda key: _layer_index(key) == 23
        ),
    }

    report = {
        "weights": str(weights),
        "target_yaml": str(MODEL_YAML.relative_to(ROOT)),
        "target_state_tensors": len(target_state),
        "inherited_state_tensors": len(matched),
        "state_tensor_ratio": len(matched) / len(target_state),
        "target_state_elements": target_state_elements,
        "inherited_state_elements": inherited_state_elements,
        "state_element_ratio": inherited_state_elements / target_state_elements,
        "target_parameter_tensors": len(target_named_parameters),
        "inherited_parameter_tensors": len(matched_parameter_names),
        "target_parameter_elements": target_parameter_elements,
        "inherited_parameter_elements": inherited_parameter_elements,
        "parameter_element_ratio": inherited_parameter_elements
        / target_parameter_elements,
        "missing_keys": missing,
        "unexpected_keys": source_only,
        "shape_mismatches": shape_mismatches,
        "unmatched_target_by_module": _summarize_groups(missing, target_state),
        "source_only_by_module": _summarize_groups(source_only, source_state),
        "critical_sections": critical_sections,
        "neck_abnormal_unmatched": critical_sections["neck"]["tensor_ratio"] < 1.0,
        "detect_abnormal_unmatched": critical_sections["detect"]["tensor_ratio"] < 1.0,
    }

    if apply:
        target_model.load(str(weights))
        report["applied"] = True
    else:
        report["applied"] = False
    return report
