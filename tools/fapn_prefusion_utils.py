"""Build, transfer, audit, and training-safety helpers for FaPN-Prefusion."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from collections.abc import Iterable
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ULTRALYTICS_VERSION = "8.4.92"
SOURCE_WEIGHTS_DEFAULT = ROOT / "yolo11n.pt"
VARIANTS: dict[str, dict[str, Any]] = {
    "baseline": {
        "experiment_name": "yolo11n_fapn_prefusion_640",
        "yaml": ROOT / "experiments" / "yolo11n_fapn_prefusion.yaml",
        "init_pt": ROOT / "artifacts" / "yolo11n_fapn_prefusion_pretrained_init.pt",
        "transfer_report": ROOT / "artifacts" / "fapn_prefusion_weight_transfer.json",
        "manifest": ROOT / "artifacts" / "yolo11n_fapn_prefusion_init_manifest.json",
        "profile": ROOT / "artifacts" / "fapn_prefusion_profile.json",
        "inceptiondw": False,
    },
    "inceptiondw": {
        "experiment_name": "yolo11n_inceptiondw_fapn_prefusion_640",
        "yaml": ROOT / "experiments" / "yolo11n_inceptiondw_fapn_prefusion.yaml",
        "init_pt": ROOT / "artifacts" / "yolo11n_inceptiondw_fapn_prefusion_pretrained_init.pt",
        "transfer_report": ROOT / "artifacts" / "inceptiondw_fapn_prefusion_weight_transfer.json",
        "manifest": ROOT / "artifacts" / "yolo11n_inceptiondw_fapn_prefusion_init_manifest.json",
        "profile": ROOT / "artifacts" / "inceptiondw_fapn_prefusion_profile.json",
        "inceptiondw": True,
    },
}

PREFUSION_INDICES = (12, 13, 17, 18)
FSM_INDICES = (12, 17)
ALIGN_INDICES = (13, 18)
TOP_DOWN_C3K2_INDICES = (15, 20)
PAN_PARAMETER_INDICES = (21, 23, 24, 26)
DETECT_INDEX = 27

# Official YOLO11 source layer -> FaPN-Prefusion target layer. Parameterless
# Upsample/Concat nodes are checked structurally rather than copied.
SOURCE_TO_TARGET_LAYER_MAP = {
    **{index: index for index in range(11)},
    13: 15,
    16: 20,
    17: 21,
    19: 23,
    20: 24,
    22: 26,
    23: 27,
}
TARGET_TO_SOURCE_LAYER_MAP = {
    target: source for source, target in SOURCE_TO_TARGET_LAYER_MAP.items()
}

CRITICAL_TENSOR_KEYS = {
    "backbone_layer0": "model.0.conv.weight",
    "backbone_layer4": "model.4.cv1.conv.weight",
    "backbone_layer6": "model.6.cv1.conv.weight",
    "backbone_layer10": "model.10.cv1.conv.weight",
    "top_down_c3k2_first": "model.15.cv1.conv.weight",
    "top_down_c3k2_second": "model.20.cv1.conv.weight",
    "pan_c3k2_first": "model.23.cv1.conv.weight",
    "pan_c3k2_second": "model.26.cv1.conv.weight",
    "detect_box": "model.27.cv2.0.0.conv.weight",
    "fsm": "model.12.conv_attention.weight",
    "offset_mask": "model.13.dcn.conv_offset_mask.weight",
    "depthwise_dcn": "model.13.dcn.dcn.weight",
}


def require_ultralytics_version() -> str:
    """Reject parser/trainer versions outside the pinned experiment runtime."""

    import ultralytics

    version = getattr(ultralytics, "__version__", "unknown")
    if version != EXPECTED_ULTRALYTICS_VERSION:
        raise RuntimeError(
            f"FaPN-Prefusion requires ultralytics=={EXPECTED_ULTRALYTICS_VERSION}; "
            f"found {version}."
        )
    return version


def register_modules() -> None:
    """Register repository modules without editing site-packages."""

    require_ultralytics_version()
    from custom_modules.register import register_fapn_prefusion_modules

    register_fapn_prefusion_modules()


def variant_config(variant: str) -> dict[str, Any]:
    if variant not in VARIANTS:
        raise KeyError(f"Unknown variant {variant!r}; expected one of {sorted(VARIANTS)}.")
    return VARIANTS[variant]


def build_model(variant: str, *, seed: int = 0):
    """Build a deterministic YAML model with Ultralytics verbosity disabled."""

    register_modules()
    from ultralytics import YOLO

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return YOLO(str(variant_config(variant)["yaml"]), verbose=False)


def build_official_model(*, nc: int = 80, seed: int = 0):
    """Build the installed official YOLO11n, optionally with a fair nc override."""

    require_ultralytics_version()
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        official = YOLO("yolo11n.yaml", verbose=False)
        if nc == 80:
            return official
        cfg = deepcopy(official.model.yaml)
        cfg["nc"] = nc
        network = DetectionModel(cfg, ch=3, nc=nc, verbose=False)
        official.model = network
        official.overrides["model"] = "yolo11n.yaml"
        return official


def read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=False)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tensor(tensor: torch.Tensor) -> str:
    """Hash dtype, shape, and raw contiguous bytes of one tensor."""

    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape)).encode("ascii"))
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _relative_or_absolute(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _git_commit() -> str:
    """Read HEAD without invoking Git from formal training code."""

    git_path = ROOT / ".git"
    try:
        if git_path.is_file():
            value = git_path.read_text(encoding="utf-8").split(":", 1)[1].strip()
            git_dir = Path(value) if Path(value).is_absolute() else (ROOT / value).resolve()
        else:
            git_dir = git_path
        head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
        if not head.startswith("ref: "):
            return head
        ref_name = head.removeprefix("ref: ")
        loose = git_dir / ref_name
        if loose.is_file():
            return loose.read_text(encoding="utf-8").strip()
        packed = git_dir / "packed-refs"
        if packed.is_file():
            for line in packed.read_text(encoding="utf-8").splitlines():
                if line and not line.startswith(("#", "^")):
                    commit, name = line.split(" ", 1)
                    if name == ref_name:
                        return commit
    except (OSError, IndexError, ValueError):
        pass
    return "unknown"


def _iter_tensors(value: Any) -> Iterable[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_tensors(item)


def _shape_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return list(value.shape)
    if isinstance(value, dict):
        return {key: _shape_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_shape_tree(item) for item in value]
    return type(value).__name__


def topology_report() -> dict[str, Any]:
    """Audit both YAML files against official and validated experiment sources."""

    baseline = read_yaml(VARIANTS["baseline"]["yaml"])
    inception = read_yaml(VARIANTS["inceptiondw"]["yaml"])
    validated_inception = read_yaml(ROOT / "experiments" / "yolo11n_inceptiondw_c3k2_p23.yaml")
    official = build_official_model(nc=80).model.yaml
    layers = baseline["backbone"] + baseline["head"]
    inception_layers = inception["backbone"] + inception["head"]
    differences = [
        index
        for index, (left, right) in enumerate(zip(layers, inception_layers))
        if left != right
    ]
    expected_from = {
        11: -1,
        12: 6,
        13: [12, 11],
        14: [12, 13],
        15: -1,
        16: -1,
        17: 4,
        18: [17, 16],
        19: [17, 18],
        20: -1,
        21: -1,
        22: [-1, 15],
        23: -1,
        24: -1,
        25: [-1, 10],
        26: -1,
        27: [20, 23, 26],
    }
    checks = {
        "both_nc_equal_one": baseline["nc"] == inception["nc"] == 1,
        "baseline_backbone_is_official": baseline["backbone"] == official["backbone"],
        "inception_backbone_is_validated": inception["backbone"] == validated_inception["backbone"],
        "heads_identical": baseline["head"] == inception["head"],
        "only_inception_layers_2_4_differ": differences == [2, 4],
        "layer_count_is_28": len(layers) == len(inception_layers) == 28,
        "from_relations_exact": all(layers[index][0] == value for index, value in expected_from.items()),
        "two_nearest_upsamples": all(
            layers[index][2] == "nn.Upsample" and layers[index][3][-1] == "nearest"
            for index in (11, 16)
        ),
        "detect_p3_p4_p5": layers[27][0] == [20, 23, 26],
    }
    return {
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "variant_differences": differences,
        "expected_from": {str(key): value for key, value in expected_from.items()},
    }


def structure_report(yolo_model, variant: str) -> dict[str, Any]:
    """Inspect module placement, channels, initialization, and forbidden scope."""

    from torchvision.ops import DeformConv2d
    from ultralytics.nn.modules import C3k2, Concat, Conv, Detect

    from custom_modules.c3k2_inceptiondw import C3k2_InceptionDW
    from custom_modules.fapn import FaPNOutputConv
    from custom_modules.fapn_prefusion import (
        FaPNAlignmentOnly,
        FaPNDepthwiseModulatedDeformConv2d,
        FaPNFeatureSelectionKeep,
    )

    network = yolo_model.model
    top = network.model
    fsms = [module for module in network.modules() if isinstance(module, FaPNFeatureSelectionKeep)]
    aligns = [module for module in network.modules() if isinstance(module, FaPNAlignmentOnly)]
    dcns = [
        module
        for module in network.modules()
        if isinstance(module, FaPNDepthwiseModulatedDeformConv2d)
    ]
    deform_ops = [module for module in network.modules() if isinstance(module, DeformConv2d)]
    inception_indices = [
        index for index, layer in enumerate(top) if isinstance(layer, C3k2_InceptionDW)
    ]
    forbidden = [
        f"{name}:{module.__class__.__name__}"
        for name, module in network.named_modules()
        if isinstance(module, FaPNOutputConv)
        or any(
            token in module.__class__.__name__.lower()
            for token in ("dysample", "dcnv3", "dcnv4", "sru", "cru", "scconv")
        )
    ]
    dcn_rules = [
        {
            "high_channels": module.high_channels,
            "groups": module.dcn.groups,
            "deformable_groups": module.deformable_groups,
            "offset_channels": module.offset_channels,
            "mask_channels": module.mask_channels,
            "offset_mask_weight_zero": bool(torch.count_nonzero(module.conv_offset_mask.weight) == 0),
            "offset_mask_bias_zero": bool(torch.count_nonzero(module.conv_offset_mask.bias) == 0),
            "dcn_noncenter_zero": bool(
                torch.count_nonzero(
                    module.dcn.weight
                    - torch.nn.functional.pad(
                        module.dcn.weight[:, :, 1:2, 1:2], (1, 1, 1, 1)
                    )
                )
                == 0
            ),
            "dcn_center_is_two": bool(
                torch.all(module.dcn.weight[:, 0, 1, 1] == 2.0)
            ),
            "dcn_bias_zero": bool(torch.count_nonzero(module.dcn.bias) == 0),
        }
        for module in dcns
    ]
    checks = {
        "28_top_level_layers": len(top) == 28,
        "two_fsm": len(fsms) == 2,
        "two_alignment_only": len(aligns) == 2,
        "two_depthwise_dcnv2": len(dcns) == len(deform_ops) == 2,
        "fsm_indices": all(isinstance(top[index], FaPNFeatureSelectionKeep) for index in FSM_INDICES),
        "alignment_indices": all(isinstance(top[index], FaPNAlignmentOnly) for index in ALIGN_INDICES),
        "nearest_upsample_indices": all(
            isinstance(top[index], nn.Upsample) and top[index].mode == "nearest"
            for index in (11, 16)
        ),
        "top_down_concat_indices": all(isinstance(top[index], Concat) for index in (14, 19)),
        "top_down_c3k2_indices": all(
            isinstance(top[index], C3k2) and not isinstance(top[index], C3k2_InceptionDW)
            for index in TOP_DOWN_C3K2_INDICES
        ),
        "pan_stride_convs": all(
            isinstance(top[index], Conv) and tuple(top[index].conv.stride) == (2, 2)
            for index in (21, 24)
        ),
        "pan_concat_indices": all(isinstance(top[index], Concat) for index in (22, 25)),
        "pan_c3k2_indices": all(
            isinstance(top[index], C3k2) and not isinstance(top[index], C3k2_InceptionDW)
            for index in (23, 26)
        ),
        "detect_inputs": isinstance(top[DETECT_INDEX], Detect)
        and list(top[DETECT_INDEX].f) == [20, 23, 26],
        "detect_stride": list(map(float, network.stride)) == [8.0, 16.0, 32.0],
        "fsm_channels_preserved": [module.in_channels for module in fsms] == [128, 128]
        and all(module.in_channels == module.out_channels for module in fsms),
        "alignment_channels": [module.out_channels for module in aligns] == [256, 128],
        "controller_channels_unscaled": all(module.controller_channels == 64 for module in aligns),
        "dcn_rules": len(dcn_rules) == 2
        and all(
            item["groups"] == item["high_channels"]
            and item["deformable_groups"] == 8
            and item["offset_channels"] == 144
            and item["mask_channels"] == 72
            and item["offset_mask_weight_zero"]
            and item["offset_mask_bias_zero"]
            and item["dcn_noncenter_zero"]
            and item["dcn_center_is_two"]
            and item["dcn_bias_zero"]
            for item in dcn_rules
        ),
        "gamma_initialization": all(
            abs(float(module.gamma_s.detach()) - 0.1) < 1e-7 for module in fsms
        )
        and all(abs(float(module.gamma_a.detach()) - 0.1) < 1e-7 for module in aligns),
        "inception_scope": inception_indices == ([2, 4] if variant == "inceptiondw" else []),
        "no_forbidden_modules": not forbidden,
    }
    return {
        "variant": variant,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "counts": {
            "fsm": len(fsms),
            "alignment_only": len(aligns),
            "depthwise_modulated_dcnv2": len(dcns),
        },
        "dcn_rules": dcn_rules,
        "inceptiondw_indices": inception_indices,
        "forbidden_modules": forbidden,
    }


def forward_report(yolo_model, *, imgsz: int = 640) -> dict[str, Any]:
    """Run an inference pass and record the complete top-level shape signature."""

    network = yolo_model.model.cpu().eval()
    top_shapes: dict[int, Any] = {}
    detect_inputs: list[list[int]] = []
    hooks = []
    for index, layer in enumerate(network.model[:-1]):
        hooks.append(
            layer.register_forward_hook(
                lambda _module, _inputs, output, layer_index=index: top_shapes.__setitem__(
                    layer_index, _shape_tree(output)
                )
            )
        )

    def capture_detect(_module, inputs) -> None:
        detect_inputs.extend([list(feature.shape) for feature in inputs[0]])

    hooks.append(network.model[DETECT_INDEX].register_forward_pre_hook(capture_detect))
    generator = torch.Generator(device="cpu").manual_seed(11)
    x = torch.randn(1, 3, imgsz, imgsz, generator=generator)
    try:
        with torch.inference_mode():
            output = network(x)
    finally:
        for hook in hooks:
            hook.remove()

    tensors = list(_iter_tensors(output))
    p3_size, p4_size, p5_size = imgsz // 8, imgsz // 16, imgsz // 32
    expected = {
        4: [1, 128, p3_size, p3_size],
        6: [1, 128, p4_size, p4_size],
        10: [1, 256, p5_size, p5_size],
        11: [1, 256, p4_size, p4_size],
        12: [1, 128, p4_size, p4_size],
        13: [1, 256, p4_size, p4_size],
        14: [1, 384, p4_size, p4_size],
        15: [1, 128, p4_size, p4_size],
        16: [1, 128, p3_size, p3_size],
        17: [1, 128, p3_size, p3_size],
        18: [1, 128, p3_size, p3_size],
        19: [1, 256, p3_size, p3_size],
        20: [1, 64, p3_size, p3_size],
        23: [1, 128, p4_size, p4_size],
        26: [1, 256, p5_size, p5_size],
    }
    expected_detect = [expected[20], expected[23], expected[26]]
    checks = {
        "input_640": list(x.shape) == [1, 3, imgsz, imgsz],
        "key_node_shapes": all(top_shapes.get(index) == shape for index, shape in expected.items()),
        "detect_p3_p4_p5_shapes": detect_inputs == expected_detect,
        "outputs_finite": bool(tensors) and all(bool(torch.isfinite(item).all()) for item in tensors),
    }
    return {
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "input_shape": list(x.shape),
        "top_level_shapes": {str(key): value for key, value in sorted(top_shapes.items())},
        "detect_input_shapes": detect_inputs,
        "output_shape_tree": _shape_tree(output),
    }


def backward_report(yolo_model, *, imgsz: int = 256) -> dict[str, Any]:
    """Run a differentiable CPU pass and audit every new parameter gradient."""

    network = yolo_model.model.cpu().train()
    network.zero_grad(set_to_none=True)
    generator = torch.Generator(device="cpu").manual_seed(17)
    x = torch.randn(1, 3, imgsz, imgsz, generator=generator, requires_grad=True)
    output = network(x)
    tensors = list(_iter_tensors(output))
    if not tensors:
        raise RuntimeError("Detection model returned no tensors for backward audit.")
    loss = sum(value.float().square().mean() for value in tensors)
    loss.backward()
    prefixes = tuple(f"model.{index}." for index in PREFUSION_INDICES)
    parameters = {
        name: parameter
        for name, parameter in network.named_parameters()
        if name.startswith(prefixes)
    }
    missing = sorted(name for name, parameter in parameters.items() if parameter.grad is None)
    nonfinite = sorted(
        name
        for name, parameter in parameters.items()
        if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
    )
    zero = sorted(
        name
        for name, parameter in parameters.items()
        if parameter.grad is not None and bool(torch.count_nonzero(parameter.grad) == 0)
    )
    checks = {
        "loss_finite": bool(torch.isfinite(loss)),
        "input_gradient_finite": x.grad is not None and bool(torch.isfinite(x.grad).all()),
        "all_new_parameters_have_gradients": not missing,
        "all_new_gradients_finite": not nonfinite,
        "all_model_parameters_finite": all(bool(torch.isfinite(p).all()) for p in network.parameters()),
    }
    network.zero_grad(set_to_none=True)
    network.eval()
    return {
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "input_shape": list(x.shape),
        "loss": float(loss.detach()),
        "new_parameter_tensors": len(parameters),
        "missing_gradient_keys": missing,
        "nonfinite_gradient_keys": nonfinite,
        "zero_gradient_keys": zero,
    }


def _state_layer_index(key: str) -> int | None:
    match = re.match(r"model\.(\d+)\.", key)
    return int(match.group(1)) if match else None


def _replace_layer_index(key: str, index: int) -> str:
    return re.sub(r"^model\.\d+\.", f"model.{index}.", key, count=1)


def _breakdown(
    target_state: dict[str, torch.Tensor],
    target_parameters: dict[str, nn.Parameter],
    inherited: set[str],
    predicate,
) -> dict[str, int | float]:
    state_keys = [key for key in target_state if predicate(key)]
    parameter_keys = [key for key in target_parameters if predicate(key)]
    state_inherited = [key for key in state_keys if key in inherited]
    parameter_inherited = [key for key in parameter_keys if key in inherited]
    total_elements = sum(target_parameters[key].numel() for key in parameter_keys)
    inherited_elements = sum(target_parameters[key].numel() for key in parameter_inherited)
    return {
        "total_state_tensors": len(state_keys),
        "inherited_state_tensors": len(state_inherited),
        "state_tensor_ratio": len(state_inherited) / len(state_keys) if state_keys else 1.0,
        "total_parameter_tensors": len(parameter_keys),
        "inherited_parameter_tensors": len(parameter_inherited),
        "parameter_tensor_ratio": (
            len(parameter_inherited) / len(parameter_keys) if parameter_keys else 1.0
        ),
        "total_parameter_elements": total_elements,
        "inherited_parameter_elements": inherited_elements,
        "parameter_element_ratio": inherited_elements / total_elements if total_elements else 1.0,
    }


def semantic_weight_transfer(
    target_yolo,
    weights: str | Path = SOURCE_WEIGHTS_DEFAULT,
    *,
    variant: str,
    apply: bool = False,
) -> dict[str, Any]:
    """Transfer official weights through explicit semantic layer mapping."""

    require_ultralytics_version()
    from ultralytics import YOLO

    weights = Path(weights).expanduser().resolve() if Path(weights).expanduser().exists() else str(weights)
    source_yolo = YOLO(str(weights), verbose=False)
    source_state = source_yolo.model.float().state_dict()
    target_network = target_yolo.model
    target_state = target_network.state_dict()
    target_parameters = dict(target_network.named_parameters())
    inherited_values: dict[str, torch.Tensor] = {}
    target_to_source: dict[str, str] = {}
    source_to_target: dict[str, str] = {}
    shape_mismatches: list[dict[str, Any]] = []
    unmapped_target: list[str] = []

    for target_key, target_value in target_state.items():
        target_index = _state_layer_index(target_key)
        source_index = TARGET_TO_SOURCE_LAYER_MAP.get(target_index) if target_index is not None else None
        if source_index is None:
            unmapped_target.append(target_key)
            continue
        source_key = _replace_layer_index(target_key, source_index)
        source_value = source_state.get(source_key)
        if source_value is None:
            unmapped_target.append(target_key)
            continue
        if tuple(source_value.shape) != tuple(target_value.shape):
            shape_mismatches.append(
                {
                    "source_key": source_key,
                    "target_key": target_key,
                    "source_shape": list(source_value.shape),
                    "target_shape": list(target_value.shape),
                }
            )
            continue
        inherited_values[target_key] = source_value
        target_to_source[target_key] = source_key
        source_to_target[source_key] = target_key

    if apply:
        load_result = target_network.load_state_dict(inherited_values, strict=False)
        unexpected_after_load = list(load_result.unexpected_keys)
    else:
        unexpected_after_load = []

    inherited = set(inherited_values)
    parameter_keys = set(target_parameters)
    inherited_parameter_keys = inherited & parameter_keys
    target_parameter_elements = sum(value.numel() for value in target_parameters.values())
    inherited_parameter_elements = sum(target_parameters[key].numel() for key in inherited_parameter_keys)

    index_predicate = lambda indices: (
        lambda key: (_state_layer_index(key) in indices)
    )
    breakdown = {
        "backbone": _breakdown(target_state, target_parameters, inherited, index_predicate(set(range(11)))),
        "top_down_c3k2_first": _breakdown(
            target_state, target_parameters, inherited, index_predicate({15})
        ),
        "top_down_c3k2_second": _breakdown(
            target_state, target_parameters, inherited, index_predicate({20})
        ),
        "pan": _breakdown(
            target_state, target_parameters, inherited, index_predicate(set(PAN_PARAMETER_INDICES))
        ),
        "detect_box": _breakdown(
            target_state,
            target_parameters,
            inherited,
            lambda key: key.startswith("model.27.cv2."),
        ),
        "detect_dfl": _breakdown(
            target_state,
            target_parameters,
            inherited,
            lambda key: key.startswith("model.27.dfl."),
        ),
        "detect_box_dfl": _breakdown(
            target_state,
            target_parameters,
            inherited,
            lambda key: key.startswith("model.27.cv2.") or key.startswith("model.27.dfl."),
        ),
        "detect_classification": _breakdown(
            target_state,
            target_parameters,
            inherited,
            lambda key: key.startswith("model.27.cv3."),
        ),
        "prefusion_new": _breakdown(
            target_state, target_parameters, inherited, index_predicate(set(PREFUSION_INDICES))
        ),
    }
    new_prefusion = sorted(
        key for key in target_parameters if _state_layer_index(key) in PREFUSION_INDICES
    )
    inception_random = sorted(
        key
        for key in target_parameters
        if variant == "inceptiondw"
        and _state_layer_index(key) in {2, 4}
        and key not in inherited
    )
    untouched_shape_pairs = {
        "top_down_c3k2_first": (13, 15),
        "top_down_c3k2_second": (16, 20),
        "pan_conv_first": (17, 21),
        "pan_c3k2_first": (19, 23),
        "pan_conv_second": (20, 24),
        "pan_c3k2_second": (22, 26),
    }
    shape_audit = {}
    for label, (source_index, target_index) in untouched_shape_pairs.items():
        target_shapes = {
            key.removeprefix(f"model.{target_index}."): list(value.shape)
            for key, value in target_state.items()
            if key.startswith(f"model.{target_index}.")
        }
        source_shapes = {
            key.removeprefix(f"model.{source_index}."): list(value.shape)
            for key, value in source_state.items()
            if key.startswith(f"model.{source_index}.")
        }
        shape_audit[label] = {
            "all_shapes_equal": target_shapes == source_shapes,
            "source_layer": source_index,
            "target_layer": target_index,
            "tensor_count": len(target_shapes),
        }

    unmatched_target = sorted(key for key in target_state if key not in inherited)
    unmatched_source = sorted(key for key in source_state if key not in source_to_target)
    report = {
        "variant": variant,
        "source_weights": str(weights),
        "mapping_strategy": "explicit semantic layer and exact subkey/shape mapping",
        "source_to_target_layer_map": {
            str(key): value for key, value in SOURCE_TO_TARGET_LAYER_MAP.items()
        },
        "total_state_tensors": len(target_state),
        "total_parameter_tensors": len(target_parameters),
        "inherited_state_tensors": len(inherited),
        "inherited_parameter_tensors": len(inherited_parameter_keys),
        "target_parameter_elements": target_parameter_elements,
        "inherited_parameter_elements": inherited_parameter_elements,
        "randomly_initialized_parameter_elements": (
            target_parameter_elements - inherited_parameter_elements
        ),
        "parameter_element_inheritance_ratio": (
            inherited_parameter_elements / target_parameter_elements
        ),
        "breakdown": breakdown,
        "new_prefusion_random_parameter_keys": new_prefusion,
        "inceptiondw_random_parameter_keys": inception_random,
        "shape_mismatches": shape_mismatches,
        "unmapped_target_keys": sorted(unmapped_target),
        "unmatched_target_keys": unmatched_target,
        "unmatched_source_keys": unmatched_source,
        "source_key_to_target_key": source_to_target,
        "target_key_to_source_key": target_to_source,
        "unchanged_module_shape_audit": shape_audit,
        "unexpected_after_load": unexpected_after_load,
        "applied": apply,
    }
    strict_checks = {
        "prefusion_is_random": breakdown["prefusion_new"]["inherited_parameter_tensors"] == 0,
        "top_down_fully_inherited": all(
            breakdown[name]["parameter_element_ratio"] == 1.0
            for name in ("top_down_c3k2_first", "top_down_c3k2_second")
        ),
        "pan_fully_inherited": breakdown["pan"]["parameter_element_ratio"] == 1.0,
        "unchanged_shapes_equal": all(
            item["all_shapes_equal"] for item in shape_audit.values()
        ),
        "no_unexpected_after_load": not unexpected_after_load,
    }
    if variant == "baseline":
        strict_checks["backbone_fully_inherited"] = breakdown["backbone"]["parameter_element_ratio"] == 1.0
    else:
        untouched_unmatched = [
            key
            for key in unmatched_target
            if (_state_layer_index(key) in set(range(11)) - {2, 4})
        ]
        strict_checks["unchanged_backbone_fully_inherited"] = not untouched_unmatched
        strict_checks["inception_new_parameters_present"] = bool(inception_random)
    report["strict_checks"] = strict_checks
    report["all_strict_checks_passed"] = all(strict_checks.values())
    return report


def _checkpoint_payload(network: nn.Module, *, source_weights: str | Path) -> dict[str, Any]:
    """Create a non-resume Ultralytics checkpoint containing the real init model."""

    model = deepcopy(network).cpu().float().eval()
    model.args = {
        "task": "detect",
        "imgsz": 640,
        "data": None,
        "single_cls": False,
    }
    model.task = "detect"
    return {
        "date": datetime.now(timezone.utc).isoformat(),
        "version": EXPECTED_ULTRALYTICS_VERSION,
        "license": "AGPL-3.0 (https://ultralytics.com/license)",
        "docs": "https://docs.ultralytics.com",
        "epoch": -1,
        "best_fitness": None,
        "model": model,
        "ema": None,
        "updates": None,
        "optimizer": None,
        "train_args": dict(model.args),
        "train_metrics": {},
        "train_results": {},
        "prefusion_initialization": {
            "source_weights": str(source_weights),
            "git_commit": _git_commit(),
        },
    }


def load_init_model(init_pt: str | Path):
    """Load an initialization checkpoint after local module registration."""

    register_modules()
    from ultralytics import YOLO

    return YOLO(str(Path(init_pt).expanduser().resolve()), verbose=False)


def _artifact_paths(variant: str, output_dir: str | Path | None) -> dict[str, Path]:
    config = variant_config(variant)
    if output_dir is None:
        return {
            key: Path(config[key])
            for key in ("init_pt", "transfer_report", "manifest", "profile")
        }
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return {
        "init_pt": root / Path(config["init_pt"]).name,
        "transfer_report": root / Path(config["transfer_report"]).name,
        "manifest": root / Path(config["manifest"]).name,
        "profile": root / Path(config["profile"]).name,
    }


def prepare_initialization(
    variant: str,
    *,
    weights: str | Path = SOURCE_WEIGHTS_DEFAULT,
    output_dir: str | Path | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Transfer official weights, save the real init checkpoint, and manifest it."""

    config = variant_config(variant)
    paths = _artifact_paths(variant, output_dir)
    weights_path = Path(weights).expanduser().resolve()
    if not weights_path.is_file():
        raise FileNotFoundError(f"Official YOLO11n weights not found: {weights_path}")

    target = build_model(variant, seed=seed)
    transfer = semantic_weight_transfer(target, weights_path, variant=variant, apply=True)
    if not transfer["all_strict_checks_passed"]:
        raise RuntimeError(f"Strict weight-transfer audit failed: {transfer['strict_checks']}")
    write_json(paths["transfer_report"], transfer)

    paths["init_pt"].parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        _checkpoint_payload(target.model, source_weights=weights_path),
        paths["init_pt"],
    )
    reloaded = load_init_model(paths["init_pt"])
    reloaded_state = reloaded.model.state_dict()
    original_state = target.model.state_dict()
    if set(reloaded_state) != set(original_state) or any(
        not torch.equal(reloaded_state[key].cpu(), original_state[key].cpu())
        for key in original_state
    ):
        raise RuntimeError("Reloaded initialization checkpoint differs from the transferred model.")
    missing_critical = [key for key in CRITICAL_TENSOR_KEYS.values() if key not in reloaded_state]
    if missing_critical:
        raise KeyError(f"Critical manifest tensors are missing: {missing_critical}")

    manifest = {
        "schema_version": 1,
        "variant": variant,
        "experiment_name": config["experiment_name"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "init_pt_absolute_path": str(paths["init_pt"].resolve()),
        "init_pt_relative_path": _relative_or_absolute(paths["init_pt"]),
        "model_yaml_absolute_path": str(Path(config["yaml"]).resolve()),
        "model_yaml_relative_path": _relative_or_absolute(Path(config["yaml"])),
        "model_parameter_count": sum(parameter.numel() for parameter in reloaded.model.parameters()),
        "yaml_sha256": sha256_file(config["yaml"]),
        "git_commit": _git_commit(),
        "ultralytics_version": require_ultralytics_version(),
        "torch_version": torch.__version__,
        "source_weights_absolute_path": str(weights_path),
        "source_weights_sha256": sha256_file(weights_path),
        "init_pt_sha256": sha256_file(paths["init_pt"]),
        "critical_tensor_hashes": {
            label: {
                "key": key,
                "shape": list(reloaded_state[key].shape),
                "dtype": str(reloaded_state[key].dtype),
                "sha256": sha256_tensor(reloaded_state[key]),
            }
            for label, key in CRITICAL_TENSOR_KEYS.items()
        },
        "weight_transfer_report": _relative_or_absolute(paths["transfer_report"]),
    }
    write_json(paths["manifest"], manifest)
    validation = validate_init_manifest(paths["init_pt"], paths["manifest"], model=reloaded)
    if not validation["all_checks_passed"]:
        raise RuntimeError(f"Initialization manifest validation failed: {validation}")
    return {
        "variant": variant,
        "init_pt": str(paths["init_pt"]),
        "transfer_report": str(paths["transfer_report"]),
        "manifest": str(paths["manifest"]),
        "weight_transfer": transfer,
        "manifest_payload": manifest,
        "validation": validation,
    }


def validate_init_manifest(
    init_pt: str | Path,
    manifest_path: str | Path,
    *,
    model=None,
) -> dict[str, Any]:
    """Validate the checkpoint file, model structure, and critical hashes."""

    init_pt = Path(init_pt).expanduser().resolve()
    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    loaded = model or load_init_model(init_pt)
    state = loaded.model.state_dict()
    tensor_checks = {
        label: item["key"] in state and sha256_tensor(state[item["key"]]) == item["sha256"]
        for label, item in manifest["critical_tensor_hashes"].items()
    }
    yaml_path = Path(variant_config(manifest["variant"])["yaml"])
    checks = {
        "init_pt_exists": init_pt.is_file(),
        "init_pt_sha256": sha256_file(init_pt) == manifest["init_pt_sha256"],
        "yaml_sha256": sha256_file(yaml_path) == manifest["yaml_sha256"],
        "parameter_count": sum(p.numel() for p in loaded.model.parameters())
        == manifest["model_parameter_count"],
        "critical_tensor_hashes": all(tensor_checks.values()),
        "detect_stride": list(map(float, loaded.model.stride)) == [8.0, 16.0, 32.0],
        "all_finite": all(
            bool(torch.isfinite(value).all())
            for value in state.values()
            if value.is_floating_point()
        ),
    }
    return {
        "checks": checks,
        "tensor_checks": tensor_checks,
        "all_checks_passed": all(checks.values()),
    }


def _unwrap_training_model(model: nn.Module) -> nn.Module:
    seen: set[int] = set()
    while id(model) not in seen:
        seen.add(id(model))
        if hasattr(model, "module"):
            model = model.module
            continue
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
            continue
        break
    return model


def verify_prefusion_trainer_initialization(
    trainer,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Abort before epoch 1 if Trainer did not receive the manifested init.pt."""

    manifest_path = manifest_path or getattr(trainer, "prefusion_manifest_path", None)
    manifest_path = manifest_path or os.environ.get("FAPN_PREFUSION_MANIFEST")
    if not manifest_path:
        raise RuntimeError("FaPN-Prefusion trainer verifier requires a manifest path.")
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    args_model = getattr(getattr(trainer, "args", None), "model", None)
    args_path = Path(str(args_model)).expanduser().resolve() if args_model else None
    network = _unwrap_training_model(trainer.model)
    state = network.state_dict()
    tensor_checks = {
        label: item["key"] in state and sha256_tensor(state[item["key"]]) == item["sha256"]
        for label, item in manifest["critical_tensor_hashes"].items()
    }
    checks = {
        "args_model_is_pt": args_path is not None and args_path.suffix.lower() == ".pt",
        "args_model_exists": args_path is not None and args_path.is_file(),
        "args_model_is_manifested_init": args_path is not None
        and args_path.is_file()
        and sha256_file(args_path) == manifest["init_pt_sha256"],
        "parameter_count": sum(parameter.numel() for parameter in network.parameters())
        == manifest["model_parameter_count"],
        "critical_tensor_hashes": all(tensor_checks.values()),
        "detect_stride": list(map(float, network.stride)) == [8.0, 16.0, 32.0],
        "all_finite": all(
            bool(torch.isfinite(value).all())
            for value in state.values()
            if value.is_floating_point()
        ),
    }
    audit = {
        "event": "on_pretrain_routine_end",
        "args_model": str(args_model),
        "checks": checks,
        "tensor_checks": tensor_checks,
        "all_checks_passed": all(checks.values()),
    }
    if not audit["all_checks_passed"]:
        raise RuntimeError(
            "FaPN-Prefusion Trainer initialization audit failed before epoch 1:\n"
            + json.dumps(audit, indent=2, ensure_ascii=False)
        )
    trainer.prefusion_initialization_audit = audit
    print("FaPN-Prefusion Trainer initialization audit: PASSED")
    return audit


def install_safe_prefusion_flops(profile_json: str | Path):
    """Temporarily replace only ``torch_utils.get_flops`` with a saved value."""

    profile_path = Path(profile_json).expanduser().resolve()
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    gflops = float(payload["gflops"])
    if not (gflops > 0.0):
        raise ValueError(f"Invalid GFLOPs in {profile_path}: {gflops}")
    import ultralytics.utils.torch_utils as torch_utils

    original = torch_utils.get_flops

    def saved_prefusion_flops(_model, imgsz=640):
        if _model is not None:
            actual_parameters = sum(parameter.numel() for parameter in _model.parameters())
            if actual_parameters != int(payload["parameters"]):
                raise RuntimeError(
                    f"Prefusion profile/model mismatch: profile has {payload['parameters']} "
                    f"parameters, model has {actual_parameters}."
                )
        requested = imgsz if isinstance(imgsz, (list, tuple)) else [imgsz, imgsz]
        base = payload.get("imgsz", 640)
        return gflops * float(requested[0]) / base * float(requested[1]) / base

    saved_prefusion_flops._prefusion_profile = str(profile_path)
    torch_utils.get_flops = saved_prefusion_flops

    def restore() -> None:
        if torch_utils.get_flops is saved_prefusion_flops:
            torch_utils.get_flops = original

    return restore


def prepare_formal_run_directory(project: str | Path, name: str) -> tuple[Path, Path | None]:
    """Refuse checkpoint overwrite and preserve incomplete runs as backups."""

    project = Path(project).expanduser().resolve()
    run_dir = project / name
    if not run_dir.exists():
        return run_dir, None
    if not run_dir.is_dir():
        raise FileExistsError(f"Run path exists and is not a directory: {run_dir}")
    last_pt = run_dir / "weights" / "last.pt"
    if last_pt.is_file():
        raise FileExistsError(
            f"Refusing to overwrite resumable run {run_dir}. Use official resume: "
            f"model = YOLO(r'{last_pt}'); model.train(resume=True)"
        )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = run_dir.with_name(f"{run_dir.name}_crashed_backup_{timestamp}")
    counter = 1
    while backup.exists():
        backup = run_dir.with_name(f"{run_dir.name}_crashed_backup_{timestamp}_{counter}")
        counter += 1
    project.mkdir(parents=True, exist_ok=True)
    shutil.move(str(run_dir), str(backup))
    return run_dir, backup


def compare_parameter_shapes_with_official(yolo_model) -> dict[str, Any]:
    """Check unchanged top-down/PAN parameter shapes against official YOLO11n."""

    official = build_official_model(nc=1)
    target_state = yolo_model.model.state_dict()
    source_state = official.model.state_dict()
    pairs = {
        "top_down_first": (13, 15),
        "top_down_second": (16, 20),
        "pan_conv_first": (17, 21),
        "pan_c3k2_first": (19, 23),
        "pan_conv_second": (20, 24),
        "pan_c3k2_second": (22, 26),
    }
    checks = {}
    for name, (source_index, target_index) in pairs.items():
        source = {
            key.removeprefix(f"model.{source_index}."): tuple(value.shape)
            for key, value in source_state.items()
            if key.startswith(f"model.{source_index}.")
        }
        target = {
            key.removeprefix(f"model.{target_index}."): tuple(value.shape)
            for key, value in target_state.items()
            if key.startswith(f"model.{target_index}.")
        }
        checks[name] = source == target
    return {"checks": checks, "all_checks_passed": all(checks.values())}
