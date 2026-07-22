"""Profile FaPN-Prefusion without Ultralytics' deepcopy/get_flops path."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torchvision.ops import DeformConv2d

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.fapn_prefusion_utils import (
    VARIANTS,
    build_model,
    build_official_model,
    variant_config,
    write_json,
)


def _count_deform_conv(module: DeformConv2d, _inputs, output: torch.Tensor) -> None:
    """Count DCNv2 convolution MACs using its declared channel groups."""

    kernel_macs = (
        module.kernel_size[0]
        * module.kernel_size[1]
        * module.in_channels
        // module.groups
    )
    module.total_ops += torch.DoubleTensor([output.numel() * kernel_macs])


def _profile_network(network, *, imgsz: int = 640, probe_imgsz: int = 64) -> dict[str, Any]:
    """Run THOP directly (no deepcopy), then scale spatial MACs to imgsz."""

    from thop import profile

    if imgsz % probe_imgsz:
        raise ValueError("imgsz must be divisible by probe_imgsz for audited spatial scaling.")
    network = network.cpu().eval()
    sample = torch.zeros(1, 3, probe_imgsz, probe_imgsz)
    try:
        with torch.inference_mode():
            macs, _thop_parameters, layer_info = profile(
                network,
                inputs=(sample,),
                custom_ops={DeformConv2d: _count_deform_conv},
                verbose=False,
                ret_layer_info=True,
            )
    finally:
        # THOP 2.0.20 removes these buffers only from modules with registered
        # counting hooks. Composite/unhandled modules (including Ultralytics'
        # shared default activation instance) otherwise leak them into models
        # built later in the same process and corrupt state-key audits.
        for module in network.modules():
            module._buffers.pop("total_ops", None)
            module._buffers.pop("total_params", None)
    spatial_scale = (imgsz / probe_imgsz) ** 2
    scaled_macs = float(macs * spatial_scale)
    root_info = layer_info["model"]
    children = root_info[2]
    per_top_level = []
    for index, layer in enumerate(network.model):
        operations, _parameters, _children = children[str(index)]
        per_top_level.append(
            {
                "index": index,
                "from": layer.f,
                "type": layer.__class__.__name__,
                "parameters": sum(parameter.numel() for parameter in layer.parameters()),
                "gflops": float(operations * spatial_scale * 2 / 1e9),
            }
        )
    parameters = sum(parameter.numel() for parameter in network.parameters())
    trainable = sum(
        parameter.numel() for parameter in network.parameters() if parameter.requires_grad
    )
    return {
        "imgsz": imgsz,
        "profile_probe_imgsz": probe_imgsz,
        "spatial_scale_factor": spatial_scale,
        "parameters": parameters,
        "trainable_parameters": trainable,
        "macs": scaled_macs,
        "gflops": scaled_macs * 2 / 1e9,
        "per_top_level_module": per_top_level,
        "method": {
            "deepcopy_used": False,
            "thop_direct_forward": True,
            "deform_conv_custom_op": (
                "output_elements * kernel_h * kernel_w * in_channels / groups"
            ),
            "gflops_convention": "2 FLOPs per MAC, matching Ultralytics model_info",
            "dcn_sampling_note": (
                "Reports convolution-equivalent DCNv2 MACs; interpolation/addressing overhead "
                "is not included, matching common THOP conventions."
            ),
        },
    }


def profile_variant(variant: str, *, imgsz: int = 640) -> dict[str, Any]:
    """Profile one variant and compare it with fair nc=1 and stock nc=80 YOLO11n."""

    custom = _profile_network(build_model(variant).model, imgsz=imgsz)
    official_nc1 = _profile_network(build_official_model(nc=1).model, imgsz=imgsz)
    official_nc80 = _profile_network(build_official_model(nc=80).model, imgsz=imgsz)
    prefusion_indices = {12, 13, 17, 18}
    custom["variant"] = variant
    custom["experiment_name"] = variant_config(variant)["experiment_name"]
    custom["prefusion_parameters"] = sum(
        item["parameters"]
        for item in custom["per_top_level_module"]
        if item["index"] in prefusion_indices
    )
    custom["prefusion_gflops"] = sum(
        item["gflops"]
        for item in custom["per_top_level_module"]
        if item["index"] in prefusion_indices
    )
    custom["comparison"] = {
        "official_yolo11n_nc1": {
            "parameters": official_nc1["parameters"],
            "gflops": official_nc1["gflops"],
            "parameter_delta": custom["parameters"] - official_nc1["parameters"],
            "gflops_delta": custom["gflops"] - official_nc1["gflops"],
        },
        "official_yolo11n_nc80": {
            "parameters": official_nc80["parameters"],
            "gflops": official_nc80["gflops"],
            "parameter_delta": custom["parameters"] - official_nc80["parameters"],
            "gflops_delta": custom["gflops"] - official_nc80["gflops"],
        },
    }
    return custom


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["baseline", "inceptiondw", "all"], default="all")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    variants = VARIANTS if args.variant == "all" else (args.variant,)
    results = {}
    for variant in variants:
        payload = profile_variant(variant, imgsz=args.imgsz)
        output = (
            args.output_dir / Path(variant_config(variant)["profile"]).name
            if args.output_dir
            else Path(variant_config(variant)["profile"])
        )
        write_json(output, payload)
        results[variant] = payload
        print(
            f"{variant}: parameters={payload['parameters']} "
            f"GFLOPs={payload['gflops']:.6f} -> {output}"
        )
    print(
        json.dumps(
            {
                key: {
                    "parameters": value["parameters"],
                    "gflops": value["gflops"],
                    "profile": str(
                        args.output_dir / Path(variant_config(key)["profile"]).name
                        if args.output_dir
                        else variant_config(key)["profile"]
                    ),
                }
                for key, value in results.items()
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
