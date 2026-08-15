"""RTMDet-Tiny adapter for cross-architecture transfer experiments.

The detector, assignment strategy, losses, head, and standard CSPNeXt blocks
remain those of the official MMDetection RTMDet implementation.  This module
only supplies the architecture-specific adapters required to transfer the
paper's final four components:

* VGUP is applied to RGB images before the CSPNeXt stem;
* the second spatial operator in shallow P2/P3 CSPNeXt bottlenecks is replaced
  by InceptionDW while the first 3x3 convolution and residual path are kept;
* the feature pyramid is shifted from P3/P4/P5 to P2/P3/P4 and its two
  top-down nearest-neighbour operators are replaced by DySample;
* one bounded CA-SCAM is placed on each P2/P3/P4 neck output immediately
  before the unchanged RTMDet head.

The OpenMMLab imports are optional at module import time so the repository's
pure-PyTorch components can still be linted on a lightweight CPU workstation.
Constructing the RTMDet backbone, neck, or sync hook requires MMDetection
3.3.0, MMEngine, and MMCV.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor, nn

from custom_modules.dysample import DySample
from custom_modules.inceptiondw import InceptionDWConv2d
from custom_modules.vgup import VGUPPreprocessor

_OPENMMLAB_IMPORT_ERROR: ModuleNotFoundError | None = None

try:
    from mmcv.cnn import ConvModule, build_activation_layer, build_norm_layer
    from mmengine.hooks import Hook
    from mmdet.models.backbones.cspnext import CSPNeXt
    from mmdet.models.necks.cspnext_pafpn import CSPNeXtPAFPN
    from mmdet.registry import HOOKS, MODELS
except ModuleNotFoundError as error:  # pragma: no cover - local fallback only
    _OPENMMLAB_IMPORT_ERROR = error

    class _UnavailableOpenMMLabBase(nn.Module):
        """Placeholder that raises only when an OpenMMLab class is built."""

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            super().__init__()
            raise ModuleNotFoundError(
                "RTMDet adapters require mmdet, mmengine, and full mmcv."
            ) from _OPENMMLAB_IMPORT_ERROR

    class _NoOpRegistry:
        def register_module(self, *_args: Any, **_kwargs: Any):
            def decorator(cls):
                return cls

            return decorator

    class Hook:  # type: ignore[no-redef]
        pass

    CSPNeXt = _UnavailableOpenMMLabBase  # type: ignore[misc,assignment]
    CSPNeXtPAFPN = _UnavailableOpenMMLabBase  # type: ignore[misc,assignment]
    HOOKS = _NoOpRegistry()  # type: ignore[assignment]
    MODELS = _NoOpRegistry()  # type: ignore[assignment]


def _require_openmmlab() -> None:
    if _OPENMMLAB_IMPORT_ERROR is not None:
        raise ModuleNotFoundError(
            "RTMDet adapters require mmdet, mmengine, and full mmcv."
        ) from _OPENMMLAB_IMPORT_ERROR


class RTMDetConvBNAct(nn.Module):
    """Single-GPU BN variant of the Conv-BN-SiLU unit used by CA-SCAM."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        *,
        use_bn: bool = True,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=padding,
            # Both the Ultralytics Conv path and SCAM's official no-BN
            # projection use bias=False.
            bias=False,
        )
        self.bn = (
            nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
            if use_bn
            else nn.Identity()
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.conv(x)))


class RTMDetCASCAM(nn.Module):
    """Framework-neutral bounded CA-SCAM with the final-model equations."""

    def __init__(self, in_channels: int, max_delta: float = 0.1) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}.")
        if not 0.0 < max_delta <= 1.0:
            raise ValueError(f"max_delta must be in (0, 1], got {max_delta}.")

        self.in_channels = int(in_channels)
        self.max_delta = float(max_delta)
        self.k = RTMDetConvBNAct(in_channels, 1)
        self.v = RTMDetConvBNAct(in_channels, in_channels)
        self.m = RTMDetConvBNAct(in_channels, in_channels, use_bn=False)
        self.m2 = RTMDetConvBNAct(2, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.local_mean = nn.AvgPool2d(
            kernel_size=3,
            stride=1,
            padding=1,
            count_include_pad=False,
        )
        self.contrast_proj = nn.Conv2d(1, 1, 3, 1, 1, bias=True)
        self.contrast_logit = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.contrast_proj.weight)
        nn.init.zeros_(self.contrast_proj.bias)

    def calibration_beta(self) -> Tensor:
        return self.max_delta * torch.tanh(self.contrast_logit)

    def compute_context_residual(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(
                f"CA-SCAM expected BCHW with C={self.in_channels}, "
                f"got {tuple(x.shape)}."
            )
        batch, channels, height, width = x.shape
        avg = self.avg_pool(x).softmax(dim=1).view(batch, 1, 1, channels)
        maximum = self.max_pool(x).softmax(dim=1).view(batch, 1, 1, channels)
        k = self.k(x).view(batch, 1, -1, 1).softmax(dim=2)
        v = self.v(x).view(batch, 1, channels, -1)
        channel_context = torch.matmul(v, k).view(batch, channels, 1, 1)
        spatial_context = torch.cat(
            (
                torch.matmul(avg, v).view(batch, 1, height, width),
                torch.matmul(maximum, v).view(batch, 1, height, width),
            ),
            dim=1,
        )
        return self.m(channel_context) * self.m2(spatial_context).sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        residual = self.compute_context_residual(x)
        local_contrast = torch.abs(x - self.local_mean(x)).mean(
            dim=1,
            keepdim=True,
        )
        contrast_map = self.contrast_proj(local_contrast).sigmoid()
        return x + (1.0 + self.calibration_beta() * contrast_map) * residual


if _OPENMMLAB_IMPORT_ERROR is None:

    class _InceptionDWDepthwiseModule(nn.Module):
        """InceptionDW spatial operator with OpenMMLab norm/activation."""

        def __init__(
            self,
            channels: int,
            *,
            square_kernel_size: int,
            band_kernel_size: int,
            branch_ratio: float,
            norm_cfg: dict[str, Any],
            act_cfg: dict[str, Any],
        ) -> None:
            super().__init__()
            self.inception = InceptionDWConv2d(
                channels,
                square_kernel_size=square_kernel_size,
                band_kernel_size=band_kernel_size,
                branch_ratio=branch_ratio,
            )
            self.bn = build_norm_layer(norm_cfg, channels)[1]
            self.activate = build_activation_layer(act_cfg)

        def forward(self, x: Tensor) -> Tensor:
            return self.activate(self.bn(self.inception(x)))


    class RTMDetInceptionDWProjection(nn.Module):
        """Replace only CSPNeXtBlock.conv2's spatial depthwise operation."""

        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            *,
            square_kernel_size: int = 3,
            band_kernel_size: int = 11,
            branch_ratio: float = 0.125,
            norm_cfg: dict[str, Any] | None = None,
            act_cfg: dict[str, Any] | None = None,
        ) -> None:
            super().__init__()
            norm_cfg = norm_cfg or dict(type="BN", momentum=0.03, eps=0.001)
            act_cfg = act_cfg or dict(type="SiLU", inplace=True)
            self.depthwise_conv = _InceptionDWDepthwiseModule(
                in_channels,
                square_kernel_size=square_kernel_size,
                band_kernel_size=band_kernel_size,
                branch_ratio=branch_ratio,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            )
            # The official CSPNeXt pointwise projection is retained, including
            # its state-dict path, so its COCO-pretrained tensors can transfer.
            self.pointwise_conv = ConvModule(
                in_channels,
                out_channels,
                1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            )

        def forward(self, x: Tensor) -> Tensor:
            return self.pointwise_conv(self.depthwise_conv(x))


@MODELS.register_module()
class RTMDetVGUPCSPNeXt(CSPNeXt):
    """Three-stage CSPNeXt backbone producing P2/P3/P4 with VGUP."""

    def __init__(
        self,
        *args: Any,
        shallow_inception_stages: Sequence[int] = (1, 2),
        vgup_bpw_segments: int = 8,
        vgup_prediction_size: int = 128,
        inception_square_kernel_size: int = 3,
        inception_band_kernel_size: int = 11,
        inception_branch_ratio: float = 0.125,
        input_mean_bgr: Sequence[float] = (103.53, 116.28, 123.675),
        input_std_bgr: Sequence[float] = (57.375, 57.12, 58.395),
        **kwargs: Any,
    ) -> None:
        _require_openmmlab()
        norm_cfg = kwargs.get(
            "norm_cfg",
            dict(type="BN", momentum=0.03, eps=0.001),
        )
        act_cfg = kwargs.get("act_cfg", dict(type="SiLU", inplace=True))
        super().__init__(*args, **kwargs)
        self.vgup = VGUPPreprocessor(
            in_channels=3,
            bpw_segments=vgup_bpw_segments,
            prediction_size=vgup_prediction_size,
            use_global_gate=True,
            use_spatial_gate=True,
        )
        self.register_buffer(
            "_input_mean_bgr",
            torch.tensor(input_mean_bgr, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_input_std_bgr",
            torch.tensor(input_std_bgr, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

        replaced = 0
        for stage_index in shallow_inception_stages:
            if stage_index < 1 or not hasattr(self, f"stage{stage_index}"):
                raise ValueError(
                    f"Invalid shallow InceptionDW stage: {stage_index}."
                )
            stage = getattr(self, f"stage{stage_index}")
            csp_layer = stage[-1]
            if not hasattr(csp_layer, "blocks"):
                raise TypeError(f"stage{stage_index} does not end in CSPLayer.")
            for block in csp_layer.blocks:
                original = block.conv2
                in_channels = original.pointwise_conv.conv.in_channels
                out_channels = original.pointwise_conv.conv.out_channels
                block.conv2 = RTMDetInceptionDWProjection(
                    in_channels,
                    out_channels,
                    square_kernel_size=inception_square_kernel_size,
                    band_kernel_size=inception_band_kernel_size,
                    branch_ratio=inception_branch_ratio,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                )
                replaced += 1
        if replaced == 0:
            raise RuntimeError("No shallow CSPNeXt bottleneck was replaced.")
        self.inceptiondw_replacements = replaced

    def _apply_vgup_to_normalized_bgr(self, x: Tensor) -> Tensor:
        mean = self._input_mean_bgr.to(dtype=x.dtype, device=x.device)
        std = self._input_std_bgr.to(dtype=x.dtype, device=x.device)
        raw_bgr = (x * std + mean).clamp(0.0, 255.0)
        rgb = raw_bgr[:, (2, 1, 0)] / 255.0
        enhanced_rgb = self.vgup(rgb)
        enhanced_bgr = enhanced_rgb[:, (2, 1, 0)] * 255.0
        return (enhanced_bgr - mean) / std

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        return super().forward(self._apply_vgup_to_normalized_bgr(x))


@MODELS.register_module()
class RTMDetDPLSCASCAMPAFPN(CSPNeXtPAFPN):
    """P2/P3/P4 CSPNeXt-PAFPN with two DySample and three CA-SCAM."""

    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int,
        *args: Any,
        dysample_groups: int = 4,
        dysample_style: str = "lp",
        dysample_scope: bool = False,
        cascam_max_delta: float = 0.1,
        **kwargs: Any,
    ) -> None:
        _require_openmmlab()
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.dysamples = nn.ModuleList(
            DySample(
                int(in_channels[index - 1]),
                scale=2,
                style=dysample_style,
                groups=dysample_groups,
                dyscope=dysample_scope,
            )
            for index in range(len(in_channels) - 1, 0, -1)
        )
        self.cascam_layers = nn.ModuleList(
            RTMDetCASCAM(out_channels, max_delta=cascam_max_delta)
            for _ in in_channels
        )

    def forward(self, inputs: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        if len(inputs) != len(self.in_channels):
            raise ValueError(
                f"Expected {len(self.in_channels)} feature levels, "
                f"got {len(inputs)}."
            )

        inner_outs = [inputs[-1]]
        for index in range(len(self.in_channels) - 1, 0, -1):
            module_index = len(self.in_channels) - 1 - index
            feature_high = self.reduce_layers[module_index](inner_outs[0])
            inner_outs[0] = feature_high
            upsampled = self.dysamples[module_index](feature_high)
            feature_low = inputs[index - 1]
            inner_out = self.top_down_blocks[module_index](
                torch.cat((upsampled, feature_low), dim=1)
            )
            inner_outs.insert(0, inner_out)

        outs = [inner_outs[0]]
        for index in range(len(self.in_channels) - 1):
            downsampled = self.downsamples[index](outs[-1])
            outs.append(
                self.bottom_up_blocks[index](
                    torch.cat((downsampled, inner_outs[index + 1]), dim=1)
                )
            )

        for index, out_conv in enumerate(self.out_convs):
            outs[index] = self.cascam_layers[index](out_conv(outs[index]))
        return tuple(outs)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_audited_rtmdet_checkpoint(
    model: nn.Module,
    checkpoint: str | Path,
    *,
    variant: str,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load every same-name/same-shape official tensor and report the handoff."""

    source_path = str(checkpoint)
    if source_path.startswith(("http://", "https://")):
        document = torch.hub.load_state_dict_from_url(
            source_path,
            map_location="cpu",
            check_hash=False,
        )
        checkpoint_sha256 = None
    else:
        resolved = Path(checkpoint).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        document = torch.load(resolved, map_location="cpu", weights_only=False)
        source_path = str(resolved)
        checkpoint_sha256 = _sha256(resolved)

    source_state = document.get("state_dict", document)
    source_state = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in source_state.items()
        if isinstance(value, Tensor)
    }
    target = model.module if hasattr(model, "module") else model
    target_state = target.state_dict()
    loaded = {
        key: value
        for key, value in source_state.items()
        if key in target_state and tuple(value.shape) == tuple(target_state[key].shape)
    }
    incompatible = target.load_state_dict(loaded, strict=False)

    component_counts: dict[str, dict[str, int]] = {}
    for key in target_state:
        component = key.split(".", 1)[0]
        bucket = component_counts.setdefault(component, {"loaded": 0, "total": 0})
        bucket["total"] += 1
        bucket["loaded"] += int(key in loaded)

    shape_mismatches = [
        {
            "key": key,
            "source": list(value.shape),
            "target": list(target_state[key].shape),
        }
        for key, value in source_state.items()
        if key in target_state and tuple(value.shape) != tuple(target_state[key].shape)
    ]
    report = {
        "variant": variant,
        "checkpoint": source_path,
        "checkpoint_sha256": checkpoint_sha256,
        "loaded_tensors": len(loaded),
        "total_tensors": len(target_state),
        "loaded_numel": int(sum(target_state[key].numel() for key in loaded)),
        "total_numel": int(sum(value.numel() for value in target_state.values())),
        "component_tensors": component_counts,
        "shape_mismatches": shape_mismatches,
        "missing_after_load": list(incompatible.missing_keys),
        "unexpected_after_load": list(incompatible.unexpected_keys),
    }

    required = (
        "backbone.stem.0.conv.weight",
        "backbone.stage3.0.conv.weight",
        "bbox_head.rtm_reg.0.weight",
    )
    absent = [key for key in required if key not in loaded]
    if absent:
        raise AssertionError(f"Required official tensors were not inherited: {absent}")
    if variant == "official" and component_counts.get("neck", {}).get("loaded", 0) == 0:
        raise AssertionError("Official baseline did not inherit any neck tensor.")
    if variant == "ours" and any(key.startswith("backbone.stage4.") for key in target_state):
        raise AssertionError("The migrated P2/P3/P4 backbone must not retain P5 stage4.")

    if report_path is not None:
        output = Path(report_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(
        f"Loaded/Total tensors: {report['loaded_tensors']}/"
        f"{report['total_tensors']} ({variant})",
        flush=True,
    )
    return report


def audit_rtmdet_structure(
    model: nn.Module,
    *,
    variant: str,
    input_size: int = 128,
) -> dict[str, Any]:
    """Run a lightweight forward/backward and assert the intended pyramid."""

    target = model.module if hasattr(model, "module") else model
    expected_strides = (8, 16, 32) if variant == "official" else (4, 8, 16)
    target.train()
    sample = torch.randn(1, 3, input_size, input_size, requires_grad=True)
    features = target.extract_feat(sample)
    actual_strides = tuple(input_size // int(feature.shape[-1]) for feature in features)
    if actual_strides != expected_strides:
        raise AssertionError(
            f"Unexpected feature strides {actual_strides}; expected {expected_strides}."
        )
    cls_scores, bbox_preds = target.bbox_head(features)
    surrogate = sum(item.float().mean() for item in (*cls_scores, *bbox_preds))
    surrogate.backward()
    gradient_tensors = sum(
        parameter.grad is not None for parameter in target.parameters()
    )
    if gradient_tensors == 0:
        raise AssertionError("No parameter received a backward gradient.")

    report: dict[str, Any] = {
        "variant": variant,
        "input_size": input_size,
        "feature_shapes": [list(feature.shape) for feature in features],
        "feature_strides": list(actual_strides),
        "head_strides": [
            int(stride[0]) for stride in target.bbox_head.prior_generator.strides
        ],
        "gradient_tensors": gradient_tensors,
    }
    if tuple(report["head_strides"]) != expected_strides:
        raise AssertionError(
            f"Head strides {report['head_strides']} do not match {expected_strides}."
        )
    if variant == "ours":
        backbone = target.backbone
        neck = target.neck
        if hasattr(backbone, "stage4"):
            raise AssertionError("P5 stage4 is still present in the migrated backbone.")
        if getattr(backbone, "inceptiondw_replacements", 0) < 2:
            raise AssertionError("Both shallow P2/P3 CSPNeXt blocks must use InceptionDW.")
        if len(neck.dysamples) != 2 or len(neck.cascam_layers) != 3:
            raise AssertionError("DPLS requires 2 DySample and 3 CA-SCAM modules.")
        offset_stds = [
            float(module.offset.weight.detach().float().std().item())
            for module in neck.dysamples
        ]
        if any(value >= 0.01 for value in offset_stds):
            raise AssertionError(
                f"DySample small-offset initialization was overwritten: {offset_stds}"
            )
        if any(
            torch.count_nonzero(module.contrast_proj.weight).item() != 0
            or torch.count_nonzero(module.contrast_proj.bias).item() != 0
            or torch.count_nonzero(module.contrast_logit).item() != 0
            for module in neck.cascam_layers
        ):
            raise AssertionError("CA-SCAM bounded calibration must start at beta=0.")
        report.update(
            {
                "inceptiondw_replacements": backbone.inceptiondw_replacements,
                "dysample_modules": len(neck.dysamples),
                "dysample_offset_stds": offset_stds,
                "cascam_modules": len(neck.cascam_layers),
                "cascam_zero_initialized": True,
                "vgup_present": isinstance(backbone.vgup, VGUPPreprocessor),
            }
        )
    target.zero_grad(set_to_none=True)
    return report


@HOOKS.register_module()
class RTMDetPretrainedTransferHook(Hook):
    """Load audited COCO tensors after init and before EMA/epoch one.

    MMEngine initializes model weights inside ``Runner.train``.  Loading the
    checkpoint before that call would therefore be silently overwritten.  A
    high-priority ``before_train`` hook is the correct lifecycle boundary.  A
    true resume already restores model and optimizer state, so it is skipped.
    """

    priority = "VERY_HIGH"

    def __init__(
        self,
        checkpoint: str,
        variant: str,
        report_path: str | None = None,
    ) -> None:
        _require_openmmlab()
        if variant not in {"official", "ours"}:
            raise ValueError(f"Unknown RTMDet transfer variant: {variant}")
        self.checkpoint = checkpoint
        self.variant = variant
        self.report_path = report_path

    def before_train(self, runner: Any) -> None:
        if getattr(runner, "_resume", False):
            runner.logger.info(
                "Resume is active; official pretrained transfer is skipped."
            )
            return
        report = load_audited_rtmdet_checkpoint(
            runner.model,
            self.checkpoint,
            variant=self.variant,
            report_path=self.report_path,
        )
        runner.logger.info(
            "Audited pretrained handoff complete: Loaded/Total tensors %d/%d",
            report["loaded_tensors"],
            report["total_tensors"],
        )


@HOOKS.register_module()
class RTMDetDriveSyncHook(Hook):
    """Atomically mirror recoverable RTMDet artifacts to Google Drive."""

    priority = "VERY_LOW"

    def __init__(self, drive_dir: str, interval: int = 1) -> None:
        _require_openmmlab()
        if interval <= 0:
            raise ValueError("interval must be positive.")
        self.drive_dir = Path(drive_dir)
        self.interval = int(interval)
        self._copied: dict[str, tuple[int, int]] = {}

    @staticmethod
    def _eligible(path: Path) -> bool:
        return path.name == "last_checkpoint" or path.suffix.lower() in {
            ".pth",
            ".json",
            ".log",
            ".txt",
            ".csv",
            ".py",
            ".yaml",
            ".yml",
        }

    def _sync(self, runner: Any) -> None:
        source_root = Path(runner.work_dir).resolve()
        destination_root = self.drive_dir
        destination_root.mkdir(parents=True, exist_ok=True)
        copied = 0
        for source in source_root.rglob("*"):
            if not source.is_file() or not self._eligible(source):
                continue
            stat = source.stat()
            relative = source.relative_to(source_root)
            signature = (stat.st_size, stat.st_mtime_ns)
            key = relative.as_posix()
            if self._copied.get(key) == signature:
                continue
            destination = destination_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + ".tmp")
            shutil.copyfile(source, temporary)
            os.replace(temporary, destination)
            self._copied[key] = signature
            copied += 1
        if copied:
            runner.logger.info(
                "Synced %d recoverable artifact(s) to %s",
                copied,
                destination_root,
            )

    def after_train_epoch(self, runner: Any) -> None:
        if (runner.epoch + 1) % self.interval == 0:
            self._sync(runner)

    def after_val_epoch(self, runner: Any, metrics: dict | None = None) -> None:
        self._sync(runner)

    def after_run(self, runner: Any) -> None:
        self._sync(runner)


__all__ = [
    "RTMDetCASCAM",
    "RTMDetDPLSCASCAMPAFPN",
    "RTMDetDriveSyncHook",
    "RTMDetPretrainedTransferHook",
    "RTMDetVGUPCSPNeXt",
    "audit_rtmdet_structure",
    "load_audited_rtmdet_checkpoint",
]
