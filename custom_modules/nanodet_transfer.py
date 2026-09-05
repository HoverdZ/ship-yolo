"""NanoDet-Plus-m-416 adapters for the paired cross-architecture experiment.

The detector head, GFL/DSLA losses, auxiliary training branch, GhostPAN
blocks, and ShuffleNetV2 blocks are imported from the official NanoDet
v1.0.0 source tree.  The migrated variant changes only the four components
that form the final ship detector:

* VGUP processes normalized BGR input before the official ShuffleNetV2 stem;
* InceptionDW replaces the second spatial operator in one non-downsampling
  block at each shallow stage, while every stage's first stride-2 block is
  retained unchanged;
* removing only the parameter-free max-pool shifts the native P3-P5 backbone
  outputs to P2-P4, and the two GhostPAN top-down upsamplers become DySample;
* bounded CA-SCAM calibrates the three P2-P4 PAN outputs, while the native
  extra P5 path is left unchanged.

NanoDet is an optional dependency so this repository remains importable on a
lightweight Ultralytics-only workstation.  Constructing either model requires
the official RangiLyu/nanodet v1.0.0 checkout to be on ``sys.path``.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, Sequence

import torch
from torch import Tensor, nn

from custom_modules.dysample import DySample
from custom_modules.inceptiondw import InceptionDWConv2d
from custom_modules.vgup import VGUPPreprocessor

NanoDetVariant = Literal["official", "ours"]

_NANODET_IMPORT_ERROR: ModuleNotFoundError | None = None

try:
    from nanodet.model.arch.nanodet_plus import NanoDetPlus
    from nanodet.model.backbone.shufflenetv2 import ShuffleNetV2
    from nanodet.model.fpn.ghost_pan import GhostPAN
    from nanodet.model.head.nanodet_plus_head import NanoDetPlusHead
    from nanodet.model.head.simple_conv_head import SimpleConvHead
except ModuleNotFoundError as error:  # pragma: no cover - local fallback
    _NANODET_IMPORT_ERROR = error

    class _UnavailableNanoDetBase(nn.Module):
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            super().__init__()
            raise ModuleNotFoundError(
                "NanoDet adapters require the official RangiLyu/nanodet "
                "v1.0.0 source tree on sys.path."
            ) from _NANODET_IMPORT_ERROR

    NanoDetPlus = _UnavailableNanoDetBase  # type: ignore[assignment]
    ShuffleNetV2 = _UnavailableNanoDetBase  # type: ignore[assignment]
    GhostPAN = _UnavailableNanoDetBase  # type: ignore[assignment]
    NanoDetPlusHead = _UnavailableNanoDetBase  # type: ignore[assignment]
    SimpleConvHead = _UnavailableNanoDetBase  # type: ignore[assignment]


def _require_nanodet() -> None:
    if _NANODET_IMPORT_ERROR is not None:
        raise ModuleNotFoundError(
            "NanoDet adapters require the official RangiLyu/nanodet "
            "v1.0.0 source tree on sys.path."
        ) from _NANODET_IMPORT_ERROR


class NanoDetConvBNAct(nn.Module):
    """Single-GPU Conv-BN-LeakyReLU projection used inside CA-SCAM."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        *,
        use_bn: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.conv(x)))


class NanoDetCASCAM(nn.Module):
    """Framework-neutral bounded CA-SCAM using NanoDet activations."""

    def __init__(self, in_channels: int, max_delta: float = 0.1) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be positive.")
        if not 0.0 < max_delta <= 1.0:
            raise ValueError("max_delta must be in (0, 1].")
        self.in_channels = int(in_channels)
        self.max_delta = float(max_delta)
        self.k = NanoDetConvBNAct(in_channels, 1)
        self.v = NanoDetConvBNAct(in_channels, in_channels)
        self.m = NanoDetConvBNAct(in_channels, in_channels, use_bn=False)
        self.m2 = NanoDetConvBNAct(2, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.local_mean = nn.AvgPool2d(3, 1, 1, count_include_pad=False)
        self.contrast_proj = nn.Conv2d(1, 1, 3, 1, 1, bias=True)
        self.contrast_logit = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.contrast_proj.weight)
        nn.init.zeros_(self.contrast_proj.bias)

    def calibration_beta(self) -> Tensor:
        return self.max_delta * torch.tanh(self.contrast_logit)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(
                f"CA-SCAM expected BCHW with C={self.in_channels}, got {tuple(x.shape)}."
            )
        batch, channels, height, width = x.shape
        average = self.avg_pool(x).softmax(dim=1).view(batch, 1, 1, channels)
        maximum = self.max_pool(x).softmax(dim=1).view(batch, 1, 1, channels)
        key = self.k(x).view(batch, 1, -1, 1).softmax(dim=2)
        value = self.v(x).view(batch, 1, channels, -1)
        channel_context = torch.matmul(value, key).view(batch, channels, 1, 1)
        spatial_context = torch.cat(
            (
                torch.matmul(average, value).view(batch, 1, height, width),
                torch.matmul(maximum, value).view(batch, 1, height, width),
            ),
            dim=1,
        )
        residual = self.m(channel_context) * self.m2(spatial_context).sigmoid()
        local_contrast = torch.abs(x - self.local_mean(x)).mean(dim=1, keepdim=True)
        contrast_map = self.contrast_proj(local_contrast).sigmoid()
        return x + (1.0 + self.calibration_beta() * contrast_map) * residual


if _NANODET_IMPORT_ERROR is None:

    class NanoDetVGUPShuffleNetV2(ShuffleNetV2):
        """ShuffleNetV2 P2-P4 backbone with shallow InceptionDW and VGUP."""

        def __init__(
            self,
            *,
            shallow_stages: Sequence[int] = (2, 3),
            vgup_bpw_segments: int = 8,
            vgup_prediction_size: int = 128,
            inception_square_kernel_size: int = 3,
            inception_band_kernel_size: int = 11,
            inception_branch_ratio: float = 0.125,
        ) -> None:
            super().__init__(
                model_size="1.0x",
                out_stages=(2, 3, 4),
                with_last_conv=False,
                activation="LeakyReLU",
                pretrain=False,
            )

            # conv1 remains stride 2 and the first block of every stage remains
            # stride 2.  Removing this parameter-free pool alone yields native
            # P2/P3/P4 outputs without discarding pretrained stage4 weights.
            self.maxpool = nn.Identity()
            self.vgup = VGUPPreprocessor(
                in_channels=3,
                bpw_segments=vgup_bpw_segments,
                prediction_size=vgup_prediction_size,
                use_global_gate=True,
                use_spatial_gate=True,
            )
            self.register_buffer(
                "_input_mean_bgr",
                torch.tensor((103.53, 116.28, 123.675), dtype=torch.float32).view(
                    1, 3, 1, 1
                ),
                persistent=False,
            )
            self.register_buffer(
                "_input_std_bgr",
                torch.tensor((57.375, 57.12, 58.395), dtype=torch.float32).view(
                    1, 3, 1, 1
                ),
                persistent=False,
            )

            replaced: list[str] = []
            for stage_index in shallow_stages:
                stage = getattr(self, f"stage{stage_index}", None)
                if stage is None or len(stage) < 2:
                    raise ValueError(
                        f"ShuffleNetV2 stage{stage_index} has no eligible "
                        "non-downsampling block."
                    )
                if int(stage[0].stride) != 2:
                    raise AssertionError(
                        f"stage{stage_index}[0] must remain the stride-2 block."
                    )
                block = stage[1]
                if int(block.stride) != 1:
                    raise AssertionError(
                        f"stage{stage_index}[1] must be non-downsampling."
                    )
                original = block.branch2[3]
                if not isinstance(original, nn.Conv2d):
                    raise TypeError(
                        f"Unexpected spatial operator at stage{stage_index}[1]."
                    )
                if original.stride != (1, 1) or original.groups != original.in_channels:
                    raise AssertionError(
                        "Only a stride-1 depthwise spatial operator may be replaced."
                    )
                block.branch2[3] = InceptionDWConv2d(
                    original.in_channels,
                    square_kernel_size=inception_square_kernel_size,
                    band_kernel_size=inception_band_kernel_size,
                    branch_ratio=inception_branch_ratio,
                )
                replaced.append(f"stage{stage_index}.1.branch2.3")
            self.inceptiondw_replacements = tuple(replaced)

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


    class NanoDetDPLSCASCAMGhostPAN(GhostPAN):
        """Official GhostPAN with DPLS top-down sampling and bounded CA-SCAM."""

        def __init__(
            self,
            *,
            in_channels: Sequence[int] = (116, 232, 464),
            out_channels: int = 96,
            dysample_groups: int = 4,
            cascam_max_delta: float = 0.1,
        ) -> None:
            super().__init__(
                in_channels=list(in_channels),
                out_channels=out_channels,
                kernel_size=5,
                num_extra_level=1,
                use_depthwise=True,
                activation="LeakyReLU",
            )
            self.upsample = nn.Identity()
            self.dysamples = nn.ModuleList(
                DySample(
                    out_channels,
                    scale=2,
                    style="lp",
                    groups=dysample_groups,
                    dyscope=False,
                )
                for _ in range(len(in_channels) - 1)
            )
            self.cascam_layers = nn.ModuleList(
                NanoDetCASCAM(out_channels, max_delta=cascam_max_delta)
                for _ in in_channels
            )

        def forward(self, inputs: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
            if len(inputs) != len(self.in_channels):
                raise ValueError(
                    f"Expected {len(self.in_channels)} input levels, got {len(inputs)}."
                )
            reduced = [
                layer(feature)
                for feature, layer in zip(inputs, self.reduce_layers)
            ]

            inner_outs = [reduced[-1]]
            for index in range(len(self.in_channels) - 1, 0, -1):
                module_index = len(self.in_channels) - 1 - index
                high = inner_outs[0]
                low = reduced[index - 1]
                upsampled = self.dysamples[module_index](high)
                fused = self.top_down_blocks[module_index](
                    torch.cat((upsampled, low), dim=1)
                )
                inner_outs.insert(0, fused)

            outs = [inner_outs[0]]
            for index in range(len(self.in_channels) - 1):
                downsampled = self.downsamples[index](outs[-1])
                outs.append(
                    self.bottom_up_blocks[index](
                        torch.cat((downsampled, inner_outs[index + 1]), dim=1)
                    )
                )

            # Keep NanoDet-Plus's native extra P5 path unchanged.
            for extra_in, extra_out in zip(
                self.extra_lvl_in_conv, self.extra_lvl_out_conv
            ):
                outs.append(extra_in(reduced[-1]) + extra_out(outs[-1]))

            for index, calibration in enumerate(self.cascam_layers):
                outs[index] = calibration(outs[index])
            return tuple(outs)


class NanoDetPlusPairModel(nn.Module):
    """NanoDet-Plus forward contract using only official head/loss classes."""

    def __init__(
        self,
        backbone: nn.Module,
        fpn: nn.Module,
        head: nn.Module,
        aux_head: nn.Module,
        *,
        detach_epoch: int = 10,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.fpn = fpn
        self.aux_fpn = copy.deepcopy(fpn)
        self.head = head
        self.aux_head = aux_head
        self.detach_epoch = int(detach_epoch)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def forward(self, x: Tensor) -> Tensor:
        features = self.backbone(x)
        return self.head(self.fpn(features))

    def forward_train(
        self, metadata: dict[str, Any]
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        features = self.backbone(metadata["img"])
        pyramid = self.fpn(features)
        if self.epoch >= self.detach_epoch:
            aux_pyramid = self.aux_fpn([item.detach() for item in features])
            dual = (
                torch.cat((main.detach(), aux), dim=1)
                for main, aux in zip(pyramid, aux_pyramid)
            )
        else:
            aux_pyramid = self.aux_fpn(features)
            dual = (
                torch.cat((main, aux), dim=1)
                for main, aux in zip(pyramid, aux_pyramid)
            )
        predictions = self.head(pyramid)
        aux_predictions = self.aux_head(dual)
        loss, states = self.head.loss(
            predictions,
            metadata,
            aux_preds=aux_predictions,
        )
        return predictions, loss, states


def _loss_config() -> SimpleNamespace:
    return SimpleNamespace(
        loss_qfl=SimpleNamespace(
            name="QualityFocalLoss",
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0,
        ),
        loss_dfl=SimpleNamespace(
            name="DistributionFocalLoss",
            loss_weight=0.25,
        ),
        loss_bbox=SimpleNamespace(name="GIoULoss", loss_weight=2.0),
    )


def build_nanodet_plus_pair_model(
    variant: NanoDetVariant,
    *,
    num_classes: int = 1,
) -> NanoDetPlusPairModel:
    """Build one member of the controlled NanoDet-Plus comparison pair."""

    _require_nanodet()
    if variant not in {"official", "ours"}:
        raise ValueError(f"Unknown NanoDet experiment variant: {variant!r}.")
    if num_classes != 1:
        raise ValueError("The frozen ship dataset contains exactly one class.")

    if variant == "official":
        backbone = ShuffleNetV2(
            model_size="1.0x",
            out_stages=(2, 3, 4),
            with_last_conv=False,
            activation="LeakyReLU",
            pretrain=False,
        )
        fpn = GhostPAN(
            in_channels=[116, 232, 464],
            out_channels=96,
            kernel_size=5,
            num_extra_level=1,
            use_depthwise=True,
            activation="LeakyReLU",
        )
        strides = [8, 16, 32, 64]
    else:
        backbone = NanoDetVGUPShuffleNetV2()
        fpn = NanoDetDPLSCASCAMGhostPAN()
        strides = [4, 8, 16, 32]

    head = NanoDetPlusHead(
        num_classes=num_classes,
        input_channel=96,
        feat_channels=96,
        stacked_convs=2,
        kernel_size=5,
        strides=strides,
        activation="LeakyReLU",
        reg_max=7,
        norm_cfg=dict(type="BN"),
        loss=_loss_config(),
    )
    aux_head = SimpleConvHead(
        num_classes=num_classes,
        input_channel=192,
        feat_channels=192,
        stacked_convs=4,
        strides=strides,
        activation="LeakyReLU",
        reg_max=7,
    )
    return NanoDetPlusPairModel(
        backbone,
        fpn,
        head,
        aux_head,
        detach_epoch=10,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_checkpoint_state(document: Any) -> dict[str, Tensor]:
    if isinstance(document, dict) and isinstance(document.get("state_dict"), dict):
        state = document["state_dict"]
    elif isinstance(document, dict):
        state = document
    else:
        raise TypeError("NanoDet checkpoint must contain a state dictionary.")

    tensor_items = {key: value for key, value in state.items() if torch.is_tensor(value)}
    average_items = {
        key[len("avg_model.") :]: value
        for key, value in tensor_items.items()
        if key.startswith("avg_model.")
    }
    if average_items:
        tensor_items = average_items

    normalised: dict[str, Tensor] = {}
    for key, value in tensor_items.items():
        clean = key
        changed = True
        while changed:
            changed = False
            for prefix in ("module.", "model."):
                if clean.startswith(prefix):
                    clean = clean[len(prefix) :]
                    changed = True
        normalised[clean] = value
    return normalised


def load_audited_nanodet_checkpoint(
    model: nn.Module,
    checkpoint: str | Path,
    *,
    variant: NanoDetVariant,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load every same-name/same-shape official tensor and audit the transfer."""

    resolved = Path(checkpoint).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    try:
        document = torch.load(resolved, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch versions before ``weights_only``.
        document = torch.load(resolved, map_location="cpu")
    source = _normalise_checkpoint_state(document)
    target = model.state_dict()
    merged = dict(target)
    loaded: list[str] = []
    partially_loaded: list[dict[str, Any]] = []
    shape_mismatch: list[dict[str, Any]] = []
    unused_source: list[str] = []

    for name, tensor in source.items():
        if name not in target:
            unused_source.append(name)
            continue
        if tuple(tensor.shape) != tuple(target[name].shape):
            # NanoDet-Plus predicts [classes, 4 * (reg_max + 1)] in one
            # convolution.  A naive shape filter would discard the pretrained
            # regression distribution whenever nc changes from COCO 80 to ship
            # 1.  Copy COCO's semantically matching ``boat`` row (index 8) and
            # all 32 regression rows into the one-class output instead.
            if (
                name.startswith("head.gfl_cls.")
                and tensor.ndim == target[name].ndim
                and tuple(tensor.shape[1:]) == tuple(target[name].shape[1:])
                and tensor.shape[0] == 80 + 4 * 8
                and target[name].shape[0] == 1 + 4 * 8
            ):
                adapted = target[name].clone()
                adapted[0].copy_(tensor[8].to(dtype=adapted.dtype))
                adapted[1:].copy_(tensor[80:].to(dtype=adapted.dtype))
                merged[name] = adapted
                loaded.append(name)
                partially_loaded.append(
                    {
                        "name": name,
                        "class_mapping": "COCO boat[8] -> ship[0]",
                        "regression_mapping": "source[80:112] -> target[1:33]",
                    }
                )
                continue
            shape_mismatch.append(
                {
                    "name": name,
                    "source_shape": list(tensor.shape),
                    "target_shape": list(target[name].shape),
                }
            )
            continue
        merged[name] = tensor.to(dtype=target[name].dtype)
        loaded.append(name)

    model.load_state_dict(merged, strict=True)
    for prefix in ("backbone.", "fpn.", "head.cls_convs."):
        if not any(name.startswith(prefix) for name in loaded):
            raise AssertionError(
                f"Official checkpoint did not initialize required prefix {prefix!r}."
            )

    report: dict[str, Any] = {
        "variant": variant,
        "checkpoint": str(resolved),
        "checkpoint_sha256": _sha256(resolved),
        "loaded_tensors": len(loaded),
        "total_tensors": len(target),
        "source_tensors": len(source),
        "loaded_ratio": len(loaded) / max(len(target), 1),
        "source_loaded_ratio": len(loaded) / max(len(source), 1),
        "loaded_names": sorted(loaded),
        "partially_loaded": partially_loaded,
        "shape_mismatch": shape_mismatch,
        "missing_target": sorted(set(target) - set(loaded)),
        "unused_source": sorted(unused_source),
    }
    if report_path is not None:
        destination = Path(report_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    print(
        f"Loaded/Total tensors: {report['loaded_tensors']}/"
        f"{report['total_tensors']} ({variant}); official source coverage "
        f"{report['loaded_tensors']}/{report['source_tensors']}",
        flush=True,
    )
    return report


def audit_nanodet_pair_structure(
    model: NanoDetPlusPairModel,
    *,
    variant: NanoDetVariant,
    input_size: int = 64,
) -> dict[str, Any]:
    """Run a small CPU forward pass and verify every intended feature stride."""

    expected = (8, 16, 32, 64) if variant == "official" else (4, 8, 16, 32)
    previous_mode = model.training
    model.eval()
    with torch.no_grad():
        sample = torch.zeros(1, 3, input_size, input_size)
        backbone_features = model.backbone(sample)
        pyramid = model.fpn(backbone_features)
        predictions = model.head(pyramid)
    actual = tuple(input_size // int(feature.shape[-1]) for feature in pyramid)
    if actual != expected:
        raise AssertionError(f"Feature strides {actual}; expected {expected}.")
    if tuple(int(value) for value in model.head.strides) != expected:
        raise AssertionError("Detection-head strides do not match pyramid strides.")

    report: dict[str, Any] = {
        "variant": variant,
        "input_size": input_size,
        "backbone_shapes": [list(item.shape) for item in backbone_features],
        "pyramid_shapes": [list(item.shape) for item in pyramid],
        "feature_strides": list(actual),
        "prediction_shape": list(predictions.shape),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }
    if variant == "ours":
        backbone = model.backbone
        neck = model.fpn
        if not isinstance(backbone.maxpool, nn.Identity):
            raise AssertionError("DPLS requires removal of the parameter-free max-pool.")
        for stage_index in (2, 3):
            stage = getattr(backbone, f"stage{stage_index}")
            if int(stage[0].stride) != 2:
                raise AssertionError("The first stride-2 block was modified.")
            if not isinstance(stage[1].branch2[3], InceptionDWConv2d):
                raise AssertionError(
                    f"stage{stage_index}[1] does not contain InceptionDW."
                )
        if len(neck.dysamples) != 2 or len(neck.cascam_layers) != 3:
            raise AssertionError("Expected two DySample and three CA-SCAM modules.")
        offset_stds = [
            float(module.offset.weight.detach().float().std().item())
            for module in neck.dysamples
        ]
        if any(value >= 0.01 for value in offset_stds):
            raise AssertionError("DySample small-offset initialization was overwritten.")
        if any(
            torch.count_nonzero(module.contrast_proj.weight).item() != 0
            or torch.count_nonzero(module.contrast_proj.bias).item() != 0
            or torch.count_nonzero(module.contrast_logit).item() != 0
            for module in neck.cascam_layers
        ):
            raise AssertionError("CA-SCAM calibration must start from beta=0.")
        report.update(
            {
                "vgup_present": isinstance(backbone.vgup, VGUPPreprocessor),
                "inceptiondw_replacements": list(
                    backbone.inceptiondw_replacements
                ),
                "first_stride2_blocks_preserved": True,
                "dysample_modules": len(neck.dysamples),
                "dysample_offset_stds": offset_stds,
                "cascam_modules": len(neck.cascam_layers),
                "native_extra_p5_preserved": len(pyramid) == 4,
            }
        )
    model.train(previous_mode)
    return report


__all__ = [
    "NanoDetCASCAM",
    "NanoDetDPLSCASCAMGhostPAN",
    "NanoDetPlusPairModel",
    "NanoDetVGUPShuffleNetV2",
    "audit_nanodet_pair_structure",
    "build_nanodet_plus_pair_model",
    "load_audited_nanodet_checkpoint",
]
