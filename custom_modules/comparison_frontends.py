"""End-to-end image-processing front-ends reproduced for the comparison study.

Every class here is a first-layer RGB preprocessor: it takes the detector input
image in [0, 1] and returns an image of the same shape. They are registered as
ordinary layers, so Ultralytics' ``parse_model`` default branch passes the YAML
arguments through unchanged and keeps 3 output channels (no parser patch).

All front-ends run in FP32 even when the trainer uses AMP, so that divisions,
powers and min-max normalisations behave as in the original FP32 code.

Sources
-------
GDIPPreprocessor
    Kalwar et al., "GDIP: Gated Differentiable Image Processing for
    Object-Detection in Adverse Conditions", ICRA 2023.
    Ported from https://github.com/Gatedip/GDIP-Yolo (MIT License,
    Copyright (c) 2022 Gatedip), files ``model/gated_dip_modified_customEncoder.py``
    and ``model/vision_encoder.py`` (the block used by ``Yolov3GatedDIP``).
AGEPreprocessor
    Zheng et al., "TinyDark-YOLO for adaptive and lightweight object detection
    in low-light conditions", Scientific Reports 16:25712 (2026), Section
    "Adaptive gamma enhancement module (AGE)". Reproduced from the paper text;
    the authors' Zenodo archive (doi:10.5281/zenodo.20618962) was not reachable
    when this file was written.

AIEPreprocessor
    Wang et al., "A joint adaptive image enhancement and detection network
    for robust maritime surveillance under hazy weather conditions",
    Reliability Engineering and System Safety 277 (2027) 113265, Sec. 3.2.
    The released code (github.com/wsw1996/AIE-DE-ship-detection) does not
    contain the AIE module, so it is reconstructed from the paper.
    ``RFCAConv``/``h_swish`` are copied from that release's
    ``ultralytics/nn/modules/rfa.py`` (RFAConv family, author Xin Zhang).

Deviations from the original code are listed in
``experiments/frontend_comparison/README.md``.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def _require_rgb(image: torch.Tensor, name: str) -> None:
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError(f"{name} expects a [B,3,H,W] RGB tensor, got {tuple(image.shape)}.")
    if not image.is_floating_point():
        raise TypeError(f"{name} expects a floating-point image.")


def _resize_square(image: torch.Tensor, size: int) -> torch.Tensor:
    if size <= 0 or tuple(image.shape[-2:]) == (size, size):
        return image
    return F.interpolate(image, size=(size, size), mode="bilinear", align_corners=False)


def _autocast_off(device_type: str):
    return torch.autocast(device_type=device_type, enabled=False)


# ---------------------------------------------------------------------------
# GDIP
# ---------------------------------------------------------------------------
class GDIPVisionEncoder(nn.Module):
    """``VisionEncoder`` from GDIP-Yolo: 5 unpadded 3x3 convs (64..1024),
    3x3/2 average pooling after the first four, global pooling and a
    ReLU-activated linear projection."""

    def __init__(self, encoder_output_dim: int = 256) -> None:
        super().__init__()
        channels = (3, 64, 128, 256, 512, 1024)
        layers: list[nn.Module] = []
        for index in range(5):
            layers.append(nn.Conv2d(channels[index], channels[index + 1], kernel_size=3, stride=1))
            layers.append(nn.ReLU(True))
            if index < 4:
                layers.append(nn.AvgPool2d((3, 3), (2, 2)))
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear_proj = nn.Sequential(nn.Linear(1024, encoder_output_dim), nn.ReLU(True))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.linear_proj(self.pool(self.features(image)).flatten(1))


class GDIPPreprocessor(nn.Module):
    """GDIP block (7 gated differentiable image-processing operations).

    YAML args: ``[prediction_size, encoder_output_dim, per_image_norm]``.

    ``prediction_size`` resizes the encoder input (the operations still act on
    the full-resolution image). ``per_image_norm=True`` applies the original
    min-max normalisations per image instead of over the whole batch.
    """

    num_gates = 7
    blur_kernel_size = 13
    blur_sigma_range = (0.1, 5.0)

    def __init__(
        self,
        prediction_size: int = 256,
        encoder_output_dim: int = 256,
        per_image_norm: bool = True,
    ) -> None:
        super().__init__()
        if prediction_size < 0:
            raise ValueError("prediction_size must be >= 0.")
        self.prediction_size = int(prediction_size)
        self.per_image_norm = bool(per_image_norm)
        dim = int(encoder_output_dim)
        self.encoder = GDIPVisionEncoder(encoder_output_dim=dim)
        self.gate_module = nn.Linear(dim, self.num_gates, bias=True)
        self.wb_module = nn.Linear(dim, 3, bias=True)
        self.gamma_module = nn.Linear(dim, 1, bias=True)
        self.sharpning_module = nn.Linear(dim, 1, bias=True)
        self.defogging_module = nn.Linear(dim, 1, bias=True)
        self.contrast_module = nn.Linear(dim, 1, bias=True)
        self.tone_module = nn.Linear(dim, 8, bias=True)

    # -- helpers (same formulas as the official implementation) -------------
    @staticmethod
    def tanh01(x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x) * 0.5 + 0.5

    def tanh_range(self, x: torch.Tensor, left: float, right: float) -> torch.Tensor:
        return self.tanh01(x) * (right - left) + left

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.per_image_norm:
            low = x.amin(dim=(1, 2, 3), keepdim=True)
            high = x.amax(dim=(1, 2, 3), keepdim=True)
        else:
            low, high = x.min(), x.max()
        return (x - low) / (high - low).clamp_min(1e-8)

    @staticmethod
    def _gate(gate: torch.Tensor) -> torch.Tensor:
        return gate.view(-1, 1, 1, 1)

    @staticmethod
    def rgb2lum(img: torch.Tensor) -> torch.Tensor:
        return 0.27 * img[:, 0] + 0.67 * img[:, 1] + 0.06 * img[:, 2]

    @staticmethod
    def dark_channel(x: torch.Tensor) -> torch.Tensor:
        return x.min(dim=1, keepdim=True)[0]

    @staticmethod
    def atmospheric_light(x: torch.Tensor, dark: torch.Tensor, top_k: int = 1000) -> torch.Tensor:
        """Vectorised form of the official double loop.

        The official loop sums ``indices[:, 1:numpx]`` (it starts at 1) and
        divides by ``numpx``; this reproduces that exactly.
        """
        batch, _, height, width = x.shape
        size = height * width
        numpx = int(max(math.floor(size / top_k), 1))
        darkvec = dark.reshape(batch, size)
        imvec = x.reshape(batch, 3, size)
        indices = darkvec.argsort(dim=1)[:, size - numpx:]
        indices = indices[:, 1:]
        if indices.shape[1] == 0:
            atm = x.new_zeros(batch, 3)
        else:
            picked = torch.gather(imvec, 2, indices.unsqueeze(1).expand(-1, 3, -1))
            atm = picked.sum(dim=2) / numpx
        return atm.view(batch, 3, 1, 1)

    def blur(self, x: torch.Tensor) -> torch.Tensor:
        """Gaussian blur, kernel 13. Training samples sigma in [0.1, 5.0] per
        call like ``torchvision.transforms.GaussianBlur``; evaluation uses the
        midpoint so validation is deterministic."""
        low, high = self.blur_sigma_range
        if self.training:
            sigma = float(torch.empty(1).uniform_(low, high).item())
        else:
            sigma = 0.5 * (low + high)
        radius = self.blur_kernel_size // 2
        coords = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
        kernel_1d = torch.exp(-(coords**2) / (2.0 * sigma * sigma))
        kernel_1d = kernel_1d / kernel_1d.sum()
        channels = x.shape[1]
        kx = kernel_1d.view(1, 1, 1, -1).repeat(channels, 1, 1, 1)
        ky = kernel_1d.view(1, 1, -1, 1).repeat(channels, 1, 1, 1)
        padded = F.pad(x, (radius, radius, radius, radius), mode="reflect")
        out = F.conv2d(padded, kx, groups=channels)
        return F.conv2d(out, ky, groups=channels)

    # -- operations ---------------------------------------------------------
    def defog(self, x, latent, gate):
        omega = self.tanh_range(self.defogging_module(latent), 0.1, 1.0).view(-1, 1, 1, 1)
        dark_i = self.dark_channel(x)
        a = self.atmospheric_light(x, dark_i)
        i = self.dark_channel(x / a.clamp_min(1e-6))
        t = 1.0 - omega * i
        j = (x - a) / torch.clamp(t, min=0.01) + a
        return self.normalize(j) * self._gate(gate)

    def white_balance(self, x, latent, gate):
        log_wb_range = 0.5
        wb = torch.exp(self.tanh_range(self.wb_module(latent), -log_wb_range, log_wb_range))
        color_scaling = 1.0 / (1e-5 + 0.27 * wb[:, 0] + 0.67 * wb[:, 1] + 0.06 * wb[:, 2])
        wb = color_scaling.unsqueeze(1) * wb
        out = wb.view(-1, 3, 1, 1) * x
        return self.normalize(out) * self._gate(gate)

    def gamma_balance(self, x, latent, gate):
        log_gamma = math.log(2.5)
        gamma = torch.exp(self.tanh_range(self.gamma_module(latent), -log_gamma, log_gamma)).view(-1, 1, 1, 1)
        g = torch.pow(torch.clamp(x, min=1e-4), gamma)
        return self.normalize(g) * self._gate(gate)

    def sharpning(self, x, latent, gate):
        blurred = self.blur(x)
        y = self.tanh_range(self.sharpning_module(latent), 0.1, 1.0).view(-1, 1, 1, 1)
        s = x + y * (x - blurred)
        return self.normalize(s) * self._gate(gate)

    def identity(self, x, gate):
        return x * self._gate(gate)

    def contrast(self, x, latent, gate):
        alpha = torch.tanh(self.contrast_module(latent)).view(-1, 1, 1, 1)
        luminance = torch.clamp(self.rgb2lum(x), 0.0, 1.0).unsqueeze(1)
        contrast_lum = -torch.cos(math.pi * luminance) * 0.5 + 0.5
        contrast_image = x / (luminance + 1e-6) * contrast_lum
        contrast_image = (1 - alpha) * x + alpha * contrast_image
        return self.normalize(contrast_image) * self._gate(gate)

    def tone(self, x, latent, gate):
        curve_steps = 8
        tone_curve = self.tanh_range(self.tone_module(latent).reshape(-1, 1, curve_steps), 0.5, 2.0)
        tone_curve_sum = torch.sum(tone_curve, dim=2) + 1e-30
        total = torch.zeros_like(x)
        for i in range(curve_steps):
            total = total + torch.clamp(x - 1.0 * i / curve_steps, 0, 1.0 / curve_steps) * tone_curve[:, :, i].view(-1, 1, 1, 1)
        total = total * (curve_steps / tone_curve_sum.view(-1, 1, 1, 1))
        return self.normalize(total) * self._gate(gate)

    def forward(self, image: torch.Tensor, return_debug: bool = False):
        _require_rgb(image, "GDIPPreprocessor")
        input_dtype = image.dtype
        with _autocast_off(image.device.type):
            x = image.float()
            latent = self.encoder(_resize_square(x, self.prediction_size))
            gate = self.tanh_range(self.gate_module(latent), 0.01, 1.0)
            out = (
                self.white_balance(x, latent, gate[:, 0])
                + self.gamma_balance(x, latent, gate[:, 1])
                + self.identity(x, gate[:, 2])
                + self.sharpning(x, latent, gate[:, 3])
                + self.defog(x, latent, gate[:, 4])
                + self.contrast(x, latent, gate[:, 5])
                + self.tone(x, latent, gate[:, 6])
            )
            out = self.normalize(out)
        out = out.to(input_dtype)
        if return_debug:
            return out, {"gates": gate}
        return out


# ---------------------------------------------------------------------------
# AGE (TinyDark-YOLO)
# ---------------------------------------------------------------------------
class AGEPreprocessor(nn.Module):
    """Adaptive Gamma Enhancement with the CNN-Gamma predictor.

    YAML args: ``[prediction_size, gamma_min, gamma_max, negative_slope]``.

    Paper: the image is resized to 256x256, CNN-Gamma (five 3x3 stride-2
    conv + LeakyReLU blocks with 16/32/32/32/32 channels and two fully
    connected layers, 165K parameters) predicts one gamma per image, which a
    sigmoid maps into [0.01, 0.99]; ``O = I ** gamma`` on the full image.
    The hidden FC width (64) is inferred from the stated 165K parameters
    (this design gives 164,033). The LeakyReLU slope is not stated.
    """

    def __init__(
        self,
        prediction_size: int = 256,
        gamma_min: float = 0.01,
        gamma_max: float = 0.99,
        negative_slope: float = 0.1,
    ) -> None:
        super().__init__()
        if prediction_size != 256:
            raise ValueError("CNN-Gamma's 2048-d flatten assumes a 256x256 input.")
        if not 0.0 < gamma_min < gamma_max:
            raise ValueError("Require 0 < gamma_min < gamma_max.")
        self.prediction_size = int(prediction_size)
        self.gamma_min = float(gamma_min)
        self.gamma_max = float(gamma_max)
        channels = (3, 16, 32, 32, 32, 32)
        layers: list[nn.Module] = []
        for index in range(5):
            layers.append(nn.Conv2d(channels[index], channels[index + 1], 3, 2, 1))
            layers.append(nn.LeakyReLU(negative_slope, inplace=True))
        self.features = nn.Sequential(*layers)
        self.fc = nn.Sequential(
            nn.Linear(32 * 8 * 8, 64),
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.Linear(64, 1),
        )

    def predict_gamma(self, image: torch.Tensor) -> torch.Tensor:
        features = self.features(_resize_square(image, self.prediction_size))
        raw = self.fc(features.flatten(1))
        return self.gamma_min + (self.gamma_max - self.gamma_min) * torch.sigmoid(raw)

    def forward(self, image: torch.Tensor, return_debug: bool = False):
        _require_rgb(image, "AGEPreprocessor")
        input_dtype = image.dtype
        with _autocast_off(image.device.type):
            x = image.float()
            gamma = self.predict_gamma(x)
            out = torch.pow(x.clamp(min=1e-6, max=1.0), gamma.view(-1, 1, 1, 1))
        out = out.to(input_dtype)
        if return_debug:
            return out, {"gamma": gamma}
        return out




# ---------------------------------------------------------------------------
# AIE (AIEDN) — reconstruction from the paper
# ---------------------------------------------------------------------------
class _HSigmoid(nn.Module):
    def __init__(self, inplace: bool = True) -> None:
        super().__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class _HSwish(nn.Module):
    def __init__(self, inplace: bool = True) -> None:
        super().__init__()
        self.sigmoid = _HSigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class RFCAConv(nn.Module):
    """Receptive-field coordinate-attention convolution, verbatim from the
    AIEDN release (``rfa.py``); ``einops.rearrange`` replaced by an
    equivalent reshape/permute."""

    def __init__(self, inp: int, oup: int, kernel_size: int, stride: int, reduction: int = 32) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.generate = nn.Sequential(
            nn.Conv2d(inp, inp * (kernel_size**2), kernel_size, padding=kernel_size // 2,
                      stride=stride, groups=inp, bias=False),
            nn.BatchNorm2d(inp * (kernel_size**2)),
            nn.ReLU(),
        )
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = _HSwish()
        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv = nn.Sequential(nn.Conv2d(inp, oup, kernel_size, stride=kernel_size))

    def forward(self, x):
        b, c = x.shape[0:2]
        k = self.kernel_size
        feature = self.generate(x)
        h, w = feature.shape[2:]
        # 'b c (n1 n2) h w -> b c (h n1) (w n2)'
        feature = feature.view(b, c, k, k, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, c, h * k, w * k)
        x_h = self.pool_h(feature)
        x_w = self.pool_w(feature).permute(0, 1, 3, 2)
        y = self.act(self.bn1(self.conv1(torch.cat([x_h, x_w], dim=2))))
        h, w = feature.shape[2:]
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        return self.conv(feature * a_w * a_h)


class _ChannelLayerNorm(nn.Module):
    """Per-pixel LayerNorm over channels with bias (Restormer 'WithBias')."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        mu = x.mean(1, keepdim=True)
        var = x.var(1, keepdim=True, unbiased=False)
        x = (x - mu) / torch.sqrt(var + 1e-5)
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class _ChannelSelfAttention(nn.Module):
    """Eqs. (7)-(8): LN -> 1x1 conv -> 3x3 group conv (3C groups) -> Q,K,V;
    channel ("transposed") self-attention softmax(QK^T/alpha)V, 1x1 conv,
    residual. Q and K are L2-normalised as in Restormer's MDTA."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = _ChannelLayerNorm(dim)
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.qkv_dw = nn.Conv2d(dim * 3, dim * 3, 3, padding=1, groups=dim * 3, bias=False)
        self.temperature = nn.Parameter(torch.ones(1))
        self.proj = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv_dw(self.qkv(self.norm(x))).chunk(3, dim=1)
        q = F.normalize(q.reshape(b, c, h * w), dim=-1)
        k = F.normalize(k.reshape(b, c, h * w), dim=-1)
        v = v.reshape(b, c, h * w)
        attn = torch.softmax((q @ k.transpose(-2, -1)) * self.temperature, dim=-1)
        out = (attn @ v).reshape(b, c, h, w)
        return self.proj(out) + x


class _CGE(nn.Module):
    """Content-Guided Enhancement, Eqs. (2)-(9). Channel/spatial/pixel
    attention follow DEA-Net's content-guided attention (the paper's [34])."""

    def __init__(self, dim: int, reduction: int = 8, splits: int = 4) -> None:
        super().__init__()
        if dim % splits:
            raise ValueError(f"CGE dim {dim} must be divisible by {splits}.")
        self.splits = splits
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, max(dim // reduction, 1), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(dim // reduction, 1), dim, 1),
        )
        self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode="reflect")
        self.pa_group = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode="reflect", groups=dim)
        self.pa_point = nn.Conv2d(dim, dim, 1)
        self.prompt_conv = nn.Conv2d(dim, dim, 3, padding=1)
        self.fuse = nn.Conv2d(2 * dim, dim, 1)
        chunk = dim // splits
        self.attn = nn.ModuleList(_ChannelSelfAttention(chunk) for _ in range(splits))

    def forward(self, d: torch.Tensor, prompt: torch.Tensor) -> torch.Tensor:
        w_c = self.ca(d)
        w_s = self.sa(torch.cat([d.mean(1, keepdim=True), d.amax(1, keepdim=True)], dim=1))
        f_coa = (w_c + w_s) * d                                            # Eq. (4)
        b, c, h, w = d.shape
        shuffled = torch.stack([f_coa, d], dim=2).reshape(b, 2 * c, h, w)  # channel shuffle
        f_rw = torch.sigmoid(self.pa_point(self.pa_group(shuffled)))       # Eq. (5)
        prompt = F.interpolate(prompt, size=(h, w), mode="bilinear", align_corners=False)
        f_w = self.fuse(torch.cat([d, self.prompt_conv(prompt * f_rw)], dim=1))  # Eq. (6)
        parts = f_w.chunk(self.splits, dim=1)
        return torch.cat([blk(p) for blk, p in zip(self.attn, parts)], dim=1)  # Eqs. (7)-(9)


class AIEPreprocessor(nn.Module):
    """Adaptive Image Enhancement module of AIEDN (reconstruction).

    YAML args: ``[base_channels, prompt_size]``.

    Encoder: RFCAConv embedding (3->C, full resolution) and four encoding
    layers (RFCAConv 3x3 stride 1 + stride-2 3x3 conv; the fourth layer has
    no down-sampling), giving C, 2C (1/2), 4C (1/4), 8C (1/8).
    DIG: a learnable content prompt P1 (8C x prompt_size^2) and stacked 3x3
    stride-2 transposed convs give P2 (4C) and P3 (2C).
    Decoder: three CGE layers at 1/8, 1/4, 1/2 with 2x transposed-conv
    up-sampling, skip concatenation and a 3x3 conv; the last up-sampled map is
    concatenated with the full-resolution embedding (2C), projected to 3
    channels and added to the input image (Eq. 10).

    Defaults C=6 and prompt 80x80 reproduce the reported cost of AIE on
    YOLOv8 (about 0.48M parameters and 5.1 GFLOPs at 640): this module has
    about 0.44M parameters and about 5.8 GFLOPs (conv layers, 640x640).
    """

    def __init__(self, base_channels: int = 6, prompt_size: int = 80) -> None:
        super().__init__()
        c = int(base_channels)
        if c < 2 or (2 * c) % 4:
            raise ValueError("base_channels must make 2C divisible by 4 (CGE splits).")
        self.embed = RFCAConv(3, c, 3, 1)
        widths = (c, 2 * c, 4 * c, 8 * c)
        self.enc = nn.ModuleList(RFCAConv(widths[i], widths[i], 3, 1) for i in range(4))
        self.down = nn.ModuleList(nn.Conv2d(widths[i], widths[i + 1], 3, 2, 1) for i in range(3))
        self.prompt = nn.Parameter(torch.zeros(1, widths[3], prompt_size, prompt_size))
        nn.init.trunc_normal_(self.prompt, std=0.02)
        self.dig = nn.ModuleList([
            nn.ConvTranspose2d(widths[3], widths[2], 3, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose2d(widths[2], widths[1], 3, stride=2, padding=1, output_padding=1),
        ])
        self.cge = nn.ModuleList(_CGE(widths[3 - i]) for i in range(3))
        self.up = nn.ModuleList(nn.ConvTranspose2d(widths[3 - i], widths[2 - i], 2, stride=2) for i in range(3))
        self.merge = nn.ModuleList(
            nn.Sequential(nn.Conv2d(2 * widths[2 - i], widths[2 - i], 3, padding=1), nn.ReLU(inplace=True))
            for i in range(2)
        )
        self.out = nn.Conv2d(2 * c, 3, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        _require_rgb(image, "AIEPreprocessor")
        h, w = image.shape[-2:]
        if h % 8 or w % 8:
            raise ValueError("AIEPreprocessor needs H and W divisible by 8.")
        f0 = self.embed(image)
        skips, x = [], f0
        for i in range(3):
            x = self.enc[i](x)
            x = self.down[i](x)
            skips.append(x)                      # 1/2 (2C), 1/4 (4C), 1/8 (8C)
        x = self.enc[3](x)                       # latent, 1/8, 8C
        prompts = [self.prompt.expand(image.shape[0], -1, -1, -1)]
        for layer in self.dig:
            prompts.append(layer(prompts[-1]))
        for i in range(3):
            x = self.cge[i](x, prompts[i])
            x = self.up[i](x)
            if i < 2:
                x = self.merge[i](torch.cat([x, skips[1 - i]], dim=1))
        x = torch.cat([x, f0], dim=1)            # full resolution, 2C
        return image + self.out(x)


__all__ = ["AGEPreprocessor", "AIEPreprocessor", "GDIPPreprocessor", "GDIPVisionEncoder", "RFCAConv"]
