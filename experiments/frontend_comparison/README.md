# Front-end comparison: independent enhancement vs. end-to-end processing

All methods are attached to the same stock YOLO11n (P3/P4/P5, native Detect)
and trained with the shared protocol in
[training_config.yaml](../training_config.yaml). The proposed module is
compared as `F6` in [vgup_filter_ablation](../vgup_filter_ablation/).

## End-to-end front-ends (trained jointly with the detector)

| ID | Method | Config | Source |
|---|---|---|---|
| C1 | ERUP (WACV 2025) | [C1_yolo11n_erup.yaml](C1_yolo11n_erup.yaml) | `custom_modules/erup.py` (paper reproduction; no official code) |
| C2 | GDIP (ICRA 2023) | [C2_yolo11n_gdip.yaml](C2_yolo11n_gdip.yaml) | ported from [Gatedip/GDIP-Yolo](https://github.com/Gatedip/GDIP-Yolo) (MIT) |
| C3 | AGE from TinyDark-YOLO (Sci. Rep. 2026) | [C3_yolo11n_age.yaml](C3_yolo11n_age.yaml) | reproduced from the paper |
| C4 | AIE from AIEDN (RESS 2027) | [C4_yolo11n_aie.yaml](C4_yolo11n_aie.yaml) | reconstructed from the paper; `RFCAConv` copied from the AIEDN release |

Each front-end is layer 0, so the official `yolo11n.pt` layers shift by one
exactly as for the VGUP experiments. The GDIP and AGE classes use
Ultralytics' default `parse_model` branch (arguments passed through,
3 output channels), so `register.py` only adds their names.

### Shared choices

- Parameter-prediction encoders of ERUP, GDIP and AGE see a 256x256
  bilinear resize of the input; all filters act on the full 640x640 image.
  For AGE this is the paper's own step (1); for ERUP and GDIP it replaces
  the full-size input of the original code. ERUP's paper does not state the encoder input
  size; `ERUPPreprocessor` keeps the old behaviour when `prediction_size=0`.
- ERUP, GDIP and AGE compute in FP32 inside AMP training; AIE is a
  conventional network and runs under AMP like the detector.

### Deviations from the original code (GDIP)

| Original | Here | Reason |
|---|---|---|
| Encoder on the full 448x448 input | Encoder on a 256x256 resize | shared choice above |
| Atmospheric light: Python double loop over pixels | Vectorised `gather`; still sums `indices[1:numpx]` and divides by `numpx` | identical values, avoids ~400 GPU launches per image |
| Min-max normalisation over the whole batch tensor | Per image (`per_image_norm=true`) | batch-global normalisation makes each output depend on the other images in the batch; set `false` for the original behaviour |
| `GaussianBlur(13, sigma=(0.1, 5.0))` samples sigma at every call, also at test time | Same sampling in training; fixed sigma 2.55 (midpoint) in eval | deterministic validation |
| `x / a` | `x / a.clamp_min(1e-6)` | avoids division by zero on black images |
| `.cuda()` hard-coded | device of the input | portability |

### Reproduction notes (AGE)

From the paper: input resized to 256x256; CNN-Gamma = five 3x3 stride-2
conv + LeakyReLU blocks (16/32/32/32/32 channels) and two FC layers;
165K parameters; one gamma per image mapped by a sigmoid into
[0.01, 0.99]; `O = I ** gamma` on the original resolution. Not stated, chosen
here: FC hidden width 64 (gives 164,033 parameters, matching 165K) and
LeakyReLU slope 0.1. TinyDark inserts AGE in place of the first backbone
convolution; here it is placed before the unchanged YOLO11n like every other
front-end.

### Reconstruction notes (AIE)

The AIEDN repository ships a modified Ultralytics without the AIE module,
and Sec. 3.2 leaves several sizes unstated or inconsistent. Choices:

| Item | Paper | Here |
|---|---|---|
| Base width C | not stated | 6, chosen so the conv cost (about 5.8 GFLOPs at 640) matches the reported 5.1 GFLOPs added to YOLOv8 |
| Initial content prompt P1 | size stated only symbolically | learnable 8C x 80 x 80 tensor (the 1/8 bottleneck size at 640); with C=6 the module has about 0.44M parameters vs. the reported 0.48M |
| DIG | stacked 3x3 stride-2 transposed convs | same; channels halve per level to match each decoder width |
| Encoder | RFCAConv embedding + 4 layers of RFCAConv + down-sampling | same; the 4th layer has no down-sampling (latent at 1/8), which makes the stated decoder sizes (D2: 1/4, 4C; D3: 1/2, 2C; F_m: full, 2C) consistent |
| CGE channel/spatial/pixel attention | Eqs. (2)-(5), following DEA-Net | DEA-Net's CGA layers (7x7 reflect convs, group conv with C groups) plus the stated 1x1 conv |
| Self-attention in CGE | LN, 1x1 conv, 3x3 group conv, softmax(QK^T/alpha)V, 1x1 conv, residual; n = 4 splits | same, channel ("transposed") attention per split with L2-normalised Q, K (Restormer MDTA) |
| Up-sampling and fusion convs | "up-sampling" and "a convolution" | 2x2 stride-2 transposed conv; 3x3 conv + ReLU |
| Output conv | not stated | zero-initialised, so the module starts as the identity (I_en = I_o + F_h) |

## Independent enhancement (frozen official weights, offline)

The detector is retrained on the enhanced images. Run
[tools/offline_enhance.py](../../tools/offline_enhance.py) once per method;
it writes a full dataset copy (`images`, copied `labels`, `data.yaml`) and
`enhancement_timing.json`, and fails if any output is missing or changes size.

| Method | Repository | Weights | Command essentials |
|---|---|---|---|
| GLADE-Net (Ecol. Inform. 2025) | [riotu-lab/GLADE-Net](https://github.com/riotu-lab/GLADE-Net) | Phase 1 `SateHaze1k.ckpt`, Phase 2 `SateHaze1k.pth` (Google Drive links in its README) | `--method glade --weights SateHaze1k.ckpt --weights2 SateHaze1k.pth --fp16` |
| AdaIR (ICLR 2025) | [c-yn/AdaIR](https://github.com/c-yn/AdaIR) | `adair5d.ckpt` (5-task, includes low-light) | `--method adair --weights adair5d.ckpt --batch 4 --fp16` |
| IPC-Dehaze (CVPR 2025) | [Jiayi-Fu/IPC-Dehaze](https://github.com/Jiayi-Fu/IPC-Dehaze) | `predictor.pth`, `critic.pth` | `--method ipc --weights predictor.pth --weights2 critic.pth` |

Notes:
- All run at 640x640. GLADE-Net Phase 1 uses `--tile 640`, i.e. one grid per
  image (the official script with the tile equal to the image size).
- GLADE-Net Phase 2: the released `SateHaze1k.pth` contains the 5x5
  Gaussian buffer `get_g_nopadding.kernel` of the first-level Laplacian
  extractor (paper Sec. 3.2: only the first pyramid level is used), but the
  active default in `architecture.py` is a 3x3 directional gradient and the
  official loader silently drops unmatched keys. The script copies
  `Phase_2/code`, swaps in the commented "level one" class verbatim, and
  aborts unless a strict load succeeds and the loaded kernel equals the
  expected 5x5 Gaussian. The cloned repository itself is not modified.
- AdaIR FP16 keeps its FFT in FP32 (cuFFT half precision needs power-of-two
  sizes). AdaIR pads to a multiple of 16 instead of the official crop, so
  label coordinates stay valid.
- IPC-Dehaze and GLADE-Net Phase 2 run in FP32 through their official
  scripts.
- Colab dependencies (install without touching torch): GLADE-Net needs
  `pytorch-lightning timm einops opencv-python`; AdaIR needs `einops`;
  IPC-Dehaze needs `pyiqa timm opencv-python` (its `basicsr` is vendored).
  Do not install GLADE-Net's pinned `torch==2.2.2`.
- Always start with `--limit 20 --dst <scratch>` to check images and the
  per-image time before the full run.
- The timing file is wall-clock time including file I/O. Measure model
  latency separately for the paper's speed column.
