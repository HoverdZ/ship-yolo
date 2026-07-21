# YOLO11n-InceptionDW-C3k2-P23

## Purpose

This formal ablation asks whether the official InceptionNeXt
`InceptionDWConv2d` spatial operator can reduce the cost of the shallow
YOLO11n backbone while retaining the official P3/P4/P5 detector topology. It
does not claim an accuracy improvement before training and evaluation.

The experiment targets `ultralytics==8.4.92`.

## Exact replacement scope

The YAML is an exact copy of the official YOLO11 detector topology except for
two module names:

- Backbone layer 2: `C3k2` -> `C3k2_InceptionDW`
- Backbone layer 4: `C3k2` -> `C3k2_InceptionDW`

Each target block has one internal bottleneck after YOLO11n depth scaling. The
first 3x3 Ultralytics `Conv`, the shortcut condition, the outer C3k2 split and
fusion 1x1 convolutions, and the outer `e=0.25` setting are retained.

The official core is channel preserving, while Ultralytics 8.4.92 constructs
the target bottleneck as `c -> c/2 -> c`. To retain both the original first
3x3 shape and the residual width, the replacement second operator is
decomposed as:

```text
unactivated 1x1 channel adapter (c/2 -> c)
-> InceptionDWConv2d
-> BatchNorm2d
-> SiLU
```

The adapter only restores the channel-expansion role of the original second
convolution. It adds no normalization, activation, attention, gate, or other
mechanism. The InceptionDW wrapper itself has exactly the required
`InceptionDWConv2d -> BatchNorm2d -> SiLU` order.

P4/P5 backbone C3k2 blocks, SPPF, C2PSA, the complete Neck, Detect, loss,
augmentation, and all stride-2 convolutions remain official YOLO11n.

## InceptionDW source and scope

The core design is adapted from:

- Paper: *InceptionNeXt: When Inception Meets ConvNeXt*, CVPR 2024
- Official repository: <https://github.com/sail-sg/inceptionnext>
- Official source: `models/inceptionnext.py`, class `InceptionDWConv2d`
- Upstream license: Apache-2.0

Defaults are fixed to `square_kernel_size=3`, `band_kernel_size=11`, and
`branch_ratio=0.125`. Channels are split into identity, 3x3 depthwise, 1x11
depthwise, and 11x1 depthwise branches. This experiment reuses only that core
operator; it is not a full InceptionNeXt backbone and does not add the other
InceptionNeXt stages or training recipe.

After YOLO11n scaling, the two core operators are:

- `model.2.m.0.cv2.inception`, split `(10, 2, 2, 2)`
- `model.4.m.0.cv2.inception`, split `(20, 4, 4, 4)`

## Files

```text
custom_modules/inceptiondw.py
custom_modules/c3k2_inceptiondw.py
custom_modules/register.py
experiments/yolo11n_inceptiondw_c3k2_p23.yaml
tools/inceptiondw_utils.py
tools/build_inceptiondw_c3k2_p23.py
tools/check_inceptiondw_c3k2_p23.py
tools/train_inceptiondw_c3k2_p23.py
tests/test_inceptiondw_c3k2_p23.py
logs/yolo11n_inceptiondw_c3k2_p23_640.md
```

## Pre-training validation

Validated on CPU with Ultralytics 8.4.92:

- YAML build succeeds.
- A 640x640 forward succeeds and all returned tensors are finite.
- Detect receives P3/P4/P5 tensors shaped `1x64x80x80`,
  `1x128x40x40`, and `1x256x20x20`.
- All top-level feature shapes match official YOLO11n.
- Exactly two custom C3k2 modules and two InceptionDW cores are present.
- P4/P5 and all Neck C3k2 modules remain official.
- No SCConv, SRU, or CRU module is present.
- The project core matches a minimal official reference for 16, 32, and 64
  channels with maximum absolute error at or below `1e-7`.

## Model statistics

Statistics use a 640 input and Ultralytics 8.4.92.

| Metric | Official YOLO11n | InceptionDW P23 | Delta | Change |
|---|---:|---:|---:|---:|
| Layers | 182 | 190 | +8 | +4.3956% |
| Parameters | 2,624,080 | 2,619,164 | -4,916 | -0.1873% |
| Trainable parameters | 2,624,064 | 2,619,148 | -4,916 | -0.1873% |
| GFLOPs | 6.614336 | 6.514240 | -0.100096 | -1.5133% |
| FP32 parameter size (MiB) | 10.010071 | 9.991318 | -0.018753 | -0.1873% |

These are architecture statistics only, not evidence of accuracy or latency
improvement.

## Official weight inheritance

Using `yolo11n.pt`:

- Source state tensors: 499
- Target state tensors: 511
- Inherited tensors: 497 (`97.2603%` of target state tensors)
- Inherited parameter elements: 2,618,320 / 2,619,164 (`99.9678%`)
- Inherited state elements: 2,634,065 / 2,634,909 (`99.9680%`)
- Both target outer 1x1 pairs and both retained first 3x3 convolutions inherit.
- Untouched backbone, Neck, and Detect have no abnormal unmatched target keys.
- The two old `cv2.conv.weight` tensors are deliberately not mapped to the
  InceptionDW branches.
- Only the two channel adapters and six new depthwise branch convolutions use
  fresh initialization.

## Commands

Build and full preflight:

```bash
pip install ultralytics==8.4.92
python tools/build_inceptiondw_c3k2_p23.py
python tools/check_inceptiondw_c3k2_p23.py --weights yolo11n.pt
pytest -q tests/test_inceptiondw_c3k2_p23.py
```

Colab formal training (not run during code preparation):

```bash
python tools/train_inceptiondw_c3k2_p23.py \
  --data /content/ship_detection/data/data.yaml \
  --project /content/drive/MyDrive/ship_detection/runs
```

Resume after a Colab disconnect. The script validates and uses
`<project>/<name>/weights/last.pt`:

```bash
python tools/train_inceptiondw_c3k2_p23.py \
  --data /content/ship_detection/data/data.yaml \
  --project /content/drive/MyDrive/ship_detection/runs \
  --resume true
```

Defaults are model
`experiments/yolo11n_inceptiondw_c3k2_p23.yaml`, `yolo11n.pt`, 150 epochs,
640 image size, batch 8, workers 2, device 0, seed 0, optimizer `auto`, and run
name `yolo11n_inceptiondw_c3k2_p23_640`.

## Not yet validated

Formal training, validation metrics, GPU memory, wall-clock throughput, and
accuracy have intentionally not been measured in this code-preparation task.
