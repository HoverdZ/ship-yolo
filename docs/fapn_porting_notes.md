# Official FaPN to YOLO11 Porting Notes

## Source and attribution

This implementation ports **FaPN: Feature-Aligned Pyramid Network for Dense
Image Prediction** by Shihua Huang, Zhichao Lu, Ran Cheng, and Cheng He,
published at ICCV 2021.

Primary sources:

- Paper: <https://openaccess.thecvf.com/content/ICCV2021/papers/Huang_FaPN_Feature-Aligned_Pyramid_Network_for_Dense_Image_Prediction_ICCV_2021_paper.pdf>
- Official repository: <https://github.com/EMI-Group/FaPN>
- Official FSM/FAM/top-down code:
  <https://github.com/EMI-Group/FaPN/blob/main/detectron2/modeling/backbone/fan.py>
- Official legacy DCNv2 reference:
  <https://github.com/EMI-Group/FaPN/blob/main/DCNv2/dcn_v2.py>
- Official repository license: Apache-2.0
- Torchvision modulated deformable convolution:
  <https://docs.pytorch.org/vision/main/generated/torchvision.ops.deform_conv2d.html>
- Ultralytics YOLO11 configuration:
  <https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/11/yolo11.yaml>

Only the relevant FaPN logic is adapted. The legacy Detectron2 tree and old
custom CUDA extension are not vendored.

## Fidelity table

| Official FaPN behavior | Current implementation | Consistent? |
|---|---|---|
| GAP + 1x1 + Sigmoid | `adaptive_avg_pool2d` + `conv_attention` + `Sigmoid` | Yes |
| `x + x * attention` | Explicit `selected` and `enhanced` tensors | Yes |
| bilinear, `align_corners=False` | `F.interpolate(..., mode="bilinear", align_corners=False)` | Yes |
| `feat_up * 2` | Preserved before the offset-feature concatenation | Yes |
| offset feature 1x1 conv | Bias-free `Conv2d(2c, c, 1)` with Xavier init | Yes |
| 3x3 modulated DCNv2 | Torchvision `DeformConv2d`, 3x3, stride 1, padding 1 | Yes, API port |
| `deformable_groups=8` | Fixed at 8; non-divisible channels raise | Yes |
| offset/mask conv zero initialization | Both weight and bias explicitly zeroed | Yes |
| ReLU after DCNv2 | `nn.ReLU(inplace=True)` | Yes |
| aligned + lateral | Direct tensor addition | Yes |
| M4 rather than T4 enters the next level | YAML layer 14 reads `[C3, layer 12 M4]` | Yes |
| Per-level 3x3 output conv | `FaPNOutputConv(c, c)`, bias enabled, Xavier init | Yes |
| Unified FaPN output channels | Logical 256, scaled by YOLO11n width to actual 64 | Yes |

## Framework adaptations

The mathematical operations above remain unchanged. The following are
engineering translations only:

1. Detectron2 `Conv2d` wrappers are represented by PyTorch `nn.Conv2d`.
   `FeatureAlign_V2` calls the official FSM with `norm=""`, so the port does
   not introduce BatchNorm or GroupNorm.
2. The retired custom DCNv2 CUDA extension is replaced by Torchvision's
   modulated `DeformConv2d`. Offset channels remain `2 * 8 * 3 * 3 = 144`,
   mask channels remain `8 * 3 * 3 = 72`, and a non-null sigmoid mask selects
   DCNv2 behavior.
3. Detectron2's feature dictionary and Python loop are expressed as explicit
   Ultralytics YAML nodes. `FaPNAlign` accepts a two-tensor YAML input list.
4. The official top-down `out_channels=256` is passed through Ultralytics
   width scaling, producing 64 channels for YOLO11n. This value remains
   divisible by eight.
5. The official code conditionally skips interpolation for already equal
   sizes. This port always calls bilinear interpolation to the lateral target
   size; equal-size interpolation is an identity, while the actual C5->C4 and
   M4->C3 recursion both use the intended 2x resize.

No DySample, P2 head/injection, DCNv3/DCNv4, ordinary-convolution fallback,
new attention, new downsampling, PAN DCN, loss change, or detection-head
replacement is introduced.

## YOLO topology mapping

Both models use this head indexing:

```text
10 C5/C2PSA
11 C5 -> FaPNLateral -> M5
12 [C4 (layer 6), M5] -> FaPNAlign -> M4
13 M4 -> FaPNOutputConv -> T4
14 [C3 (layer 4), M4 (layer 12)] -> FaPNAlign -> M3
15 M3 -> FaPNOutputConv -> T3
16 T3 -> official stride-2 Conv
17 concat with T4 (layer 13)
18 official C3k2 -> P4
19 P4 -> official stride-2 Conv
20 concat with original C5/C2PSA (layer 10)
21 official C3k2 -> P5
22 Detect(T3, P4, P5)
```

The bottom-up PAN therefore retains its two stride-2 `Conv` nodes, two
`Concat` nodes, and two output `C3k2` nodes. The first PAN C3k2 receives fewer
input channels because original FaPN uses one unified top-down width; its
module role, output width, repeat count, and placement remain unchanged.

`yolo11n_fapn.yaml` uses the exact official YOLO11n backbone.
`yolo11n_inceptiondw_fapn.yaml` copies the already validated InceptionDW
backbone verbatim; only its top-down head is changed to the same FaPN graph.

## Weight transfer

Raw state-dict names after layer 10 cannot be trusted because replacing the
top-down graph shifts PAN and Detect indices. The transfer utility uses this
explicit semantic mapping:

```text
source backbone 0-10 -> target backbone 0-10
source PAN 17       -> target PAN 16
source PAN 19       -> target PAN 18
source PAN 20       -> target PAN 19
source PAN 22       -> target PAN 21
source Detect 23    -> target Detect 22
```

Source top-down layers 11-16 are deliberately excluded. Shape checks are then
applied to every semantic candidate. The reports are:

- `artifacts/fapn_weight_transfer_baseline.json`
- `artifacts/fapn_weight_transfer_inceptiondw.json`

FaPN-only inherits 2,378,032 / 2,917,411 parameter elements (81.5117%). Its
entire backbone inherits. InceptionDW+FaPN inherits 2,372,272 / 2,912,495
parameter elements (81.4515%); the InceptionDW backbone follows the existing
official-YOLO11 initialization strategy, with 99.9380% of backbone parameter
elements inherited.

The new FaPN modules are intentionally random-initialized. One first-PAN C3k2
input projection changes shape because the unified FaPN T4 width differs from
the original YOLO top-down width. Detect classification mismatches are
expected because these ship models use `nc=1` while official `yolo11n.pt`
uses 80 classes. Shape-compatible PAN, regression, and Detect tensors still
inherit.

## Validation and model statistics

With `ultralytics==8.4.92`, Torchvision modulated DCNv2 is available on CPU.
The checks cover 640 inference, 256 forward/backward, finite tensors and
gradients, exact offset/mask channels, zero initialization, node shapes, PAN
scope, explicit weight mapping, and both model builds.

| Model | Layers | Parameters | Trainable | GFLOPs | FP32 MiB |
|---|---:|---:|---:|---:|---:|
| YOLO11n-FaPN | 179 | 2,917,411 | 2,917,395 | 9.083816 | 11.129040 |
| YOLO11n-InceptionDW-FaPN | 187 | 2,912,495 | 2,912,479 | 8.983720 | 11.110287 |

GFLOPs are reported by THOP with an explicit hook that counts each modulated
DCNv2 convolution as its equivalent 3x3 convolution. Sampling/interpolation
implementation overhead is not represented by this analytical FLOP number.

At official zero initialization, both upstream offset-feature 1x1 weights
receive a valid zero gradient on the first synthetic backward pass because
the following offset/mask generator weights are all zero. The parameters have
gradient tensors and become nonzero-gradient paths after that generator's
first update. No unexpected missing or non-finite gradient exists.

## Colab commands

FaPN-only:

```bash
python tools/train_yolo11n_fapn.py \
  --data /content/drive/MyDrive/ship_detection/data/data.yaml
```

InceptionDW + FaPN:

```bash
python tools/train_yolo11n_inceptiondw_fapn.py \
  --data /content/drive/MyDrive/ship_detection/data/data.yaml
```

Both scripts copy the configured dataset root to
`/content/datasets/ship_detection`, audit train/val/test image and label
counts, verify the pinned runtime and GPU, apply semantic pretrained weights,
write metadata, and save runs under Google Drive. Add `--resume true` to use
the validated `<project>/<name>/weights/last.pt` checkpoint.

Formal 150-epoch training has not been started locally. Accuracy, latency,
GPU memory, and dataset metrics remain unverified.
