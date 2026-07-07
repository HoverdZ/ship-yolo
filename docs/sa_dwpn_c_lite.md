# YOLO11n-SA-DWPN-C-lite

## Purpose

SA-DWPN-C-lite is a controlled extension of SA-DWPN-B. It keeps the same backbone, C2-guided P3 injection path, channel settings, and three-head Detect design, then enables spatial gate only at the two high-resolution P3-related fusion nodes.

## Difference From SA-DWPN-B

SA-DWPN-B uses five SDWF fusion nodes with `use_spatial=False` everywhere:

```text
T4 = SDWF([L4, Up(L5)], spatial off)
T3 = SDWF([L3, Up(T4), D2], spatial off)
O3 = SDWF([L3, T3, D2], spatial off)
O4 = SDWF([L4, T4, Down(O3)], spatial off)
O5 = SDWF([L5, Down(O4)], spatial off)
```

SA-DWPN-C-lite changes only:

```text
T3 = SDWF([L3, Up(T4), D2], spatial on)
O3 = SDWF([L3, T3, D2], spatial on)
```

All other SDWF nodes remain spatial off.

## Why Enable Spatial Gate Only At T3 And O3

T3 and O3 are the P3-resolution fusion nodes. They directly affect the feature map used for small-ship detection at stride 8. Spatial recalibration at these nodes can suppress local clutter such as waves, coastlines, harbor edges, and wake-like textures while preserving the main SA-DWPN-B topology.

## Why Not Enable T4, O4, And O5

T4, O4, and O5 operate at lower spatial resolutions and carry more semantic context. Enabling spatial gates there would change more of the feature hierarchy and make the ablation harder to interpret. C-lite intentionally keeps those nodes unchanged so the experiment isolates P3-level spatial filtering.

## Spatial Gate Stability

The spatial gate uses:

```text
feature
-> channel-wise average pooling
-> channel-wise max pooling
-> concatenate
-> 7x7 Conv
-> Sigmoid
-> residual spatial recalibration
```

The recalibration form is:

```python
x_out = x * (1.0 + eta * mask)
```

`eta` is a learnable parameter initialized to `0`, so the spatial gate starts as an identity-like mapping. This prevents C-lite from disrupting the validated B features at the beginning of training.

## Hypothesis

- Precision may improve by suppressing clutter-driven false positives.
- False detections from waves, wakes, and coastline textures may decrease.
- mAP50-95 may improve or remain stable.
- Recall may slightly decrease if the spatial gate suppresses weak small-ship cues.

## Required Training Consistency

SA-DWPN-C-lite must use the same dataset, image size, epochs, batch size, optimizer settings, augmentation settings, and evaluation protocol as SA-DWPN-B. The preferred initial weights are SA-DWPN-B `best.pt`, followed by a comparison against YOLO11n pretrained transfer if needed.

## Ablation Plan

- SA-DWPN-B
- SA-DWPN-C-lite
- SA-DWPN-C-full
- spatial gate only at T3
- spatial gate only at O3

## Commands

```bash
python tools/test_sa_dwpn_c_lite_build.py
python tools/test_sa_dwpn_c_lite_forward.py
python tools/test_sa_dwpn_c_lite_weight_transfer.py --weights yolo11n.pt
python tools/test_sa_dwpn_c_lite_weight_transfer.py --weights path/to/sa_dwpn_b_best.pt
python tools/train_sa_dwpn_c_lite_smoke.py --data /content/drive/MyDrive/ship_detection/data/data.yaml --weights path/to/sa_dwpn_b_best.pt
```
