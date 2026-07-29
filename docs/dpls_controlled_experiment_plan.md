# DPLS controlled experiment plan

## Scope

This document plans three later training runs. This commit performs dataset
scale analysis and structure checks only; it does not start training, load a
trained checkpoint, or run YOLO validation.

The controlled factorization is:

- D0 → D1 isolates Pyramid Level Shift (PLS).
- D1 → D2 isolates DySample.
- D0 → D2 evaluates the complete Dynamic Pyramid Level Shift (DPLS).

No run contains InceptionDW, SCAM, CA-SCAM, VGUP, another attention block, or
a modified loss.

## D0: official YOLO11n

- Backbone: official YOLO11n.
- Detection pyramid: P3/P4/P5.
- Upsampling: nearest neighbor.
- Strides: `[8, 16, 32]`.
- YAML status: already present at
  `experiments/formal_ablation_v1/A0_yolo11n.yaml`.

## D1: YOLO11n + PLS

- Backbone: official YOLO11n shallow stages remain unchanged.
- Detection pyramid: P2/P3/P4.
- Upsampling: nearest neighbor.
- Strides: `[4, 8, 16]`.
- The P5 detection path and the no-longer-consumed P5 backbone stage are
  removed.
- YAML status: new planning draft at
  `experiments/dpls_controlled/D1_yolo11n_pls_nearest.yaml`.

## D2: YOLO11n + PLS + DySample

- Backbone: identical to D1.
- Detection pyramid: P2/P3/P4.
- Upsampling: DySample at P4→P3 and P3→P2.
- Strides: `[4, 8, 16]`.
- YAML status: new planning draft at
  `experiments/dpls_controlled/D2_yolo11n_pls_dysample.yaml`.

## Fixed training protocol for the later runs

The later formal protocol must hold constant:

- the same official `yolo11n.pt` initialization policy and audited
  Loaded/Total tensor reporting;
- the same frozen dataset and train/val/test partition;
- Ultralytics 8.4.92;
- `imgsz=640`;
- batch, epochs, seed, augmentation, optimizer, data-loader settings, and
  best-model selection rule;
- validation mAP50–95 as the model-selection metric;
- each run's own validation-best `best.pt` for final reporting.

The exact batch, epoch, optimizer, and augmentation values must be copied from
one versioned formal protocol when training is authorized. They are not
defined or changed by this planning commit.

## Reserved result table

| Model | Pyramid | Upsampling | P | R | mAP50 | mAP50–95 | Params | GFLOPs |
|---|---|---|---:|---:|---:|---:|---:|---:|
| D0 | P3–P5 | Nearest | pending | pending | pending | pending | pending | pending |
| D1 | P2–P4 | Nearest | pending | pending | pending | pending | pending | pending |
| D2 | P2–P4 | DySample | pending | pending | pending | pending | pending | pending |

## Scale-stratified evaluation

The initial reporting strata are defined using horizontal annotation-box
short sides after deterministic 640-pixel letterbox mapping:

- `<8 px`;
- `8–16 px`;
- `≥16 px`.

Before evaluation, inspect the generated
`analysis/ship_scale/raw_tables/short_side_stride_bins.csv`. If one stratum
contains too few instances for stable reporting, merge adjacent intervals and
record the rule before reading model results. Do not choose intervals using
test-set performance.

The frozen-data analysis in this branch finds only 10/1/0 instances below
8 px in train/val/test. The originally proposed `<8`, `8–16`, `≥16` reporting
would therefore contain an empty test stratum. The recommended pre-registered
replacement is:

- `<16 px` (train/val/test: 373/132/114);
- `16–32 px` (1641/497/447);
- `≥32 px` (179/59/47).

This merge is based only on annotation-scale counts and is defined before
model results are read.

## Structure validation

Run:

```powershell
python tools/check_dpls_controlled_models.py
```

The checker builds random-weight structures only and verifies:

- D0 strides `[8,16,32]` with two nearest upsamplers;
- D1 strides `[4,8,16]` with two nearest upsamplers;
- D2 strides `[4,8,16]` with two DySample modules;
- D1 and D2 share the same backbone and differ only in upsampling modules;
- none of D0–D2 contains InceptionDW, SCAM, CA-SCAM, or VGUP.

It does not train or validate a detector.
