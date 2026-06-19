# Experiment Record: YOLO11n-SA-DWPN-B

## Basic Info

Experiment Name: yolo11n_sa_dwpn_b_640
Model YAML: experiments/yolo11n_sa_dwpn_b.yaml
Resolution: 640
Detect Heads: 3 (P3/P4/P5)

## Motivation

SA-DWPN-B tests C2-guided P3 detail injection and scene-adaptive dynamic weighted fusion without adding a P2 detection head. The goal is to improve remote sensing small-ship detection while keeping the detection head count and inference cost controlled.

## Modification

- C2 detail branch: C2 -> L2 -> DWDown -> D2.
- Feature alignment: C3/C4/C5 are projected to L3/L4/L5.
- Top-down fusion: T4 = SDWF([L4, Up(L5)]), T3 = SDWF([L3, Up(T4), D2]).
- Output fusion: O3 = SDWF([L3, T3, D2]), O4 = SDWF([L4, T4, Down(O3)]), O5 = SDWF([L5, Down(O4)]).
- Detect remains Detect([O3, O4, O5]).

## Weight Transfer

Pretrained: TBD
Loaded/Total tensors: TBD
Skipped shape mismatch: TBD
Missing keys: TBD
Random initialized modules: SDWF / DWDown / alignment Conv / TBD

## Build Check

model.info(): TBD
FLOPs: TBD
dummy forward: TBD
SDWF modules: TBD
Detect heads: TBD

## Training Setup

Epoch: TBD
imgsz: 640
batch: TBD
device: TBD

## Smoke Train

1 epoch smoke train: TBD

## Results

Precision: TBD
Recall: TBD
mAP50: TBD
mAP50-95: TBD

## Analysis

TBD

## Conclusion

TBD

## Next Step

TBD
