# Experiment Record

Experiment Name: `yolo11n_incdw_ascgd_base_640`

Date: 2026-07-23

Baseline: validated YOLO11n-InceptionDW backbone and original YOLO11 FPN/PAN.

Modification: ASCGD ablation A; no neck change. The model is the exact
structural baseline for the six ASCGD neck variants.

Pretrained Weights: official `yolo11n.pt`

Loaded/Total State Tensors: 446/511

Loaded/Total Parameter Elements: 2,540,240/2,585,119

Training:

- Epochs: 150 planned; not started
- imgsz: 640
- batch: 8
- workers: 2
- seed: 0

Results:

- Precision: not available
- Recall: not available
- mAP50: not available
- mAP50-95: not available

Status: CPU build/forward and exact-baseline equivalence passed. Formal
training was intentionally not started.
