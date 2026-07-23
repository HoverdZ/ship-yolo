# Experiment Record

Experiment Name: `yolo11n_incdw_ascgd_swap_640`

Date: 2026-07-23

Baseline: validated YOLO11n-InceptionDW backbone.

Modification: ASCGD ablation F; E's attention responsibilities are swapped.
P3 uses channel cross-attention, while aligned low-to-high P4/P5 paths use
window spatial cross-attention.

Pretrained Weights: official `yolo11n.pt`

Loaded/Total State Tensors: 308/584

Loaded/Total Parameter Elements: 1,746,544/3,072,864

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

Status: CPU build/forward passed. Formal training was intentionally not
started.
