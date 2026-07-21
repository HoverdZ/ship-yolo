# Experiment Record

Experiment Name: `yolo11n_inceptiondw_fapn_640`

Date: 2026-07-21

Baseline: Validated YOLO11n-InceptionDW backbone and official bottom-up PAN,
Ultralytics 8.4.92

Modification: Preserve existing InceptionDW layer 2/4 replacements exactly
and add the same original ICCV 2021 FaPN top-down graph as the FaPN-only
model. Detect remains P3/P4/P5 and `nc=1`.

Weight Transfer: Official `yolo11n.pt` with explicit semantic layer mapping

Loaded/Total: 397/483 state tensors; 196/264 parameter tensors

Loaded parameter elements: 2,372,272/2,912,495 (81.4515%)

Training:

- Epoch: 150 planned; not started
- imgsz: 640
- batch: 8
- workers: 2
- device: 0
- seed: 0
- optimizer: auto
- learning rate and augmentation: unchanged Ultralytics 8.4.92 defaults

Results:

- Precision: Not available; formal training not started
- Recall: Not available; formal training not started
- mAP50: Not available; formal training not started
- mAP50-95: Not available; formal training not started

Preflight:

- Parameters: 2,912,495
- Trainable parameters: 2,912,479
- GFLOPs: 8.983720
- 640 CPU forward: passed
- 256 CPU forward/backward: passed
- Detect(P3/P4/P5): passed
- Existing InceptionDW scope: passed
- FaPN count/initialization/channel audit: passed
- PAN topology audit: passed
- Formal training: intentionally not run
