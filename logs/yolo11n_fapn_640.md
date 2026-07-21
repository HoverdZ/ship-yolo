# Experiment Record

Experiment Name: `yolo11n_fapn_640`

Date: 2026-07-21

Baseline: Official YOLO11n backbone and bottom-up PAN, Ultralytics 8.4.92

Modification: Replace only the two-stage top-down YOLO FPN with the original
ICCV 2021 FaPN FSM/FAM recursion using modulated DCNv2. Detect remains
P3/P4/P5 and `nc=1`.

Weight Transfer: Official `yolo11n.pt` with explicit semantic layer mapping

Loaded/Total: 399/471 state tensors; 198/252 parameter tensors

Loaded parameter elements: 2,378,032/2,917,411 (81.5117%)

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

- Parameters: 2,917,411
- Trainable parameters: 2,917,395
- GFLOPs: 9.083816
- 640 CPU forward: passed
- 256 CPU forward/backward: passed
- Detect(P3/P4/P5): passed
- FaPN count/initialization/channel audit: passed
- PAN topology audit: passed
- Formal training: intentionally not run
