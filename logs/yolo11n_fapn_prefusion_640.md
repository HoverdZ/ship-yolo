# Experiment Record

Experiment Name: `yolo11n_fapn_prefusion_640`

Date: 2026-07-22

Status: pre-training preparation only; formal training not started

Baseline: official YOLO11n, Ultralytics 8.4.92, `nc=1`

Modification: insert channel-preserving FaPN FSM and depthwise DCNv2
AlignmentOnly before each original YOLO top-down Concat; retain nearest
upsample, top-down C3k2, PAN, and P3/P4/P5 Detect.

Weight Transfer: official `yolo11n.pt` only

Loaded/Total state tensors: 448/519

Loaded/Total parameter tensors: 223/276

Loaded/Total parameter elements: 2,546,000/2,933,255 (86.7978%)

Training: not started

Planned Epoch: 150

imgsz: 640

batch: 8

Results:

Precision: not available

Recall: not available

mAP50: not available

mAP50-95: not available

Analysis: CPU structure, 640 forward, 256 backward, identity initialization,
weight-transfer, checkpoint reload, manifest, and Trainer corruption tests pass.

Conclusion: ready for the documented Colab GPU preflight and formal run.

Next Step: run CUDA AMP probe, then Cell 5 without changing the protocol.
