# Experiment Record

Experiment Name: `yolo11n_inceptiondw_fapn_prefusion_640`

Date: 2026-07-22

Status: pre-training preparation only; formal training not started

Baseline: validated YOLO11n-InceptionDW backbone layers 2/4, with official
YOLO11n as the only pretrained source; Ultralytics 8.4.92, `nc=1`.

Modification: retain the validated InceptionDW implementation unchanged and
insert the same FaPN-Prefusion neck as the baseline variant. PAN and P3/P4/P5
Detect remain official.

Weight Transfer: official `yolo11n.pt` only; no trained InceptionDW `best.pt`

Loaded/Total state tensors: 446/531

Loaded/Total parameter tensors: 221/288

Loaded/Total parameter elements: 2,540,240/2,928,339 (86.7468%)

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
