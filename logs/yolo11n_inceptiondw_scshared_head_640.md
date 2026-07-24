# Experiment Record

Experiment Name: yolo11n_inceptiondw_scshared_head_640
Date: prepared 2026-07-24; formal training not started

Baseline: yolo11n_inceptiondw_c3k2_p23_640

Modification:
- Keep C3k2_InceptionDW only at backbone layers 2 and 4.
- Keep the native YOLO11 neck and Detect inputs P3/P4/P5.
- Replace native Detect with SCSharedDetect.
- Use per-level 1x1 adapters, two shared 3x3 Conv-GN-SiLU blocks,
  independent positive scale calibration, and separate DFL/class outputs.
- Do not add P2, DCNv2, attention, neck changes, or loss changes.

Weight Transfer:
Loaded/Total: to be copied from the generated initialization manifest
Mapped native Detect outputs: to be copied from the generated initialization manifest

Training:
Epoch: planned 80 screening + optional resume to 150 total
imgsz: 640
batch: 8

Results:
Precision: not trained
Recall: not trained
mAP50: not trained
mAP50-95: not trained

Analysis:
Training has not started. Do not treat preflight forward checks as experimental results.

Conclusion:
Pending the 80-epoch screening curve.

Next Step:
Run the Colab preflight and the 80-epoch stage, then compare with the existing
InceptionDW first-80-epoch curve before enabling continuation.
