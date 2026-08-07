# Experiment Record

Experiment Name: DFINE_N
Date: pending Colab execution

Baseline: Official Peterande/D-FINE Nano (HGNetv2-B0), pinned to commit
`7fe2f8889f0b7b817f20c315b40fc15a4fb64ae6`.

Modification: No detector modification. The frozen YOLO-HBB ship dataset is
deterministically converted to COCO with zero-based raw category IDs required
by the official custom-data loader. Input multiscale is disabled to keep the
training and evaluation size fixed at 640 × 640.

Weight Transfer: Official D-FINE-N COCO checkpoint (`dfine_n_coco.pth`).
Loaded/Total: pending runtime tensor audit

Training:
Epoch: 150
imgsz: 640
batch: 8
seed: 0

Results:
Precision: pending
Recall: pending
mAP50: pending
mAP50-95: pending
AP75: pending

Analysis: pending

Conclusion: pending

Next Step: Execute the formal Colab notebook, preserve the test split, and
populate only measured metrics and local complexity artifacts.
