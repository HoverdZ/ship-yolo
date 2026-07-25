# Experiment Record

Experiment Name: `yolo11n-c3cross-p23`

Baseline: `yolo11n_baseline_recheck_640`

Modification:

- Preserve ordinary `3x3, stride=2 Conv` at backbone layers 1 and 3.
- Replace only the following P2/P3 C3k2 blocks at layers 2 and 4 with
  `C3k2CrossConv`.
- Keep P4/P5, Neck, and Detect unchanged.

Weight Transfer:

- Pretrained weights used: yes, hybrid initialization.
- Baseline `best.pt`: all ordinary exact-name, exact-shape tensors.
- Full C3Cross `best.pt`: complete `model.2.*` and `model.4.*` override.
- Loaded/Total: recorded by `hybrid_initialization_audit.json` at runtime.

Training:

- Epoch: 30 maximum, with epoch-15 gate
- imgsz: 640
- batch: 8
- optimizer: AdamW
- seed: 0

Results:

- Precision: pending
- Recall: pending
- mAP50: pending
- AP75: pending
- mAP50-95: pending

Analysis:

- This 30-epoch hybrid-initialized run is a screening experiment.
- Promotion requires mAP50-95 >= 0.324, Recall >= 0.705, and mAP50 >= 0.770.

Conclusion: pending

Next Step:

- If promoted, run one 20-epoch low-LR fine-tune.
- Otherwise abandon C3Cross and do not run another 150-epoch experiment.
