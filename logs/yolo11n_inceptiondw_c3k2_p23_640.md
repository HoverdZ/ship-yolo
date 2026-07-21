# Experiment Record

Experiment Name: `yolo11n_inceptiondw_c3k2_p23_640`

Date: 2026-07-21

Baseline: Official YOLO11n, `ultralytics==8.4.92`

Modification: Replace only backbone layers 2 and 4 with
`C3k2_InceptionDW`. Within each depth-scaled target bottleneck, retain the
first 3x3 Ultralytics Conv and residual, then use the required channel adapter
and `InceptionDWConv2d -> BatchNorm2d -> SiLU` for the second spatial
operator. All P4/P5, Neck, and Detect modules remain official.

Weight Transfer: Yes, from official `yolo11n.pt`

Loaded/Total: 497/511 state tensors

Loaded parameter elements: 2,618,320/2,619,164 (99.9678%)

Training:

- Epoch: 150 planned; not started
- imgsz: 640
- batch: 8
- workers: 2
- device: 0
- seed: 0
- optimizer: auto

Results:

- Precision: Not available; formal training not started
- Recall: Not available; formal training not started
- mAP50: Not available; formal training not started
- mAP50-95: Not available; formal training not started

Pre-training checks:

- YAML build: passed
- 640x640 CPU forward: passed
- P3/P4/P5 Detect input shapes: passed
- Finite output check: passed
- Official feature-shape parity: passed
- Scoped module/count checks: passed
- Official implementation consistency (16/32/64 channels): passed
- Weight-transfer audit: passed
- SCConv/SRU/CRU absence: passed

Model comparison:

| Metric | Official YOLO11n | InceptionDW P23 | Delta |
|---|---:|---:|---:|
| Layers | 182 | 190 | +8 |
| Parameters | 2,624,080 | 2,619,164 | -4,916 |
| Trainable parameters | 2,624,064 | 2,619,148 | -4,916 |
| GFLOPs | 6.614336 | 6.514240 | -0.100096 |
| FP32 parameter size (MiB) | 10.010071 | 9.991318 | -0.018753 |

Analysis: Engineering preflight confirms the intended architecture and safe
official-weight inheritance. No accuracy conclusion can be drawn without the
formal run.

Conclusion: Code is ready for the next-round Colab formal training command;
training was intentionally not started in this task.

Next Step: Run the documented 150-epoch command on the intended ship dataset
and preserve `args.yaml`, `results.csv`, `summary.json`, checkpoints, and this
experiment record.
