# Ocean Engineering formal experiment framework

This framework prepares code and reproducibility artifacts only. It does not
contain trained results and does not start training by default.

## Audited structure facts

- Official YOLO11n detects P3/P4/P5 with strides 8/16/32.
- PLS truncates the deep P5 path, builds a P2/P3/P4 detector, and uses two
  nearest-neighbor upsamplers.
- DPLS keeps the same P2/P3/P4 topology and changes only those two
  upsamplers to DySample.
- SCAM or CA-SCAM is instantiated independently on the final P2, P3, and P4
  features immediately before Detect.
- VGUP is model layer 0. Its global scalar gate controls acceptance of the
  BPW residual:
  `I_bpw = I + g_global * (BPW(I) - I)`.
- Its spatial gate controls the KBL residual at each pixel:
  `I_out = I_bpw + G_spatial * (KBL(I_bpw) - I_bpw)`.
- A removed gate uses an effective value of one while the corresponding
  BPW/KBL operator stays active. Gate ablations therefore test gating, not
  simultaneous operator deletion.
- The current ERUP input module has 6,781,042 trainable parameters. Complete
  VGUP has 77,396, a ratio of 0.0114135851 (1.1414%). The reproducible
  calculation is in `docs/experiments/erup_vgup_parameter_audit.csv`.
- The YOLOv8 adaptation retains its native C2f blocks and channel/depth
  scaling. It does not transplant YOLO11 C3k2 or C2PSA indices.

## CA-SCAM internal ablation

The variants follow the actual computation rather than adding unrelated
modules:

1. CI0/R03: original SCAM context residual.
2. CI1/R05A: local contrast, spatial projection, and fixed beta=0.1.
3. CI2/R05B: the same contrast path with a learnable unbounded beta,
   initialized at zero.
4. CI3/R04: the complete method with
   `beta = 0.1 * tanh(contrast_logit)`.

The learnable CI2 and CI3 variants are exact SCAM at initialization because
beta starts at zero. CI1 intentionally applies the fixed calibration from
the first update. Comparing CI2 with CI3 isolates the effect of bounding the
learnable residual amplitude.

## Official initialization

Every topology starts independently from its matching official checkpoint
(`yolo11n.pt` or `yolov8n.pt`). No previous ablation `best.pt` is accepted.
Topology changes make raw layer-number matching unsafe, so the protocol uses
an explicit source-layer map plus shape checks. Detect branches have an
explicit channel-compatible mapping. Every loaded target/source tensor key
is saved in `pretrained_transfer_report.json`.

Ultralytics 8.4.92 requires a truthy in-memory checkpoint marker and
`pretrained=True` to hand the already initialized model to DetectionTrainer.
This is not a second download/override: before epoch 1, the callback compares
every trainer tensor with the audited state and aborts on any difference.

## Execution order

1. Open the Notebook for a canonical run under `notebooks/formal/`.
2. Keep `RUN_TRAINING=False` through environment, dataset copy, model build,
   forward/backward, stride, complexity, and inheritance checks.
3. Verify the printed RUN_ID, YAML, data YAML, official weight, fixed Git
   commit, parameters, and output directories.
4. Set `RUN_TRAINING=True` and execute the foreground training cell.
5. If interrupted, rerun from the top. A matching `last.pt` and state file
   resume; inconsistent residuals are rejected.
6. Finalize validation, per-image evidence, manifest, checksums, Drive mirror,
   and ZIP.
7. Generate tables and real-hook visualizations. Human review makes the final
   representative-image selection.

The test set is never used for model selection. It remains disabled by
default until the final evaluation protocol is explicitly approved.
