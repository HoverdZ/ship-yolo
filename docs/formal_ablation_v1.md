# Formal cumulative ablation v1

This workflow prepares six independent Colab runs for the paper:

| ID | Structure | Detect levels | YAML |
|---|---|---|---|
| A0 | YOLO11n | P3/P4/P5 | `experiments/formal_ablation_v1/A0_yolo11n.yaml` |
| A1 | A0 + shallow InceptionDW | P3/P4/P5 | `A1_inceptiondw.yaml` |
| A2 | A1 + DPLS | P2/P3/P4 | `A2_inceptiondw_dpls.yaml` |
| A3 | A2 + three SCAM blocks | P2/P3/P4 | `A3_inceptiondw_dpls_scam.yaml` |
| A4 | A3 + complete VGUP | P2/P3/P4 | `A4_inceptiondw_dpls_scam_vgup.yaml` |
| A5 | A4 with CA-SCAM replacing SCAM | P2/P3/P4 | `A5_inceptiondw_dpls_ca_scam_vgup.yaml` |

InceptionDW is limited to the P2/P3 backbone C3k2 blocks. Its bottleneck
retains the first standard convolution and replaces only the second spatial
convolution. DPLS is the already-tested DySample plus pyramid-level shift:
P5 detection and its redundant backbone stage are removed, and Detect consumes
strides 4, 8, and 16. A5 contains CA-SCAM only; it does not stack SCAM and
CA-SCAM.

## Run sequence

1. Open one notebook under `colab/formal_ablation_v1/`.
2. In the first code cell, normally edit only `DRIVE_DATA_YAML`. Leave
   `DRIVE_DATA_ROOT=None` unless automatic resolution cannot locate the data.
3. Add a read-only `GITHUB_TOKEN` to Colab Secrets and enable notebook access.
   The token is read at runtime, never printed or stored in the repository.
4. Select a GPU runtime and run cells from top to bottom.
5. Verify dataset audit, `Loaded/Total tensors`, model structure, and the CPU
   forward/backward smoke test before the formal training cell starts.
6. Keep `RUN_TEST_EVALUATION=False` for the six ablations. Enable it only after
   the final structure is selected using validation mAP50-95.

Training uses the official `YOLO.train(...)` API in the active notebook
process. No child process wraps training, so Ultralytics epoch logs remain
visible. `RUN_TRAINING=True` by default.

## Dataset and fairness

The Drive dataset is read-only. The runner resolves absolute and relative YAML
paths and falls back to the YAML parent when a historical `/content/...` path
does not exist in the new account. It audits image, label, empty-label,
instance, invalid-row, and class counts for train/val/test. It never cleans
labels or modifies splits.

The full resolved dataset is copied independently to Colab local storage using
16 `shutil.copyfile` threads. Two progress bars report completed files and
bytes. Source/destination file sizes are checked, and a new local YAML points
to `/content/datasets/ship_clean_v1`.

All six experiments use Python 3.12.x, Ultralytics 8.4.92, 150 epochs,
640 pixels, batch 8, workers 2, seed 0, `cache=disk`,
`deterministic=False`, plots, and ten-epoch checkpoints. Remaining
Ultralytics arguments use the same 8.4.92 defaults.

Every model is initialized only from the same official `yolo11n.pt`. A1–A3
use exact-name/exact-shape matching. A4/A5 insert an RGB preprocessor at layer
0, so their official source keys use an audited deterministic one-layer index
shift before the same shape check. Only tensors that actually originate in
the official checkpoint count as inherited. Newly introduced module
parameters remain randomly initialized. No trained ablation checkpoint is
used to initialize another ablation.

## Mirroring and recovery

Local runs are `/content/formal_runs/<EXP_ID>` and Drive mirrors are
`MyDrive/ShipPaper/formal_ablation_v1/<EXP_ID>`. At each epoch and checkpoint,
the callback snapshots `results.csv`, `args.yaml`, the console log, state,
`last.pt`, `best.pt`, and periodic `epoch*.pt`. A same-process worker writes
snapshots to temporary Drive files and atomically replaces the destination.
Mirror failures are printed but do not silently terminate training.

`RUNNING.lock`, `experiment_state.json`, `FAILED.json`, and `COMPLETED.ok`
describe state. A completed run is never overwritten. An incomplete run can
resume only from its own `weights/last.pt`, with the experiment ID verified.
Cross-experiment resume is rejected.

## Paper artifacts

Every completed run contains validation metrics including AP75, a complexity
and latency report, environment records, best-epoch summary, per-image
predictions, per-image TP/FP/FN, deterministic visual selection, detections,
errors, heatmaps, model-specific statistics, `run_manifest.json`, and SHA-256
checksums. Grad-CAM and Grad-CAM++ failures are recorded and fall back to a
feature-energy heatmap rather than fabricated evidence.

The A4 report includes per-image VGUP global/spatial gate and BPW/KBL parameter
statistics plus original/BPW/gated-BPW/KBL/final images. A5 adds CA-SCAM beta,
contrast, and residual statistics. Large validation-wide feature tensors are
never saved.

Run the cross-experiment tools after all six packages are collected:

```powershell
python tools/paper_artifacts/collect_run_artifacts.py <root>
python tools/paper_artifacts/generate_paper_tables.py <root>
python tools/paper_artifacts/select_visual_examples.py <root>
```

The result includes CSV, XLSX, Markdown, and LaTeX tables plus cumulative and
accuracy/complexity plots. The source validation split—not test—is used for
selection.

## Validation

```powershell
pytest -q
python tools/validate_formal_notebooks.py
git diff --check
git status --short
```

No 150-epoch run is started by these checks.
