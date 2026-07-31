# Ocean Engineering formal experiment framework

This framework prepares code and reproducibility artifacts without containing
trained results. Each generated Notebook uses seed 0 once and starts foreground
training immediately after all preflight checks pass.

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
2. Run the first operation cell. It mounts Drive, reads the private-repository
   credential only from the Colab Secret `GITHUB_TOKEN`, installs
   Ultralytics 8.4.92, and checks out the frozen Git commit.
3. Run the second operation cell. It copies the dataset with live file/byte
   progress, then checks the dataset, model, forward/backward path, stride,
   complexity, and official-weight inheritance. Training starts directly in
   the current kernel only after every check passes; there is no user-editable
   training switch or multi-seed loop.
4. Verify the printed RUN_ID, YAML, data YAML, official weight, fixed Git
   commit, effective parameters, and output directories as training begins.
5. If interrupted, rerun from the top. A matching `last.pt` and state file
   resume; inconsistent residuals are rejected.
6. Finalize validation, per-image evidence, manifest, checksums, Drive mirror,
   and ZIP.
7. The final operation cell refreshes all currently available result tables.
   Real-hook visualizations remain separate tools so representative images can
   be selected after completed runs are reviewed.

The test set is never used for model selection. It remains disabled by
default until the final evaluation protocol is explicitly approved.

## Paper skeleton and result insertion

`tools/build_experiment_manuscript_skeleton.py` creates synchronized Chinese
and English experiment chapters in editable Markdown and DOCX, plus an
English LaTeX version. All unknown measurements use exact
`{{PENDING_...}}` tokens. It never inserts guessed values or pre-authors a
directional result.

After formal runs are complete,
`tools/update_manuscript_experiment_results.py <run_root>` performs a
read-only preview by default. Only `--apply` modifies manuscripts, and every
changed file is copied to a timestamped backup first. The updater accepts
only identity-matched `run_manifest.json` files whose status is `completed`;
unavailable fields remain unresolved.

## Pre-push validation

Run the following before publishing experiment-framework changes:

```powershell
pytest -q
python tools/check_formal_experiment_models.py
python tools/validate_formal_experiment_notebooks.py
git diff --check
git status --short
```

The model check builds all 14 unique topologies with `nc=1`, explicitly
transfers matching official pretrained tensors, validates Detect strides,
and executes a finite CPU forward/backward pass. S00 and S01 reuse the
already-audited R00 and R10 topologies; their dataset validity is checked
separately after the external dataset is selected.
