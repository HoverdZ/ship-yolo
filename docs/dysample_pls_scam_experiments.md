# YOLO11n cumulative DySample / PLS / SCAM experiments

## Scope

This preparation adds three independent cumulative models for low-cost
small-ship detection screening. No formal training or accuracy result is
claimed in this change. Every future run starts from the same official
`yolo11n.pt`; one experiment's `best.pt` must never initialize another.

## Sources and original contexts

| Component | Paper / original context | Official implementation |
| --- | --- | --- |
| InceptionDW | InceptionNeXt (CVPR 2024), an Inception-style ConvNeXt backbone | [sail-sg/inceptionnext](https://github.com/sail-sg/inceptionnext), commit `3f9769c6b3fcf903d1dc2f436eddc6963cacb535` |
| DySample | *Learning to Upsample by Learning to Sample* (ICCV 2023), a generic feature upsampler | [tiny-smart/dysample](https://github.com/tiny-smart/dysample), commit `81a1de5caa95d55a0f5488425fa53ec7ef47f8f0` |
| PLS | LiM-YOLO P2/P3/P4 level-shift idea in its YOLOv9-E/OBB-oriented model | [egshkim/LiM-YOLO](https://github.com/egshkim/LiM-YOLO), commit `034ce444e1a08c839590ee0f806c3feb46e2f682` |
| SCAM | FFCA-YOLO small-object detector for remote-sensing imagery | [yemu1138178251/FFCA-YOLO](https://github.com/yemu1138178251/FFCA-YOLO), commit `874a00da12266b4ee1abc3b6494c193972488956` |

## Adaptation boundaries

- InceptionDW reuses the already validated repository implementation only at
  backbone P2/P3 (`model.2` and `model.4`). The first ordinary 3x3 convolution
  inside each Bottleneck remains intact; InceptionDW replaces the later spatial
  operation.
- DySample follows `tiny-smart/dysample` commit
  `81a1de5caa95d55a0f5488425fa53ec7ef47f8f0` with `scale=2`,
  `style=lp`, `groups=4`, and `dyscope=False`. It is channel preserving.
- Pyramid Level Shift removes the P5 backbone/detection path and moves Detect
  from P3/P4/P5 to P2/P3/P4. YOLO11 SPPF, C2PSA, Detect, and DFL remain.
  LiM-YOLO's YOLOv9-E, OBB, GN-CBLinear, and reversible branches are not
  transferred.
- SCAM follows `yemu1138178251/FFCA-YOLO` commit
  `874a00da12266b4ee1abc3b6494c193972488956`. Three independent blocks are
  placed immediately before Detect. Its no-BN `m` projection, channel/spatial
  matrix multiplications, multiplicative gate, and `x + y` residual are kept;
  no channel reduction is added.

## Independent configurations

| Experiment | YAML | Detect levels | Strides |
| --- | --- | --- | --- |
| InceptionDW + DySample | `experiments/yolo11n_inceptiondw_dysample.yaml` | P3/P4/P5 | 8/16/32 |
| + PLS | `experiments/yolo11n_inceptiondw_dysample_pls.yaml` | P2/P3/P4 | 4/8/16 |
| + PLS + SCAM | `experiments/yolo11n_inceptiondw_dysample_pls_scam.yaml` | P2/P3/P4 | 4/8/16 |

## CPU statistics and official-weight inheritance

The following values were generated locally with Ultralytics 8.4.92 and
official `yolo11n.pt`. Inheritance is exact-name plus exact-shape; every loaded
tensor is verified after `load_state_dict`, and all unmatched keys remain in
the JSON audit instead of being hidden.

| Experiment | Parameters | GFLOPs at 640 | Loaded / total state tensors | Tensor ratio | Parameter-element ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| InceptionDW + DySample | 2,631,516 | 6.5314 | 497 / 517 | 96.13% | 99.50% |
| + PLS | 1,295,004 | 14.9771 | 126 / 457 | 27.57% | 23.46% |
| + PLS + SCAM | 1,369,710 | 15.5114 | 126 / 514 | 24.51% | 22.18% |

PLS intentionally changes layer numbering and topology, so exact-name
inheritance is much lower than experiment one. This is reported rather than
silently compensated with semantic remapping, because the task requires
name-and-shape matching against the same official checkpoint.

## Baseline-controlled training settings

Formal runs use the real baseline recipe: 150 epochs, image size 640, batch 8,
workers 2, seed 0, deterministic mode, optimizer `auto`, patience 150,
`cache=ram`, and `close_mosaic=10`. All remaining augmentation and loss
arguments are explicitly pinned in `tools/train_cumulative_models.py`.

Run exactly one experiment:

```powershell
python tools/train_cumulative_models.py `
  --experiment incdw_dysample `
  --data C:\path\to\data.yaml `
  --weights yolo11n.pt `
  --name yolo11n_incdw_dysample_640
```

The entrypoint builds the selected YAML, copies only exact-name/exact-shape
tensors from official `yolo11n.pt`, writes `inheritance_report.json`, saves and
reloads the initialization checkpoint, and verifies the Trainer did not
discard it before training starts.

## Validation

Run:

```powershell
pytest -q
python tools/check_cumulative_models.py --weights yolo11n.pt --imgsz 640 --try-1024 --print-network
git diff --check
git status
```

The checker records parameters, GFLOPs, model structure, Detect input shapes,
CPU forward/backward, state-dict roundtrip, optional 1024 forward, compatibility
with native YOLO11n and the existing InceptionDW model, and official-weight
inheritance. Generated evidence is stored under
`reports/cumulative_models/`.

All three 640 forwards, minimal CPU backwards, strict state-dict roundtrips,
and optional 1024 shape forwards passed. Native YOLO11n and the existing
InceptionDW YAML also still build with strides 8/16/32.

The Google Colab workflow is
`colab/YOLO11n_Cumulative_DySample_PLS_SCAM.ipynb`. It installs
`ultralytics==8.4.92`, securely clones this private branch using a Colab
secret, copies Drive data with 16 concurrent `shutil.copyfile` workers and
file/byte progress, runs the preflight, and leaves formal training disabled by
default.
