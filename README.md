# ship-yolo

## Current screening experiment

`YOLO11n-InceptionDW-SCSharedHead` keeps the validated InceptionDW P2/P3
backbone change and replaces only the native P3/P4/P5 Detect head with a
scale-calibrated shared Conv-GN head. See
`docs/scshared_head_design.md` for the paper comparison, weight-transfer scope,
preflight checks, and staged 80/150-epoch protocol. Formal training has not
started.

## InceptionDW small-ship experiments: SPDDown and P2 Gaussian auxiliary loss

The `feature/spddown-p2-gaussian-aux` experiment suite keeps the validated
InceptionDW backbone and native YOLO11 Neck, then prepares two independent
single-variable ablations:

- targeted `SPDDown` only at C2-to-C3 downsampling;
- a training-only P2 Gaussian heatmap loss with native P3/P4/P5 inference.

See [the design and pre-training audit](docs/spddown_p2_gaussian_aux_design.md)
and `configs/spddown_p2aux_protocol.yaml`. No formal training is started by the
repository code or notebook setup cells.

The corrected P2-only Colab entrypoint is
`notebooks/YOLO11n_InceptionDW_P2GaussianAux_AMPStable.ipynb`. It uses a new
run directory and must not resume the earlier checkpoint whose auxiliary loss
became non-finite under FP16.

Remote Sensing Ship Detection based on YOLO11.

This repository manages reproducible research experiments for remote sensing ship detection using Ultralytics YOLO11. It is intended for long-term SCI paper experimentation, including baseline construction, model-structure improvement, ablation studies, experiment logging, and paper-related documentation.

## Research Goal

The main goal of this project is to support a high-quality SCI Q2/Q1 paper on remote sensing ship detection. The repository is organized around reproducibility, traceable experiment evolution, and maintainable model improvements rather than industrial deployment.

## Current Research Directions

- Improve small-object ship detection in remote sensing imagery.
- Enhance low-level detail preservation through P2 detection branches.
- Explore semantic guidance for shallow feature maps.
- Study lightweight multi-scale feature fusion.
- Evaluate dynamic detection heads and related attention mechanisms.
- Build clear ablation chains for SCI manuscript writing.

## Experiment Evolution Route

```text
YOLO11 Baseline
↓
P2 Detection Head
↓
SemanticGuideP2
↓
SemanticGuideP3
↓
WeightedAdd Fusion
↓
MSWPN-lite
↓
DyHead（待探索）
```

## Repository Structure

```text
ship-yolo/
├── README.md
├── AGENTS.md
├── experiments/
│   └── .gitkeep
├── custom_modules/
│   └── .gitkeep
├── logs/
│   └── .gitkeep
├── papers/
│   └── .gitkeep
└── docs/
    ├── .gitkeep
    └── experiment_template.md
```

### Directory Usage

- `experiments/`: Stores independent experiment YAML files, including baseline and ablation configurations.
- `custom_modules/`: Stores custom network modules, fusion blocks, detection heads, and other model modifications.
- `logs/`: Stores experiment records, training summaries, metric tables, and reproducibility notes.
- `papers/`: Stores paper drafts, figures, tables, and literature notes when needed.
- `docs/`: Stores reusable templates, research notes, and repository-level documentation.

## Experiment Management Rules

All experiments must be reproducible and traceable.

- Each new experiment must create an independent YAML file.
- Existing experiment YAML files must not be overwritten.
- Each experiment should have a matching experiment record based on `docs/experiment_template.md`.
- Every experiment record must include model changes, weight transfer status, core training settings, and final metrics.
- Historical experiment records must not be deleted.
- Custom modules must be placed under `custom_modules/`.
- Changes should follow the minimum-intrusion principle to keep the YOLO11 baseline easy to compare against.
- Ultralytics source code is not included at the current stage.

The current repository only manages:

- Experiment configurations (`*.yaml`)
- Custom modules
- Experiment logs and records
- Research and paper documentation

Full Ultralytics source integration will be considered later only if it becomes necessary for long-term research maintenance.

## Required Experiment Record Fields

Each experiment should record:

- Experiment name
- Modification details
- Whether pretrained weights are used
- Loaded/Total tensors
- Epoch
- Image size (`imgsz`)
- Batch size
- Final Precision
- Recall
- mAP50
- mAP50-95

## Roadmap

- Establish YOLO11 baseline results on the remote sensing ship dataset.
- Add and evaluate a P2 detection head for small ships.
- Introduce SemanticGuideP2 and compare with the P2-only variant.
- Extend semantic guidance to P3 and analyze feature-level effects.
- Evaluate WeightedAdd fusion against standard feature fusion.
- Develop and test MSWPN-lite for lightweight multi-scale enhancement.
- Explore DyHead and compare its cost-performance tradeoff.
- Build complete ablation tables and visualization materials for SCI manuscript submission.

## Current Planned Experiment: WAFPN-v1-640

The current planned experiment is `wafpn_v1_640`.

- This experiment keeps the standard three detection heads: P3, P4, and P5.
- It replaces the 4 Concat fusion nodes in the FPN/PAN Neck with static learnable WeightedAdd fusion.
- The goal is to verify whether weighted additive fusion is more suitable than original concatenation fusion for remote sensing small-ship detection.
- Only the pre-training engineering scaffold is complete at this stage. Training has not started.

## Current Planned Experiment: YOLO11n-SA-DWPN-B

SA-DWPN-B (`yolo11n_sa_dwpn_b_640`) is complete as the validated base variant for the SA-DWPN line.

- This experiment keeps three detection heads: P3, P4, and P5.
- It injects C2 detail into P3 through a downsampled branch instead of adding a P2 detection head.
- It replaces standard Neck fusion with SDWF, a scene-adaptive dynamic weighted fusion module.
- Spatial gating is implemented in code but disabled in the first YAML to keep the first ablation stable.
- B is the baseline variant for the SA-DWPN series; training metrics should be preserved in `logs/` for later comparison.

## Current Planned Experiment: YOLO11n-SA-DWPN-C-lite

The current added experiment is `yolo11n_sa_dwpn_c_lite_640`.

- C-lite is built strictly on SA-DWPN-B.
- It keeps Detect(O3, O4, O5), with three heads P3/P4/P5.
- It enables spatial gate only at the P3-related `T3` and `O3` SDWF nodes.
- It keeps spatial gate disabled at `T4`, `O4`, and `O5`.
- It does not add P2 Detect, DCNv3, DyHead, loss changes, or data-augmentation changes.

Recommended checks:

```bash
python tools/test_sa_dwpn_c_lite_build.py
python tools/test_sa_dwpn_c_lite_forward.py
python tools/test_sa_dwpn_c_lite_weight_transfer.py --weights yolo11n.pt
python tools/train_sa_dwpn_c_lite_smoke.py --data /content/drive/MyDrive/ship_detection/data/data.yaml --weights path/to/sa_dwpn_b_best.pt
```

## SA-DWPN Formal Ablations

Formal SA-DWPN ablations are governed by `configs/sa_dwpn_protocol.yaml`. The current variants are:

- `b`: all five SDWF spatial gates disabled.
- `c_lite`: spatial gates enabled at T3 and O3.
- `t3_only`: spatial gate enabled only at T3.
- `o3_only`: spatial gate enabled only at O3.

Use the unified training entrypoint:

```bash
python tools/train_sa_dwpn_variant.py --variant t3_only --data /content/ship_detection_local/data/data_local.yaml --weights /content/ship-yolo/yolo11n.pt --project /content/drive/MyDrive/ship_detection/runs
```

Analysis and archiving tools:

```bash
python tools/inspect_sa_dwpn_gates.py --weights path/to/best.pt --output artifacts/gate_analysis/c_lite
python tools/visualize_sa_dwpn_heatmaps.py --data /path/to/data.yaml --output artifacts/heatmaps/c_lite --model c_lite path/to/best.pt
python tools/export_run_artifacts.py --run-dir /content/drive/MyDrive/ship_detection/runs/yolo11n_sa_dwpn_c_lite_640 --destination results/sa_dwpn_c_lite_640_main
```

Heatmaps are auxiliary interpretation artifacts only; they do not prove causality by themselves.

## Prepared Experiment: YOLO11n-InceptionDW-C3k2-P23

The `experiment/inceptiondw-c3k2-p23` line prepares a formal shallow-backbone
ablation for Ultralytics 8.4.92. It replaces only backbone C3k2 layers 2 and 4;
P4/P5, SPPF, C2PSA, the full Neck, and Detect remain official YOLO11n.

The implementation uses the Apache-2.0 official InceptionNeXt
`InceptionDWConv2d` core with fixed 3x3, 1x11, and 11x1 depthwise branches.
After YOLO11n depth scaling, exactly two InceptionDW cores are present.

Pre-training checks:

```bash
python tools/build_inceptiondw_c3k2_p23.py
python tools/check_inceptiondw_c3k2_p23.py --weights yolo11n.pt
pytest -q tests/test_inceptiondw_c3k2_p23.py
```

Formal training entrypoint (training is not started by the preparation task):

```bash
python tools/train_inceptiondw_c3k2_p23.py \
  --data /path/to/ship_detection/data.yaml \
  --project /path/to/persistent/runs
```

See `docs/experiments/yolo11n_inceptiondw_c3k2_p23.md` for the exact scope,
weight-inheritance audit, model statistics, and safe resume command.

## Prepared Experiments: Original FaPN

Two ICCV 2021 FaPN experiments are prepared without starting formal training:

- `yolo11n_fapn_640`: official YOLO11n backbone plus original FaPN top-down.
- `yolo11n_inceptiondw_fapn_640`: the validated InceptionDW backbone plus the
  identical original FaPN top-down.

Both retain the YOLO11 bottom-up PAN topology and Detect(P3, P4, P5), use
`nc=1`, keep `deformable_groups=8`, and use Torchvision modulated DCNv2 rather
than vendoring the legacy CUDA extension.

```bash
python tools/check_fapn_models.py --weights yolo11n.pt
python tools/transfer_fapn_weights.py --weights yolo11n.pt
pytest -q tests/test_fapn.py
```

Formal Colab entrypoints:

```bash
python tools/train_yolo11n_fapn.py --data /content/drive/MyDrive/ship_detection/data/data.yaml
python tools/train_yolo11n_inceptiondw_fapn.py --data /content/drive/MyDrive/ship_detection/data/data.yaml
```

See `docs/fapn_porting_notes.md` for official-source mapping, topology,
initialization, semantic weight transfer, statistics, and resume behavior.

## Prepared Experiments: FaPN-Prefusion

`feature/fapn-prefusion` isolates FaPN's shallow feature selection and
high-feature DCNv2 alignment before the original YOLO11 fusion. Both official
Nearest Upsample nodes, both top-down Concat+C3k2 blocks, the complete PAN,
native T3/T4 channels, and Detect(P3,P4,P5) remain unchanged.

- `yolo11n_fapn_prefusion_640`: official YOLO11n backbone.
- `yolo11n_inceptiondw_fapn_prefusion_640`: the already validated InceptionDW
  replacements at backbone layers 2 and 4 only.

Neither formal 150-epoch experiment has been started. Prepare and audit the
real pretrained initialization checkpoints before training:

```bash
python tools/fapn_prefusion_profile.py --variant all --imgsz 640
python tools/prepare_fapn_prefusion_init.py --variant all --weights yolo11n.pt
python tools/check_fapn_prefusion_models.py --variant all --weights yolo11n.pt
python -m pytest tests/test_fapn_prefusion.py -q
```

See `docs/fapn_prefusion_design.md` for the exact ablation and
`docs/colab_fapn_prefusion.md` for the six-cell formal Colab workflow. Formal
training always starts from the manifested `*_pretrained_init.pt`, never from
the custom YAML.

## Prepared Experiments: ASCGD-Neck

Seven controlled neck ablations reuse the validated InceptionDW backbone
without changing its implementation or layer-2/layer-4 placement. Variant E
is the full asymmetric design: window spatial cross-attention distributes
public semantics to P3, while cross-covariance channel attention distributes
shallow geometry to P4 and P5. All variants retain Detect(P3,P4,P5).

```bash
python tools/build_ascgd_variants.py --variant all
python tools/check_ascgd.py --all
python -m pytest tests/test_ascgd.py -q
```

The first formal Colab run is intentionally E only:

```bash
python tools/train_ascgd_colab.py \
  --variant e_full \
  --data /content/datasets/ship/data.yaml \
  --project /content/drive/MyDrive/ship_detection/organized_experiments \
  --name yolo11n_incdw_ascgd_full_640
```

See `docs/ascgd_neck.md` for the exact A-G differences, initialization audit,
resume protection, L4 benchmark, and result-summary commands. Formal training
is not started during repository preparation.

## Prepared Experiments: BADC / SCG / SGTA

Three independent 80-epoch screens reuse the first 80 epochs of the existing
150-epoch InceptionDW run as the matched reference:

- `badc`: background-aware directional contrast at shallow P2/P3.
- `scg`: bounded P4 semantic confirmation replacing the native P3 concat.
- `sgta`: training-only scale-adaptive CIoU/NWD assignment and regression.

The complete `full` model is implemented and audited but blocked in the default
Colab notebook until the three independent screens are reviewed.

```bash
python tools/check_badc_scg_sgta.py --weights yolo11n.pt --imgsz 640
python -m pytest tests/test_badc_scg_sgta.py -q
python tools/build_badc_scg_sgta_colab.py
```

Training uses a 150-epoch scheduler, pauses after epoch 80, and preserves the
raw optimizer/scheduler/EMA state as `weights/stage80_resume.pt`. The optional
continuation resumes that checkpoint through epoch 150; it is not a new
70-epoch fine-tune. See `docs/badc_scg_sgta_design.md`.
