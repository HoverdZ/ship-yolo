# AGENTS.md

This repository is for SCI paper experiments on remote sensing ship detection, not for industrial deployment.

## Repository Objective

The project manages experiments based on Ultralytics YOLO11 for remote sensing ship detection. The main priorities are reproducibility, clear ablation logic, long-term maintainability, and paper-ready experiment records.

## Working Rules for Codex

- All experiments must be reproducible.
- Every new experiment must create an independent YAML file.
- Do not overwrite existing experiments.
- Do not delete historical experiment records.
- New custom network modules must be placed in `custom_modules/`.
- Keep modifications minimally invasive whenever possible.
- Avoid uploading or vendoring Ultralytics source code at the current stage.
- Manage only experiment configurations, custom modules, logs, and documentation unless explicitly instructed otherwise.

## Required Experiment Record Fields

Every experiment must record:

- Experiment name
- Modification details
- Whether pretrained weights are used
- Loaded/Total tensors
- Epoch
- imgsz
- batch size
- Final Precision
- Recall
- mAP50
- mAP50-95

## Experiment File Policy

- Use a new YAML file for each baseline, improvement, or ablation experiment.
- File names should be descriptive and stable.
- Existing YAML files are historical records and must not be overwritten.
- If an experiment needs revision, create a new versioned YAML file instead of modifying the previous one.

## Module Policy

- Place all custom modules in `custom_modules/`.
- Keep module boundaries clear so each paper contribution can be isolated and ablated.
- Prefer small, reviewable changes over broad framework edits.

## Logging Policy

- Store experiment records in `logs/` using the template in `docs/experiment_template.md`.
- Record both successful and failed experiments when they provide useful ablation evidence.
- Preserve all historical logs and metric records.

## WAFPN Experiment Rules

- Any WAFPN experiment must pass structure checks before training.
- Do not start training without a successful `model.info()` and dummy forward pass.
- Always record weight transfer as Loaded/Total tensors.
- Keep three detection heads unless the experiment name explicitly contains `P2` or `4head`.
- WAFPN-v1 uses static learnable weights only. Do not mix in dynamic weights, sigmoid attention, spatial attention, DyHead, DCNv3, or full MSWPN behavior.

## SA-DWPN Experiment Rules

- Any SA-DWPN experiment must pass build, `model.info()`, dummy forward, and weight-transfer checks before full training.
- Keep Detect(P3, P4, P5) unless the experiment name explicitly contains `P2` or `4head`.
- C2 can guide P3 only through a downsampled branch; it must not connect directly to Detect in SA-DWPN-B.
- Do not enable spatial gating in the first SA-DWPN-B YAML.
- Do not add DCNv3, Transformer blocks, loss changes, or data augmentation changes to the first SA-DWPN-B experiment.
- Record smoke-train status before starting long training.

## SA-DWPN-C-lite Rules

- C-lite may differ from B only by enabling spatial gate at T3 and O3.
- T4, O4, and O5 must keep `use_spatial=False`.
- B YAML must remain unchanged and must continue to build with zero spatial gates enabled.
- C-lite must pass build, forward, and weight-transfer checks before smoke training.
- Prefer SA-DWPN-B `best.pt` as the initial weight source for C-lite, then compare against YOLO11n pretrained transfer if needed.
- Do not start long training until the 1-epoch smoke run succeeds.

## Formal SA-DWPN Ablation Protocol

- Formal ablations must follow `configs/sa_dwpn_protocol.yaml` unless a documented protocol revision is created.
- Formal models must start from the same official `yolo11n.pt` initialization.
- Except for the target variable, do not change structure, loss, optimizer policy, augmentation, image size, batch size, epoch count, dataset split, or Ultralytics version.
- New YAML files must include structural-difference tests.
- New training scripts must support safe resume.
- If resume validation fails, raise an error; never fall back to `coco8.yaml` or default training.
- Do not use temporary Colab monkey patches for Ultralytics source changes; register custom modules through `custom_modules/register.py`.
- Do not commit datasets, caches, `runs/`, ordinary `.pt` checkpoints, or exported model weights.
- Every experiment must preserve `args.yaml`, `results.csv`, `summary.json`, and an experiment note when available.
- Do not fabricate missing experiment data.
- After code changes, run available tests and report skipped items honestly in the PR.
