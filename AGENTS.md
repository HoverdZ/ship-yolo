# AGENTS.md

This repository is for SCI paper experiments on remote sensing ship detection, not for industrial deployment.

## Repository Objective

The project manages experiments based on Ultralytics YOLO11 for remote sensing ship detection. The main priorities are reproducibility, clear ablation logic, long-term maintainability, and paper-ready experiment records.

## Working Rules for Codex

- All experiments must be reproducible.
- Every new experiment must create an independent YAML file.
- Do not overwrite existing experiments.
- Keep the paper-facing `main` branch limited to configurations and analyses reported in the manuscript. Historical exploration is retained on `archive/experimental-exploration`.
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

## Formal Experiment Protocol

- Formal ablations must follow `experiments/formal_training_config.yaml` and `experiments/formal_experiment_registry.yaml` unless a documented protocol revision is created.
- Formal models must start from the same official `yolo11n.pt` initialization.
- Except for the target variable, do not change structure, loss, optimizer policy, augmentation, image size, batch size, epoch count, dataset split, or Ultralytics version.
- New YAML files must include structural-difference tests.
- New training scripts must support safe resume.
- If resume validation fails, raise an error; never fall back to `coco8.yaml` or default training.
- Training scripts must avoid blocking themselves after pre-training validation. A run directory may be reused only when it contains no files or only `protocol.yaml` and `resolved_args.json`; once training artifacts exist, require valid resume or explicit `--exist-ok`.
- Do not use temporary Colab monkey patches for Ultralytics source changes; register custom modules through `custom_modules/register.py`.
- Do not commit caches or `runs/`. Paper-release checkpoints are the only exception and must be stored in `paper_artifacts/model_weights/` through Git LFS.
- Every experiment must preserve `args.yaml`, `results.csv`, `summary.json`, and an experiment note when available.
- Do not fabricate missing experiment data.
- After code changes, run available tests and report skipped items honestly in the PR.
