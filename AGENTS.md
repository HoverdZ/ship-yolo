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
