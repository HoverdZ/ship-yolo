# ship-yolo

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
