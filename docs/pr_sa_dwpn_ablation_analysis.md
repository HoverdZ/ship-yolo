# PR: Add SA-DWPN ablation, gate analysis, and heatmap tools

## Summary

This PR adds formal SA-DWPN ablation infrastructure on a feature branch without changing the existing B and C-lite semantics.

## New Files

- `configs/sa_dwpn_protocol.yaml`
- `experiments/yolo11n_sa_dwpn_c_t3_only.yaml`
- `experiments/yolo11n_sa_dwpn_c_o3_only.yaml`
- `tools/sa_dwpn_utils.py`
- `tools/train_sa_dwpn_variant.py`
- `tools/inspect_sa_dwpn_gates.py`
- `tools/visualize_sa_dwpn_heatmaps.py`
- `tools/export_run_artifacts.py`
- `analysis/gradcam_targets.py`
- `analysis/detection_matching.py`
- `analysis/gate_visualization.py`
- `tests/test_sa_dwpn_static.py`
- `results/sa_dwpn_c_lite_640_main/summary.json`
- `docs/experiments/sa_dwpn_c_lite_640_main.md`
- `requirements-viz.txt`

## Modified Files

- `.gitignore`
- `AGENTS.md`
- `README.md`
- `custom_modules/register.py`
- `tools/test_sa_dwpn_c_lite_forward.py`
- `tools/test_sa_dwpn_c_lite_weight_transfer.py`

## Structural Differences

- B: spatial positions `[]`
- C-lite: spatial positions `[2, 3]`
- T3-only: spatial positions `[2]`
- O3-only: spatial positions `[3]`

All four variants keep 5 SDWF nodes, 1 DWDown node, and Detect input `[22, 24, 26]`.

## Training Protocol

Formal ablations use `configs/sa_dwpn_protocol.yaml`: imgsz 640, 150 epochs, batch 8, seed 0, deterministic training, official `yolo11n.pt` initialization, and one structural variable changed per ablation.

## Tests Run

- `python -m compileall .` - passed
- `python tools/train_sa_dwpn_variant.py --help` - passed
- `python tools/inspect_sa_dwpn_gates.py --help` - passed
- `python tools/visualize_sa_dwpn_heatmaps.py --help` - passed
- `python tools/export_run_artifacts.py --help` - passed
- Static YAML/protocol check - passed
- Missing data YAML safety check - passed

## Not Run In Codex Environment

- `pytest -q`: local environment does not have pytest installed.
- model build / dummy forward: local environment does not have torch or ultralytics installed.
- weight-transfer tests: local environment does not have torch, ultralytics, or `yolo11n.pt`.
- heatmap generation: no model weights or validation images are available in this environment.

## Minimal Colab Commands

```bash
python tools/train_sa_dwpn_variant.py --variant t3_only --data /content/ship_detection_local/data/data_local.yaml --weights /content/ship-yolo/yolo11n.pt --project /content/drive/MyDrive/ship_detection/runs
python tools/train_sa_dwpn_variant.py --variant o3_only --data /content/ship_detection_local/data/data_local.yaml --weights /content/ship-yolo/yolo11n.pt --project /content/drive/MyDrive/ship_detection/runs
python tools/inspect_sa_dwpn_gates.py --weights /path/to/best.pt --output artifacts/gate_analysis/c_lite
```

## Real Experiment Data Still Needed

Real `args.yaml`, `results.csv`, figures, checkpoints, gate masks, and heatmaps should be exported from Google Drive runs. This PR records only the user-reported C-lite validation summary and does not fabricate missing run artifacts.
