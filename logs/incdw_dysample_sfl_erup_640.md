# incdw_dysample_sfl_erup

- Structure: existing InceptionDW + DySample + PLS/SFL, with paper-level ERUP before the first detector Conv; no SCAM
- YAML: `experiments/yolo11n_incdw_dysample_sfl_erup.yaml`
- Pretrained weights: official `yolo11n.pt` through the matching PLS base; ERUP is independently initialized
- Official -> base Loaded/Total tensors: 126/457
- Base detector -> target Loaded/Total tensors: 457/457
- Epoch/imgsz/batch: 150/640/8 (prepared, not trained)
- Parameters/GFLOPs: 8,076,046 / 77.2257
- Precision/Recall/mAP50/mAP50-95: pending; no formal training
