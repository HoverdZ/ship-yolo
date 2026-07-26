# incdw_dysample_sfl_scam_erup

- Structure: existing InceptionDW + DySample + PLS/SFL + 3 SCAM blocks, with paper-level ERUP before the first detector Conv
- YAML: `experiments/yolo11n_incdw_dysample_sfl_scam_erup.yaml`
- Pretrained weights: official `yolo11n.pt` through the matching PLS+SCAM base; ERUP is independently initialized
- Official -> base Loaded/Total tensors: 126/514
- Base detector -> target Loaded/Total tensors: 514/514
- Epoch/imgsz/batch: 150/640/8 (prepared, not trained)
- Parameters/GFLOPs: 8,150,752 / 77.7599
- Precision/Recall/mAP50/mAP50-95: pending; no formal training
