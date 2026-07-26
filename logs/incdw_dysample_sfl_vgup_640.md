# incdw_dysample_sfl_vgup

- Structure: existing InceptionDW + DySample + PLS/SFL, with complete VGUP before the first detector Conv; no SCAM
- YAML: `experiments/yolo11n_incdw_dysample_sfl_vgup.yaml`
- Pretrained weights: official `yolo11n.pt` through the matching PLS base; VGUP is independently initialized
- Official -> base Loaded/Total tensors: 126/457
- Base detector -> target Loaded/Total tensors: 457/457
- Epoch/imgsz/batch: 150/640/8 (prepared, not trained)
- Parameters/GFLOPs: 1,372,400 / 18.4067
- Precision/Recall/mAP50/mAP50-95: pending; no formal training
