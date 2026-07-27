# incdw_dysample_pls_ca_scam_vgup_640

- Experiment name: `incdw_dysample_pls_ca_scam_vgup`
- Structure: successful InceptionDW + DySample + PLS + VGUP topology; layers
  22/23/24 replace independent SCAM blocks with equivalent-initialized
  CA-SCAM (`max_delta=0.1`).
- Pretrained weights: yes, official `yolo11n.pt`; no historical `best.pt`.
- Loaded/Total tensors: expected `562/571` (local/Colab preflight audit).
- New tensors: exactly 9 CA tensors, zero initialized.
- Epochs: 150
- Image size: 640
- Batch: 8
- Precision: pending formal Colab training
- Recall: pending formal Colab training
- mAP50: pending formal Colab training
- AP75: pending formal Colab validation
- mAP50-95: pending formal Colab training
- Baseline for controlled comparison: successful SCAM model best at epoch
  143, P `0.85610`, R `0.73692`, mAP50 `0.80119`, mAP50-95 `0.34787`.
- Test split: sealed until final model selection.
