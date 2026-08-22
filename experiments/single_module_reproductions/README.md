# YOLO11n single-method reproductions

Each YAML in this directory starts from the repository's official YOLO11n
baseline and introduces exactly one method:

- `yolo11n_fconv.yaml`: FConv on the P3, P4, and P5 backbone features before
  the unchanged PAN. Because three layers are inserted before the native head,
  use `remap_yolo11n_state_dict_for_fconv()` when transferring an official
  YOLO11n state dict so original layers 11-23 map to 14-26.
- `yolo11n_hhspp.yaml`: HHSPP replaces only the layer-9 SPPF.
- `yolo11n_hhspp_local_detail.yaml`: HHSPP is preserved at layer 9 and gains
  one zero-scaled, context-gated local-detail compensation branch.
- `yolo11n_focal_ciou.yaml`: the architecture is unchanged; only the native
  bbox CIoU term becomes `IoU^0.5 * (1-CIoU)`.
- `yolo11n_dre.yaml`: a DRENet RCAN branch reconstructs Selective Degradation
  targets only in training. Evaluation/export never invokes the enhancer;
  `DREDetect.switch_to_deploy()` can permanently strip its training-only
  parameters from a deployment copy.
- `yolo11n_hilo.yaml`: only the self-attention interaction inside layer-10
  C2PSA is exchanged for HiLo; the C2 split and FFN remain unchanged.

These files are not combination experiments. They do not alter YOLO11 DFL,
TaskAlignedAssigner, detection scales, PAN topology, or augmentation policy.

## Source provenance

- FConv: AMFC-DEIM, DOI `10.1109/JSTARS.2026.3653626`; no public author
  implementation was available, so `custom_modules/fconv.py` is paper-derived.
- HHSPP and Focal CIoU: `chenjiajiechen/DPCSANet-2025`, commit
  `0d8eb2e6035b375150edca48ed304e56ef4c1ff1` (GPL-3.0), reconciled with the
  final equations in the Electronics paper.
- DRE: `WindVChen/DRENet`, commit
  `a187dbe0f623b521a62c6176c7cafaa7322f5f66` (GPL-3.0), with the paper's
  two-task learned loss balance rather than the repository's later four-term
  experiment residue.
- HiLo: `ziplab/LITv2`, commit
  `7501c8662991cad6db780b61fac0886b56a76588` (Apache-2.0); only its attention
  is adapted, not RT-DETR's AIFI or Ship-DETR's other changes.
