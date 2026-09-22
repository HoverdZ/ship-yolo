# Experiment index

## Current SIP-Det configurations

The canonical model is [M7: YOLO11n + VGUP + LDPP + CGDR](final_ablation/M7_yolo11n_vgup_ldpp_cgdr.yaml).

| Directory | Purpose |
|---|---|
| [final_ablation](final_ablation/README.md) | Complete eight-combination index; four LDPP-enabled YAMLs live here |
| [ldpp_ablation](ldpp_ablation/README.md) | Four LDPP-disabled controls referenced by the same index |
| [ldpp_allocation_study](ldpp_allocation_study/README.md) | Scale-specific regression convolution allocation; H2 reuses M2 |
| [architecture_transferability](architecture_transferability/README.md) | Current host-aware LDPP transfers alongside historical DPLS configurations |
| [vgup_gate_study](vgup_gate_study/) | VGUP gate variants; component study |
| [cgdr_detail_ablation](cgdr_detail_ablation/) | CGDR detail-branch alternatives; component study |

## Precursor, comparison and historical experiments

| Directory | Purpose and interpretation |
|---|---|
| [dpls_lightweight](dpls_lightweight/) | Lightweight operator screening on the precursor DPLS architecture |
| [systematic_ablation](systematic_ablation/) | Earlier DPLS-based component combinations; not the current LDPP factorial map |
| [model_ablation](model_ablation/) | Earlier cumulative architecture experiments, including InceptionDW and CA-SCAM |
| [component_ablation](component_ablation/) | Earlier operator and scale-path studies |
| [transfer_models](transfer_models/) | Earlier transfer experiments, including the InceptionDW/DPLS/CA-SCAM RTMDet variant |
| [comparison_models](comparison_models/) | External model comparisons and protocol |
| [external_baselines](external_baselines/) | External baseline configurations |
| [single_module_reproductions](single_module_reproductions/README.md) | Reference module reproductions |

Historical studies can support design choices when the comparison protocol is appropriate, but their results must retain their original architecture labels. DPLS and LDPP are not interchangeable.

See [training_config.yaml](training_config.yaml) for the shared recorded protocol and [model versions](../docs/MODEL_VERSIONS.md) for naming and checkpoint provenance.
