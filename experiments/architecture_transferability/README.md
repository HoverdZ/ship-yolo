# VGUP + DPLS + CGDR architecture transfer

This directory contains structure-only transfer configurations. It does not
contain training notebooks, optimizer settings, data paths, or checkpoint
transfer code.

## Authoritative hosts

| Host | Source implementation | Configuration |
|---|---|---|
| YOLOv8n | Ultralytics YOLOv8, used through Ultralytics 8.4.92 | `yolov8n_vgup_dpls_cgdr.yaml` |
| YOLO11n | Ultralytics 8.4.92 | `../systematic_ablation/yolo11n_vgup_dpls_cgdr.yaml` |
| YOLOv12n | `sunsmarterjie/yolov12` at `01a22c0603e0eaa6d9bd62120a391e744d92cea2` | `yolov12n_author_vgup_dpls_cgdr.yaml` |
| YOLOv13n | `iMoonLab/yolov13` release tag at `70f23ede45ee00a30cf6139c3d1ea7abe3df4eec` | `yolov13n_author_vgup_dpls_cgdr.yaml` |

YOLOv12 and YOLOv13 must be constructed inside their original-author forks.
They must not be substituted with similarly named models from the Ultralytics
main repository. After importing the selected author fork and adding this
repository to `sys.path`, call the matching helper before constructing YAML:

```python
from custom_modules.author_fork_registration import (
    register_yolov12_author_modules,
    register_yolov13_author_modules,
)

# Call exactly one in a fresh process/runtime.
register_yolov12_author_modules()
# register_yolov13_author_modules()
```

## Adaptation policy

- **YOLOv8n:** native C2f blocks remain unchanged. The P3--P5 pyramid is
  shifted to P2--P4, both top-down resizing nodes use DySample, and CGDR
  replaces the highest retained P4 SPPF node.
- **YOLO11n:** the canonical final configuration is retained without a second
  copy in this directory.
- **YOLOv12n:** the author Turbo grouped P2/P3 downsampling and A2C2f blocks are
  preserved. The pyramid is shifted to P2--P4. Because the author topology has
  no SPPF, CGDR is inserted immediately after the highest retained P4 A2C2f.
- **YOLOv13n:** native DSC3k2, DSConv, A2C2f, HyperACE, FullPAD, P4 and P5 paths
  are preserved. CGDR refines the backbone P4 feature before it enters P5 and
  HyperACE. Two DySample nodes implement the added P4-to-P3 and P3-to-P2 DPLS
  path, while zero-initialized native FullPAD gates protect the original P3
  route. Detect receives P2, P3, P4 and P5.

YOLOv13 uses `VGUPPrecisionSafePreprocessor`. It is the complete VGUP, not the
older KBL-only host adaptation: BPW, KBL, global gate and spatial gate all
remain. Only BPW/KBL and gate arithmetic run in a local FP32 precision island;
the encoder and detector continue to follow the surrounding AMP policy.
