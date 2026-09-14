# VGUP + LDPP + CGDR: host-aware architecture transfer

This directory contains structure-only transfer configurations. It does not
contain training notebooks, optimizer settings, data paths, or checkpoint
transfer code.

## Authoritative hosts

| Host | Source implementation | Configuration |
|---|---|---|
| YOLOv8n | Ultralytics 8.4.92 | `yolov8n_vgup_ldpp_cgdr.yaml` |
| YOLO11n | Ultralytics 8.4.92 | `../final_ablation/M7_yolo11n_vgup_ldpp_cgdr.yaml` (canonical, unchanged) |
| YOLOv12n | [sunsmarterjie/yolov12](https://github.com/sunsmarterjie/yolov12/tree/01a22c0603e0eaa6d9bd62120a391e744d92cea2) at `01a22c0603e0eaa6d9bd62120a391e744d92cea2` | `yolov12n_author_vgup_ldpp_cgdr.yaml` |
| YOLOv13n | [iMoonLab/yolov13](https://github.com/iMoonLab/yolov13/tree/70f23ede45ee00a30cf6139c3d1ea7abe3df4eec) at `70f23ede45ee00a30cf6139c3d1ea7abe3df4eec` | `yolov13n_author_kbllite_ldpp_cgdr.yaml` |

YOLOv12 and YOLOv13 must be constructed inside their original-author forks.
They both report version **8.3.63** but are different codebases. They must not
be substituted with similarly named models from Ultralytics main. The helper
verifies the active package's Git origin, exact commit, version and parser
features. It extends registration in memory only, never author source files
or site-packages. Author patch version 3 adds `AuthorLDPPDetect` to both the
Detect channel-append branch and the legacy-selection branch; repeat calls
are idempotent. Existing VGUP, KBL-Lite, CGDR and DySample registration remains.

In a **fresh process**, put the chosen author clone first on `sys.path` and
add ship-yolo before importing either `ultralytics` or custom modules:

```python
import sys
from pathlib import Path

ship = Path("/absolute/path/to/ship-yolo")
sys.path[:0] = ["/absolute/path/to/pinned-yolov12", str(ship)]
from custom_modules.author_fork_registration import register_yolov12_author_modules
register_yolov12_author_modules()
from ultralytics import YOLO
model = YOLO(str(ship / "experiments/architecture_transferability/"
                "yolov12n_author_vgup_ldpp_cgdr.yaml"), task="detect", verbose=False)
```

For v13, start another process, select its pinned clone, call
`register_yolov13_author_modules()` and use
`yolov13n_author_kbllite_ldpp_cgdr.yaml`. Never switch author forks in a running
process or import the standard registration there. No `reg_max` or `end2end`
arguments belong in author YAML: `[nc]` becomes `(nc, channels)`.

For v8/11 in standard Ultralytics **8.4.92**, run from ship-yolo:

```python
from custom_modules.register import register_custom_modules
register_custom_modules()
from ultralytics import YOLO
model = YOLO("experiments/architecture_transferability/yolov8n_vgup_ldpp_cgdr.yaml",
             task="detect", verbose=False)
# For YOLO11 use experiments/final_ablation/M7_yolo11n_vgup_ldpp_cgdr.yaml.
```

Standard patch version 26 registers `C2f_DWConvLite` in both base and repeat
sets: channels and depth/width scaling are host-controlled, with repetitions
inside C2f. The author-only head is not imported by this registration path.

## Adaptation policy

- **YOLOv8n (C2f-aware LDPP):** the native C2f backbone remains unchanged. The P3--P5 pyramid is
  shifted to P2--P4, both top-down resizing nodes use DySample, and CGDR
  replaces the highest retained P4 SPPF node. At P3 fusion nodes 11 and 17,
  `C2f_DWConvLite` changes only Bottleneck.cv2 to native DWConv, retaining
  C2f's inner expansion, cv1, projections, concatenation and shortcut. P2 and
  final P4 fusion remain native C2f. Both bottom-up transitions use DWConv.
  Canonical `LDPPDetect` receives nodes 14/17/20 (P2/P3/P4).
- **YOLO11n:** the canonical final configuration is retained without a second
  copy in this directory: two DySample, two C3k2_DWConvLite, two direct DWConv,
  and LDPPDetect on P2/P3/P4 (nodes 15/18/21).
- **YOLOv12n:** the author Turbo grouped P2/P3 downsampling and A2C2f blocks are
  preserved. The pyramid is shifted to P2--P4. Because the author topology has
  no SPPF, CGDR is inserted immediately after the highest retained P4 A2C2f.
  A2C2f fusion and area-attention internals are untouched; only the two
  bottom-up transitions become DWConv. Three-scale `AuthorLDPPDetect` receives
  nodes 14/17/20 (P2/P3/P4).
- **YOLOv13n:** the previously validated host adaptation is retained. Native
  DSC3k2, DSConv, A2C2f, HyperACE, FullPAD, P4 and P5 paths remain intact.
  CGDR refines backbone P4 before it enters P5 and HyperACE. P3-centered
  DPLS-13 uses two zero-initialized native FullPAD gates and the author's
  nearest-neighbour resizing, then adds a P2 output. Detect receives P2, P3,
  P4 and P5. LDPP reuses these native DSC3k2/DSConv spatial operators instead
  of inserting YOLO11 containers. The only change from the validated old
  v13 YAML's executable graph is the four-scale `AuthorLDPPDetect` at nodes
  41/38/29/33. P2 and P4 context are aligned to P3, fused via FullPAD into
  P3*, and P3* is upsampled and fused with backbone P2 to produce P2*.

YOLOv13 deliberately uses the stable `KBLLitePreprocessor` from the earlier
host experiment instead of forcing complete VGUP into this fork. It retains
KBL and the spatial visibility gate, removes BPW and the global gate, and runs
only KBL dynamic filtering in a local FP32 precision island. This is an
architecture-specific stability decision; YOLOv8n, YOLO11n and YOLOv12n keep
the complete VGUP.

## Shared LDPP definition, not identical YAML

LDPP preserves three functions: high-resolution detail access, lightweight
neck spatial computation, and scale-aware regression capacity. For ships
with short sides below 20 px at 640, P2 provides a stride-4 detail path;
retaining dense mixing at P2/P3 is intended to protect localization capacity.
VGUP/KBL-Lite and CGDR retain visibility processing and context-guided detail
refinement for haze and confusing sea clutter. These are design motivations,
not evidence that the new LDPP transfers improve accuracy.

| Host | Detect inputs / strides | Regression from fine to coarse |
|---|---|---|
| v8 / canonical v11 / author v12 | P2/P3/P4: 4/8/16 | Dense+DS, Dense+DS, DS+DS |
| author v13 | P2/P3/P4/P5: 4/8/16/32 | Dense+DS, Dense+DS, DS+DS, DS+DS |

Dense is native Conv3x3; DS is native DWConv3x3 followed by Conv1x1. Each
tower ends with a 1x1 predictor to 64 regression logits (`4 * reg_max`).
All heads use `reg_max=16` and regression hidden width
`max(16, first_input_channels // 4, 4 * reg_max)`, at least 64. There is no
extra input stem. The parent-created stock classification `cv3` is never
reconstructed: v8 retains its legacy classification, and v11/v12/v13 use
their own host-selected classification. Author forward, DFL, anchor creation,
decoding, bias initialization and detection loss are inherited unchanged;
the pinned author heads are non-end-to-end.

This is **host-aware transferability**, not a claim that four implementations
are identical. In particular, v13 uses a documented VGUP stability adapter,
keeps P5 and its own feature coordination, and v12 preserves A2C2f.
All old `*_dpls_cgdr.yaml` files remain historical configurations. Previously
reported DPLS migration results must not be relabeled as results of these new
LDPP models. Baselines and existing results are unchanged; new effectiveness
claims require subsequent controlled training/evaluation.

## Minimal validation scope

Use syntax checks and one model construction per host (separate processes
for author forks), checking the above head inputs and strides. Construction
includes the host's internal stride-initialization forward; no additional
random forwards, backward, training, benchmark or FLOP profiling are needed.
CPU construction does not establish CUDA/AMP training stability or accuracy.

Implementation check (2026-09-14): Python 3.12.7, PyTorch 2.13.0+cpu;
the four changed/new Python files passed `py_compile`. Each of the four
listed models was constructed once with `verbose=False`; all reported the
expected strides above. v8 selected `legacy=True`, while v11/v12/v13 selected
`legacy=False`. Both author heads reported hidden width 64 and end2end=False.
Repeated registration in each active environment succeeded. The pinned author
forks used their own built-in scaled-dot-product-attention CPU fallback;
no source patch or operator substitution was made. No training, backward,
benchmark, baseline run or additional random-input forward was performed.
