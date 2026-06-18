# WAFPN-v1 Integration Notes

## Motivation

WAFPN-v1 is designed for the next remote sensing ship detection experiment after the YOLO11 baseline, P2 head, and SemanticGuide explorations. Prior results suggest that high-resolution information is useful, but directly adding a P2 detection head can introduce shallow-feature noise. WAFPN-v1 keeps the standard three detection heads and tests whether static learnable weighted fusion in the Neck improves feature selection for small ships.

## Module Summary

`WeightedAdd2` fuses two same-shape feature maps with two learnable scalar weights. The weights are initialized to 1, passed through ReLU to remain non-negative, and normalized with `eps=1e-4`.

`AlignWeightedAdd2` aligns two inputs before fusion. It applies 1x1 Conv to both inputs, resizes the second feature map to the first feature map spatial size when needed, calls `WeightedAdd2`, and optionally applies a lightweight 3x3 Conv block after fusion.

This module is intentionally static and simple. It does not use sigmoid attention, spatial attention, dynamic weights, DyHead, DCNv3, or the full MSWPN structure.

## Ultralytics Source Integration

This repository does not vendor the full Ultralytics source code. To run the WAFPN-v1 YAML in a local or Colab Ultralytics source environment, integrate the module manually or with Codex in that source tree.

Recommended integration steps:

1. Copy `custom_modules/weighted_add.py` into `ultralytics/nn/modules/weighted_add.py`, or another custom module location that is importable by Ultralytics.
2. Add imports in `ultralytics/nn/modules/__init__.py`:

```python
from .weighted_add import AlignWeightedAdd2, WeightedAdd2
```

3. Add imports in `ultralytics/nn/tasks.py`:

```python
from ultralytics.nn.modules import AlignWeightedAdd2, WeightedAdd2
```

4. Update `parse_model()` in `ultralytics/nn/tasks.py` so `AlignWeightedAdd2` can resolve channels from two input branches. The experiment YAML uses `args: [c_out]`; `parse_model()` should convert that to:

```python
c1, c2 = ch[f[0]], ch[f[1]]
c_out = make_divisible(min(args[0], max_channels) * width, 8)
args = [c1, c2, c_out, *args[1:]]
```

5. Ensure `AlignWeightedAdd2` is included in the appropriate module handling path so its output channel is registered as `c_out`.

The exact `parse_model()` patch depends on the Ultralytics version in Colab. Keep the patch minimal and isolated.

## Pre-Training Checks

Before training, run the structure check script in the modified Ultralytics environment:

- `model.info()`
- dummy forward with `torch.zeros(1, 3, 640, 640)`
- Detect head count check
- confirmation that the model contains `AlignWeightedAdd2`
- weight transfer check with Loaded/Total tensors

Do not start training until `model.info()` and dummy forward pass both succeed.

## Current Scope

This round only adds the WAFPN-v1 scaffold:

- custom module file
- experiment YAML
- integration notes
- structure check reference script
- experiment record

This round does not upload the full Ultralytics source code and does not write Colab training code.
