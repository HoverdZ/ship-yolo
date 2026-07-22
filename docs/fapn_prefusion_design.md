# FaPN-Prefusion design and pre-training audit

Status: code preparation and CPU validation complete; no formal 150-epoch run
has been started. `FaPN-Prefusion` is a temporary experiment name, not a paper
module name.

## Sources checked

The implementation was compared directly with:

- Official FaPN `FeatureSelectionModule`, `FeatureAlign_V2`, and `FAN.forward`:
  <https://github.com/EMI-Group/FaPN/blob/main/detectron2/modeling/backbone/fan.py>
- Official YOLO11 detector configuration:
  <https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/11/yolo11.yaml>
- The installed Ultralytics 8.4.92 `nn/tasks.py`, `engine/model.py`,
  `engine/trainer.py`, and `utils/torch_utils.py`.
- Torchvision `DeformConv2d` and functional `deform_conv2d` documentation:
  <https://docs.pytorch.org/vision/main/generated/torchvision.ops.DeformConv2d.html>
  and
  <https://docs.pytorch.org/vision/main/generated/torchvision.ops.deform_conv2d.html>.

Torchvision documents that a non-`None` mask selects modulated DCNv2, offset
channels encode offset groups, and the convolution weight shape/groups encode
channel grouping. Prefusion therefore uses 8 offset groups but a depthwise
content convolution (`groups=high_channels`).

## Three top-down structures

### Official YOLO11n

```text
C5(256,20x20) -> nearest -> U5(256,40x40)
C4(128,40x40) --------------------------- concat(384) -> C3k2 -> T4(128,40x40)

T4(128,40x40) -> nearest -> U4(128,80x80)
C3(128,80x80) --------------------------- concat(256) -> C3k2 -> T3(64,80x80)
```

### Previous original-FaPN replacement

```text
C5(256) -> 1x1 -> M5(64)
C4(128) -> FSM(64); FAM(M5,C4) -> M4(64) -> output 3x3 -> T4(64)
C3(128) -> FSM(64); FAM(M4,C3) -> M3(64) -> output 3x3 -> T3(64)
```

This is a faithful replacement experiment, but it also removes YOLO's two
Concat+C3k2 fusions, changes T4 from 128 to 64 channels, and changes the
distribution consumed by PAN. Its audited 150-epoch result was approximately
Precision 0.794, Recall 0.698, mAP50 0.750, and mAP50-95 0.293. It is treated
as a credible negative result rather than an implementation failure.

### FaPN-Prefusion

```text
C5 -> official nearest -> U5(256)
C4 -> FSM-Keep -> S4(128)
[S4,U5] -> AlignmentOnly -> A5(256)
[S4,A5] -> official Concat(384) -> official C3k2 -> T4(128)

T4 -> official nearest -> U4(128)
C3 -> FSM-Keep -> S3(128)
[S3,U4] -> AlignmentOnly -> A4(128)
[S3,A4] -> official Concat(256) -> official C3k2 -> T3(64)
```

The complete official PAN and `Detect(T3,P4,P5)` follow unchanged.

## Exact layer and shape audit at 640

| Index | From | Operation | Output |
|---:|---|---|---|
| 0-10 | official | YOLO11n backbone | C3=128x80x80, C4=128x40x40, C5=256x20x20 |
| 11 | -1 | nearest upsample | 256x40x40 |
| 12 | 6 | `FaPNFeatureSelectionKeep` | 128x40x40 |
| 13 | [12,11] | `FaPNAlignmentOnly` | 256x40x40 |
| 14 | [12,13] | Concat | 384x40x40 |
| 15 | -1 | official top-down C3k2 | T4=128x40x40 |
| 16 | -1 | nearest upsample | 128x80x80 |
| 17 | 4 | `FaPNFeatureSelectionKeep` | 128x80x80 |
| 18 | [17,16] | `FaPNAlignmentOnly` | 128x80x80 |
| 19 | [17,18] | Concat | 256x80x80 |
| 20 | -1 | official top-down C3k2 | T3=64x80x80 |
| 21 | -1 | PAN stride-2 Conv | 64x40x40 |
| 22 | [-1,15] | PAN Concat | 192x40x40 |
| 23 | -1 | PAN C3k2 | P4=128x40x40 |
| 24 | -1 | PAN stride-2 Conv | 128x20x20 |
| 25 | [-1,10] | PAN Concat | 384x20x20 |
| 26 | -1 | PAN C3k2 | P5=256x20x20 |
| 27 | [20,23,26] | Detect | strides [8,16,32] |

Both YAMLs share indices 11-27 exactly. The InceptionDW variant differs only
at the already validated backbone layers 2 and 4.

## What comes from FaPN and what is adapted

| Category | Prefusion decision |
|---|---|
| Retained from FaPN | GAP -> 1x1 -> Sigmoid feature selection |
| Retained from FaPN | Joint low/high offset controller |
| Retained from FaPN | High controller feature multiplied by 2.0 |
| Retained from FaPN | Offset, modulation mask, 8 deformable groups, DCNv2 alignment |
| YOLO interface adaptation | FSM output keeps the original shallow channels |
| YOLO interface adaptation | The already-nearest-upsampled feature is the FAM high input |
| YOLO interface adaptation | Controller paths project to fixed 64 channels only |
| YOLO interface adaptation | Full high channels pass through a depthwise DCNv2 |
| Deliberately removed | FSM's projection of every pyramid level to one 64-channel width |
| Deliberately removed | FAM's internal `aligned + lateral` content fusion |
| Deliberately removed | FAM's post-DCN ReLU |
| Deliberately removed | Original FaPN per-level output 3x3 convolution |
| Restored from YOLO | Nearest upsampling, Concat, top-down C3k2, native T3/T4 widths |
| Kept from YOLO | Complete PAN, three-scale Detect, loss and training policy |

The removed Add and ReLU are not omissions: AlignmentOnly is responsible only
for geometry. Native Concat+C3k2 remains responsible for content fusion and
channel expression.

## Selection and alignment equations

For a shallow feature `x`:

```text
attention = sigmoid(conv_attention(GAP(x)))
selected = x + gamma_s * (x * attention), gamma_s=0.1
```

No projection, MLP, normalization, activation, dropout, softmax, or channel
change is added.

For `[selected_low, upsampled_high]`:

```text
q_low  = low_projection(selected_low)             # -> 64
q_high = high_projection(upsampled_high)           # -> 64
control = offset_feature(concat(q_low, 2*q_high))  # 128 -> 64
offset_y, offset_x, mask_logits = chunk(conv3x3(control), 3)
offset = concat(offset_y, offset_x)                # 144 channels
mask = sigmoid(mask_logits)                        # 72 channels
aligned = depthwise_DCNv2(upsampled_high, offset, mask)
output = upsampled_high + gamma_a*(aligned-upsampled_high), gamma_a=0.1
```

There is no second upsample inside AlignmentOnly.

## Why depthwise DCNv2, not DCNv4

Torchvision's maintained operator supplies the required modulated DCNv2
semantics without the original custom CUDA extension. DCNv4 would introduce a
different operator and confound the FaPN ablation. A full-channel DCN would
also add unnecessary channel mixing already handled by YOLO's C3k2. Depthwise
DCNv2 isolates spatial alignment, preserves 256/128 content channels, and
keeps offset groups independently fixed at 8.

## Identity initialization

The offset/mask convolution starts at zero, so offsets are zero and masks are
0.5. Every depthwise 3x3 kernel is zero except its center value 2.0, and bias
is zero. Thus `2.0 * 0.5 * center_sample = input`. CPU tests measured maximum
absolute DCN and AlignmentOnly identity errors below `1e-5` (currently exact
within the tested float32 inputs). This protects the pretrained YOLO feature
distribution before the new controller learns.

## Model size and profile

The profiler runs THOP directly on a fresh model, never deep-copies it, and
uses an explicit depthwise `DeformConv2d` custom operation. Values follow the
Ultralytics convention of two FLOPs per MAC.

| Model (nc=1) | Parameters | GFLOPs | Delta params vs official nc=1 | Delta GFLOPs |
|---|---:|---:|---:|---:|
| Official YOLO11n | 2,590,035 | 6.440602 | - | - |
| FaPN-Prefusion | 2,933,255 | 8.879360 | +343,220 | +2.438758 |
| InceptionDW + FaPN-Prefusion | 2,928,339 | 8.779264 | +338,304 | +2.338662 |

The Prefusion modules themselves contain 343,220 parameters and 2.438758
GFLOPs. DCNv2 sampling/addressing overhead is excluded, as in normal THOP
convolution-equivalent accounting.

## Official weight transfer

The source is only official `yolo11n.pt`; no trained InceptionDW checkpoint is
used.

| Audit | Baseline Prefusion | InceptionDW Prefusion |
|---|---:|---:|
| Inherited state tensors / target | 448 / 519 | 446 / 531 |
| Inherited parameter tensors / target | 223 / 276 | 221 / 288 |
| Inherited parameter elements / target | 2,546,000 / 2,933,255 | 2,540,240 / 2,928,339 |
| Element inheritance rate | 86.7978% | 86.7468% |
| Random parameter elements | 387,255 | 388,099 |
| Backbone element inheritance | 100% | 99.9380% (new InceptionDW internals) |
| Both top-down C3k2 | 100% | 100% |
| PAN | 100% | 100% |
| Detect box branch | 100% | 100% |
| Detect DFL | 100% | 100% |
| Shape-compatible Detect classification | 4,928 / 48,963 | 4,928 / 48,963 |
| Prefusion parameters | 0% inherited by design | 0% inherited by design |

The final `nc=80 -> nc=1` classification output tensors are never forced into
the target. Reports list every source-to-target key, mismatch, unmatched key,
and new parameter.

## How pretrained weights reach Trainer

```text
custom YAML -> deterministic target model
official yolo11n.pt -> explicit semantic mapping -> strict audit
transferred target model -> save real *_pretrained_init.pt
reload init.pt -> tensor equality + manifest SHA256 audit
YOLO(INIT_PT) -> official Model.train -> Trainer rebuild/load
on_pretrain_routine_end -> compare Trainer tensors with manifest
first optimizer step (only if every check passes)
```

The previous failure mode was to audit one in-memory transferred model and
then pass the YAML to training, causing Trainer to build another random model.
That path is now impossible in the formal entry: `args.model` must be a `.pt`,
its file SHA256 must equal the manifest, model parameter count and critical
tensor hashes must match, stride must be `[8,16,32]`, and every floating tensor
must be finite. Deliberate tensor corruption is covered by a failing test.

## Safe model-info path

Ultralytics 8.4.92 `model_info()` calls `get_flops()`, which deep-copies the
model before THOP. The previous Torchvision DCN experiment could hard-restart
a Colab kernel in that path. `fapn_prefusion_profile.py` computes the profile
in advance without deepcopy. `install_safe_prefusion_flops()` temporarily
replaces only `ultralytics.utils.torch_utils.get_flops` with the saved value;
official `model_info`, LOGGER, tqdm, Trainer, and training loops remain intact.

Only the known deterministic warning matching
`compute_grad_input does not have a deterministic implementation` is filtered.
AMP, deterministic mode, CUDA, cuDNN, and official progress output remain on.

## Files and commands

```text
custom_modules/fapn_prefusion.py
experiments/yolo11n_fapn_prefusion.yaml
experiments/yolo11n_inceptiondw_fapn_prefusion.yaml
tools/fapn_prefusion_utils.py
tools/fapn_prefusion_profile.py
tools/prepare_fapn_prefusion_init.py
tools/check_fapn_prefusion_models.py
tools/probe_fapn_prefusion_amp.py
tools/train_fapn_prefusion.py
tests/test_fapn_prefusion.py
docs/colab_fapn_prefusion.md
```

```bash
python tools/fapn_prefusion_profile.py --variant all --imgsz 640
python tools/prepare_fapn_prefusion_init.py --variant all --weights yolo11n.pt
python tools/check_fapn_prefusion_models.py --variant all --weights yolo11n.pt
python -m pytest tests/test_fapn_prefusion.py -q
```

The `.pt` initialization files are intentionally ignored by Git. Their JSON
manifests and transfer/profile reports are tracked.

## Not yet validated

No formal training, GPU AMP probe on this Windows CPU host, L4 memory use,
throughput, Precision, Recall, mAP50, or mAP50-95 has been measured in this
preparation task. The Colab workflow must run the CUDA AMP probe before formal
training.
