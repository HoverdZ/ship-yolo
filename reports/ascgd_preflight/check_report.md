# ASCGD preflight report

Overall status: **PASS**

Formal 150-epoch training was not started.

## Variant checks

| Variant | Build/forward | Detect shapes | Finite | Params | GFLOPs |
|---|---:|---:|---:|---:|---:|
| a_base | True | True | True | 2585119 | 6.340506 |
| b_gather | True | True | True | 2701837 | 7.034784 |
| c_sca | True | True | True | 2827572 | 8.524115 |
| d_cca | True | True | True | 2890518 | 7.390317 |
| e_full | True | True | True | 3016253 | 8.879648 |
| f_swap | True | True | True | 3072864 | 8.473922 |
| g_symmetric | True | True | True | 3418767 | 10.540686 |

## Cross-variant checks

- A equals validated InceptionDW baseline: True
- All backbones match: True
- Window padding/reverse and non-standard shape check: True
- Rectangular E forward (non-window-multiple feature sizes): True
- E backward gradients present and finite: True
- E inherited parameter elements: 1746544/3016253 (backbone 99.9380%, Detect 89.7799%)
- CUDA AMP: not_run

## Profiling note

GFLOPs use Ultralytics 8.4.92 THOP conventions. Standard convolution operators are counted; THOP may not include the explicit attention matrix multiplications. Any profiler exception is recorded rather than silently skipped.

## Remaining GPU-only validation

L4 AMP stability, FP32/FP16 latency, throughput, peak memory, and formal accuracy remain unmeasured on this CPU-only host.
