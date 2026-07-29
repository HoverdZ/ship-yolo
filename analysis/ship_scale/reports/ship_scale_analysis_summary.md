# Deterministic ship-scale analysis summary

## Training-set lower-tail short-side statistics

- Q2.5: 11.2500 px
- Q5: 12.5000 px
- Q10: 13.7500 px

## Training-set short sides below feature strides

- <4 px: 0.0000%
- <8 px: 0.4560%
- <16 px: 17.0087%
- <32 px: 91.8377%

## Training-set Q5 dilution statistics

| Level | Stride | Intervals spanned | Dilution rate / % |
|---|---:|---:|---:|
| P2 | 4 | 3.1250 | 0.0000 |
| P3 | 8 | 1.5625 | 0.0000 |
| P4 | 16 | 0.7812 | 21.8750 |
| P5 | 32 | 0.3906 | 60.9375 |

## Horizontal-box elongation

- Mean aspect ratio: 1.3098
- Median aspect ratio: 1.2500
- Ratio ≥2: 3.1464%
- Ratio ≥4: 0.0000%
- Ratio ≥8: 0.0000%

## Split comparison

| Pair | Empirical-CDF maximum distance | Left median / px | Right median / px |
|---|---:|---:|---:|
| train vs val | 0.0308 | 21.2500 | 21.2500 |
| train vs test | 0.0292 | 21.2500 | 21.2500 |
| val vs test | 0.0305 | 21.2500 | 21.2500 |

## Objective interpretation

- Quantile candidate: Use q=5% as the central lower-tail descriptor and retain q=2.5% and q=10% as sensitivity checks; all three produce the same P2/P3 versus P4/P5 qualitative ordering.
- Scale motivation: Across q=2.5%, 5%, and 10%, P3 has zero dilution while P4 and P5 have positive dilution. The descriptor therefore supports testing removal of the coarse P5 path, but it does not by itself establish that P2 is better than P3.
- Split shift: The largest pairwise short-side ECDF distance is 0.0308; train/val/test medians are 21.25/21.25/21.25 px. No marked split-scale displacement is apparent under this descriptor.
- Aspect ratio: The median horizontal-box aspect ratio is 1.25, and 3.15% of training instances have ratio >=2. The dataset-level DPLS motivation should emphasize object scale rather than a globally pronounced elongated-box distribution.

These descriptive statistics motivate evaluating a P2–P4 detection pyramid. They do not establish a detection-accuracy gain; D0/D1/D2 controlled experiments are required.
