# Fog-LEVIR-Ship dataset notice

## Provenance

Fog-LEVIR-Ship is the primary dataset used by this project. It is a derived version of the public **LEVIR-Ship** dataset (not “Liver_Ship”), which contains optical remote-sensing ship imagery collected by the GF-1 and GF-6 satellites.

The derived data applies the procedure described by Wang et al. (2022), *A novel method of ship detection under cloud interference for optical remote sensing images*. Perlin-noise-based visibility degradation is used to construct thin-fog, dense-fog and patchy-fog conditions. The exact construction parameters, frozen train/validation/test split and experimental use are described in the accompanying paper.

## Redistribution status

The dataset images are **not bundled in this public repository**. The official LEVIR-Ship page states that:

- its annotations are available under CC BY 4.0;
- the dataset authors do not own the copyright of the source satellite images;
- use of those images must comply with the Terms of Use of the China Centre for Resources Satellite Data and Application.

Because those terms do not establish unrestricted public redistribution through this repository, users should obtain LEVIR-Ship from its official source and reproduce the fog-processing procedure described in the paper.

## Expected local layout

```text
Fog-LEVIR-Ship/
  train/
    images/
    labels/
  val/
    images/
    labels/
  test/
    images/
    labels/
  data.yaml
```

The detector uses one class:

```yaml
names:
  0: ship
```

## References

- Chen, W. et al. (2022). LEVIR-Ship official project: <https://github.com/WindVChen/LEVIR-Ship>
- Wang, W., Zhang, X., Sun, W., and Huang, M. (2022). *A novel method of ship detection under cloud interference for optical remote sensing images*. Remote Sensing, 14(15), 3731. <https://doi.org/10.3390/rs14153731>
