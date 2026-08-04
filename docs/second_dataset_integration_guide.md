# HRSC2016-MS second-dataset integration guide

S00 and S01 use the private Drive archive
`/content/drive/MyDrive/ship_detection/data_2/HRSC2016_MS_YOLO.zip`.
The archive is the audited YOLO-HBB conversion of HRSC2016-MS and contains
the frozen `train`, `val`, and `test` splits.

1. The Notebook copies the single ZIP to Colab local storage with byte
   progress, then performs path-safe extraction with file and byte progress.
2. A local runtime `data.yaml` is generated. The obsolete absolute `path`
   embedded when the conversion ZIP was created is never trusted.
3. The integration validator checks image readability, image/label pairing,
   empty labels, normalized boxes, class mapping, exact cross-split image
   hashes, and the frozen counts without modifying the Drive archive.
4. Expected counts are 610/460/610 images and 2453/1953/3249 ship instances
   for train/val/test, respectively (1680 images and 7655 instances total).
5. Both models start independently from official `yolo11n.pt`. S01 never
   loads a primary-dataset checkpoint and uses the current final structure:
   InceptionDW + DPLS + bounded CA-SCAM + complete VGUP.
6. S00 and S01 retain the same 150-epoch formal training settings used by the
   primary-data experiment matrix. The test split remains sealed during
   checkpoint selection.

Required checks:

- image count and readable image verification;
- label count, orphan/missing labels, and empty-label images;
- class-name mapping to the paper's `ship` target;
- normalized finite YOLO boxes with positive width/height;
- train/val/test identity and exact-image-hash duplicate audit;
- annotation conversion manifest and archive checksum;
- dataset license note and the formal HRSC2016-MS citation:
  Chen et al., *Remote Sensing* 14(21), 5460 (2022),
  https://doi.org/10.3390/rs14215460.

Optional zero-shot evaluations (primary-trained model on the second dataset,
and vice versa) are separate cross-dataset tests. They do not replace the
independently trained S00/S01 comparison and must not be used for checkpoint
selection.
