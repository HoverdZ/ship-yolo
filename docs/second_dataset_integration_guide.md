# Second-dataset integration guide

S00 and S01 are intentionally blocked until a dataset is selected. Do not
guess a dataset name or download one automatically.

1. Copy `datasets/external_dataset_template.yaml` to a local, untracked
   dataset descriptor and complete every field, including license and
   citation.
2. Run the integration validator. It reports image/label counts, empty
   labels, corrupt images, invalid boxes, class mapping, split overlap, and
   YOLO-format compatibility without changing the source.
3. Create a YOLO data YAML whose `train`, `val`, and `test` entries preserve
   the chosen dataset's frozen split. Store data outside Git.
4. Set the S00/S01 data path through the Notebook configuration cell. Both
   models start independently from `yolo11n.pt`; S01 must not load R10.
5. Train S00 and S01 with the same formal settings unless a documented
   dataset-specific protocol revision is required.

Required checks:

- image count and readable image verification;
- label count, orphan/missing labels, and empty-label images;
- class-name mapping to the paper's `ship` target;
- normalized finite YOLO boxes with positive width/height;
- train/val/test identity and near-duplicate audit;
- annotation conversion manifest and checksums;
- dataset license, redistribution conditions, and formal citation.

Optional zero-shot evaluations (primary-trained model on the second dataset,
and vice versa) are separate cross-dataset tests. They do not replace the
independently trained S00/S01 comparison and must not be used for checkpoint
selection.
