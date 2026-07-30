# Experiment Record

Experiment Name: C3_pkiconv_p23
Date: Pending

Baseline: Official YOLO11n

Modification: Replace only Bottleneck.cv2 inside the P2/P3 backbone C3k2 blocks with the dense 3/5/7/9/11 Poly Kernel Inception convolutional mixer. Preserve Bottleneck.cv1 and every other layer.

Weight Transfer: Official yolo11n.pt, exact-name exact-shape transfer with pre-epoch trainer handoff audit
Loaded/Total: Pending notebook preflight

Training:
Epoch: 150
imgsz: 640
batch: 8

Results:
Precision: Pending
Recall: Pending
mAP50: Pending
mAP50-95: Pending

Analysis: Pending

Conclusion: Pending

Next Step: Compare against the same baseline and the other convolution screens under the fixed protocol.
