"""X05: official MMDetection RTMDet-Tiny ship baseline, 150 epochs."""

_base_ = "./rtmdet_tiny_ship_common.py"

experiment_id = "X05_RTMDet_Tiny_Official_150ep"
architecture_change = "None; official RTMDet-Tiny CSPNeXt + CSPNeXt-PAFPN."
pretrained_source = (
    "Official MMDetection RTMDet-Tiny COCO checkpoint; all same-name, "
    "same-shape tensors are loaded and audited at runtime."
)
