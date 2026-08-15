"""X06: final method transferred to native MMDetection RTMDet-Tiny."""

_base_ = "./rtmdet_tiny_ship_common.py"

custom_imports = dict(
    imports=["custom_modules.rtmdet_transfer"],
    allow_failed_imports=False,
)
experiment_id = "X06_RTMDet_Tiny_InceptionDW_DPLS_CA_SCAM_VGUP_150ep"
architecture_change = (
    "VGUP input; shallow P2/P3 CSPNeXt conv2 spatial InceptionDW; "
    "P2/P3/P4 DPLS with two DySample; three bounded CA-SCAM before the "
    "unchanged RTMDetSepBNHead."
)
pretrained_source = (
    "Official MMDetection RTMDet-Tiny COCO checkpoint; every compatible "
    "native tensor is loaded, while new or shape-changed tensors are reported."
)

model = dict(
    backbone=dict(
        _delete_=True,
        type="RTMDetVGUPCSPNeXt",
        arch="P5",
        arch_ovewrite=[
            [64, 128, 3, True, False],
            [128, 256, 6, True, False],
            [256, 512, 6, True, False],
        ],
        expand_ratio=0.5,
        deepen_factor=0.167,
        widen_factor=0.375,
        out_indices=(1, 2, 3),
        channel_attention=True,
        shallow_inception_stages=(1, 2),
        vgup_bpw_segments=8,
        vgup_prediction_size=128,
        inception_square_kernel_size=3,
        inception_band_kernel_size=11,
        inception_branch_ratio=0.125,
        input_mean_bgr=(103.53, 116.28, 123.675),
        input_std_bgr=(57.375, 57.12, 58.395),
        norm_cfg={{_base_.norm_cfg}},
        act_cfg={{_base_.act_cfg}},
        init_cfg=None,
    ),
    neck=dict(
        _delete_=True,
        type="RTMDetDPLSCASCAMPAFPN",
        in_channels=[48, 96, 192],
        out_channels=96,
        num_csp_blocks=1,
        expand_ratio=0.5,
        dysample_groups=4,
        dysample_style="lp",
        dysample_scope=False,
        cascam_max_delta=0.1,
        norm_cfg={{_base_.norm_cfg}},
        act_cfg={{_base_.act_cfg}},
        # Preserve DySample's small-offset initialization and CA-SCAM's
        # zero-initialized bounded calibration instead of recursively applying
        # CSPNeXtPAFPN's generic Kaiming rule to every newly inserted Conv2d.
        init_cfg=None,
    ),
    bbox_head=dict(
        anchor_generator=dict(
            type="MlvlPointGenerator",
            offset=0,
            strides=[4, 8, 16],
        )
    ),
)
