"""Shared official RTMDet-Tiny recipe for the two ship experiments.

Global controls are fixed at 640 input, 150 epochs, batch 8, and seed 0.  The
optimizer, losses, dynamic assigner, EMA, augmentation family, and head remain
RTMDet-native.  Both variants inherit this file, so architecture is the only
between-run variable.
"""

default_scope = "mmdet"
backend_args = None
data_root = "/content/ship_detection/rtmdet_coco/"
metainfo = dict(classes=("ship",), palette=[(220, 20, 60)])

norm_cfg = dict(type="BN", momentum=0.03, eps=0.001)
act_cfg = dict(type="SiLU", inplace=True)

model = dict(
    type="RTMDet",
    data_preprocessor=dict(
        type="DetDataPreprocessor",
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
        bgr_to_rgb=False,
        batch_augments=None,
    ),
    backbone=dict(
        type="CSPNeXt",
        arch="P5",
        expand_ratio=0.5,
        deepen_factor=0.167,
        widen_factor=0.375,
        out_indices=(2, 3, 4),
        channel_attention=True,
        norm_cfg=norm_cfg,
        act_cfg=act_cfg,
        init_cfg=None,
    ),
    neck=dict(
        type="CSPNeXtPAFPN",
        in_channels=[96, 192, 384],
        out_channels=96,
        num_csp_blocks=1,
        expand_ratio=0.5,
        norm_cfg=norm_cfg,
        act_cfg=act_cfg,
    ),
    bbox_head=dict(
        type="RTMDetSepBNHead",
        num_classes=1,
        in_channels=96,
        stacked_convs=2,
        feat_channels=96,
        anchor_generator=dict(
            type="MlvlPointGenerator",
            offset=0,
            strides=[8, 16, 32],
        ),
        bbox_coder=dict(type="DistancePointBBoxCoder"),
        loss_cls=dict(
            type="QualityFocalLoss",
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0,
        ),
        loss_bbox=dict(type="GIoULoss", loss_weight=2.0),
        with_objectness=False,
        exp_on_reg=False,
        share_conv=True,
        pred_kernel_size=1,
        norm_cfg=norm_cfg,
        act_cfg=act_cfg,
    ),
    train_cfg=dict(
        assigner=dict(type="DynamicSoftLabelAssigner", topk=13),
        allowed_border=-1,
        pos_weight=-1,
        debug=False,
    ),
    test_cfg=dict(
        nms_pre=30000,
        min_bbox_size=0,
        score_thr=0.001,
        nms=dict(type="nms", iou_threshold=0.65),
        max_per_img=300,
    ),
)

train_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="CachedMosaic",
        img_scale=(640, 640),
        pad_val=114.0,
        max_cached_images=20,
        random_pop=False,
    ),
    dict(
        type="RandomResize",
        scale=(1280, 1280),
        ratio_range=(0.5, 2.0),
        keep_ratio=True,
    ),
    dict(type="RandomCrop", crop_size=(640, 640)),
    dict(type="YOLOXHSVRandomAug"),
    dict(type="RandomFlip", prob=0.5),
    dict(type="Pad", size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(
        type="CachedMixUp",
        img_scale=(640, 640),
        ratio_range=(1.0, 1.0),
        max_cached_images=10,
        random_pop=False,
        pad_val=(114, 114, 114),
        prob=0.5,
    ),
    dict(type="PackDetInputs"),
]

train_pipeline_stage2 = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="RandomResize",
        scale=(640, 640),
        ratio_range=(0.5, 2.0),
        keep_ratio=True,
    ),
    dict(type="RandomCrop", crop_size=(640, 640)),
    dict(type="YOLOXHSVRandomAug"),
    dict(type="RandomFlip", prob=0.5),
    dict(type="Pad", size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(type="PackDetInputs"),
]

test_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="Resize", scale=(640, 640), keep_ratio=True),
    dict(type="Pad", size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="PackDetInputs",
        meta_keys=(
            "img_id",
            "img_path",
            "ori_shape",
            "img_shape",
            "scale_factor",
        ),
    ),
]

train_dataloader = dict(
    batch_size=8,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=True),
    batch_sampler=None,
    dataset=dict(
        type="CocoDataset",
        data_root=data_root,
        ann_file="annotations/instances_train.json",
        data_prefix=dict(img="images/train/"),
        metainfo=metainfo,
        filter_cfg=dict(filter_empty_gt=False, min_size=1),
        pipeline=train_pipeline,
        backend_args=backend_args,
    ),
)

val_dataloader = dict(
    batch_size=8,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="CocoDataset",
        data_root=data_root,
        ann_file="annotations/instances_val.json",
        data_prefix=dict(img="images/val/"),
        metainfo=metainfo,
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args,
    ),
)

test_dataloader = dict(
    batch_size=8,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="CocoDataset",
        data_root=data_root,
        ann_file="annotations/instances_test.json",
        data_prefix=dict(img="images/test/"),
        metainfo=metainfo,
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args,
    ),
)

val_evaluator = dict(
    type="CocoMetric",
    ann_file=data_root + "annotations/instances_val.json",
    metric="bbox",
    format_only=False,
    classwise=False,
    backend_args=backend_args,
)
test_evaluator = dict(
    type="CocoMetric",
    ann_file=data_root + "annotations/instances_test.json",
    metric="bbox",
    format_only=False,
    classwise=False,
    backend_args=backend_args,
)

max_epochs = 150
stage2_num_epochs = 10
base_lr = 0.004

train_cfg = dict(
    type="EpochBasedTrainLoop",
    max_epochs=max_epochs,
    val_interval=1,
)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

optim_wrapper = dict(
    type="AmpOptimWrapper",
    loss_scale="dynamic",
    optimizer=dict(type="AdamW", lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0,
        bias_decay_mult=0,
        bypass_duplicate=True,
    ),
)
param_scheduler = [
    dict(
        type="LinearLR",
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=1000,
    ),
    dict(
        type="CosineAnnealingLR",
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
]
auto_scale_lr = dict(enable=True, base_batch_size=256)

default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=20),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(
        type="CheckpointHook",
        interval=1,
        save_last=True,
        save_best="coco/bbox_mAP",
        rule="greater",
        max_keep_ckpts=3,
    ),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="DetVisualizationHook"),
)
custom_hooks = [
    dict(
        type="EMAHook",
        ema_type="ExpMomentumEMA",
        momentum=0.0002,
        update_buffers=True,
        priority=49,
    ),
    dict(
        type="PipelineSwitchHook",
        switch_epoch=max_epochs - stage2_num_epochs,
        switch_pipeline=train_pipeline_stage2,
    ),
]

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)
vis_backends = [dict(type="LocalVisBackend")]
visualizer = dict(
    type="DetLocalVisualizer",
    vis_backends=vis_backends,
    name="visualizer",
)
log_processor = dict(type="LogProcessor", window_size=50, by_epoch=True)
log_level = "INFO"
load_from = None
resume = False
randomness = dict(seed=0, deterministic=False, diff_rank_seed=False)
launcher = "none"
work_dir = "/content/formal_runs/RTMDet_Tiny/seed_0"
