_base_ = ["../schedule_1x.py", "../default_runtime.py", "./yolox_tta.py"]

img_scale = (640, 640)  # width, height

# model settings
model = dict(
    type="YOLOX",
    data_preprocessor=dict(
        type="DetDataPreprocessor",
        pad_size_divisor=32,
        batch_augments=[
            dict(type="BatchSyncRandomResize", random_size_range=(480, 800), size_divisor=32, interval=10)
        ],
    ),
    backbone=dict(
        type="CSPDarknet",
        deepen_factor=0.33,
        widen_factor=0.5,
        out_indices=(2, 3, 4),
        use_depthwise=False,
        spp_kernal_sizes=(5, 9, 13),
        norm_cfg=dict(type="BN", momentum=0.03, eps=0.001),
        act_cfg=dict(type="Swish"),
    ),
    neck=dict(
        type="YOLOXPAFPN",
        in_channels=[128, 256, 512],
        out_channels=128,
        num_csp_blocks=1,
        use_depthwise=False,
        upsample_cfg=dict(scale_factor=2, mode="nearest"),
        norm_cfg=dict(type="BN", momentum=0.03, eps=0.001),
        act_cfg=dict(type="Swish"),
    ),
    bbox_head=dict(
        type="YOLOXHead",
        num_classes=80,
        in_channels=128,
        feat_channels=128,
        stacked_convs=2,
        strides=(8, 16, 32),
        use_depthwise=False,
        norm_cfg=dict(type="BN", momentum=0.03, eps=0.001),
        act_cfg=dict(type="Swish"),
        loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=True, reduction="sum", loss_weight=1.0),
        loss_bbox=dict(type="IoULoss", mode="square", eps=1e-16, reduction="sum", loss_weight=5.0),
        loss_obj=dict(type="CrossEntropyLoss", use_sigmoid=True, reduction="sum", loss_weight=1.0),
        loss_l1=dict(type="L1Loss", reduction="sum", loss_weight=1.0),
    ),
    train_cfg=dict(assigner=dict(type="SimOTAAssigner", center_radius=2.5)),
    # In order to align the source code, the threshold of the val phase is
    # 0.01, and the threshold of the test phase is 0.001.
    test_cfg=dict(score_thr=0.01, nms=dict(type="nms", iou_threshold=0.65)),
)

loading_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
]

multi_image_mix_dataset = dict(
    mosaic=dict(
        img_scale=img_scale,
        center_ratio_range=(0.5, 1.5),
        bbox_clip_border=True,
        pad_val=114.0,
        prob=0.5,
    ),
    # TODO: add random affine    dict(
    # RandomAffine=dict(
    #     scaling_ratio_range=(0.1, 2),
    #     # img_scale is (width, height)
    #     border=(-img_scale[0] // 2, -img_scale[1] // 2),
    # ),
    mixup=dict(
        img_scale=img_scale,
        ratio_range=(0.8, 1.6),
        flip_ratio=0.5,
        pad_val=114.0,
        max_iters=15,
        bbox_clip_border=True,
    ),
)

train_pipeline = [
    dict(type="YOLOXHSVRandomAug"),
    dict(type="RandomFlip", prob=0.5),
    # According to the official implementation, multi-scale
    # training is not considered here but in the
    # 'mmdet/models/detectors/yolox.py'.
    # Resize and Pad are for the last 15 epochs when Mosaic,
    # RandomAffine, and MixUp are closed by YOLOXModeSwitchHook.
    dict(type="Resize", scale=img_scale, keep_ratio=True),
    dict(
        type="Pad",
        pad_to_square=True,
        # If the image is three-channel, the pad value needs
        # to be set separately for each channel.
        pad_val=dict(img=(114.0, 114.0, 114.0)),
    ),
    dict(type="FilterAnnotations", min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type="PackDetInputs"),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=img_scale, keep_ratio=True),
    dict(type="Pad", pad_to_square=True, pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="PackDetInputs", meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor")),
]
