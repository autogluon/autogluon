_base_ = ["../_base_/default_runtime.py"]
# model settings
model = dict(
    type="YOLOV3",
    backbone=dict(
        type="Darknet",
        depth=53,
        out_indices=(3, 4, 5),
        init_cfg=dict(type="Pretrained", checkpoint="open-mmlab://darknet53"),
    ),
    neck=dict(type="YOLOV3Neck", num_scales=3, in_channels=[1024, 512, 256], out_channels=[512, 256, 128]),
    bbox_head=dict(
        type="YOLOV3Head",
        num_classes=600,
        in_channels=[512, 256, 128],
        out_channels=[1024, 512, 256],
        anchor_generator=dict(
            type="YOLOAnchorGenerator",
            base_sizes=[
                [(116, 90), (156, 198), (373, 326)],
                [(30, 61), (62, 45), (59, 119)],
                [(10, 13), (16, 30), (33, 23)],
            ],
            strides=[32, 16, 8],
        ),
        bbox_coder=dict(type="YOLOBBoxCoder"),
        featmap_strides=[32, 16, 8],
        loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0, reduction="sum"),
        loss_conf=dict(type="CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0, reduction="sum"),
        loss_xy=dict(type="CrossEntropyLoss", use_sigmoid=True, loss_weight=2.0, reduction="sum"),
        loss_wh=dict(type="MSELoss", loss_weight=2.0, reduction="sum"),
    ),
    # training and testing settings
    train_cfg=dict(assigner=dict(type="GridAssigner", pos_iou_thr=0.5, neg_iou_thr=0.5, min_pos_iou=0)),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        conf_thr=0.005,
        nms=dict(type="nms", iou_threshold=0.45),
        max_per_img=100,
    ),
)
# dataset settings

img_norm_cfg = dict(mean=[0, 0, 0], std=[255.0, 255.0, 255.0], to_rgb=True)
train_pipeline = [
    dict(type="LoadImageFromFile", to_float32=True),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="PhotoMetricDistortion"),
    dict(type="Expand", mean=img_norm_cfg["mean"], to_rgb=img_norm_cfg["to_rgb"], ratio_range=(1, 2)),
    dict(type="MinIoURandomCrop", min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9), min_crop_size=0.3),
    dict(type="Resize", img_scale=[(320, 320), (608, 608)], keep_ratio=True),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(608, 608),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=32),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]
data = dict(train=dict(pipeline=train_pipeline), val=dict(pipeline=test_pipeline), test=dict(pipeline=test_pipeline))

# fp16 settings
# fp16 = dict(loss_scale='dynamic')
