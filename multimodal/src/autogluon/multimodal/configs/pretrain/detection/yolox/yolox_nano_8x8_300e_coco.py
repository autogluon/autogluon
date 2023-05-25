_base_ = "./yolox_s_8x8_300e_coco.py"


# model settings
model = dict(
    backbone=dict(deepen_factor=0.33, widen_factor=0.25, use_depthwise=True),
    neck=dict(in_channels=[64, 128, 256], out_channels=64, num_csp_blocks=1, use_depthwise=True),
    bbox_head=dict(in_channels=64, feat_channels=64, use_depthwise=True),
)

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(416, 416),  # Use 416x416 for inference
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Pad", pad_to_square=True, pad_val=dict(img=(114.0, 114.0, 114.0))),
            dict(type="DefaultFormatBundle"),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]
