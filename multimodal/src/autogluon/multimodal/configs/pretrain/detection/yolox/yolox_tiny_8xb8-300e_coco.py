_base_ = "./yolox_s_8xb8-300e_coco.py"

# model settings
model = dict(
    data_preprocessor=dict(
        batch_augments=[dict(type="BatchSyncRandomResize", random_size_range=(320, 640), size_divisor=32, interval=10)]
    ),
    backbone=dict(deepen_factor=0.33, widen_factor=0.375),
    neck=dict(in_channels=[96, 192, 384], out_channels=96),
    bbox_head=dict(in_channels=96, feat_channels=96),
)

img_scale = (640, 640)  # width, height

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(416, 416), keep_ratio=True),
    dict(type="Pad", pad_to_square=True, pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="PackDetInputs", meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor")),
]
