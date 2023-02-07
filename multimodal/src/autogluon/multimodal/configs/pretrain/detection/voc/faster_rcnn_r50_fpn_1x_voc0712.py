_base_ = ["../faster_rcnn/faster_rcnn_r50_fpn.py", "./voc0712.py", "../default_runtime.py"]
model = dict(roi_head=dict(bbox_head=dict(num_classes=20)))
# optimizer
optimizer = dict(type="SGD", lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
# actual epoch = 3 * 3 = 9
lr_config = dict(policy="step", step=[3])
# runtime settings
runner = dict(type="EpochBasedRunner", max_epochs=4)  # actual epoch = 4 * 3 = 12
