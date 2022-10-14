"""
The example to train an object detection model in AutoMM.

An example to finetune an MMDetection model on COCO:
    python detection_train.py \
        --train_path coco17/annotations/instances_train2017.json
        --test_path coco17/annotations/instances_val2017.json
        --checkpoint_name yolov3_mobilenetv2_320_300e_coco
        --num_classes 80
        --lr <learning_rate>
        --wd <weight_decay>
        --epochs <epochs>

An example to finetune an MMDetection model on VOC:
    First, use this script to convert the VOC dataset to COCO format:
    https://github.com/open-mmlab/mmdetection/blob/9d3e162459590eee4cfc891218dfbb5878378842/tools/dataset_converters/pascal_voc.py
    Then, run:
    python detection_train.py \
        --train_path /media/data/datasets/voc/VOCdevkit/VOCCOCO/voc07_trainval.json
        --test_path /media/data/datasets/voc/VOCdevkit/VOCCOCO/voc07_test.json
        --checkpoint_name yolov3_mobilenetv2_320_300e_coco
        --num_classes 20
        --lr <learning_rate>
        --wd <weight_decay>
        --epochs <epochs>

Note that for now it's required to install nightly build torchmetrics.
This will be solved in next pr. (MeanAveragePrecision will be moved to AG temporarily.)
"""

import argparse

from autogluon.multimodal import MultiModalPredictor

def detection_train(
    train_path,
    test_path=None,
    checkpoint_name="faster_rcnn_r50_fpn_2x_coco",
    num_classes=80,
    lr=1e-3,
    wd=1e-4,
    epochs=50,
):

    # TODO: add val_path
    # TODO: remove hardcode for num_classes

    predictor = MultiModalPredictor(
        label="rois_label",
        hyperparameters={
            "model.mmdet_image.checkpoint_name": checkpoint_name,
            "env.num_gpus": 1,
        },
        pipeline="object_detection",
        output_shape=num_classes,
    )

    import time

    start = time.time()
    predictor.fit(
        train_path,
        hyperparameters={
            "optimization.learning_rate": lr,
            "optimization.weight_decay": wd,
            "optimization.max_epochs": epochs,
            # "env.per_gpu_batch_size": 4, # decrease it when model is large
        },
    )
    fit_end = time.time()

    if test_path is not None:
        print("time usage for fit: %.2f" % (fit_end - start))
        predictor.evaluate(test_path)
        print("time usage for eval: %.2f" % (time.time() - fit_end))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", default="/media/data/datasets/voc/VOCdevkit/VOCCOCO/voc07_trainval.json", type=str)
    parser.add_argument("--test_path", default="/media/data/datasets/voc/VOCdevkit/VOCCOCO/voc07_test.json", type=str)
    parser.add_argument("--checkpoint_name", default="yolov3_mobilenetv2_320_300e_coco", type=str)
    parser.add_argument("--num_classes", default=20, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--wd", default=1e-3, type=float)
    parser.add_argument("--epochs", default=50, type=int)
    args = parser.parse_args()

    detection_train(
        train_path=args.train_path,
        test_path=args.test_path,
        checkpoint_name=args.checkpoint_name,
        num_classes=args.num_classes,
        lr=args.lr,
        wd=args.wd,
        epochs=args.epochs,
    )