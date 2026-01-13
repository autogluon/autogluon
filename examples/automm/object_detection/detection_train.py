"""
The example to train an object detection model in AutoMM.

An example to finetune an MMDetection model on COCO:
    python detection_train.py \
        --train_path coco17/annotations/instances_train2017.json \
        --test_path coco17/annotations/instances_val2017.json \
        --checkpoint_name yolov3_mobilenetv2_320_300e_coco \
        --num_classes 80 \
        --lr <learning_rate> \
        --wd <weight_decay> \
        --epochs <epochs>

An example to finetune an MMDetection model on VOC:
    First, use this script to convert the VOC dataset to COCO format:
    https://github.com/open-mmlab/mmdetection/blob/9d3e162459590eee4cfc891218dfbb5878378842/tools/dataset_converters/pascal_voc.py
    Then, run:
    python detection_train.py \
        --train_path ./VOCdevkit/VOC2007/Annotations/cocotrain.json \
        --test_path ./VOCdevkit/VOC2007/Annotations/test_cocoformat.json \
        --checkpoint_name yolov3_mobilenetv2_320_300e_coco \
        --num_classes 20 \
        --lr <learning_rate> \
        --wd <weight_decay> \
        --epochs <epochs>
"""

import argparse
import os

from autogluon.multimodal import MultiModalPredictor
from autogluon.multimodal.utils import get_detection_classes


def detection_train(
    train_path,
    val_path=None,
    test_path=None,
    checkpoint_name="faster_rcnn_r50_fpn_2x_coco",
    num_classes=80,
    lr=1e-3,
    epochs=50,
    num_gpus=4,
    val_metric=None,
    per_gpu_batch_size=8,
):
    # TODO: add val_path
    # TODO: remove hardcode for num_classes

    # TODO: move this code to predictor
    classes = None
    eval_tool = None
    VOC_format = False
    if os.path.isdir(train_path):
        classes = get_detection_classes(train_path)
        eval_tool = "torchmetrics"
        VOC_format = True

    predictor = MultiModalPredictor(
        label="label",
        hyperparameters={
            "model.mmdet_image.checkpoint_name": checkpoint_name,
            "env.num_gpus": num_gpus,
            "optim.val_metric": val_metric,
        },
        problem_type="object_detection",
        num_classes=num_classes,
        classes=classes,
    )

    import time

    start = time.time()
    predictor.fit(
        train_path,
        tuning_data=val_path,
        hyperparameters={
            "optim.lr": lr / 100,  # we use two stage and lr_mult=100 for detection
            "optim.max_epochs": epochs,
            "env.per_gpu_batch_size": per_gpu_batch_size,  # decrease it when model is large
        },
    )
    fit_end = time.time()
    print("time usage for fit: %.2f" % (fit_end - start))

    if test_path is not None:
        if (not eval_tool) or eval_tool == "pycocotools" or (eval_tool == "torchmetrics" and num_gpus == 1):
            print(predictor.evaluate(test_path, eval_tool=eval_tool))
            print("time usage for eval: %.2f" % (time.time() - fit_end))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", default="./VOCdevkit/VOC2007/Annotations/train_cocoformat.json", type=str)
    parser.add_argument("--val_path", default=None, type=str)
    parser.add_argument("--test_path", default=None, type=str)
    parser.add_argument("--checkpoint_name", default="yolov3_mobilenetv2_320_300e_coco", type=str)
    parser.add_argument("--num_classes", default=20, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--num_gpus", default=4, type=int)
    parser.add_argument("--per_gpu_batch_size", default=8, type=int)
    parser.add_argument("--val_metric", default=None, type=str)
    args = parser.parse_args()

    detection_train(
        train_path=args.train_path,
        val_path=args.val_path,
        test_path=args.test_path,
        checkpoint_name=args.checkpoint_name,
        num_classes=args.num_classes,
        lr=args.lr,
        epochs=args.epochs,
        num_gpus=args.num_gpus,
        val_metric=args.val_metric,  # "mAP" or "direct_loss" or None (use default: "direct_loss")
        per_gpu_batch_size=args.per_gpu_batch_size,
    )
