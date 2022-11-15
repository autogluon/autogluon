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
        --train_path ./VOCdevkit/VOC2007/Annotations/train_cocoformat.json \
        --test_path ./VOCdevkit/VOC2007/Annotations/test_cocoformat.json \
        --checkpoint_name yolov3_mobilenetv2_320_300e_coco \
        --num_classes 20 \
        --lr <learning_rate> \
        --wd <weight_decay> \
        --epochs <epochs>

Note that for now it's required to install nightly build torchmetrics.
This will be solved in next pr. (MeanAveragePrecision will be moved to AG temporarily.)
"""

import argparse
import os

from autogluon.multimodal import MultiModalPredictor
from autogluon.multimodal.utils import get_voc_classes


def watercolor_benchmark(
    checkpoint_name,
    lr,
    per_gpu_batch_size,
):

    train_path = "/media/code/detdata/DetBenchmark/watercolor/Annotations/train_cocoformat.json"
    val_path = "/media/code/detdata/DetBenchmark/watercolor/Annotations/val_cocoformat.json"
    test_path = "/media/code/detdata/DetBenchmark/watercolor/Annotations/test_cocoformat.json"
    epochs = 20
    val_metric = "map"

    predictor = MultiModalPredictor(
        label="label",
        hyperparameters={
            "model.mmdet_image.checkpoint_name": checkpoint_name,
            "env.num_gpus": 4,
            "optimization.val_metric": val_metric,
        },
        problem_type="object_detection",
        sample_data_path=train_path,
    )

    import time
    start = time.time()
    predictor.fit(
        train_path,
        tuning_data=val_path,
        hyperparameters={
            "optimization.learning_rate": lr/100, # we use two stage and lr_mult=100 for detection
            "optimization.max_epochs": epochs,
            "optimization.patience": 10,
            "env.per_gpu_batch_size": per_gpu_batch_size,  # decrease it when model is large
        },
    )
    fit_end = time.time()
    print("time usage for fit: %.2f" % (fit_end - start))

    predictor.evaluate(test_path)
    print("time usage for eval: %.2f" % (time.time() - fit_end))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_name", type=str)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--per_gpu_batch_size", type=int)
    args = parser.parse_args()

    watercolor_benchmark(
        checkpoint_name=args.checkpoint_name,
        lr=args.lr,
        per_gpu_batch_size=args.per_gpu_batch_size,
    )
