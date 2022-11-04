"""
The example to evaluate a pretrained object detection model in COCO format.

An example to evaluate a pretrained model on COCO dataset:
    python eval_pretrained_coco_format.py \
        --test_path coco17/annotations/instances_val2017.json \
        --checkpoint_name yolov3_mobilenetv2_320_300e_coco

An example to evaluate a pretrained model on VOC dataset (COCO format):
    python eval_pretrained_coco_format.py \
        --test_path VOCdevkit/VOCCOCO/voc07_test.json \
        --checkpoint_name faster_rcnn_r50_fpn_1x_voc0712
"""

import argparse

from autogluon.multimodal import MultiModalPredictor


def tutorial_script_for_eval_pretrained_coco_format():
    # this code block is used in tutorial
    checkpoint_name = "yolov3_mobilenetv2_320_300e_coco"
    num_gpus = -1  # here we use all available GPUs

    predictor = MultiModalPredictor(
        hyperparameters={
            "model.mmdet_image.checkpoint_name": checkpoint_name,
            "env.num_gpus": num_gpus,
        },
        problem_type="object_detection",
    )

    test_path = "coco17/annotations/instances_val2017.json"

    predictor.evaluate(test_path)


def eval_pretrained_coco_format(
    checkpoint_name="yolov3_mobilenetv2_320_300e_coco",
    test_path="coco17/annotations/instances_val2017.json",
    num_gpus=-1,
):
    # TODO: replace pipeline with problem type
    predictor = MultiModalPredictor(
        hyperparameters={
            "model.mmdet_image.checkpoint_name": checkpoint_name,
            "env.num_gpus": num_gpus,
        },
        pipeline="object_detection",
    )

    result = predictor.evaluate(test_path)

    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_path", default="coco17/annotations/instances_val2017.json", type=str)
    parser.add_argument("--checkpoint_name", default="yolov3_mobilenetv2_320_300e_coco", type=str)
    parser.add_argument("--num_gpus", default=-1, type=int)
    args = parser.parse_args()

    eval_pretrained_coco_format(
        test_path=args.test_path,
        checkpoint_name=args.checkpoint_name,
        num_gpus=args.num_gpus,
    )
