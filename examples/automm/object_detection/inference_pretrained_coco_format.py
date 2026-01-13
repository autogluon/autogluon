"""
The example to evaluate a pretrained object detection model in COCO format.
An example to evaluate a pretrained model on COCO dataset:
    python inference_pretrained_coco_format.py \
        --test_path coco17/annotations/instances_val2017.json \
        --checkpoint_name yolov3_mobilenetv2_320_300e_coco \
An example to evaluate a pretrained model on VOC dataset (COCO format):
    python inference_pretrained_coco_format.py \
        --test_path ./VOCdevkit/VOC2007/Annotations/test_cocoformat.json \
        --checkpoint_name faster_rcnn_r50_fpn_1x_voc0712 \
If you want to save results, enable the following:
--save_results
If you want to specify a save result path, add the following:
--result_path VOCdevkit/VOC2007/results.txt
"""

import argparse

from autogluon.multimodal import MultiModalPredictor
from autogluon.multimodal.utils import COCODataset, from_coco


def tutorial_script_for_eval_pretrained_coco_format():
    # this code block is used in tutorial
    checkpoint_name = "yolov3_mobilenetv2_320_300e_coco"
    num_gpus = -1  # here we use all available GPUs

    # construct the predictor
    predictor = MultiModalPredictor(
        hyperparameters={
            "model.mmdet_image.checkpoint_name": checkpoint_name,
            "env.num_gpus": num_gpus,
        },
        problem_type="object_detection",
    )

    test_path = "coco17/annotations/instances_val2017.json"

    pred = predictor.predict(test_path, save_results=True)


def eval_pretrained_coco_format(
    checkpoint_name="yolov3_mobilenetv2_320_300e_coco",
    test_path="coco17/annotations/instances_val2017.json",
    num_gpus=-1,
    save_results=True,
):
    predictor = MultiModalPredictor(
        hyperparameters={
            "model.mmdet_image.checkpoint_name": checkpoint_name,
            "env.num_gpus": num_gpus,
        },
        pipeline="object_detection",
    )

    pred = predictor.predict(test_path, save_results=save_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_path", default="coco17/annotations/instances_val2017.json", type=str)
    parser.add_argument("--checkpoint_name", default="yolov3_mobilenetv2_320_300e_coco", type=str)
    parser.add_argument("--num_gpus", default=-1, type=int)
    parser.add_argument("--save_results", action="store_true")
    args = parser.parse_args()

    eval_pretrained_coco_format(
        test_path=args.test_path,
        checkpoint_name=args.checkpoint_name,
        num_gpus=args.num_gpus,
        save_results=args.save_results,
    )
