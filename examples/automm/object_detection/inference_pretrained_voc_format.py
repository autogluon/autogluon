"""
The example to evaluate a pretrained object detection model in VOC format.
An example to evaluate a pretrained model on VOC dataset (VOC format):
    python inference_pretrained_voc_format.py \
        --test_path VOCdevkit/VOC2007 \
        --checkpoint_name faster_rcnn_r50_fpn_1x_voc0712 \
If you want to save results, enable the following:
--save_results
If you want to specify a save result path, add the following:
--result_path VOCdevkit/VOC2007/results.txt
"""

import argparse
import os

import numpy as np

from autogluon.multimodal import MultiModalPredictor
from autogluon.multimodal.utils import from_voc


def tutorial_script_for_eval_pretrained_voc_format():
    # this code block is used in tutorial
    checkpoint_name = "faster_rcnn_r50_fpn_1x_voc0712"
    num_gpus = -1  # here we use all available GPUs

    # construct the predictor
    predictor = MultiModalPredictor(
        hyperparameters={
            "model.mmdet_image.checkpoint_name": checkpoint_name,
            "env.num_gpus": num_gpus,
        },
        problem_type="object_detection",
    )

    test_path = "VOCdevkit/VOC2007"

    pred = predictor.predict(test_path, save_results=True)


def eval_pretrained_voc_format(
    checkpoint_name="faster_rcnn_r50_fpn_1x_voc0712",
    test_path="VOCdevkit/VOC2007",
    num_gpus=-1,
    save_results=True,
    result_path=None,
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
    parser.add_argument("--test_path", default="VOCdevkit/VOC2007", type=str)
    parser.add_argument("--checkpoint_name", default="faster_rcnn_r50_fpn_1x_voc0712", type=str)
    parser.add_argument("--num_gpus", default=1, type=int)
    parser.add_argument("--save_results", action="store_true")
    args = parser.parse_args()

    eval_pretrained_voc_format(
        test_path=args.test_path,
        checkpoint_name=args.checkpoint_name,
        num_gpus=args.num_gpus,
        save_results=args.save_results,
    )
