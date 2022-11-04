"""
The example to evaluate a pretrained object detection model in VOC format.

An example to evaluate a pretrained model on VOC dataset (VOC format):
    python eval_pretrained_coco_format.py \
        --test_path VOCdevkit/VOC2007 \
        --checkpoint_name faster_rcnn_r50_fpn_1x_voc0712
"""

import argparse

from autogluon.multimodal import MultiModalPredictor


def tutorial_script_for_eval_pretrained_voc_format():
    # this code block is used in tutorial
    checkpoint_name = "faster_rcnn_r50_fpn_1x_voc0712"
    num_gpus = -1  # here we use all available GPUs

    predictor = MultiModalPredictor(
        hyperparameters={
            "model.mmdet_image.checkpoint_name": checkpoint_name,
            "env.num_gpus": num_gpus,
        },
        problem_type="object_detection",
    )

    test_path = "VOCdevkit/VOC2007"

    result = predictor.evaluate(test_path)

    print(result)


def eval_pretrained_voc_format(
    checkpoint_name="faster_rcnn_r50_fpn_1x_voc0712",
    test_path="VOCdevkit/VOC2007",
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
    parser.add_argument("--test_path", default="VOCdevkit/VOC2007", type=str)
    parser.add_argument("--checkpoint_name", default="faster_rcnn_r50_fpn_1x_voc0712", type=str)
    parser.add_argument("--num_gpus", default=-1, type=int)
    args = parser.parse_args()

    eval_pretrained_voc_format(
        test_path=args.test_path,
        checkpoint_name=args.checkpoint_name,
        num_gpus=args.num_gpus,
    )
