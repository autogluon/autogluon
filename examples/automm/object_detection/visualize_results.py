"""
The example to evaluate a pretrained object detection model in VOC format.
An example to evaluate a pretrained model on VOC dataset (VOC format):
    python inference_pretrained_voc_format.py \
        --test_path VOCdevkit/VOC2007 \
        --checkpoint_name faster_rcnn_r50_fpn_1x_voc0712 \
        --result_path VOCdevkit/VOC2007/results.npy
"""

import argparse
import numpy as np
import os

from autogluon.multimodal import MultiModalPredictor
from autogluon.multimodal.utils import from_voc, visualize_results, from_coco


def tutorial_script_for_visualize_detection_results():
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

    df = from_voc(test_path)[:10][["image"]]
    pred = predictor.predict(df, as_pandas=False)


def visualize_detection_results(
        checkpoint_name="faster_rcnn_r50_fpn_1x_voc0712",
        test_path="VOCdevkit/VOC2007",
        num_gpus=-1,
        visualization_result_dir="VOCdevkit/VOC2007/visualizations"
):
    # TODO: remove label
    # TODO: replace pipeline with problem type
    predictor = MultiModalPredictor(
        # label="label",
        hyperparameters={
            "model.mmdet_image.checkpoint_name": checkpoint_name,
            "env.num_gpus": num_gpus,
        },
        pipeline="object_detection",
    )

    # result = predictor.evaluate(test_path)

    # df = from_voc(test_path)[:10][["image"]]
    df = from_coco(test_path)[:10][["image"]]
    pred = predictor.predict(df, as_pandas=False)


    visualized_image = visualize_results(pred[0], df.iloc[0]["image"], test_path, visualization_result_dir)



    # pred = predictor.predict(test_path, as_pandas=False, result_path=result_path)
    # dump testing results in npy
    # np.save(result_path, pred)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_path", default="VOCdevkit/VOC2007", type=str)
    parser.add_argument("--checkpoint_name", default="faster_rcnn_r50_fpn_1x_voc0712", type=str)
    parser.add_argument("--num_gpus", default=1, type=int)
    parser.add_argument("--visualization_result_dir", default="VOCdevkit/VOC2007/visualizations", type=str)
    args = parser.parse_args()

    visualize_detection_results(
        test_path=args.test_path,
        checkpoint_name=args.checkpoint_name,
        num_gpus=args.num_gpus,
        visualization_result_dir=args.visualization_result_dir
    )
