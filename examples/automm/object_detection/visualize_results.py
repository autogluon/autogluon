"""
The example to visualize detection results in COCO dataset (COCO format):
    python visualize_results.py \
    --test_path ~/yongxinw-workspace/tools/coco17/annotations/instances_val2017.json \
    --checkpoint_name vfnet_x101_64x4d_fpn_mdconv_c3-c5_mstrain_2x_coco \
    --conf_threshold 0.4
If you want to specify a folder to save visualizations, add the following:
--visualization_result_dir VOCdevkit/VOC2007/visualizations
"""

import argparse

from autogluon.multimodal import MultiModalPredictor
from autogluon.multimodal.utils import from_coco_or_voc, visualize_detection


def tutorial_script_for_visualize_detection_results():
    # this code block is used in tutorial
    checkpoint_name = "vfnet_x101_64x4d_fpn_mdconv_c3-c5_mstrain_2x_coco"
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
    visualization_result_dir = "coco17/visualizations"
    conf_threshold = 0.4

    df = from_coco_or_voc(test_path)[:10][["image"]]

    pred = predictor.predict(df)

    visualize_detection(
        pred=pred,
        detection_classes=predictor.classes,
        conf_threshold=conf_threshold,
        visualization_result_dir=visualization_result_dir,
    )


def visualize_detection_results(
    checkpoint_name: str = "faster_rcnn_r50_fpn_1x_voc0712",
    test_path: str = "VOCdevkit/VOC2007",
    num_gpus: int = -1,
    visualization_result_dir: str = "VOCdevkit/VOC2007/visualizations",
    conf_threshold: float = 0.3,
):
    predictor = MultiModalPredictor(
        hyperparameters={
            "model.mmdet_image.checkpoint_name": checkpoint_name,
            "env.num_gpus": num_gpus,
        },
        pipeline="object_detection",
    )

    df = from_coco_or_voc(test_path)[:10][["image"]]

    pred = predictor.predict(df)

    visualize_detection(
        pred=pred,
        detection_classes=predictor.classes,
        conf_threshold=conf_threshold,
        visualization_result_dir=visualization_result_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_path", default="coco17/annotations/instances_val2017.json", type=str)
    parser.add_argument("--checkpoint_name", default="vfnet_x101_64x4d_fpn_mdconv_c3-c5_mstrain_2x_coco", type=str)
    parser.add_argument("--num_gpus", default=1, type=int)
    parser.add_argument("--visualization_result_dir", default="coco17/visualizations/", type=str)
    parser.add_argument("--visualization_conf_threshold", default=0.3, type=float)
    args = parser.parse_args()

    visualize_detection_results(
        test_path=args.test_path,
        checkpoint_name=args.checkpoint_name,
        num_gpus=args.num_gpus,
        visualization_result_dir=args.visualization_result_dir,
        conf_threshold=args.visualization_conf_threshold,
    )
