"""
The example to finetune an object detection model in AutoMM in COCO format.

An example to finetune an MMDetection model on COCO:
    python finetune_coco_format.py \
        --train_path coco17/annotations/instances_train2017.json \
        --test_path coco17/annotations/instances_val2017.json \
        --checkpoint_name yolov3_mobilenetv2_320_300e_coco \
        --lr <learning_rate> \
        --epochs <epochs>

An example to finetune an MMDetection model on VOC:
    First, use this script to convert the VOC dataset to COCO format:
    https://github.com/open-mmlab/mmdetection/blob/9d3e162459590eee4cfc891218dfbb5878378842/tools/dataset_converters/pascal_voc.py
    Then, run:
    python finetune_coco_format.py \
        --train_path /media/data/datasets/voc/VOCdevkit/VOCCOCO/voc07_trainval.json \
        --test_path /media/data/datasets/voc/VOCdevkit/VOCCOCO/voc07_test.json \
        --checkpoint_name yolov3_mobilenetv2_320_300e_coco \
        --lr <learning_rate> \
        --epochs <epochs>
"""

import argparse
import time

from autogluon.multimodal import MultiModalPredictor


def tutorial_script_for_save_load_predictor():
    test_path = "./VOCdevkit/VOCCOCO/voc07_test.json"
    load_path = "/media/code/autogluon/examples/automm/object_detection/AutogluonModels/ag-20221104_185342"

    print("load a predictor from save path...")
    predictor = MultiModalPredictor.load(load_path)

    predictor.evaluate(test_path)

    print("load a predictor from save path and change num_gpus to 1...")
    predictor_single_gpu = MultiModalPredictor.load(load_path)
    predictor_single_gpu.set_num_gpus(num_gpus=1)

    predictor_single_gpu.evaluate(test_path)


def detection_load_predictor_and_eval(test_path, load_path, num_gpus):
    print(f"loading a predictor from save path and change num_gpus to {num_gpus}...")
    predictor_single_gpu = MultiModalPredictor.load(load_path)
    predictor_single_gpu.set_num_gpus(num_gpus=num_gpus)

    predictor_single_gpu.evaluate(test_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_path", default=None, type=str)
    parser.add_argument("--load_path", default=None, type=str)
    parser.add_argument("--num_gpus", default=4, type=int)
    args = parser.parse_args()

    detection_load_predictor_and_eval(
        test_path=args.test_path,
        load_path=args.load_path,
        num_gpus=args.num_gpus,
    )


if __name__ == "__main__":
    main()
