"""
The example to load a trained predictor and evaluate.

An example:
    python load_predictor.py \
        --test_path <test_path> \
        --load_path <load_path> \
        --num_gpus <num_gpus>

"""

import argparse
import time

from autogluon.multimodal import MultiModalPredictor


def tutorial_script_for_save_load_predictor():
    test_path = "./VOCdevkit/VOC2007/Annotations/test_cocoformat.json"
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
