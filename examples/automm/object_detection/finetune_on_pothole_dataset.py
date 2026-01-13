"""
The example to finetune an object detection model on pothole dataset.
"""

import argparse
import os

from finetune_coco_format import finetune_coco_format

from autogluon.core.utils.loaders import load_zip
from autogluon.multimodal import MultiModalPredictor


def download_pothole_dataset():
    zip_file = "https://automl-mm-bench.s3.amazonaws.com/object_detection/dataset/pothole.zip"
    download_dir = "./pothole"

    load_zip.unzip(zip_file, unzip_dir=download_dir)
    data_dir = os.path.join(download_dir, "pothole")
    train_path = os.path.join(data_dir, "Annotations", "usersplit_train_cocoformat.json")
    val_path = os.path.join(data_dir, "Annotations", "usersplit_val_cocoformat.json")
    test_path = os.path.join(data_dir, "Annotations", "usersplit_test_cocoformat.json")

    return train_path, val_path, test_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--presets", default=None, type=str)
    parser.add_argument("-c", "--checkpoint_name", default=None, type=str)
    parser.add_argument("-n", "--num_gpus", default=None, type=int)
    parser.add_argument("-l", "--lr", default=None, type=float)
    parser.add_argument("-e", "--epochs", default=None, type=int)
    parser.add_argument("-b", "--per_gpu_batch_size", default=None, type=int)
    args = parser.parse_args()

    train_path, val_path, test_path = download_pothole_dataset()

    finetune_coco_format(
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        presets=args.presets,
        checkpoint_name=args.checkpoint_name,
        lr=args.lr,
        epochs=args.epochs,
        num_gpus=args.num_gpus,
        per_gpu_batch_size=args.per_gpu_batch_size,
    )


if __name__ == "__main__":
    main()
