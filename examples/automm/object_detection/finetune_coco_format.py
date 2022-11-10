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


def tutorial_script_for_finetune_fast_coco_format():
    train_path = "./VOCdevkit/VOCCOCO/voc12_train.json"
    test_path = "./VOCdevkit/VOCCOCO/voc07_test.json"
    checkpoint_name = "yolov3_mobilenetv2_320_300e_coco"
    num_gpus = -1

    predictor = MultiModalPredictor(
        hyperparameters={
            "model.mmdet_image.checkpoint_name": checkpoint_name,
            "env.num_gpus": num_gpus,
        },
        problem_type="object_detection",
        sample_data_path=train_path,
    )


    start = time.time()
    predictor.fit(
        train_path,
        hyperparameters={
            "optimization.learning_rate": 1e-4, # we use two stage and detection head has 100x lr
            "optimization.max_epochs": 5,
            "env.per_gpu_batch_size": 32,  # decrease it when model is large
        },
    )
    end = time.time()

    print("This finetuning takes %.2f seconds." % (end - start))

    predictor.evaluate(test_path)



def tutorial_script_for_finetune_high_performance_coco_format():
    checkpoint_name = "vfnet_x101_64x4d_fpn_mdconv_c3-c5_mstrain_2x_coco"
    num_gpus = -1
    val_metric = "map"

    train_path = "./VOCdevkit/VOCCOCO/voc12_train.json"
    test_path = "./VOCdevkit/VOCCOCO/voc07_test.json"

    predictor = MultiModalPredictor(
        hyperparameters={
            "model.mmdet_image.checkpoint_name": checkpoint_name,
            "env.num_gpus": num_gpus,
        },
        problem_type="object_detection",
        sample_data_path=train_path,
        val_metric=val_metric,
    )

    start = time.time()
    predictor.fit(
        train_path,
        hyperparameters={
            "optimization.learning_rate": 1e-5, # we use two stage and detection head has 100x lr
            "optimization.max_epochs": 20,
            "env.per_gpu_batch_size": 2,  # decrease it when model is large
        },
    )
    end = time.time()

    print("This finetuning takes %.2f seconds." % (end - start))

    predictor.evaluate(test_path)


def detection_train(
    train_path,
    val_path=None,
    test_path=None,
    checkpoint_name="faster_rcnn_r50_fpn_2x_coco",
    lr=1e-4,
    epochs=50,
    num_gpus=4,
    val_metric=None,
    per_gpu_batch_size=8,
):

    predictor = MultiModalPredictor(
        hyperparameters={
            "model.mmdet_image.checkpoint_name": checkpoint_name,
            "env.num_gpus": num_gpus,
        },
        problem_type="object_detection",
        sample_data_path=train_path,
        val_metric=val_metric,
    )

    import time

    start = time.time()
    predictor.fit(
        train_path,
        tuning_data=val_path,
        hyperparameters={
            "optimization.learning_rate": lr, # we use two stage and lr_mult=100 for detection
            "optimization.max_epochs": epochs,
            "env.per_gpu_batch_size": per_gpu_batch_size,  # decrease it when model is large
        },
    )
    fit_end = time.time()
    print("time usage for fit: %.2f" % (fit_end - start))

    if test_path is not None:
        predictor.evaluate(test_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_path", default="/media/data/datasets/voc/VOCdevkit/VOCCOCO/voc07_trainval.json", type=str
    )
    parser.add_argument("--val_path", default=None, type=str)
    parser.add_argument("--test_path", default=None, type=str)
    parser.add_argument("--checkpoint_name", default="yolov3_mobilenetv2_320_300e_coco", type=str)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--num_gpus", default=4, type=int)
    parser.add_argument("--per_gpu_batch_size", default=8, type=int)
    parser.add_argument("--val_metric", default=None, type=str)
    args = parser.parse_args()

    detection_train(
        train_path=args.train_path,
        val_path=args.val_path,
        test_path=args.test_path,
        checkpoint_name=args.checkpoint_name,
        lr=args.lr,
        epochs=args.epochs,
        num_gpus=args.num_gpus,
        val_metric=args.val_metric,  # "mAP" or "direct_loss" or None (use default: "direct_loss")
        per_gpu_batch_size=args.per_gpu_batch_size,
    )

if __name__ == "__main__":
    # main()
    tutorial_script_for_finetune_fast_coco_format()
