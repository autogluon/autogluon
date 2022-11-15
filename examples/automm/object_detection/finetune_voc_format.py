"""
The example to finetune an object detection model in AutoMM in VOC format.

An example to finetune an MMDetection model on watercolor in VOC format:
    python finetune_coco_format.py \
        --train_path coco17/annotations/instances_train2017.json \
        --test_path coco17/annotations/instances_val2017.json \
        --checkpoint_name yolov3_mobilenetv2_320_300e_coco \
        --lr <learning_rate> \
        --epochs <epochs>
"""

import argparse
import time

from autogluon.multimodal import MultiModalPredictor


def tutorial_script_for_finetune_voc_format():
    train_path = "./watercolor"
    checkpoint_name = "yolov3_mobilenetv2_320_300e_coco"
    num_gpus = -1

    predictor = MultiModalPredictor(
        hyperparameters={
            "model.mmdet_image.checkpoint_name": checkpoint_name,
            "env.num_gpus": num_gpus,
        },
        problem_type="object_detection",
        sample_data_path=train_path,
        save_path="yolov3_watercolor",
    )

    lr = 1e-4
    epochs = 5
    per_gpu_batch_size = 32

    start = time.time()

    predictor.fit(
        train_path,
        hyperparameters={
            "optimization.learning_rate": lr, # we use two stage and detection head has 100x lr
            "optimization.max_epochs": epochs,
            "env.per_gpu_batch_size": per_gpu_batch_size,  # decrease it when model is large
        },
    )

    print("This finetuning takes %.2f seconds." % (time.time() - start))

def detection_train(
    train_path,
    val_path=None,
    save_path=None,
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
            "optimization.val_metric": val_metric,
        },
        problem_type="object_detection",
        sample_data_path=train_path,
        save_path=save_path,
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_path", default="./VOCdevkit/VOC2007/Annotations/train_cocoformat.json", type=str
    )
    parser.add_argument("--val_path", default=None, type=str)
    parser.add_argument("--save_path", default=None, type=str)
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
        save_path=args.save_path,
        checkpoint_name=args.checkpoint_name,
        lr=args.lr,
        epochs=args.epochs,
        num_gpus=args.num_gpus,
        val_metric=args.val_metric,  # "mAP" or "direct_loss" or None (use default: "direct_loss")
        per_gpu_batch_size=args.per_gpu_batch_size,
    )

if __name__ == "__main__":
    main()