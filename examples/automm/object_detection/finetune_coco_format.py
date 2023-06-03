"""
The example to finetune an object detection model in a COCO format dataset.
See finetune_on_pothole_dataset.py for an example on our provided dataset.
"""

import argparse

from autogluon.multimodal import MultiModalPredictor


def finetune_coco_format(
    train_path,
    val_path=None,
    test_path=None,
    preset=None,
    checkpoint_name=None,
    num_gpus=None,
    val_metric=None,
    lr=None,
    epochs=None,
    per_gpu_batch_size=None,
):
    assert train_path is not None, "train_path must be provided and cannot be None"

    hyperparameters = {}
    if checkpoint_name is not None:
        hyperparameters["model.mmdet_image.checkpoint_name"] = checkpoint_name
    if num_gpus is not None:
        hyperparameters["env.num_gpus"] = num_gpus
    if val_metric is not None:
        hyperparameters["optimization.val_metric"] = val_metric
    if lr is not None:
        hyperparameters["optimization.learning_rate"] = lr
    if epochs is not None:
        hyperparameters["optimization.max_epochs"] = epochs
    if per_gpu_batch_size is not None:
        hyperparameters["env.per_gpu_batch_size"] = per_gpu_batch_size

    predictor = MultiModalPredictor(
        hyperparameters=hyperparameters,
        problem_type="object_detection",
        sample_data_path=train_path,
        preset=preset,
    )
    predictor.fit(train_path, tuning_data=val_path)
    print("time usage for fit: %.2f" % (predictor._total_train_time))

    if test_path is not None:
        predictor.evaluate(test_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", default=None, type=str)
    parser.add_argument("--val_path", default=None, type=str)
    parser.add_argument("--test_path", default=None, type=str)
    parser.add_argument("-p", "--preset", default=None, type=str)
    parser.add_argument("-c", "--checkpoint_name", default=None, type=str)
    parser.add_argument("-n", "--num_gpus", default=None, type=int)
    parser.add_argument("-v", "--val_metric", default=None, type=str)
    parser.add_argument("-l", "--lr", default=None, type=float)
    parser.add_argument("-e", "--epochs", default=None, type=int)
    parser.add_argument("-b", "--per_gpu_batch_size", default=None, type=int)
    args = parser.parse_args()

    finetune_coco_format(
        train_path=args.train_path,
        val_path=args.val_path,
        test_path=args.test_path,
        preset=args.preset,
        checkpoint_name=args.checkpoint_name,
        lr=args.lr,
        epochs=args.epochs,
        num_gpus=args.num_gpus,
        val_metric=args.val_metric,  # "mAP" or "direct_loss" or None (use default: "direct_loss")
        per_gpu_batch_size=args.per_gpu_batch_size,
    )


if __name__ == "__main__":
    main()
