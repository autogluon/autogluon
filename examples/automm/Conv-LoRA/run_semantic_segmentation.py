import argparse
import os

import pandas as pd

from autogluon.multimodal import MultiModalPredictor


def get_validation_metric(dataset_name):
    if dataset_name == "isic2017":
        validation_metric = "iou"

    elif dataset_name == "SBU-shadow":
        validation_metric = "ber"

    elif dataset_name == "polyp":
        validation_metric = "sm"

    elif dataset_name == "camo_sem_seg":
        validation_metric = "sm"

    elif dataset_name == "road_segmentation":
        validation_metric = "iou"

    elif dataset_name == "leaf_disease_segmentation":
        validation_metric = "iou"
    return validation_metric


def expand_path(df, dataset_dir):
    for col in ["image", "label"]:
        df[col] = df[col].apply(lambda ele: os.path.join(dataset_dir, ele))
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This script support converting voc format xmls to coco format json")
    parser.add_argument(
        "--task",
        type=str,
        default="leaf_disease_segmentation",
        choices=["polyp", "leaf_disease_segmentation", "camo_sem_seg", "isic2017", "road_segmentation", "SBU-shadow"],
    )
    parser.add_argument("--seed", type=int, default=42686693)
    parser.add_argument("--rank", type=int, default=3)
    parser.add_argument("--expert_num", type=int, default=8)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--ckpt_path", type=str, default="outputs", help="Checkpoint path.")
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args()

    dataset_name = args.task
    dataset_dir = os.path.join(f"datasets/{dataset_name}", dataset_name)
    os.makedirs(args.output_dir, exist_ok=True)

    # prepare dataframes
    train_df = expand_path(pd.read_csv(os.path.join(dataset_dir, f"train.csv")), dataset_dir)
    val_df = expand_path(pd.read_csv(os.path.join(dataset_dir, f"val.csv")), dataset_dir)

    if dataset_name not in ["polyp", "camo_sem_seg"]:  # have multiple test sets.
        test_df = expand_path(pd.read_csv(os.path.join(dataset_dir, f"test.csv")), dataset_dir)

    # validation metric for each dataset.
    validation_metric = get_validation_metric(dataset_name)

    hyperparameters = {}
    hyperparameters.update(
        {
            "optimization.lora.r": args.rank,
            "optimization.lora.conv_lora_expert_num": args.expert_num,
            "env.num_gpus": args.num_gpus,
        }
    )

    # create predictor
    predictor = MultiModalPredictor(
        problem_type="semantic_segmentation",
        validation_metric=validation_metric,
        eval_metric=validation_metric,
        path=os.path.join(args.output_dir, "models"),
        hyperparameters=hyperparameters,
        label="label",
    )

    if args.eval:
        predictor = predictor.load(args.ckpt_path)

    else:
        # Training
        predictor.fit(train_data=train_df, tuning_data=val_df, seed=args.seed)

    # Evaluation
    metric_file = os.path.join(args.output_dir, "metrics.txt")
    f = open(metric_file, "a")
    if dataset_name == "isic2017":
        res = predictor.evaluate(
            test_df,
            metrics=[
                "iou",
            ],
        )
    elif dataset_name == "SBU-shadow":
        res = predictor.evaluate(
            test_df,
            metrics=[
                "ber",
            ],
        )
    elif dataset_name == "polyp":
        for test_dataset in ["CVC-ClinicDB", "Kvasir"]:
            test_df = expand_path(pd.read_csv(os.path.join(dataset_dir, f"test_{test_dataset}.csv")), dataset_dir)
            res = predictor.evaluate(test_df, metrics=["sm", "fm", "em", "mae"])
            print(f"Evaluation results for test dataset {test_dataset}: ", res)
            f.write(f"Evaluation results for test dataset {test_dataset}: {res} \n")
    elif dataset_name == "camo_sem_seg":
        for test_dataset in ["CAMO"]:
            test_df = expand_path(pd.read_csv(os.path.join(dataset_dir, f"test_{test_dataset}.csv")), dataset_dir)
            res = predictor.evaluate(test_df, metrics=["sm", "fm", "em", "mae"])
            print(f"Evaluation results for test dataset {test_dataset}: ", res)
            f.write(f"Evaluation results for test dataset {test_dataset}: {res} \n")
    elif dataset_name == "road_segmentation":
        res = predictor.evaluate(test_df, metrics=["iou"])
    elif dataset_name == "leaf_disease_segmentation":
        res = predictor.evaluate(test_df, metrics=["iou"])
    if dataset_name not in ["polyp", "camo_sem_seg"]:
        print(f"Evaluation results for test dataset {dataset_name}: ", res)
        f.write(f"Evaluation results for test dataset {dataset_name}: {res} \n")
    f.close()
