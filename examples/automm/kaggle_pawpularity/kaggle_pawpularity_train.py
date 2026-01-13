import argparse
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import StratifiedKFold

from autogluon.multimodal import MultiModalPredictor


def get_args():
    parser = argparse.ArgumentParser(
        description="The example for Kaggle competition Petfinder Pawpularity with MultiModalPredictor."
    )
    parser.add_argument("--data_path", type=str, help="The path of the competiton dataset.", default="./data/")
    parser.add_argument("--label_column", type=str, help="The column name of label.", default="Pawpularity")
    parser.add_argument("--problem_type", type=str, help="The problem type.", default="regression")
    parser.add_argument("--eval_metric", type=str, help="The evaluation metric.", default="rmse")
    parser.add_argument("--fusion", type=bool, help="Whether using model fusion.", default=True)
    parser.add_argument(
        "--timm_image_checkpoint_name",
        type=str,
        help="The name of model for images in timm.",
        default="swin_large_patch4_window7_224",
    )
    parser.add_argument(
        "--image_train_transforms",
        type=str,
        help="The types for transforming images.",
        default="['resize_shorter_side','center_crop','randaug']",
    )
    parser.add_argument(
        "--categorical_convert_to_text", type=bool, help="Whether convert categorical to text.", default=False
    )
    parser.add_argument("--per_gpu_batch_size", type=int, help="The batch size in gpu.", default=16)
    parser.add_argument("--precision", type=str, help="The precision in the training.", default="32")
    parser.add_argument("--lr", type=float, help="The learning rate in the training.", default=2e-5)
    parser.add_argument("--weight_decay", type=float, help="The weight decay in the training.", default=0)
    parser.add_argument("--lr_decay", type=float, help="The learning rate decay in the training.", default=1)
    parser.add_argument("--max_epochs", type=int, help="The max training epochs in the training.", default=5)
    parser.add_argument("--warmup_steps", type=float, help="The warmup steps of the training epochs.", default=0)
    parser.add_argument(
        "--loss_func", type=str, help="The loss function of the training epochs.", default="bcewithlogitsloss"
    )
    parser.add_argument("--folds", type=int, help="The folds of the training.", default=5)
    parser.add_argument("--seed", type=int, help="The random seed.", default=1)
    parser.add_argument("--trial", type=int, help="The id of multiple runs.", default=0)
    args = parser.parse_args()

    if args.fusion:
        args.model_names = None
        args.save_path = "pawpularity_{}_fusion_lr_{}_decay_{}_bsz_{}_mepoch_{}_warmup_{}_trial_{}".format(
            args.timm_image_checkpoint_name,
            args.lr,
            args.lr_decay,
            args.per_gpu_batch_size,
            args.max_epochs,
            args.warmup_steps,
            args.trial,
        )
    else:
        args.model_names = "['timm_image']"
        args.save_path = "pawpularity_{}_timm_only_lr_{}_decay_{}_bsz_{}_mepoch_{}_warmup_{}_trial_{}".format(
            args.timm_image_checkpoint_name,
            args.lr,
            args.lr_decay,
            args.per_gpu_batch_size,
            args.max_epochs,
            args.warmup_steps,
            args.trial,
        )

    return args


def load_data(data_path: str):
    """
    Load training and testing datasets and split folds.
    Parameters
    ----------
    data_path
        The path of training and testing files.

    Return
    ----------
    The pure training dataset and the folds split information and the testing dataset.
    """
    # Load training dataset.
    train_df = pd.read_csv(os.path.join(data_path, "train.csv"))
    train_df.rename(columns={"Id": "Image Path"}, inplace=True)
    train_df["Image Path"] = train_df["Image Path"].apply(lambda s: os.path.join(data_path, "train", s + ".jpg"))

    # Create the split information of folds.
    num_bins = int(np.ceil(2 * ((len(train_df)) ** (1.0 / 3))))
    train_df_bins = pd.cut(train_df["Pawpularity"], bins=num_bins, labels=False)
    train_df_fold = pd.Series([-1] * len(train_df))
    strat_kfold = StratifiedKFold(n_splits=N_FOLDS, shuffle=True)
    for i, (_, train_index) in enumerate(strat_kfold.split(train_df.index, train_df_bins)):
        train_df_fold[train_index] = i
    train_df_fold = train_df_fold.astype("int")

    # Load testing dataset.
    test_df = pd.read_csv(os.path.join(data_path, "test.csv"))
    test_df.rename(columns={"Id": "Image Path"}, inplace=True)
    test_df["Image Path"] = test_df["Image Path"].apply(lambda s: os.path.join(data_path, "test", s + ".jpg"))

    return train_df, train_df_fold, test_df


if __name__ == "__main__":
    args = get_args()

    data_path = args.data_path  # The path of the training and testing data.
    save_path = args.save_path  # The path of saving the model.
    save_standalone_path = (
        save_path + "_standalone"
    )  # The path of saving the standalone model which includes downloaded model.

    N_FOLDS = args.folds  # The number of folds.

    all_score = []  # The result of folds.
    train_df, train_df_fold, _ = load_data(data_path)

    for i in range(N_FOLDS):
        # The predictor in use.
        predictor = MultiModalPredictor(
            label=args.label_column,  # label indicates the target value
            problem_type=args.problem_type,  # problem_type indicates the type of the problem. It can choose "multiclass", # "binary" or "regression"
            eval_metric=args.eval_metric,  # eval_metric indicates the evaluation index of the model
            path=save_path,
            verbosity=4,  # verbosity controls how much information is printed.
        )

        # Training process.
        training_df = train_df[train_df_fold != i]
        valid_df = train_df[train_df_fold == i]
        predictor.fit(
            train_data=training_df,
            tuning_data=valid_df,
            save_path=save_path + f"_fold{i}",
            hyperparameters={
                "model.names": args.model_names,
                "model.timm_image.checkpoint_name": args.timm_image_checkpoint_name,
                "model.timm_image.train_transforms": args.image_train_transforms,
                "data.categorical.convert_to_text": args.categorical_convert_to_text,
                "env.per_gpu_batch_size": args.per_gpu_batch_size,
                "env.precision": args.precision,
                "optim.lr": args.lr,
                "optim.weight_decay": args.weight_decay,
                "optim.lr_decay": args.lr_decay,
                "optim.max_epochs": args.max_epochs,
                "optim.warmup_steps": args.warmup_steps,
                "optim.loss_func": args.loss_func,
            },
            seed=args.seed,
        )

        # Manual Validating process.
        valid_pred = predictor.predict(data=valid_df)
        score = root_mean_squared_error(valid_df["Pawpularity"].values, valid_pred)
        print(f"Fold {i} | Score: {score}")
        predictor.save(
            path=save_standalone_path + f"_fold{i}",
            standalone=True,
        )
        all_score.append(score)

        del predictor
        torch.cuda.empty_cache()

    print(f"all-scores: {all_score}")
    print(f"mean_rmse: {np.mean(all_score)}")
