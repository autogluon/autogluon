import argparse
import random

import numpy as np
import pandas as pd
import torch as th
from kaggle_feedback_prize_preprocess import read_and_process_data
from sklearn.model_selection import StratifiedKFold

from autogluon.multimodal import MultiModalPredictor


def get_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="The example for Kaggle competition Feedback Prize - Predicting Effective Arguments."
    )
    parser.add_argument("--data_path", type=str, help="The path of the competiton dataset.", default="./data/")
    parser.add_argument("--model_path", type=str, help="The path of the model artifacts.", default="./model/")
    parser.add_argument(
        "--label_name", type=str, help="The column name of predictive label.", default="discourse_effectiveness"
    )
    parser.add_argument("--problem_type", type=str, help="The problem type.", default="multiclass")
    parser.add_argument("--eval_metric", type=str, help="The evaluation metric.", default="log_loss")
    parser.add_argument("--learning_rate", type=float, help="The learning rate in the training.", default=5e-5)
    parser.add_argument("--max_epochs", type=int, help="The max training epochs in the training.", default=7)
    parser.add_argument(
        "--text_backbone", type=str, help="Pretrained backbone for finetuning.", default="microsoft/deberta-v3-large"
    )
    parser.add_argument("--folds", type=int, help="The folds of the training.", default=5)
    parser.add_argument("--seed", type=int, help="The random seed.", default=42)
    args = parser.parse_args()

    backbone_model = args.text_backbone.replace("/", "-")
    args.save_path = args.model_path + "feedback-{}/{}-cv{}-lr-{}-mepoch-{}".format(
        backbone_model,
        backbone_model,
        args.folds,
        args.learning_rate,
        args.max_epochs,
    )
    return args


def get_hparams(args: argparse.ArgumentParser) -> dict:
    hparams = {
        "model.hf_text.checkpoint_name": args.text_backbone,
        "data.text.normalize_text": True,
        "optimization.learning_rate": args.learning_rate,
        "optimization.max_epochs": args.max_epochs,
    }

    return hparams


def set_seed(seed: int) -> None:
    th.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def train(
    train_df: pd.DataFrame, val_df: pd.DataFrame, args: argparse.ArgumentParser, path: str
) -> MultiModalPredictor:
    hparams = get_hparams(args)

    predictor = MultiModalPredictor(
        label=args.label_name,
        problem_type=args.problem_type,
        eval_metric=args.eval_metric,
        path=path,
        verbosity=3,
    ).fit(
        train_data=train_df,
        tuning_data=val_df,
        presets="best_quality",
        hyperparameters=hparams,
    )

    return predictor


if __name__ == "__main__":
    args = get_args()
    set_seed(args.seed)

    train_df = read_and_process_data(args.data_path, "train.csv", is_train=True)

    y_train = train_df[args.label_name]
    X_train = train_df.drop(args.label_name, axis=1)

    # K fold cross validation
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True)
    losses = []
    for i, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_t, X_v = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_t, y_v = y_train.iloc[train_idx], y_train.iloc[val_idx]

        train_df = pd.concat([X_t, y_t], axis=1)
        val_df = pd.concat([X_v, y_v], axis=1)
        path = args.save_path + f"_{i}"

        predictor = train(train_df, val_df, args, path)
        predictor.save(path, standalone=True)
