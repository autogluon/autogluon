import argparse
import json
import os
import random

import numpy as np
import pandas as pd

from autogluon.multimodal import MultiModalPredictor
from autogluon.tabular import TabularPredictor
from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config


def get_parser():
    parser = argparse.ArgumentParser(
        description="The Basic Example of using AutoGluon Multimodal for Text Prediction."
    )
    parser.add_argument("--train_file", type=str, help="The training CSV file.", default=None)
    parser.add_argument("--test_file", type=str, help="The testing CSV file.", default=None)
    parser.add_argument("--sample_submission", type=str, help="The sample submission CSV file.", default=None)
    parser.add_argument("--seed", type=int, help="The seed", default=123)
    parser.add_argument("--eval_metric", type=str, help="The metric used to evaluate the model.", default=None)
    parser.add_argument(
        "--task",
        type=str,
        choices=["product_sentiment", "mercari_price", "price_of_books", "data_scientist_salary"],
        required=True,
    )
    parser.add_argument(
        "--exp_dir", type=str, default=None, help="The experiment directory where the model params will be written."
    )
    parser.add_argument(
        "--mode",
        choices=["stacking", "weighted", "single"],
        default="single",
        help="Whether to use a single model or a stack ensemble. "
        'If it is "single", If it is turned on, we will use 5-fold, 1-layer for stacking.',
    )
    parser.add_argument(
        "--preset",
        type=str,
        help="Pre-registered configurations",
        choices=["medium_quality_faster_train", "high_quality", "best_quality"],
        default=None,
    )
    return parser


def load_machine_hack_product_sentiment(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    feature_columns = ["Product_Description", "Product_Type"]
    label_column = "Sentiment"
    train_df = train_df[feature_columns + [label_column]]
    test_df = test_df[feature_columns]
    return train_df, test_df, label_column


def load_price_of_books(train_path, test_path):
    train_df = pd.read_excel(train_path, engine="openpyxl")
    test_df = pd.read_excel(test_path, engine="openpyxl")
    # Convert Reviews
    train_df.loc[:, "Reviews"] = pd.to_numeric(train_df["Reviews"].apply(lambda ele: ele[: -len(" out of 5 stars")]))
    test_df.loc[:, "Reviews"] = pd.to_numeric(test_df["Reviews"].apply(lambda ele: ele[: -len(" out of 5 stars")]))
    # Convert Ratings
    train_df.loc[:, "Ratings"] = pd.to_numeric(
        train_df["Ratings"].apply(lambda ele: ele.replace(",", "")[: -len(" customer reviews")])
    )
    test_df.loc[:, "Ratings"] = pd.to_numeric(
        test_df["Ratings"].apply(lambda ele: ele.replace(",", "")[: -len(" customer reviews")])
    )
    # Convert Price to log scale
    train_df.loc[:, "Price"] = np.log10(train_df["Price"] + 1)
    return train_df, test_df, "Price"


def load_data_scientist_salary(train_path, test_path):
    train_df = pd.read_csv(train_path, index_col=0)
    test_df = pd.read_csv(test_path, index_col=None)
    train_df.drop("company_name_encoded", axis=1, inplace=True)
    test_df.drop("company_name_encoded", axis=1, inplace=True)
    return train_df, test_df, "salary"


def load_mercari_price_prediction(train_path, test_path):
    train_df = pd.read_csv(train_path, sep="\t")
    test_df = pd.read_csv(test_path, sep="\t")

    train_cat1 = []
    train_cat2 = []
    train_cat3 = []

    test_cat1 = []
    test_cat2 = []
    test_cat3 = []

    for ele in train_df["category_name"]:
        if isinstance(ele, str):
            categories = ele.split("/", 2)
            train_cat1.append(categories[0])
            train_cat2.append(categories[1])
            train_cat3.append(categories[2])
        else:
            train_cat1.append(None)
            train_cat2.append(None)
            train_cat3.append(None)

    for ele in test_df["category_name"]:
        if isinstance(ele, str):
            categories = ele.split("/", 2)
            test_cat1.append(categories[0])
            test_cat2.append(categories[1])
            test_cat3.append(categories[2])
        else:
            test_cat1.append(None)
            test_cat2.append(None)
            test_cat3.append(None)

    # Convert to log(1 + x)
    train_df.loc[:, "price"] = np.log(train_df["price"] + 1)
    # train_df = train_df.drop('category_name', axis=1)
    train_df["cat1"] = train_cat1
    train_df["cat2"] = train_cat2
    train_df["cat3"] = train_cat3

    # test_df = test_df.drop('category_name', axis=1)
    test_df["cat1"] = test_cat1
    test_df["cat2"] = test_cat2
    test_df["cat3"] = test_cat3

    label_column = "price"
    ignore_columns = ["train_id"]
    feature_columns = []
    for column in sorted(train_df.columns):
        if column != label_column and column not in ignore_columns:
            feature_columns.append(column)
    train_df = train_df[feature_columns + [label_column]]
    test_df = test_df[feature_columns]
    return train_df, test_df, label_column


def run(args):
    if args.task == "product_sentiment":
        train_df, test_df, label_column = load_machine_hack_product_sentiment(args.train_file, args.test_file)
    elif args.task == "mercari_price":
        train_df, test_df, label_column = load_mercari_price_prediction(args.train_file, args.test_file)
    elif args.task == "price_of_books":
        train_df, test_df, label_column = load_price_of_books(args.train_file, args.test_file)
    elif args.task == "data_scientist_salary":
        train_df, test_df, label_column = load_data_scientist_salary(args.train_file, args.test_file)
    else:
        raise NotImplementedError

    hyperparameters = get_hyperparameter_config("multimodal")
    if args.preset is not None and args.mode in ["stacking", "weighted"]:
        hyperparameters["AG_TEXT_NN"]["presets"] = args.preset

    if args.mode == "stacking":
        predictor = TabularPredictor(label=label_column, eval_metric=args.eval_metric, path=args.exp_dir)
        predictor.fit(train_data=train_df, hyperparameters=hyperparameters, num_bag_folds=5, num_stack_levels=1)
    elif args.mode == "weighted":
        predictor = TabularPredictor(label=label_column, eval_metric=args.eval_metric, path=args.exp_dir)
        predictor.fit(train_data=train_df, hyperparameters=hyperparameters)
    elif args.mode == "single":
        # When no embedding is used,
        # we will just use MultiModalPredictor that will train a single model internally.
        predictor = MultiModalPredictor(label=label_column, eval_metric=args.eval_metric, path=args.exp_dir)
        predictor.fit(train_data=train_df, presets=args.preset, seed=args.seed)
    else:
        raise NotImplementedError
    if args.task == "product_sentiment":
        test_probabilities = predictor.predict_proba(test_df, as_pandas=True, as_multiclass=True)
        test_probabilities.to_csv(os.path.join(args.exp_dir, "submission.csv"), index=False)
    elif args.task == "data_scientist_salary":
        predictions = predictor.predict(test_df, as_pandas=False)
        submission = pd.read_excel(args.sample_submission, engine="openpyxl")
        submission.loc[:, label_column] = predictions
        submission.to_excel(os.path.join(args.exp_dir, "submission.xlsx"))
    elif args.task == "price_of_books":
        predictions = predictor.predict(test_df, as_pandas=False)
        submission = pd.read_excel(args.sample_submission, engine="openpyxl")
        submission.loc[:, label_column] = np.power(10, predictions) - 1
        submission.to_excel(os.path.join(args.exp_dir, "submission.xlsx"))
    elif args.task == "mercari_price":
        test_predictions = predictor.predict(test_df, as_pandas=False)
        submission = pd.read_csv(args.sample_submission)
        submission.loc[:, label_column] = np.exp(test_predictions) - 1
        submission.to_csv(os.path.join(args.exp_dir, "submission.csv"), index=False)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    run(args)
