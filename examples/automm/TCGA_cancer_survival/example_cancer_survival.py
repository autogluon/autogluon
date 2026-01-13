"""
Example script to predict the vital status of patients with Head-Neck Squamous Cell Carcinoma.
Dataset is originally from https://portal.gdc.cancer.gov/projects/TCGA-HNSC.
Paper working on similar task: https://bmcbioinformatics.biomedcentral.com/track/pdf/10.1186/s12859-019-2929-8.pdf
"""

import argparse
import os
import random
import warnings

import numpy as np
import pandas as pd
import torch as th
from sklearn.model_selection import train_test_split

from autogluon.multimodal.utils import download
from autogluon.tabular import TabularDataset, TabularPredictor
from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config

warnings.filterwarnings("ignore")


# Dataset information for TCGA dataset
INFO = {
    "name": "cancer_survival.tsv",
    "url": "s3://automl-mm-bench/life-science/clinical.tsv",
    "sha1sum": "6d19609c2a8492f767efd9f2c0b7687bcd3845a3",
}


def get_parser():
    parser = argparse.ArgumentParser(description="The Basic Example of AutoGluon for TCGA dataset.")
    parser.add_argument("--path", default="./dataset")
    parser.add_argument("--test_size", default=0.3)
    parser.add_argument("--shuffle", default=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--task", choices=["TCGA_HNSC", "adult"], default="adult")
    parser.add_argument("--mode", choices=["FT_Transformer", "all_models"], default="all_models")
    parser.add_argument("--num_gpus", type=int, default=-1)
    parser.add_argument("--num_workers", type=int, default=2)
    return parser


def data_loader(
    path="./dataset/",
):
    name = INFO["name"]
    full_path = os.path.join(path, name)
    if os.path.exists(full_path):
        print(f"Existing dataset: {name}")
    else:
        print(f"Dataset not exist. Start downloading: {name}")
        download(INFO["url"], path=full_path, sha1_hash=INFO["sha1sum"])
    df = pd.read_csv(full_path, sep="\t")
    return df


# Preprocessing steps include:
# (1) Remove "id"-related columns and columns with the same values;
# (2) Some column shared common information with the target label. Those "shortcuts" were removed.
#      e.g. We aim to predict whether patents are alive or not. The "death_date" is an invalid feature.
# (3) Split data into train/test sets, by specifying "test_size" and "shuffle".
def preprocess(df, test_size, shuffle):
    N, _ = df.shape

    df = df[df != "'--"]  # Replace missing entries with nan
    n_unique = df.nunique()

    for col, n in n_unique.items():
        if "id" in col or n <= 1:
            df.drop(col, axis=1, inplace=True)

    shortcut_col = ["days_to_death", "year_of_death"]  # Shortcut columns should be removed
    for col in shortcut_col:
        df.drop(col, axis=1, inplace=True)

    df_train, df_test = train_test_split(df, test_size=test_size, shuffle=shuffle)
    return df_train, df_test


def train(args):
    if args.task == "adult":
        df_train = TabularDataset("https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv")
        df_test = TabularDataset("https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv")
        label = "class"
    elif args.task == "TCGA_HNSC":
        df = data_loader(args.path)
        df_train, df_test = preprocess(df, args.test_size, args.shuffle)
        label = "vital_status"
    else:
        raise NotImplementedError

    metric = "accuracy"
    hyperparameters = {} if args.mode == "FT_Transformer" else get_hyperparameter_config("default")
    hyperparameters["FT_TRANSFORMER"] = {"env.num_gpus": args.num_gpus, "env.num_workers": args.num_workers}
    predictor = TabularPredictor(
        label=label,
        eval_metric=metric,
    ).fit(
        train_data=df_train,
        hyperparameters=hyperparameters,
        time_limit=900,
    )
    leaderboard = predictor.leaderboard(df_test)
    leaderboard.to_csv("./leaderboard.csv")
    return


def set_seed(seed):
    th.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    set_seed(args.seed)
    train(args)
