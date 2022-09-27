import os

import pandas as pd


def get_essay(essay_id: str, input_dir: str, is_train: bool = True) -> str:
    parent_path = input_dir + "train" if is_train else input_dir + "test"
    essay_path = os.path.join(parent_path, f"{essay_id}.txt")
    essay_text = open(essay_path, "r").read()
    return essay_text


def read_and_process_data(path: str, file: str, is_train: bool) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(path, file))
    df["essay_text"] = df["essay_id"].apply(lambda x: get_essay(x, path, is_train=is_train))
    return df
