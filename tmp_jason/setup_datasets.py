import argparse
import openml
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import subprocess

"""
Download datasets from openml and place them at datasets/ folder.
For each downloaded datasets, create synthetic versions.
"""
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--id', help="openml dataset id", type=int, default=None)
parser.add_argument('-n', '--name', help="openml dataset name", type=str, default=None)
args = parser.parse_args()

# Always download baseline datasets if they do not exist
DATASETS = {
    "adult": 179,
    "airlines": 1169,
    "australian": 40981,
    "covertype": 1596,
    "higgs": 42769,
}
if args.id and args.name:
    DATASETS[args.name] = args.id

for dataset_name, dataset_id in DATASETS.items():
    save_dir = f"dataset/{dataset_name}"
    if not os.path.exists(save_dir):
        dataset = openml.datasets.get_dataset(dataset_id)
        X, y, _, _ = dataset.get_data(dataset_format="dataframe", target=dataset.default_target_attribute)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        # change label column name to class
        y_train, y_test = y_train.rename('class'), y_test.rename('class')
        train_data = pd.concat([X_train, y_train], axis=1)
        test_data = pd.concat([X_test, y_test], axis=1)
        os.makedirs(save_dir, exist_ok=True)
        train_save_path = f"{save_dir}/train_data.csv"
        test_save_path = f"{save_dir}/test_data.csv"
        train_data.to_csv(train_save_path, index=False)
        test_data.to_csv(test_save_path, index=False)
        # synthetic dataset generation command
        for times in [0.2, 1]:
            cmd = f"python3 generate_synthetic_data.py -f {train_save_path} -g {test_save_path} -t {times} -l class"
            subprocess.run(cmd, shell=True)
