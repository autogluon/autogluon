import argparse
import os
from time import time

import pandas as pd
from datasets import load_dataset

from autogluon.multimodal import MultiModalPredictor

PAWS_TASKS = ["en", "de", "es", "fr", "ja", "ko", "zh"]


def tasks_to_id(pawsx_tasks):
    id = ""
    for task in PAWS_TASKS:
        if task in pawsx_tasks:
            id += task
    return id


def main(args):
    model_path = args.model_path
    pawsx_tasks = args.pawsx_tasks
    assert all(task in PAWS_TASKS for task in pawsx_tasks)

    datasets = {}
    val_dfs = {}
    test_dfs = {}
    for task in args.pawsx_tasks:
        datasets[task] = load_dataset("paws-x", task)
        val_dfs[task] = datasets[task]["validation"].to_pandas()
        test_dfs[task] = datasets[task]["test"].to_pandas()
        print("task %s: val %d, test %d" % (task, len(val_dfs[task]), len(test_dfs[task])))
    val_df = pd.concat(val_dfs)
    test_dfs["all"] = pd.concat(test_dfs)
    test_dfs["val"] = val_df

    result = {}
    predictor = MultiModalPredictor.load(model_path)
    start = time()
    for test_name, test_df in test_dfs.items():
        result[test_name] = predictor.evaluate(data=test_df, metrics="accuracy")
    usedtime = time() - start

    for test_name in test_dfs.keys():
        print("Distillation Result (%s):" % test_name)
        print("Model: %s" % model_path)
        for k in result[test_name]:
            print(f"For metric {k}:")
            print("Model's %s: %.6f" % (k, result[test_name][k]))
    print("Teacher Model's time: %.6f" % usedtime)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pawsx_tasks", default=["en"], type=list)
    parser.add_argument("--model_path", type=str)
    args = parser.parse_args()

    main(args)
