import argparse
import os
import subprocess

import pandas as pd
import yaml

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument(
    "--config_path", help="path to generated config path to fetch benchmark name", type=str, required=True
)
parser.add_argument("--time_limit", help="time limit of the benchmark run", type=str, required=True)

args = parser.parse_args()

config_path = args.config_path
time_limit = args.time_limit

for root, dirs, files in os.walk(config_path):
    for file in files:
        if file == "tabular_cloud_configs.yaml":
            config_file = os.path.join(root, file)

with open(config_file, "r") as f:
    config = yaml.safe_load(f)
    benchmark_name = config["benchmark_name"]

subprocess.run(
    [
        "agbench",
        "aggregate-amlb-results",
        "autogluon-ci-benchmark",
        "tabular",
        benchmark_name,
        "--constraint",
        time_limit,
    ],
    check=True,
)
subprocess.run(
    [
        "agbench",
        "clean-amlb-results",
        benchmark_name,
        f"--results-dir-input",
        f"s3://autogluon-ci-benchmark/aggregated/tabular/{benchmark_name}/",
        "--benchmark-name-in-input-path",
        "--constraints",
        time_limit,
        "--results-dir-output",
        "./results",
    ]
)

paths = []
frameworks = []
for file in os.listdir("./results"):
    if file.endswith(".csv"):
        file = os.path.join("./results", file)
        df = pd.read_csv(file)
        paths.append(os.path.basename(file))
        frameworks += list(df["framework"].unique())

subprocess.run(
    [
        "agbench",
        "evaluate-amlb-results",
        "--frameworks-run",
        f"{','.join(frameworks)}",
        "--results-dir-input",
        "./results",
        "--paths",
        f"{','.join(paths)}",
        "--no-clean-data",
    ]
)
