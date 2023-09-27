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
parser.add_argument("--branch_name", help="if it happens to be master then just push the cleaned result, do not evaluate", type=str, required=True)

args = parser.parse_args()

config_path = args.config_path
time_limit = args.time_limit
branch_name = args.branch_name

for root, dirs, files in os.walk(config_path):
    for file in files:
        if file == "tabular_cloud_configs.yaml":
            config_file = os.path.join(root, file)
            break

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

# If it is a PR then perform the evaluation w.r.t cleaned master bench results
if branch_name != "master":
    paths = []
    frameworks = []
    for file in os.listdir("./results"):
        if file.endswith(".csv"):
            file = os.path.join("./results", file)
            df = pd.read_csv(file)
            paths.append(os.path.basename(file))
            frameworks += list(df["framework"].unique())

    modified_list_paths = []
    modified_list_frameworks = []

    for path in paths:
        modified_list_paths.append('--paths')
        modified_list_paths.append(path)

    for framework in frameworks:
        modified_list_frameworks.append('--frameworks-run')
        modified_list_frameworks.append(framework)
        
    paths = modified_list_paths
    frameworks = modified_list_frameworks
    subprocess.run(
        [
            "agbench",
            "evaluate-amlb-results",
            *frameworks,
            "--results-dir-input",
            "./results",
            *paths,
            f"--results-dir-output",
            f"./evaluate",
            "--no-clean-data",
        ]
    )

    unique_framework = {}
    # Renaming the frameworks for dashboard formatting
    for file in os.listdir("./evaluate"):
        if file.endswith("dataset_all.csv"):
            file_path = os.path.join("./evaluate", file)
            df = pd.read_csv(file_path)
            for index, row in df.iterrows():
                if (row['framework'].split('_')[-1] not in unique_framework) and ("AutoGluon" in row['framework']):
                    unique_framework[row['framework']] = row['framework'].split('_')[-1]
    
    if len(unique_framework) > 1:
        unique_framework = dict(sorted(unique_framework.items(), key=lambda item: item[1]))
        earliest_timestamp = next(iter(unique_framework))
        unique_framework[earliest_timestamp] = 'AutoGluon_master'
        for index, (key, value) in enumerate(unique_framework.items()):
            if index > 0:
                unique_framework[key] = f'AutoGluon_PR_{index}'

    df['framework'] = df['framework'].map(unique_framework)
    df.to_csv(file_path, index=False)
    
    for file in os.listdir("./evaluate/pairwise/"):
        if file.endswith(".csv"):
            file_path = os.path.join("./evaluate/pairwise/", file)
            df = pd.read_csv(file_path)

    df['framework'] = df['framework'].map(unique_framework)
    df.to_csv(file_path, index=False)

    # Compare aggregated results with Master branch and return comment
    master_win_rate = 0
    for _, row in df.iterrows():
        if "master" in row['framework']:
            master_win_rate = row['winrate']

    pr_comment = "\nBenchmark Test Result - Pass\n"
    for _, row in df.iterrows():
        if ("master" not in row['framework']) and (master_win_rate >= row['winrate']):
            pr_comment = ""
            pr_comment = "\nBenchmark Test Result - Fail\n"

    with open("final_eval.txt", "w") as file:
        file.write(pr_comment)
