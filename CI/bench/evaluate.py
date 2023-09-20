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
            f"./results/evaluate",
            "--no-clean-data",
        ]
    )
    
    # High level implementation -
    # Write the print logic below
    # You already have 2 CSVs post evaluation {file_all, file_aggregated} - check where they are stored, mostly in ./results
    # If not then find the path
    # Let's read file_aggregate for now, in that read Win Rate for every framework 
    # If Win Rate of Master greater than Win Rate of PR then print it (it will eventually show up in GitHub)
    # Write this print in a file > $cwd/final_eval.txt
    # Do the artifact upload and download
    # For now read one metric from the CSV
    for file in os.listdir("./results/evaluate/pairwise/"):
        print("\nFile is: ", file)
        if file.endswith(".csv"):
            df = pd.read_csv(file)
            unique_framework = {}
            for index, row in df.iterrows():
                if row['framework'] not in unique_framework:
                    unique_framework[row['framework']] = row['winrate']

    master_win_rate = None
    for key in unique_framework:
        if "master" in key:
            master_win_rate = unique_framework[key]

    pr_comment = None
    for key in unique_framework:
        if ("master" not in key) and (master_win_rate >= unique_framework[key]):
            pr_comment = "Benchmark Test Result - Negative"

    with open("final_eval.txt", "w") as file:
        file.write(pr_comment)
