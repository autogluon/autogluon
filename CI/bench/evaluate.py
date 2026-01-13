import argparse
import os
import subprocess
from datetime import datetime

import pandas as pd
import yaml


def process_results(eval_flag: bool):
    try:
        paths = []
        frameworks = []
        for file in os.listdir("./results"):
            if file.endswith(".csv") and not file.endswith("_min.csv"):
                file = os.path.join("./results", file)
                df = pd.read_csv(file)
                paths.append(os.path.basename(file))
                frameworks += list(df["framework"].unique())

        modified_list_paths = []
        modified_list_frameworks = []

        for path in paths:
            modified_list_paths.append("--paths")
            modified_list_paths.append(path)

        for framework in frameworks:
            modified_list_frameworks.append("--frameworks-run")
            modified_list_frameworks.append(framework)

        paths = modified_list_paths
        frameworks = modified_list_frameworks
        subprocess.run(
            [
                "agbench",
                "evaluate-amlb-results",
                *frameworks,
                "--results-dir-input",
                "./results/",
                *paths,
                f"--results-dir-output",
                f"./evaluate",
                "--no-clean-data",
            ],
            check=True,
        )

        unique_framework = {}
        # Renaming the frameworks for dashboard formatting
        for file in os.listdir("./evaluate"):
            if file.endswith("dataset_valid.csv"):
                file_path = os.path.join("./evaluate", file)
                df = pd.read_csv(file_path)
                for index, row in df.iterrows():
                    if (row["framework"].split("_")[-1] not in unique_framework) and ("AutoGluon" in row["framework"]):
                        unique_framework[row["framework"]] = row["framework"].split("_")[-1]

        if len(unique_framework) > 1:
            unique_framework = dict(sorted(unique_framework.items(), key=lambda item: item[1]))
            earliest_timestamp = next(iter(unique_framework))
            for index, (key, value) in enumerate(unique_framework.items()):
                if eval_flag:
                    if index > 0:
                        unique_framework[key] = f"AutoGluon_master"
                    else:
                        unique_framework[key] = f"AutoGluon_v1.0"
                else:
                    if index > 0:
                        unique_framework[key] = f"AutoGluon_PR_{index}"
                    else:
                        unique_framework[key] = f"AutoGluon_master_branch"

        df["framework"] = df["framework"].map(unique_framework)
        df.to_csv(file_path, index=False)

        for file in os.listdir("./evaluate/pairwise/"):
            if file.endswith(".csv"):
                file_path = os.path.join("./evaluate/pairwise/", file)
                df = pd.read_csv(file_path)
                cols_to_drop = ["% Loss Reduction (median)", "Avg Fit Speed Diff"]
                df = df.drop(columns=cols_to_drop, errors="ignore")
                df = df.rename(columns={"% Loss Reduction": "% Less Avg. Errors"})

        df["framework"] = df["framework"].map(unique_framework)
        df.to_csv(file_path, index=False)
        return df
    except Exception as e:
        raise Exception(f"Failed to process results: {e}") from e


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--config_path", help="path to generated config path to fetch benchmark name", type=str, required=True
    )
    parser.add_argument("--module_name", help="module on which we run benchmark", type=str, required=True)
    parser.add_argument("--time_limit", help="time limit of the benchmark run", type=str, required=True)
    parser.add_argument(
        "--branch_name",
        help="if it happens to be master then just push the cleaned result, do not evaluate",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--benchmark_type",
        help="type of benchmark to run, tabular, timeseries, automm-text etc.",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    config_path = args.config_path
    module_name = args.module_name
    time_limit = args.time_limit
    branch_name = args.branch_name
    benchmark_type = args.benchmark_type
    df1 = pd.DataFrame()

    try:
        for root, dirs, files in os.walk(config_path):
            for file in files:
                if file == f"{module_name}_cloud_configs.yaml":
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
                module_name,
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
                f"s3://autogluon-ci-benchmark/aggregated/{module_name}/{benchmark_name}/",
                "--file-prefix",
                f"results_automlbenchmark_{time_limit}",
                "--benchmark-name-in-input-path",
                "--results-dir-output",
                "./results",
            ],
            check=True,
        )

        subprocess.run(
            [
                "sh",
                "-c",
                "rm -f ./results/*_min.csv ./results/*_min.parquet ./results/*.parquet",
            ],
            check=True,
        )

        # If branch is master Copy v1.0 results from S3
        if branch_name == "master":
            if benchmark_type.startswith("tabular") or benchmark_type.startswith("timeseries"):
                subprocess.run(
                    [
                        "aws",
                        "s3",
                        "cp",
                        "--recursive",
                        f"s3://autogluon-ci-benchmark/version_1.0/cleaned/{module_name}/",
                        "./results",
                    ],
                    check=True,
                )
            else:
                subprocess.run(
                    [
                        "aws",
                        "s3",
                        "cp",
                        "--recursive",
                        f"s3://autogluon-ci-benchmark/version_1.0/cleaned/{module_name}/{benchmark_type}/",
                        "./results",
                    ],
                    check=True,
                )

            df = process_results(eval_flag=True)

            for root, dirs, files in os.walk("./evaluate"):
                for file in files:
                    if file.startswith("AutoGluon") and file.endswith(".csv") and "pairwise" in root:
                        file_path = os.path.join(root, file)
                        df1 = pd.read_csv(
                            file_path, usecols=["framework", "Winrate", "time_train_s", "time_infer_s", "rank"]
                        )

            df1.to_csv("./report_results.csv", index=False, mode="w")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            if benchmark_type.startswith("tabular") or benchmark_type.startswith("timeseries"):
                subprocess.run(
                    [
                        "aws",
                        "s3",
                        "cp",
                        "--recursive",
                        "./evaluate",
                        f"s3://autogluon-ci-benchmark/version_1.0/evaluated/{module_name}/{timestamp}/",
                    ],
                    check=True,
                )
            else:
                subprocess.run(
                    [
                        "aws",
                        "s3",
                        "cp",
                        "--recursive",
                        "./evaluate",
                        f"s3://autogluon-ci-benchmark/version_1.0/evaluated/{module_name}/{benchmark_type}/{timestamp}/",
                    ],
                    check=True,
                )
        # If it is not master then it is a PR, perform the evaluation w.r.t cleaned master bench results
        else:
            df = process_results(eval_flag=False)

            # Compare aggregated results with Master branch and return comment
            master_win_rate = 0
            for _, row in df.iterrows():
                if "master" in row["framework"]:
                    master_win_rate = row["Winrate"]

            pr_comment = f"\nBenchmark Test Result - Pass\nEvaluation Results Path: s3://autogluon-ci-benchmark/evaluation/{module_name}/{branch_name}\n"
            for _, row in df.iterrows():
                if ("master" not in row["framework"]) and (master_win_rate >= row["Winrate"]):
                    pr_comment = ""
                    pr_comment = f"\nBenchmark Test Result - Fail\nEvaluation Results Path: s3://autogluon-ci-benchmark/evaluation/{module_name}/{branch_name}\n"

            with open("final_eval.txt", "w") as file:
                file.write(pr_comment)
    except Exception as e:
        print(f"An exception occurred: {e}")


if __name__ == "__main__":
    main()
