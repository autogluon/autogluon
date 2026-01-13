import argparse
import os

import yaml

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--module", help="module to run ag-bench on", type=str, required=True)
parser.add_argument("--repository", help="git repository to run autogluon on", type=str, required=True)
parser.add_argument("--branch", help="git branch to run autogluon on", type=str, required=True)
parser.add_argument(
    "--folds_to_run", help="number of folds to run, has to be greater than 0", type=int, required=False
)

args = parser.parse_args()

module = args.module
repository = args.repository
branch = args.branch
folds_to_run = args.folds_to_run
current_dir = os.path.dirname(__file__)

if module == "multimodal":
    framework_template_file = os.path.join(
        current_dir, f"{module}/custom_user_dir", "multimodal_frameworks_template.yaml"
    )
    framework_benchmark_file = os.path.join(os.path.dirname(framework_template_file), "multimodal_frameworks.yaml")
    constraints_file = os.path.join(current_dir, f"{module}/custom_user_dir", "multimodal_constraints.yaml")
else:
    framework_template_file = os.path.join(current_dir, f"{module}/amlb_user_dir", "frameworks_template.yaml")
    framework_benchmark_file = os.path.join(os.path.dirname(framework_template_file), "frameworks_benchmark.yaml")
    constraints_file = os.path.join(current_dir, f"{module}/amlb_user_dir", "constraints.yaml")

if folds_to_run > 0 and module != "multimodal":
    constraints = {}
    with open(constraints_file, "r") as f:
        constraints = yaml.safe_load(f)

    for constraint in constraints.values():
        constraint["folds"] = folds_to_run

    with open(constraints_file, "w") as f:
        yaml.safe_dump(constraints, f)

frameworks = {}
with open(framework_template_file, "r") as f:
    frameworks = yaml.safe_load(f)

for framework in frameworks.values():
    framework["repo"] = repository
    framework["version"] = branch

with open(framework_benchmark_file, "w") as f:
    yaml.safe_dump(frameworks, f)
