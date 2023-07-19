import argparse
import boto3
import os
import yaml


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)


parser.add_argument("--repository", help="git repository to run autogluon on", type=str, required=True)
parser.add_argument("--branch", help="git branch to run autogluon on", type=str, required=True)

args = parser.parse_args()


repository = args.repository
branch = args.branch
framework_template_file = os.path.join(os.path.dirname(__file__), "amlb_user_dir", "framework_template.yaml")
framework_benchmark_file = os.path.join(os.path.dirname(framework_template_file), "framework_benchmark.yaml")

frameworks = {}
with open(framework_template_file, "r") as f:
    frameworks = yaml.safe_load(f)

for framework in frameworks.values():
    framework["repo"] = repository
    framework["version"] = branch

with open(framework_benchmark_file, "w") as f:
    yaml.safe_dump(frameworks, f)
