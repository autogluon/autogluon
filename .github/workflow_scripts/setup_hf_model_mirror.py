import argparse
import os
import subprocess
import yaml


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--model_list_file", help="file containing list of models to download.", type=str, required=True)
parser.add_argument("--dataset_list_file", help="file containing list of datasets to download.", type=str, default=None)
parser.add_argument("--sub_folder", help="which subfolder to download models for", required=True)
parser.add_argument("--cache_dir", help="place to cache the downloaded models", default="~/.cache/huggingface/hub")
parser.add_argument("--datasets_cache_dir", help="place to cache the downloaded datasets", default="~/.cache/huggingface/datasets")

args = parser.parse_args()

model_list_file = args.model_list_file
dataset_list_file = args.dataset_list_file
sub_folder = args.sub_folder
cache_dir = os.path.expanduser(args.cache_dir)
datasets_cache_dir = os.path.expanduser(args.datasets_cache_dir)

with open(model_list_file, "r") as fp:
    models = yaml.safe_load(fp)
    sub_folder_models = models[sub_folder]

os.makedirs(cache_dir, exist_ok=True)

bucket = "s3://autogluon-hf-model-mirror"
model_prefix = "models"
for model in sub_folder_models:
    model_name = "--".join(model.split("/"))
    model_name = "--".join([model_prefix, model_name])
    model_path = os.path.join(cache_dir, model_name)
    os.makedirs(model_path, exist_ok=True)
    s3_path = bucket + "/" + model_name
    subprocess.run(["aws", "s3", "cp", s3_path, model_path, "--recursive", "--quiet"])

# Sync datasets from S3 if dataset list is provided
if dataset_list_file and os.path.exists(dataset_list_file):
    with open(dataset_list_file, "r") as fp:
        datasets = yaml.safe_load(fp)
        sub_folder_datasets = datasets.get(sub_folder, [])

    if sub_folder_datasets:
        os.makedirs(datasets_cache_dir, exist_ok=True)
        # Sync entire datasets directory from S3
        s3_datasets_path = bucket + "/datasets"
        subprocess.run(["aws", "s3", "cp", s3_datasets_path, datasets_cache_dir, "--recursive", "--quiet"])
