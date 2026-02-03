import argparse
import os
import subprocess
import yaml


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--model_list_file", help="file containing list of models to download.", type=str, required=True)
parser.add_argument("--sub_folder", help="which subfolder to download models for", required=True)
parser.add_argument("--cache_dir", help="place to cache the downloaded models", default="~/.cache/huggingface/hub")

args = parser.parse_args()

model_list_file = args.model_list_file
sub_folder = args.sub_folder
cache_dir = args.cache_dir
cache_dir = os.path.expanduser(cache_dir)

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
