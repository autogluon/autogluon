import argparse

import yaml
from huggingface_hub import snapshot_download
from huggingface_hub.utils import RepositoryNotFoundError

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--model_list_file", help="file containing list of models to download.", type=str, required=True)

args = parser.parse_args()

model_list_file = args.model_list_file

with open(model_list_file, "r") as fp:
    model_list = yaml.safe_load(fp)  # a dict containing models for different sub_folders
    model_list = list(model_list.values())
    model_list = set(sum(model_list, []))  # concatenate inner model lists

for model in model_list:
    print(f"Downloading {model}")
    try:
        snapshot_download(repo_id=model, cache_dir="/mnt/efs/")
        print(f"Finished downloading {model}")
    except RepositoryNotFoundError:
        print(f"Invalid repo_id: {model}. Failed to download")
