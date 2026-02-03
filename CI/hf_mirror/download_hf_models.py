import argparse
import os
import yaml

from huggingface_hub import snapshot_download
from huggingface_hub.utils import RepositoryNotFoundError


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--model_list_file', help='file containing list of models to download.', type=str, required=True)
parser.add_argument('--dataset_list_file', help='file containing list of datasets to download.', type=str, default=None)

args = parser.parse_args()

model_list_file = args.model_list_file
dataset_list_file = args.dataset_list_file

with open(model_list_file, 'r') as fp:
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

# Download HuggingFace datasets if dataset list is provided
if dataset_list_file and os.path.exists(dataset_list_file):
    from datasets import load_dataset
    from datasets.exceptions import DatasetNotFoundError

    with open(dataset_list_file, 'r') as fp:
        dataset_list = yaml.safe_load(fp)
        dataset_list = list(dataset_list.values())
        dataset_list = set(sum(dataset_list, []))  # concatenate inner dataset lists

    datasets_cache_dir = "/mnt/efs/datasets"
    os.makedirs(datasets_cache_dir, exist_ok=True)

    for dataset_spec in dataset_list:
        # Handle dataset specs like "glue/mrpc" -> dataset_name="glue", config="mrpc"
        parts = dataset_spec.split("/")
        if len(parts) == 2 and parts[0] in ["glue", "super_glue"]:
            dataset_name, config = parts
        else:
            dataset_name = dataset_spec
            config = None

        print(f"Downloading dataset: {dataset_spec}")
        try:
            if config:
                load_dataset(dataset_name, config, cache_dir=datasets_cache_dir)
            else:
                load_dataset(dataset_name, cache_dir=datasets_cache_dir)
            print(f"Finished downloading dataset: {dataset_spec}")
        except (DatasetNotFoundError, Exception) as e:
            print(f"Failed to download dataset {dataset_spec}: {e}")
