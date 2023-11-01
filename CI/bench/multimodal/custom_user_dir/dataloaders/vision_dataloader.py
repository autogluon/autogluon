import logging
import os

import pandas as pd
import yaml

from autogluon.bench.utils.dataset_utils import get_data_home_dir
from autogluon.common.loaders import load_zip


def path_expander(path, base_folder):
    path_l = path.split(";")
    return ";".join([os.path.abspath(os.path.join(base_folder, path)) for path in path_l])


logger = logging.getLogger(__name__)


class VisionDataLoader:
    def __init__(self, dataset_name: str, dataset_config_file: str, split: str = "train"):
        with open(dataset_config_file, "r") as f:
            config = yaml.safe_load(f)

        self.dataset_config = config[dataset_name]
        if split == "val":
            split = "validation"
        if split not in self.dataset_config["splits"]:
            logger.warning(f"Data split {split} not available.")
            self.data = None
            return

        self.name = dataset_name
        self.split = split
        self.feature_columns = self.dataset_config["feature_columns"]
        self.label_columns = self.dataset_config["label_columns"]

        url = self.dataset_config["url"].format(name=self.name)
        base_dir = get_data_home_dir()
        load_zip.unzip(url, unzip_dir=base_dir)
        self.dataset_dir = os.path.join(base_dir, self.name)

        annotation_filename = self.dataset_config["annotation"].format(name=self.name, split=self.split)
        image_path_pattern = self.dataset_config["image_path"]

        self.data = pd.read_csv(os.path.join(self.dataset_dir, annotation_filename))
        _columns_to_drop = self.data.columns.difference(self.feature_columns + self.label_columns)
        self.data.drop(columns=_columns_to_drop, inplace=True)
        image_base_path = image_path_pattern.format(name=self.name, split=self.split, value="")
        for col in self.feature_columns:
            self.data[col] = self.data[col].apply(
                lambda ele: path_expander(ele, base_folder=os.path.join(self.dataset_dir, image_base_path))
            )

    @property
    def problem_type(self):
        return self.dataset_config["problem_type"]

    @property
    def metric(self):
        return self.dataset_config["metric"]
