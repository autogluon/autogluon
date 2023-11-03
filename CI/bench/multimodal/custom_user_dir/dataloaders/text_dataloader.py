import logging
import os

import pandas as pd
import yaml

from autogluon.bench.utils.dataset_utils import get_data_home_dir
from autogluon.common.loaders._utils import download

logger = logging.getLogger(__name__)


class TextDataLoader:
    def __init__(
        self,
        dataset_name: str,
        dataset_config_file: str,
        split: str = "train",
        lang: str = "en",
        fewshot: bool = False,
        shot: int = 50,
        seed: int = 0,
    ):
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

        subsample_path = self.dataset_config["subsample_path"].format(shot=shot, seed=seed)
        url = self.dataset_config["url"].format(
            name=self.name,
            lang=lang,
            subsample_path=subsample_path if fewshot and self.split in self.dataset_config["subsample_splits"] else "",
            split=self.split,
        )
        base_dir = get_data_home_dir()
        data_dir = os.path.join(self.name, lang)
        if fewshot:
            data_dir = os.path.join(data_dir, "subsampling", f"{shot}_shot-seed{seed}")
        self.dataset_dir = os.path.join(base_dir, data_dir)
        data_path = os.path.join(self.dataset_dir, f"{split}.csv")
        download(url, path=data_path)

        self.data = pd.read_csv(
            data_path,
            header=None,
            names=self.dataset_config["data_columns"],
            sep=self.dataset_config.get("data_sep", ","),
            on_bad_lines="warn",
        )

    @property
    def problem_type(self):
        return self.dataset_config["problem_type"]

    @property
    def metric(self):
        return self.dataset_config["metric"]

    @property
    def feature_columns(self):
        return self.dataset_config["feature_columns"]

    @property
    def label_columns(self):
        return self.dataset_config["label_columns"]
