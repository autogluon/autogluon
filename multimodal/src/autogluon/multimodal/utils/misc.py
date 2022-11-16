import logging
import os

import numpy as np
import pandas as pd
import torch
from scipy.special import softmax

from ..constants import AUTOMM

logger = logging.getLogger(AUTOMM)


def logits_to_prob(logits: np.ndarray):
    """
    Convert logits to probabilities.

    Parameters
    ----------
    logits
        The logits output of a classification head.

    Returns
    -------
    Probabilities.
    """
    assert logits.ndim == 2
    prob = softmax(logits, axis=1)
    return prob


def tensor_to_ndarray(tensor: torch.Tensor):
    """
    Convert Pytorch tensor to numpy array.

    Parameters
    ----------
    tensor
        A Pytorch tensor.

    Returns
    -------
    A ndarray.
    """
    return tensor.detach().cpu().float().numpy()


def shopee_dataset(download_dir):
    """
    Download Shopee dataset for demo.

    Parameters
    ----------
    download_dir
        Path to save the dataset locally.

    Returns
    -------
    train and test set of Shopee dataset in pandas DataFrame format.
    """
    zip_file = "https://automl-mm-bench.s3.amazonaws.com/vision_datasets/shopee.zip"
    from autogluon.core.utils.loaders import load_zip

    load_zip.unzip(zip_file, unzip_dir=download_dir)

    dataset_path = os.path.join(download_dir, "shopee")
    train_data = pd.read_csv(f"{dataset_path}/train.csv")
    test_data = pd.read_csv(f"{dataset_path}/test.csv")

    def path_expander(path, base_folder):
        path_l = path.split(";")
        return ";".join([os.path.abspath(os.path.join(base_folder, path)) for path in path_l])

    train_data["image"] = train_data["image"].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
    test_data["image"] = test_data["image"].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
    return train_data, test_data
