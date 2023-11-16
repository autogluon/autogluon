import json
import logging
import os
import re

import numpy as np
import pandas as pd
import torch
from scipy.special import expit, softmax

logger = logging.getLogger(__name__)


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
    if logits.ndim == 1:
        return expit(logits)
    elif logits.ndim == 2:
        return softmax(logits, axis=1)
    else:
        raise ValueError(f"Unsupported logit dim: {logits.ndim}.")


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


def path_expander(path, base_folder):
    path_l = path.split(";")
    return ";".join([os.path.abspath(os.path.join(base_folder, path)) for path in path_l])


def _read_byte(file):
    with open(file, "rb") as image:
        f = image.read()
        b = bytearray(f)
    return b


def path_to_bytearray_expander(path, base_folder):
    path_l = path.split(";")
    return [_read_byte(os.path.abspath(os.path.join(base_folder, path))) for path in path_l]


def shopee_dataset(
    download_dir: str,
    is_bytearray=False,
):
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

    expander = path_to_bytearray_expander if is_bytearray else path_expander
    train_data["image"] = train_data["image"].apply(lambda ele: expander(ele, base_folder=dataset_path))
    test_data["image"] = test_data["image"].apply(lambda ele: expander(ele, base_folder=dataset_path))
    return train_data, test_data


def merge_spans(sent, pred, for_visualizer=False):
    """Merge subsequent predictions."""
    if isinstance(pred, str):
        # For string values, we assume that it is json-encoded string of the sentences.
        try:
            pred = json.loads(pred)
        except Exception as exp:
            raise RuntimeError(
                f"The received entity annotations is {pred}, "
                f"which can not be encoded with the json format. "
                f"Check your input again, or running `json.loads(pred)` to verify your data."
            )
    spans = {}
    last_start = -1
    last_end = -1
    last_label = ""
    for entity in pred:
        entity_group = entity["entity_group"]
        start = entity["start"]
        end = entity["end"]
        if (
            last_start >= 0
            and not for_visualizer
            and (not re.match("B-", entity_group, re.IGNORECASE))
            and (
                (re.match("I-", entity_group, re.IGNORECASE) and last_label[2:] == entity_group[2:])
                or last_label == entity_group
            )
            and (sent[last_end:start].isspace() or (last_end == start))
        ):
            last_end = end
        else:
            last_start = start
            last_end = end
            last_label = entity_group

        if re.match("B-", last_label, re.IGNORECASE) or re.match("I-", last_label, re.IGNORECASE):
            spans.update({last_start: (last_end, last_label[2:])})
        else:
            spans.update({last_start: (last_end, last_label)})
    return spans


def merge_bio_format(data, preds):
    """Merge predictions with BIO format during prediction."""
    results = []
    for sent, pred in zip(data, preds):
        results.append(
            [
                {"entity_group": value[-1], "start": key, "end": value[0]}
                for key, value in merge_spans(sent, pred).items()
            ]
        )
    return results
