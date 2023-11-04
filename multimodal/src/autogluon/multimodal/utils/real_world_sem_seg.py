import logging
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import pandas as pd
from .inference import predict, extract_from_output
from ..constants import LOGITS, LABEL
import torch
from ..optimization.utils import get_metric

logger = logging.getLogger(__name__)


def from_paths(
    img_root: str,
    gt_root: str,
):
    """
    Load dataset from file paths.

    Parameters
    ----------
    img_root
        The path to the images.
    gt_root
        The path to the ground truth images..

    Returns
    -------
    A dataframe with columns "image", "gt", and "label".
    """

    def file2id(folder_path, file_path):
        # extract relative path starting from `folder_path`
        image_id = os.path.normpath(os.path.relpath(file_path, start=folder_path))
        # remove file extension
        if "segmentation" in image_id:  # for isic only
            arr = os.path.splitext(image_id)[0].split("_")
            image_id = f"{arr[0]}_{arr[1]}"
        else:
            image_id = os.path.splitext(image_id)[0]
        return image_id

    # load entries
    d = {"image": [], "gt": []}

    img_paths = os.listdir(img_root)
    gt_paths = os.listdir(gt_root)
    # image_ext = 'png'
    img_paths = sorted(
        (f for f in img_paths),
        key=lambda file_path: file2id(img_root, file_path),
    )
    gt_paths = sorted(
        (f for f in gt_paths),
        key=lambda file_path: file2id(gt_root, file_path),
    )

    for img_path, gt_path in zip(img_paths, gt_paths):
        abs_img_path = os.path.join(img_root, img_path)
        abs_gt_path = os.path.join(gt_root, gt_path)

        d["image"].append(abs_img_path)
        d["gt"].append(abs_gt_path)

    df = pd.DataFrame(d)
    df["label"] = df.loc[:, "gt"].copy()
    return df.sort_values("image").reset_index(drop=True)


def save_sem_seg_result_df(pred: Iterable, data: Union[pd.DataFrame, Dict], result_path: Optional[str] = None):
    """
    Saving segmenatation results in pd.DataFrame format (per image)

    Parameters
    ----------
    pred
        List containing segmenatation results for one image
    data
        pandas data frame or dict containing the image information to be tested
    result_path
        path to save result
    Returns
    -------
    The segmenatation results as pandas DataFrame
    """
    if isinstance(data, dict):
        image_names = data["image"]
    else:
        image_names = data["image"].to_list()
    results = []

    for image_pred, image_name in zip(pred, image_names):
        results.append([image_name, image_pred])
    result_df = pd.DataFrame(results, columns=["image", "logits"])
    if result_path:
        result_df.to_csv(result_path, index=False)
        logger.info("Saved detection results to {}".format(result_path))
    return result_df


def setup_segmentation_train_tuning_data(predictor, max_num_tuning_data, seed, train_data, tuning_data):
    if isinstance(train_data, str):
        train_img_root, train_gt_root = train_data.split(":")
        train_data = from_paths(train_img_root, train_gt_root)
        if tuning_data is not None:
            val_img_root, val_gt_root = tuning_data.split(":")
            tuning_data = from_paths(val_img_root, val_gt_root)
            if max_num_tuning_data is not None:
                if len(tuning_data) > max_num_tuning_data:
                    tuning_data = tuning_data.sample(
                        n=max_num_tuning_data, replace=False, random_state=seed
                    ).reset_index(drop=True)
    else:
        raise TypeError(f"Expected train_data to have type str, but got type: {type(train_data)}")
    return train_data, tuning_data


def setup_segmentation_eval_data(eval_data):
    if isinstance(eval_data, str):
        eval_img_root, eval_gt_root = eval_data.split(":")
        eval_data = from_paths(eval_img_root, eval_gt_root)
    else:
        raise TypeError(f"Expected train_data to have type str, but got type: {type(eval_data)}")
    return eval_data

def evaluate_semantic_segmentation(
    predictor,
    data: Union[pd.DataFrame, dict, list, str],
    metrics: Optional[Union[str, List[str]]] = None,
    realtime: Optional[bool] = None,
):
    """
    Evaluate object detection model on a test dataset in COCO format.

    Parameters
    ----------
    predictor
        A predictor object.
    """
    outputs = predict(
        predictor=predictor,
        data=data,
        requires_label=True,
        realtime=realtime,
    )

    logits = extract_from_output(ret_type=LOGITS, outputs=outputs, as_ndarray=False)
    y_pred = logits.float()
    y_true = [ele[LABEL] for ele in outputs]

    y_true = torch.cat(y_true)

    assert len(y_true) == len(y_pred)

    results = {}
    for per_metric_name in metrics:
        per_metric, _ = get_metric(
            metric_name=per_metric_name.lower()
        )
        per_metric.update(y_pred, y_true)
        score = per_metric.compute()

        results[per_metric_name] = score.item()

    return results


