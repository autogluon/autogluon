"""
COCO format utilities for object detection.
This module provides functionality for working with COCO format datasets,
including loading, saving, and evaluating detection results.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from ...constants import (
    BBOX,
    LABEL,
    MAP,
    MAP_50,
    MAP_75,
    MAP_LARGE,
    MAP_MEDIUM,
    MAP_SMALL,
    MAR_1,
    MAR_10,
    MAR_100,
    MAR_LARGE,
    MAR_MEDIUM,
    MAR_SMALL,
    MEAN_AVERAGE_PRECISION,
)
from .bbox import bbox_clip_xyxy, bbox_xywh_to_xyxy, bbox_xyxy_to_xywh
from .image import _get_image_info, get_image_filename

logger = logging.getLogger(__name__)


class COCODataset:
    """Class for handling COCO format datasets."""

    def __init__(self, anno_file: str, category_ids: Optional[List[int]] = None):
        """
        Initialize COCO dataset handler.

        Args:
            anno_file: Path to COCO format annotation file
            category_ids: Optional list of category IDs to use
        """
        self.anno_file = anno_file
        self._load_annotations(category_ids)

    def _load_annotations(self, category_ids: Optional[List[int]] = None) -> None:
        """Load annotations from file and setup internal mappings."""
        with open(self.anno_file, "r") as f:
            data = json.load(f)

        # Build image filename to ID mapping
        self.image_filename_to_id = {get_image_filename(img["file_name"]): int(img["id"]) for img in data["images"]}

        # Set category IDs
        if category_ids is not None:
            self.category_ids = category_ids
        else:
            self.category_ids = (
                [cat["id"] for cat in data["categories"]] if "categories" in data else list(range(9999))
            )

    def get_image_id_from_path(self, image_path: str) -> int:
        """Get COCO image ID from image path."""
        return self.image_filename_to_id[get_image_filename(image_path)]

    def save_result(self, ret: List, data: pd.DataFrame, save_path: str) -> None:
        """
        Save detection results in COCO format.

        Args:
            ret: List of detection results
            data: DataFrame containing image information
            save_path: Path to save JSON results
        """
        coco_results = []

        for i, row in data.reset_index(drop=True).iterrows():
            image_id = self.get_image_id_from_path(row["image"])
            pred_result = ret[i]

            for bbox_idx in range(len(pred_result["bboxes"])):
                coco_results.append(
                    {
                        "image_id": image_id,
                        "category_id": self.category_ids[int(pred_result["labels"][bbox_idx].item())],
                        "bbox": bbox_xyxy_to_xywh(pred_result["bboxes"][bbox_idx].tolist()),
                        "score": pred_result["scores"][bbox_idx].item(),
                    }
                )

        with open(save_path, "w") as f:
            logger.info("Saving COCO results to %s", save_path)
            json.dump(coco_results, f)


def get_coco_format_classes(anno_file: str) -> Tuple[List[str], List[int]]:
    """
    Get class names and category IDs from COCO format annotation file.

    Args:
        anno_file: Path to COCO format annotation file

    Returns:
        Tuple containing:
            - List of class names
            - List of corresponding category IDs

    Raises:
        ValueError: If annotation file cannot be loaded
    """
    try:
        with open(anno_file, "r") as f:
            annotation = json.load(f)
    except Exception as e:
        raise ValueError(f"Failed to load COCO annotations from {anno_file}: {e}")

    class_names = [cat["name"] for cat in annotation["categories"]]
    category_ids = [cat["id"] for cat in annotation["categories"]]

    return class_names, category_ids


def cocoeval_torchmetrics(outputs: List[Dict]) -> Dict:
    """
    Evaluate detection outputs using torchmetrics' mAP implementation.

    Args:
        outputs: List of detection outputs per image

    Returns:
        Dictionary containing mAP metrics
    """
    map_metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=False)

    preds = []
    targets = []

    for output in outputs:
        preds.append(
            {
                "boxes": output[BBOX]["bboxes"].to("cpu"),
                "scores": output[BBOX]["scores"].to("cpu"),
                "labels": output[BBOX]["labels"].to("cpu"),
            }
        )

        targets.append(
            {
                "boxes": output[LABEL]["bboxes"].to("cpu"),
                "labels": output[LABEL]["labels"].to("cpu"),
            }
        )

    map_metric.update(preds, targets)
    return map_metric.compute()


def cocoeval_pycocotools(outputs: List[Dict], data: pd.DataFrame, anno_file: str, cache_path: str) -> np.ndarray:
    """
    Evaluate detection outputs using pycocotools' mAP implementation.

    Args:
        outputs: List of detection outputs per image
        data: DataFrame containing image information
        anno_file: Path to COCO annotation file
        cache_path: Path to cache prediction results

    Returns:
        Array containing COCO evaluation metrics
    """
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        raise ImportError(
            "pycocotools is required for COCO evaluation. "
            "Install it with: pip install pycocotools-windows (on Windows) "
            "or pip install pycocotools (on Linux/Mac)"
        )

    from .. import extract_from_output

    # Initialize COCO dataset and save predictions
    coco_dataset = COCODataset(anno_file)
    bbox_results = extract_from_output(ret_type=BBOX, outputs=outputs)
    coco_dataset.save_result(bbox_results, data, cache_path)

    # Perform evaluation
    coco_gt = COCO(anno_file)
    coco_dt = coco_gt.loadRes(cache_path)

    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    return evaluator.stats


def parse_detection_result(result: Union[Dict, np.ndarray]) -> Dict:
    """
    Parse COCO detection evaluation results into a standardized format.

    Args:
        result: Raw detection results from evaluation

    Returns:
        Dictionary containing parsed metrics
    """
    if isinstance(result, np.ndarray):
        parsed = {
            MAP: result[0],
            MAP_50: result[1],
            MAP_75: result[2],
            MAP_SMALL: result[3],
            MAP_MEDIUM: result[4],
            MAP_LARGE: result[5],
            MAR_1: result[6],
            MAR_10: result[7],
            MAR_100: result[8],
            MAR_SMALL: result[9],
            MAR_MEDIUM: result[10],
            MAR_LARGE: result[11],
        }
        parsed[MEAN_AVERAGE_PRECISION] = parsed[MAP]
        return parsed

    result[MEAN_AVERAGE_PRECISION] = result[MAP]
    return result


def cocoeval(
    outputs: List[Dict],
    data: pd.DataFrame,
    anno_file: str,
    cache_path: str,
    metrics: Optional[Union[str, List[str]]] = None,
    tool: str = "pycocotools",
) -> Dict:
    """
    Evaluate detection outputs using specified evaluation tool.

    Args:
        outputs: List of detection outputs per image
        data: DataFrame containing image information
        anno_file: Path to COCO annotation file
        cache_path: Path to cache prediction results
        metrics: Specific metrics to return
        tool: Evaluation tool to use ('pycocotools' or 'torchmetrics')

    Returns:
        Dictionary containing evaluation metrics

    Raises:
        ValueError: If unsupported evaluation tool is specified
    """
    if not tool or tool == "pycocotools":
        result = cocoeval_pycocotools(outputs, data, anno_file, cache_path)
    elif tool == "torchmetrics":
        result = cocoeval_torchmetrics(outputs)
    else:
        raise ValueError(f"Unsupported evaluation tool: {tool}")

    result = parse_detection_result(result)

    # Filter metrics if specified
    if metrics:
        if isinstance(metrics, str):
            return {metrics.lower(): result[metrics.lower()]}
        return {metric.lower(): result[metric.lower()] for metric in metrics}

    return result


def save_result_coco_format(
    data_path: str, predictions: List, category_ids: List[int], result_path: str, coco_root: Optional[str] = None
) -> None:
    """
    Save detection results in COCO format.

    Args:
        data_path: Path to COCO dataset
        predictions: List of detection predictions
        category_ids: List of category IDs
        result_path: Path to save results
        coco_root: Optional root directory for COCO dataset
    """
    from .format_converter import from_coco_or_voc

    # Initialize COCO dataset and save results
    coco_dataset = COCODataset(data_path, category_ids=category_ids)
    result_name = os.path.splitext(result_path)[0]
    json_path = f"{result_name}.json"

    data_df = from_coco_or_voc(data_path, "test", coco_root=coco_root)
    coco_dataset.save_result(predictions, data_df, json_path)

    logger.info("Saved COCO format results to %s", json_path)
