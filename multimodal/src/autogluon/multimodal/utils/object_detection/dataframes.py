"""
Data frame operations for object detection data processing.
This module contains functions for converting between different data formats and performing
operations on detection data frames.
"""

import logging
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

from .image import _get_image_info

logger = logging.getLogger(__name__)


def get_df_unique_classes(data: pd.DataFrame) -> tuple[list, dict]:
    """
    Get the unique classes and their category IDs from the dataframe for object detection.

    Args:
        data: DataFrame holding the data for object detection. Each row should contain a 'rois'
             column with detection boxes and class labels.

    Returns:
        A tuple containing:
            - class_names: list of unique class name strings
            - category_ids: dict mapping class names to their numeric IDs
    """
    unique_classes = {}

    for _, row in data.iterrows():
        rois = row["rois"]
        
        for roi in rois:
            # Unpack ROI values (assuming last element is class label)
            class_label = roi[-1]
            
            # Add new classes to the dictionary with auto-incrementing IDs
            if class_label not in unique_classes:
                # Start IDs from 1, as 0 is often reserved for background
                unique_classes[class_label] = len(unique_classes) + 1

    class_names = list(unique_classes.keys())
    return class_names, unique_classes


def from_dict(data: dict) -> pd.DataFrame:
    """
    Construct a dataframe from a data dictionary with image paths.

    Args:
        data: Dict containing the image paths in format {"image": ["img1.jpg", "img2.jpg", ...]}

    Returns:
        DataFrame with columns "image", "rois", and "label"
    """
    df_data = {
        "image": [],
        "rois": [],
        "label": []
    }

    for image in data["image"]:
        df_data["image"].append(image)
        # Add dummy ROIs and labels
        dummy_annotation = [[-1, -1, -1, -1, 0]]
        df_data["rois"].append(dummy_annotation)
        df_data["label"].append(dummy_annotation)

    df = pd.DataFrame(df_data)
    return df.sort_values("image").reset_index(drop=True)


def from_list(image_paths: List[str]) -> pd.DataFrame:
    """
    Construct a dataframe from list of image paths.

    Args:
        image_paths: List containing the image paths

    Returns:
        DataFrame with columns "image", "rois", and "label"
    """
    df_data = {
        "image": [],
        "rois": [],
        "label": []
    }

    for image_path in image_paths:
        df_data["image"].append(image_path)
        # Add dummy ROIs and labels
        dummy_annotation = [[-1, -1, -1, -1, 0]]
        df_data["rois"].append(dummy_annotation)
        df_data["label"].append(dummy_annotation)

    df = pd.DataFrame(df_data)
    return df.sort_values("image").reset_index(drop=True)


def from_str(image_path: str) -> pd.DataFrame:
    """
    Construct a dataframe from a single image path.

    Args:
        image_path: String of the image path

    Returns:
        DataFrame with columns "image", "rois", and "label"
    """
    df_data = {
        "image": [image_path],
        "rois": [[[-1, -1, -1, -1, 0]]],  # Dummy ROIs
        "label": [[[-1, -1, -1, -1, 0]]]  # Dummy labels
    }

    df = pd.DataFrame(df_data)
    return df.sort_values("image").reset_index(drop=True)


def sanity_check_dataframe(data: pd.DataFrame):
    """
    Checking if the dataframe contains valid headers and values
    Parameters
    ----------
    data
        dataframe holding the data for object detection
    Returns
    -------

    """
    if "image" not in data:
        raise ValueError(f"column 'image' not found in data column names: {data.columns.to_list()}")
    if "rois" not in data and "label" not in data:
        raise ValueError(f"Both column 'rois' and 'label' not found in data column names: {data.columns.to_list()}")
    else:
        if "rois" not in data:
            warnings.warn(
                f"column 'rois' not found in data column names: {data.columns.to_list()}. Copying from 'label' column..."
            )
            data["rois"] = data["label"]
        if "label" not in data:
            warnings.warn(
                f"column 'label' not found in data column names: {data.columns.to_list()}. Copying from 'rois' column..."
            )
            data["label"] = data["rois"]
    assert data.shape[0] > 0, "data frame is empty"


def object_detection_data_to_df(
    data: Union[pd.DataFrame, dict, list, str], 
    coco_root: Optional[str] = None
) -> pd.DataFrame:
    """
    Convert various input formats to a standardized detection dataframe.

    Args:
        data: Input data in one of these formats:
            - pd.DataFrame: DataFrame with required columns
            - dict: Dictionary with image paths
            - list: List of image paths
            - str: Path to COCO/VOC dataset or single image
        coco_root: Root directory for COCO dataset if applicable

    Returns:
        DataFrame with columns "image", "rois", and "label"

    Raises:
        TypeError: If input data type is not supported
    """
    if isinstance(data, dict):
        return from_dict(data)
    
    if isinstance(data, list):
        return from_list(data)
    
    if isinstance(data, str):
        if os.path.isdir(data) or data.endswith(".json"):
            from .format_converter import from_coco_or_voc
            return from_coco_or_voc(data, coco_root=coco_root)
        return from_str(data)
    
    if isinstance(data, pd.DataFrame):
        sanity_check_dataframe(data)
        return data

    raise TypeError(
        f"Expected data to be dict, list, str or pd.DataFrame, but got {type(data)}"
    )


def convert_result_df(
    predictions: List, 
    data: Union[pd.DataFrame, Dict], 
    detection_classes: List[str], 
    result_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Convert detection results to DataFrame format.

    Args:
        predictions: List of detection results per image
        data: Input data containing image information
        detection_classes: List of all available detection classes
        result_path: Optional path to save results CSV

    Returns:
        DataFrame with detection results per image
    """
    image_names = data["image"].tolist() if isinstance(data, pd.DataFrame) else data["image"]
    idx_to_classname = {i: classname for i, classname in enumerate(detection_classes)}
    
    results = []
    for image_pred, image_name in zip(predictions, image_names):
        boxes = []
        for i in range(len(image_pred["bboxes"])):
            boxes.append({
                "class": idx_to_classname[image_pred["labels"][i].item()],
                "class_id": image_pred["labels"][i].item(),
                "bbox": image_pred["bboxes"][i].tolist(),
                "score": image_pred["scores"][i].item(),
            })
        results.append([image_name, boxes])

    result_df = pd.DataFrame(results, columns=["image", "bboxes"])
    
    if result_path:
        result_df.to_csv(result_path, index=False)
        logger.info("Saved detection results to %s", result_path)
        
    return result_df
