"""
Dataset format conversion utilities for object detection.
Handles conversions between different dataset formats (COCO, VOC, etc.)
and provides utilities for processing annotation files.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import defusedxml.ElementTree as ET
import pandas as pd

from .bbox import bbox_clip_xyxy, bbox_xywh_to_xyxy, bbox_xyxy_to_xywh
from .coco import get_coco_format_classes
from .dataframes import get_df_unique_classes
from .image import _get_image_info
from .pycocotools_helper import try_import_pycocotools

logger = logging.getLogger(__name__)


def from_coco_or_voc(file_path: str, splits: Optional[str] = None, coco_root: Optional[str] = None) -> pd.DataFrame:
    """
    Convert data from COCO or VOC format to pandas DataFrame.

    Args:
        file_path: Path to data (COCO JSON file or VOC directory)
        splits: Dataset splits to use for VOC format
        coco_root: Root directory for COCO dataset

    Returns:
        DataFrame with standardized detection format
    """
    if os.path.isdir(file_path):
        return from_voc(root=file_path, splits=splits)
    return from_coco(file_path, coco_root=coco_root)


def from_coco(
    anno_file: str, coco_root: Optional[str] = None, min_object_area: float = 0, use_crowd: bool = False
) -> pd.DataFrame:
    """
    Load COCO format annotations into DataFrame.

    Args:
        anno_file: Path to COCO annotation JSON file
        coco_root: Root directory containing images
        min_object_area: Minimum object area to include
        use_crowd: Whether to include crowd annotations

    Returns:
        DataFrame with standardized detection format
    """
    try_import_pycocotools()
    from pycocotools.coco import COCO

    # Initialize COCO API
    coco = COCO(anno_file)

    # Determine dataset root directory
    if coco_root is None:
        coco_root = os.path.join(os.path.dirname(anno_file), "..")
        logger.info("Using default COCO root: %s", coco_root)

    # Check if annotations exist
    try:
        num_annotations = len(coco.getAnnIds())
    except KeyError:
        num_annotations = 0
        logger.warning("No annotations found in COCO file")

    # Process images and annotations
    data = {"image": [], "rois": []}
    image_ids = sorted(coco.getImgIds())

    for img_entry in coco.loadImgs(image_ids):
        # Get image path
        if "coco_url" in img_entry:
            dirname, filename = img_entry["coco_url"].split("/")[-2:]
            img_path = os.path.join(coco_root, dirname, filename)
        else:
            img_path = os.path.join(coco_root, img_entry["file_name"])

        if not os.path.exists(img_path):
            logger.warning("Skipping missing image: %s", img_path)
            continue

        # Get annotations for image
        rois, _ = _check_load_coco_bbox(coco, img_entry, min_object_area, use_crowd)

        if not rois and num_annotations > 0:
            continue

        data["image"].append(img_path)
        data["rois"].append(rois if rois else [[-1, -1, -1, -1, 0]])

    df = pd.DataFrame(data)
    df["label"] = df["rois"].copy()
    return df.sort_values("image").reset_index(drop=True)


def _check_load_coco_bbox(
    coco, img_entry: dict, min_object_area: float = 0, use_crowd: bool = False
) -> Tuple[List, List]:
    """
    Load and validate COCO bounding boxes for an image.

    Args:
        coco: COCO API instance
        img_entry: Image entry from COCO dataset
        min_object_area: Minimum object area to include
        use_crowd: Whether to include crowd annotations

    Returns:
        Tuple of (valid_boxes, is_crowd_flags)
    """
    # Get annotations for image
    img_id = img_entry["id"]
    img_id = [img_id] if not isinstance(img_id, (list, tuple)) else img_id
    ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=None)
    annotations = coco.loadAnns(ann_ids)

    valid_boxes = []
    is_crowd_flags = []
    img_width = img_entry["width"]
    img_height = img_entry["height"]

    for ann in annotations:
        # Filter annotations
        if ann["area"] < min_object_area or ann.get("ignore", 0) == 1 or (not use_crowd and ann.get("iscrowd", 0)):
            continue

        # Convert box format and clip to image bounds
        xmin, ymin, xmax, ymax = bbox_clip_xyxy(bbox_xywh_to_xyxy(ann["bbox"]), img_width, img_height)

        # Validate box
        if ann["area"] > 0 and xmax > xmin and ymax > ymin:
            cat_ids = coco.getCatIds()
            id_to_idx = dict(zip(cat_ids, range(len(cat_ids))))
            class_id = id_to_idx[coco.loadCats(ann["category_id"])[0]["id"]]
            valid_boxes.append(
                [
                    float(xmin),
                    float(ymin),
                    float(xmax),
                    float(ymax),
                    class_id,
                ]
            )
            is_crowd_flags.append(ann.get("iscrowd", 0))

    return valid_boxes, is_crowd_flags


def from_voc(
    root: str,
    splits: Optional[Union[str, Tuple[str, ...]]] = None,
    exts: Union[str, Tuple[str, ...]] = (".jpg", ".jpeg", ".png"),
) -> pd.DataFrame:
    """
    Load Pascal VOC format dataset into DataFrame.

    Args:
        root: Root directory of VOC dataset
        splits: Dataset splits to use (e.g., 'train', 'val')
        exts: Valid image extensions

    Returns:
        DataFrame with standardized detection format

    Raises:
        FileNotFoundError: If required files are missing
    """
    root_path = Path(root).expanduser()
    img_list = []

    # Get class mappings
    class_names, _ = get_voc_format_classes(root)
    name_to_idx = {name: idx for idx, name in enumerate(class_names)}

    # Get image list
    if splits:
        logger.debug("Using splits: %s for root: %s", splits, root)
        splits = [splits] if isinstance(splits, str) else splits

        for split in splits:
            split_file = root_path / "ImageSets" / "Main" / split
            if not split_file.exists():
                split_file = split_file.with_suffix(".txt")

            if not split_file.exists():
                raise FileNotFoundError(f"Split file not found: {split_file}")

            with split_file.open() as f:
                img_list.extend([line.split()[0].strip() for line in f])
    else:
        logger.debug("No split specified, using all images in %s", root_path / "JPEGImages")
        exts = [exts] if isinstance(exts, str) else exts
        for ext in exts:
            img_list.extend([p.stem for p in (root_path / "JPEGImages").glob(f"*{ext}")])

    # Process annotations
    data = {"image": [], "rois": []}
    logger.info("Processing %d images", len(img_list))

    for img_id in img_list:
        anno_path = root_path / "Annotations" / f"{img_id}.xml"

        # Parse XML annotation
        tree = ET.parse(anno_path)
        root_elem = tree.getroot()

        # Get image path and dimensions
        img_filename = root_elem.find("filename").text
        if "." not in img_filename:
            img_filename += ".jpg"

        img_path = str(root_path / "JPEGImages" / img_filename)

        size_elem = root_elem.find("size")
        width = float(size_elem.find("width").text)
        height = float(size_elem.find("height").text)

        # Process objects
        boxes = []
        for obj in root_elem.iter("object"):
            class_name = obj.find("name").text.strip().lower()
            class_idx = name_to_idx[class_name]

            bbox = obj.find("bndbox")
            xmin = max(0, float(bbox.find("xmin").text) - 1)
            ymin = max(0, float(bbox.find("ymin").text) - 1)
            xmax = min(width, float(bbox.find("xmax").text) - 1)
            ymax = min(height, float(bbox.find("ymax").text) - 1)

            if xmin >= xmax or ymin >= ymax:
                logger.warning("Invalid bbox in %s: %s", anno_path.name, bbox)
                continue

            boxes.append([xmin, ymin, xmax, ymax, class_idx])

        if boxes:
            data["image"].append(img_path)
            data["rois"].append(boxes)

    df = pd.DataFrame(data)
    df["label"] = df["rois"].copy()
    return df.sort_values("image").reset_index(drop=True)


def get_voc_format_classes(root: str) -> Tuple[List[str], List[int]]:
    """
    Get class names and IDs from VOC format dataset.

    Args:
        root: Root directory of VOC dataset

    Returns:
        Tuple of (class_names, category_ids)
    """
    root_path = Path(root)
    labels_file = root_path / "labels.txt"

    if labels_file.exists():
        with open(labels_file) as f:
            class_names = [line.rstrip().lower() for line in f]
        logger.info("Using class names from labels.txt: %s", class_names)
    else:
        logger.warning("labels.txt not found, scanning annotations directory: %s", root_path / "Annotations")
        class_names = dump_voc_classes(
            voc_annotation_path=str(root_path / "Annotations"), voc_class_names_output_path=str(labels_file)
        )

    # Create category IDs (1-based indexing)
    category_ids = list(range(1, len(class_names) + 1))
    return class_names, category_ids


def dump_voc_classes(voc_annotation_path: str, voc_class_names_output_path: Optional[str] = None) -> List[str]:
    """
    Extract unique class names from VOC annotations.

    Args:
        voc_annotation_path: Path to VOC annotation directory
        voc_class_names_output_path: Optional path to save class names

    Returns:
        List of unique class names
    """
    class_names = set()

    for xml_file in os.listdir(voc_annotation_path):
        if not xml_file.endswith(".xml"):
            continue

        tree = ET.parse(os.path.join(voc_annotation_path, xml_file))
        root = tree.getroot()

        for obj in root.iter("object"):
            class_names.add(obj.find("name").text.lower())

    sorted_names = sorted(list(class_names))

    if voc_class_names_output_path:
        with open(voc_class_names_output_path, "w") as f:
            f.write("\n".join(sorted_names))
        logger.info("Saved class names to %s", voc_class_names_output_path)

    return sorted_names


def object_detection_df_to_coco(data: pd.DataFrame, save_path: Optional[str] = None) -> Dict:
    """
    Convert detection DataFrame to COCO format.

    Args:
        data: DataFrame with detection data
        save_path: Optional path to save COCO JSON

    Returns:
        Dictionary in COCO format
    """
    coco_data = {"images": [], "type": "instances", "annotations": [], "categories": []}

    bbox_count = 0
    unique_classes = {}

    for idx, row in data.iterrows():
        # Process image
        img_info = _get_image_info(row["image"])
        if img_info is None:
            continue

        image_entry = {"file_name": row["image"], "height": img_info["height"], "width": img_info["width"], "id": idx}
        coco_data["images"].append(image_entry)

        # Process annotations
        for roi in row["rois"]:
            xmin, ymin, xmax, ymax, class_label = roi
            x, y, w, h = bbox_xyxy_to_xywh([xmin, ymin, xmax, ymax])

            annotation = {
                "area": w * h,
                "iscrowd": 0,
                "bbox": [x, y, w, h],
                "category_id": class_label,
                "ignore": 0,
                "segmentation": [],
                "image_id": idx,
                "id": bbox_count,
            }
            bbox_count += 1
            coco_data["annotations"].append(annotation)

            if class_label not in unique_classes:
                unique_classes[class_label] = len(unique_classes)

    # Add categories
    for class_name, class_id in unique_classes.items():
        coco_data["categories"].append({"supercategory": "none", "id": class_id, "name": class_name})

    if save_path and save_path.endswith(".json"):
        with open(save_path, "w") as f:
            json.dump(coco_data, f)
        logger.info("Saved COCO format data to %s", save_path)

    return coco_data


def get_detection_classes(sample_data_path):
    """
    Get class names and category IDs from detection dataset in various formats.

    Parameters
    ----------
    sample_data_path : Union[str, pd.DataFrame]
        The input can be one of:
        - str (directory): Path to root directory of VOC format data
        - str (file): Path to COCO format JSON annotation file
        - pd.DataFrame: DataFrame containing detection data with 'rois' column

    Returns
    -------
    tuple
        A tuple containing (class_names, category_ids) where:
        - class_names: list of class name strings
        - category_ids: dict mapping class names to their numeric IDs

        For VOC: IDs start from 1
        For COCO: Original category IDs from annotation file
        For DataFrame: Sequential IDs starting from 1
    """
    # Handle string paths for VOC and COCO formats
    if isinstance(sample_data_path, str):
        if os.path.isdir(sample_data_path):
            # Directory path indicates VOC format
            return get_voc_format_classes(sample_data_path)
        else:
            # File path indicates COCO format JSON
            return get_coco_format_classes(sample_data_path)
    # Handle DataFrame format
    elif isinstance(sample_data_path, pd.DataFrame):
        return get_df_unique_classes(sample_data_path)
