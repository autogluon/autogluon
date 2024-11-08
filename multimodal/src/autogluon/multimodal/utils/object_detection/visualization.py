"""
Visualization utilities for object detection results.
Provides functions for visualizing detection boxes, labels, and scores on images.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd

from .bbox import bbox_xyxy_to_xywh

logger = logging.getLogger(__name__)


def get_color(idx: int) -> Tuple[int, int, int]:
    """
    Generate a unique color for a given index.
    Uses a deterministic hash function to ensure consistent colors.

    Args:
        idx: Index to generate color for

    Returns:
        RGB color tuple
    """
    idx = idx * 3
    return (
        (37 * idx) % 255,  # Red
        (17 * idx) % 255,  # Green
        (29 * idx) % 255,  # Blue
    )


def add_bbox_with_alpha(
    im: np.ndarray,
    tl: Tuple[int, int],
    br: Tuple[int, int],
    line_color: Tuple[int, int, int],
    alpha: float = 0.5,
    line_thickness: int = 2,
) -> np.ndarray:
    """
    Draw a single bounding box with transparency on an image.

    Args:
        im: Input image
        tl: Top-left corner coordinates (x, y)
        br: Bottom-right corner coordinates (x, y)
        line_color: RGB color tuple for box
        alpha: Transparency value (0-1)
        line_thickness: Thickness of box lines

    Returns:
        Image with drawn bounding box
    """
    overlay = im.copy()
    cv2.rectangle(overlay, tl, br, line_color, thickness=line_thickness)
    return cv2.addWeighted(overlay, alpha, im, 1 - alpha, 0)


def add_text_with_bg_color(
    im: np.ndarray,
    text: str,
    tl: Tuple[int, int],
    bg_color: Tuple[int, int, int],
    alpha: float = 0.5,
    font: int = cv2.FONT_HERSHEY_DUPLEX,
    text_scale: float = 0.75,
    text_thickness: int = 1,
    text_vert_padding: Optional[int] = None,
) -> np.ndarray:
    """
    Add text with background color to an image.

    Args:
        im: Input image
        text: Text string to add
        tl: Top-left position for text
        bg_color: RGB color tuple for text background
        alpha: Transparency value (0-1)
        font: OpenCV font type
        text_scale: Text size scale
        text_thickness: Text thickness
        text_vert_padding: Vertical padding around text

    Returns:
        Image with added text
    """
    x1, y1 = tl

    # Calculate text size and padding
    text_size, _ = cv2.getTextSize(text, font, float(text_scale), text_thickness)
    text_w, text_h = text_size

    if text_vert_padding is None:
        text_vert_padding = int(text_h * 0.1)

    # Ensure text stays within image bounds
    y1 = max(y1 - text_h - text_vert_padding * 2, 0)

    # Create background rectangle
    overlay = im.copy()
    cv2.rectangle(overlay, (x1, y1), (x1 + text_w, y1 + text_h + text_vert_padding * 2), bg_color, -1)

    # Blend background
    im = cv2.addWeighted(overlay, alpha, im, 1 - alpha, 0)

    # Add text
    cv2.putText(
        im, text, (x1, y1 + text_h + text_vert_padding), font, text_scale, (255, 255, 255), thickness=text_thickness
    )

    return im


def plot_detections(
    image: np.ndarray,
    tlwhs: List[List[float]],
    obj_ids: List[int],
    idx2classname: Dict[int, str],
    conf_threshold: float,
    scores: Optional[List[float]] = None,
    text_scale: float = 0.75,
    text_thickness: int = 1,
    line_thickness: int = 2,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Plot detection results on an image.

    Args:
        image: Input image
        tlwhs: List of bounding boxes in [x, y, w, h] format
        obj_ids: List of class IDs for each box
        idx2classname: Mapping from class IDs to names
        conf_threshold: Confidence threshold for displaying detections
        scores: Optional confidence scores for each detection
        text_scale: Scale for text size
        text_thickness: Thickness of text
        line_thickness: Thickness of box lines
        alpha: Transparency of overlays

    Returns:
        Image with plotted detections
    """
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    # Adjust text scale based on image width
    text_scale = text_scale if im_w > 500 else text_scale * 0.8
    font = cv2.FONT_HERSHEY_DUPLEX

    # Add title with detection count and threshold
    title = f"num_det: {len(tlwhs)} conf: {conf_threshold:.2f}"
    im = add_text_with_bg_color(
        im=im,
        text=title,
        tl=(0, 0),
        bg_color=(0, 0, 0),
        alpha=alpha,
        font=font,
        text_scale=text_scale,
        text_thickness=text_thickness,
    )

    # Plot each detection
    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = int(obj_ids[i])

        # Create label text
        class_name = idx2classname[obj_ids[i]]
        label = f"{class_name},{scores[i]:.3f}" if scores is not None else class_name

        # Get unique color for class
        color = get_color(abs(obj_id))

        # Draw box
        im = add_bbox_with_alpha(
            im=im, tl=intbox[0:2], br=intbox[2:4], line_color=color, alpha=alpha, line_thickness=line_thickness
        )

        # Add label
        im = add_text_with_bg_color(
            im=im,
            text=label,
            tl=(intbox[0], intbox[1]),
            bg_color=color,
            alpha=0.75,
            font=font,
            text_scale=text_scale,
            text_thickness=text_thickness,
        )

    return im


def visualize_detection(
    predictions: pd.DataFrame, detection_classes: List[str], conf_threshold: float, visualization_result_dir: str
) -> List[np.ndarray]:
    """
    Visualize detection results for multiple images and save to directory.

    Args:
        predictions: DataFrame containing detection results
        detection_classes: List of class names
        conf_threshold: Confidence threshold for displaying detections
        visualization_result_dir: Directory to save visualization results

    Returns:
        List of visualized images as numpy arrays

    Raises:
        ImportError: If OpenCV is not installed
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("OpenCV is required for visualization. " "Install it with: pip install opencv-python")

    # Create output directory if needed
    os.makedirs(visualization_result_dir, exist_ok=True)

    # Create class name mappings
    classname2idx = {name: i for i, name in enumerate(detection_classes)}
    idx2classname = {i: name for i, name in enumerate(detection_classes)}

    visualized_images = []
    for _, row in predictions.iterrows():
        # Load image
        image_path = row["image"]
        image = cv2.imread(image_path)
        if image is None:
            logger.warning(f"Could not load image: {image_path}")
            continue

        # Process detections
        boxes = []
        ids = []
        scores = []

        for det in row["bboxes"]:
            if det["score"] > conf_threshold:
                boxes.append(bbox_xyxy_to_xywh(det["bbox"]))
                ids.append(classname2idx[det["class"]])
                scores.append(det["score"])

        if not boxes:
            logger.debug(f"No detections above threshold for: {image_path}")
            continue

        # Visualize detections
        vis_image = plot_detections(
            image=image,
            tlwhs=boxes,
            obj_ids=ids,
            idx2classname=idx2classname,
            conf_threshold=conf_threshold,
            scores=scores,
        )

        visualized_images.append(vis_image)

        # Save result
        output_path = os.path.join(visualization_result_dir, os.path.basename(image_path))
        cv2.imwrite(output_path, vis_image)

    logger.info("Saved visualizations to %s", visualization_result_dir)
    return visualized_images
