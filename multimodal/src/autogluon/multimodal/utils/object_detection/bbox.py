"""
Bounding box utilities for object detection.
Provides functions for converting between different bounding box formats
and performing operations like clipping and transformations.
"""

from typing import List, Optional, Tuple, Union

import numpy as np
import torch

BBoxType = Union[List[float], Tuple[float, ...], np.ndarray]


def bbox_xywh_to_xyxy(xywh: BBoxType) -> Union[Tuple[float, ...], np.ndarray]:
    """
    Convert bounding boxes from format (x, y, width, height) to (xmin, ymin, xmax, ymax).

    Args:
        xywh: Bounding boxes in (x, y, w, h) format.
             Can be list/tuple of 4 numbers or numpy array of shape (N, 4)

    Returns:
        Bounding boxes in (xmin, ymin, xmax, ymax) format,
        same type as input (tuple or numpy array)

    Raises:
        IndexError: If input doesn't have exactly 4 elements per box
        TypeError: If input type is not supported
    """
    if isinstance(xywh, (tuple, list)):
        if len(xywh) != 4:
            raise IndexError(f"Bounding box must have 4 elements, got {len(xywh)}")

        # Convert to xmin, ymin, xmax, ymax
        x, y = xywh[0], xywh[1]
        w = max(xywh[2] - 1, 0)  # Ensure non-negative width
        h = max(xywh[3] - 1, 0)  # Ensure non-negative height
        return (x, y, x + w, y + h)

    if isinstance(xywh, np.ndarray):
        if xywh.size % 4 != 0:
            raise IndexError(f"Bounding boxes must have n * 4 elements, got shape {xywh.shape}")
        return np.hstack((xywh[:, :2], xywh[:, :2] + np.maximum(0, xywh[:, 2:4] - 1)))

    raise TypeError(f"Expected list, tuple or numpy.ndarray, got {type(xywh)}")


def bbox_xyxy_to_xywh(xyxy: BBoxType) -> Union[Tuple[float, ...], np.ndarray]:
    """
    Convert bounding boxes from format (xmin, ymin, xmax, ymax) to (x, y, width, height).

    Args:
        xyxy: Bounding boxes in (xmin, ymin, xmax, ymax) format.
             Can be list/tuple of 4 numbers or numpy array of shape (N, 4)

    Returns:
        Bounding boxes in (x, y, w, h) format,
        same type as input (tuple or numpy array)

    Raises:
        IndexError: If input doesn't have exactly 4 elements per box
        TypeError: If input type is not supported
    """
    if isinstance(xyxy, (tuple, list)):
        if len(xyxy) != 4:
            raise IndexError(f"Bounding box must have 4 elements, got {len(xyxy)}")

        # Convert to x, y, width, height
        x1, y1 = xyxy[0], xyxy[1]
        w = xyxy[2] - x1
        h = xyxy[3] - y1
        return (x1, y1, w, h)

    if isinstance(xyxy, np.ndarray):
        if xyxy.size % 4 != 0:
            raise IndexError(f"Bounding boxes must have n * 4 elements, got shape {xyxy.shape}")

        # Convert array of boxes
        return np.hstack((xyxy[:, :2], xyxy[:, 2:4] - xyxy[:, :2]))

    raise TypeError(f"Expected list, tuple or numpy.ndarray, got {type(xyxy)}")


def bbox_clip_xyxy(
    xyxy: BBoxType, width: Union[int, float], height: Union[int, float]
) -> Union[Tuple[float, ...], np.ndarray]:
    """
    Clip bounding boxes to stay within image boundaries.

    Args:
        xyxy: Bounding boxes in (xmin, ymin, xmax, ymax) format
        width: Image width
        height: Image height

    Returns:
        Clipped bounding boxes in same format as input

    Raises:
        IndexError: If input doesn't have exactly 4 elements per box
        TypeError: If input type is not supported
    """
    if isinstance(xyxy, (tuple, list)):
        if len(xyxy) != 4:
            raise IndexError(f"Bounding box must have 4 elements, got {len(xyxy)}")

        # Clip coordinates to image boundaries
        x1 = np.clip(xyxy[0], 0, width - 1)
        y1 = np.clip(xyxy[1], 0, height - 1)
        x2 = np.clip(xyxy[2], 0, width - 1)
        y2 = np.clip(xyxy[3], 0, height - 1)
        return (x1, y1, x2, y2)

    if isinstance(xyxy, np.ndarray):
        if xyxy.size % 4 != 0:
            raise IndexError(f"Bounding boxes must have n * 4 elements, got shape {xyxy.shape}")

        # Clip array of boxes
        x1 = np.clip(xyxy[:, 0], 0, width - 1)
        y1 = np.clip(xyxy[:, 1], 0, height - 1)
        x2 = np.clip(xyxy[:, 2], 0, width - 1)
        y2 = np.clip(xyxy[:, 3], 0, height - 1)
        return np.stack([x1, y1, x2, y2], axis=1)

    raise TypeError(f"Expected list, tuple or numpy.ndarray, got {type(xyxy)}")


def bbox_ratio_xywh_to_index_xyxy(
    xywh: BBoxType, image_wh: Union[List[float], Tuple[float, float], np.ndarray]
) -> Union[Tuple[float, ...], np.ndarray]:
    """
    Convert bounding boxes from normalized ratios to absolute pixel coordinates.

    Converts from format (x_center_ratio, y_center_ratio, w_ratio, h_ratio)
    to (xmin, ymin, xmax, ymax) in pixel coordinates.

    Args:
        xywh: Bounding boxes in normalized ratio format
        image_wh: Image (width, height) dimensions

    Returns:
        Bounding boxes in absolute pixel coordinates

    Raises:
        IndexError: If inputs don't have correct number of elements
        TypeError: If input types are not supported
        AssertionError: If input types don't match
    """
    if isinstance(xywh, (tuple, list)):
        if not isinstance(image_wh, (tuple, list)):
            raise TypeError(f"image_wh type ({type(image_wh)}) should match xywh type ({type(xywh)})")

        if len(xywh) != 4:
            raise IndexError(f"Bounding box must have 4 elements, got {len(xywh)}")
        if len(image_wh) != 2:
            raise IndexError(f"Image dimensions must have 2 elements, got {len(image_wh)}")

        # Convert ratios to absolute coordinates
        x_center, y_center = xywh[0], xywh[1]
        w, h = xywh[2], xywh[3]
        img_w, img_h = image_wh

        # Convert to absolute pixels
        x_center *= img_w
        y_center *= img_h
        w *= img_w
        h *= img_h

        # Convert center to top-left corner
        x1 = x_center - w / 2
        y1 = y_center - h / 2

        return (x1, y1, x1 + w, y1 + h)

    if isinstance(xywh, np.ndarray):
        if not xywh.size % 4 == 0:
            raise IndexError(f"Bounding boxes must have n * 4 elements, got shape {xywh.shape}")

        if isinstance(image_wh, np.ndarray):
            if not image_wh.size % 2 == 0:
                raise IndexError(f"Image dimensions must have n * 2 elements, got shape {image_wh.shape}")
            # Handle batched image dimensions
            scale_factors = np.concatenate([image_wh, image_wh], axis=1)

        elif isinstance(image_wh, (tuple, list)):
            # Handle single image dimensions
            img_w, img_h = image_wh
            scale_factors = np.array([[img_w, img_h, img_w, img_h]])

        # Scale to absolute coordinates
        abs_coords = xywh * scale_factors

        # Convert centers to corners
        abs_coords[:, :2] -= abs_coords[:, 2:] / 2

        # Convert to xyxy format
        return np.hstack((abs_coords[:, :2], abs_coords[:, :2] + np.maximum(0, abs_coords[:, 2:] - 1)))

    raise TypeError(f"Expected list, tuple or numpy.ndarray, got {type(xywh)}")


def convert_pred_to_xywh(predictions: Optional[List[dict]]) -> Optional[List[dict]]:
    """
    Convert prediction bounding boxes from XYXY to XYWH format.

    Args:
        predictions: List of predictions, where each prediction contains 'bboxes'
                   that can be either a torch.Tensor or numpy.ndarray

    Returns:
        Modified list of predictions with bboxes in XYWH format
    """
    if not predictions:
        return predictions

    for pred in predictions:
        bboxes = pred["bboxes"]

        if isinstance(bboxes, np.ndarray):
            pred["bboxes"] = bbox_xyxy_to_xywh(bboxes)
        elif torch.is_tensor(bboxes):
            pred["bboxes"] = bbox_xyxy_to_xywh(bboxes.detach().numpy())
        else:
            raise TypeError(f"Unsupported bbox type: {type(bboxes)}. " "Expected numpy.ndarray or torch.Tensor")

    return predictions
