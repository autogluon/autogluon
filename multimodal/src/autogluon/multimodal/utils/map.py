# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import IntTensor, Tensor
from torchmetrics.metric import Metric
from torchmetrics.utilities.imports import _TORCHVISION_GREATER_EQUAL_0_8

if _TORCHVISION_GREATER_EQUAL_0_8:
    from torchvision.ops import box_area, box_convert, box_iou
else:
    box_convert = box_iou = box_area = None
    __doctest_skip__ = ["MeanAveragePrecision"]

log = logging.getLogger(__name__)


class BaseMetricResults(dict):
    """Base metric class, that allows fields for pre-defined metrics."""

    def __getattr__(self, key: str) -> Tensor:
        # Using this you get the correct error message, an AttributeError instead of a KeyError
        if key in self:
            return self[key]
        raise AttributeError(f"No such attribute: {key}")

    def __setattr__(self, key: str, value: Tensor) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        if key in self:
            del self[key]
        raise AttributeError(f"No such attribute: {key}")


class MAPMetricResults(BaseMetricResults):
    """Class to wrap the final mAP results."""

    __slots__ = ("map", "map_50", "map_75", "map_small", "map_medium", "map_large")


class MARMetricResults(BaseMetricResults):
    """Class to wrap the final mAR results."""

    __slots__ = ("mar_1", "mar_10", "mar_100", "mar_small", "mar_medium", "mar_large")


class COCOMetricResults(BaseMetricResults):
    """Class to wrap the final COCO metric results including various mAP/mAR values."""

    __slots__ = (
        "map",
        "map_50",
        "map_75",
        "map_small",
        "map_medium",
        "map_large",
        "mar_1",
        "mar_10",
        "mar_100",
        "mar_small",
        "mar_medium",
        "mar_large",
        "map_per_class",
        "mar_100_per_class",
    )


def _input_validator(preds: Sequence[Dict[str, Tensor]], targets: Sequence[Dict[str, Tensor]]) -> None:
    """Ensure the correct input format of `preds` and `targets`"""
    if not isinstance(preds, Sequence):
        raise ValueError("Expected argument `preds` to be of type Sequence")
    if not isinstance(targets, Sequence):
        raise ValueError("Expected argument `target` to be of type Sequence")
    if len(preds) != len(targets):
        raise ValueError("Expected argument `preds` and `target` to have the same length")

    for k in ["boxes", "scores", "labels"]:
        if any(k not in p for p in preds):
            raise ValueError(f"Expected all dicts in `preds` to contain the `{k}` key")

    for k in ["boxes", "labels"]:
        if any(k not in p for p in targets):
            raise ValueError(f"Expected all dicts in `target` to contain the `{k}` key")

    if any(type(pred["boxes"]) is not Tensor for pred in preds):
        raise ValueError("Expected all boxes in `preds` to be of type Tensor")
    if any(type(pred["scores"]) is not Tensor for pred in preds):
        raise ValueError("Expected all scores in `preds` to be of type Tensor")
    if any(type(pred["labels"]) is not Tensor for pred in preds):
        raise ValueError("Expected all labels in `preds` to be of type Tensor")
    if any(type(target["boxes"]) is not Tensor for target in targets):
        raise ValueError("Expected all boxes in `target` to be of type Tensor")
    if any(type(target["labels"]) is not Tensor for target in targets):
        raise ValueError("Expected all labels in `target` to be of type Tensor")

    for i, item in enumerate(targets):
        if item["boxes"].size(0) != item["labels"].size(0):
            raise ValueError(
                f"Input boxes and labels of sample {i} in targets have a"
                f" different length (expected {item['boxes'].size(0)} labels, got {item['labels'].size(0)})"
            )
    for i, item in enumerate(preds):
        if not (item["boxes"].size(0) == item["labels"].size(0) == item["scores"].size(0)):
            raise ValueError(
                f"Input boxes, labels and scores of sample {i} in predictions have a"
                f" different length (expected {item['boxes'].size(0)} labels and scores,"
                f" got {item['labels'].size(0)} labels and {item['scores'].size(0)})"
            )


def _fix_empty_tensors(boxes: Tensor) -> Tensor:
    """Empty tensors can cause problems in DDP mode, this methods corrects them."""
    if boxes.numel() == 0 and boxes.ndim == 1:
        return boxes.unsqueeze(0)
    return boxes


class MeanAveragePrecision(Metric):
    r"""
    Computes the `Mean-Average-Precision (mAP) and Mean-Average-Recall (mAR)
    <https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173>`_
    for object detection predictions.
    Optionally, the mAP and mAR values can be calculated per class.

    Predicted boxes and targets have to be in Pascal VOC format
    (xmin-top left, ymin-top left, xmax-bottom right, ymax-bottom right).
    See the :meth:`update` method for more information about the input format to this metric.

    For an example on how to use this metric check the `torchmetrics examples
    <https://github.com/PyTorchLightning/metrics/blob/master/tm_examples/detection_map.py>`_

    .. note::
        This metric is following the mAP implementation of
        `pycocotools <https://github.com/cocodataset/cocoapi/tree/master/PythonAPI/pycocotools>`_,
        a standard implementation for the mAP metric for object detection.

    .. note::
        This metric requires you to have `torchvision` version 0.8.0 or newer installed (with corresponding
        version 1.7.0 of torch or newer). Please install with ``pip install torchvision`` or
        ``pip install torchmetrics[detection]``.

    Args:
        box_format:
            Input format of given boxes. Supported formats are ``[`xyxy`, `xywh`, `cxcywh`]``.
        iou_thresholds:
            IoU thresholds for evaluation. If set to ``None`` it corresponds to the stepped range ``[0.5,...,0.95]``
            with step ``0.05``. Else provide a list of floats.
        rec_thresholds:
            Recall thresholds for evaluation. If set to ``None`` it corresponds to the stepped range ``[0,...,1]``
            with step ``0.01``. Else provide a list of floats.
        max_detection_thresholds:
            Thresholds on max detections per image. If set to `None` will use thresholds ``[1, 10, 100]``.
            Else, please provide a list of ints.
        class_metrics:
            Option to enable per-class metrics for mAP and mAR_100. Has a performance impact.
        compute_on_step:
            Forward only calls ``update()`` and returns None if this is set to False.

            .. deprecated:: v0.8
                Argument has no use anymore and will be removed v0.9.

        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        ModuleNotFoundError:
            If ``torchvision`` is not installed or version installed is lower than 0.8.0
        ValueError:
            If ``class_metrics`` is not a boolean
    """

    detection_boxes: List[Tensor]
    detection_scores: List[Tensor]
    detection_labels: List[Tensor]
    groundtruth_boxes: List[Tensor]
    groundtruth_labels: List[Tensor]

    def __init__(
        self,
        box_format: str = "xyxy",
        iou_thresholds: Optional[List[float]] = None,
        rec_thresholds: Optional[List[float]] = None,
        max_detection_thresholds: Optional[List[int]] = None,
        class_metrics: bool = False,
        compute_on_step: Optional[bool] = None,
        **kwargs: Dict[str, Any],
    ) -> None:  # type: ignore
        super().__init__(compute_on_step=compute_on_step, **kwargs)

        if not _TORCHVISION_GREATER_EQUAL_0_8:
            raise ModuleNotFoundError(
                "`MeanAveragePrecision` metric requires that `torchvision` version 0.8.0 or newer is installed."
                " Please install with `pip install torchvision>=0.8` or `pip install torchmetrics[detection]`."
            )

        allowed_box_formats = ("xyxy", "xywh", "cxcywh")
        if box_format not in allowed_box_formats:
            raise ValueError(f"Expected argument `box_format` to be one of {allowed_box_formats} but got {box_format}")
        self.box_format = box_format
        self.iou_thresholds = iou_thresholds or torch.linspace(0.5, 0.95, round((0.95 - 0.5) / 0.05) + 1).tolist()
        self.rec_thresholds = rec_thresholds or torch.linspace(0.0, 1.00, round(1.00 / 0.01) + 1).tolist()
        max_det_thr, _ = torch.sort(IntTensor(max_detection_thresholds or [1, 10, 100]))
        self.max_detection_thresholds = max_det_thr.tolist()
        self.bbox_area_ranges = {
            "all": (0**2, int(1e5**2)),
            "small": (0**2, 32**2),
            "medium": (32**2, 96**2),
            "large": (96**2, int(1e5**2)),
        }

        if not isinstance(class_metrics, bool):
            raise ValueError("Expected argument `class_metrics` to be a boolean")

        self.class_metrics = class_metrics
        self.add_state("detection_boxes", default=[], dist_reduce_fx=None)
        self.add_state("detection_scores", default=[], dist_reduce_fx=None)
        self.add_state("detection_labels", default=[], dist_reduce_fx=None)
        self.add_state("groundtruth_boxes", default=[], dist_reduce_fx=None)
        self.add_state("groundtruth_labels", default=[], dist_reduce_fx=None)

    def update(self, preds: List[Dict[str, Tensor]], target: List[Dict[str, Tensor]]) -> None:  # type: ignore
        """Add detections and ground truth to the metric.

        Args:
            preds: A list consisting of dictionaries each containing the key-values
                (each dictionary corresponds to a single image):

                - ``boxes``: ``torch.FloatTensor`` of shape ``[num_boxes, 4]`` containing ``num_boxes`` detection boxes
                  of the format specified in the constructor. By default, this method expects
                  ``[xmin, ymin, xmax, ymax]`` in absolute image coordinates.
                - ``scores``: ``torch.FloatTensor`` of shape ``[num_boxes]`` containing detection scores for the boxes.
                - ``labels``: ``torch.IntTensor`` of shape ``[num_boxes]`` containing 0-indexed detection classes
                  for the boxes.

            target: A list consisting of dictionaries each containing the key-values
                (each dictionary corresponds to a single image):

                - ``boxes``: ``torch.FloatTensor`` of shape ``[num_boxes, 4]`` containing ``num_boxes``
                  ground truth boxes of the format specified in the constructor. By default, this method expects
                  ``[xmin, ymin, xmax, ymax]`` in absolute image coordinates.
                - ``labels``: ``torch.IntTensor`` of shape ``[num_boxes]`` containing 1-indexed ground truth
                   classes for the boxes.

        Raises:
            ValueError:
                If ``preds`` is not of type ``List[Dict[str, Tensor]]``
            ValueError:
                If ``target`` is not of type ``List[Dict[str, Tensor]]``
            ValueError:
                If ``preds`` and ``target`` are not of the same length
            ValueError:
                If any of ``preds.boxes``, ``preds.scores`` and ``preds.labels`` are not of the same length
            ValueError:
                If any of ``target.boxes`` and ``target.labels`` are not of the same length
            ValueError:
                If any box is not type float and of length 4
            ValueError:
                If any class is not type int and of length 1
            ValueError:
                If any score is not type float and of length 1
        """
        _input_validator(preds, target)

        for item in preds:
            boxes = _fix_empty_tensors(item["boxes"])
            boxes = box_convert(boxes, in_fmt=self.box_format, out_fmt="xyxy")
            self.detection_boxes.append(boxes)
            self.detection_labels.append(item["labels"])
            self.detection_scores.append(item["scores"])

        for item in target:
            boxes = _fix_empty_tensors(item["boxes"])
            boxes = box_convert(boxes, in_fmt=self.box_format, out_fmt="xyxy")
            self.groundtruth_boxes.append(boxes)
            self.groundtruth_labels.append(item["labels"])

    def _get_classes(self) -> List:
        """Returns a list of unique classes found in ground truth and detection data."""
        if len(self.detection_labels) > 0 or len(self.groundtruth_labels) > 0:
            return torch.cat(self.detection_labels + self.groundtruth_labels).unique().tolist()
        return []

    def _compute_iou(self, idx: int, class_id: int, max_det: int) -> Tensor:
        """Computes the Intersection over Union (IoU) for ground truth and detection bounding boxes for the given
        image and class.

        Args:
            idx:
                Image Id, equivalent to the index of supplied samples
            class_id:
                Class Id of the supplied ground truth and detection labels
            max_det:
                Maximum number of evaluated detection bounding boxes
        """
        gt = self.groundtruth_boxes[idx]
        det = self.detection_boxes[idx]
        gt_label_mask = self.groundtruth_labels[idx] == class_id
        det_label_mask = self.detection_labels[idx] == class_id
        if len(gt_label_mask) == 0 or len(det_label_mask) == 0:
            return Tensor([])
        gt = gt[gt_label_mask]
        det = det[det_label_mask]
        if len(gt) == 0 or len(det) == 0:
            return Tensor([])

        # Sort by scores and use only max detections
        scores = self.detection_scores[idx]
        scores_filtered = scores[self.detection_labels[idx] == class_id]
        inds = torch.argsort(scores_filtered, descending=True)
        det = det[inds]
        if len(det) > max_det:
            det = det[:max_det]

        # generalized_box_iou
        ious = box_iou(det, gt)
        return ious

    def __evaluate_image_gt_no_preds(
        self, gt: Tensor, gt_label_mask: Tensor, area_range: Tuple[int, int], nb_iou_thrs: int
    ) -> Dict[str, Any]:
        """Some GT but no predictions."""
        # GTs
        gt = gt[gt_label_mask]
        nb_gt = len(gt)
        areas = box_area(gt)
        ignore_area = (areas < area_range[0]) | (areas > area_range[1])
        gt_ignore, _ = torch.sort(ignore_area.to(torch.uint8))
        gt_ignore = gt_ignore.to(torch.bool)

        # Detections
        nb_det = 0
        det_ignore = torch.zeros((nb_iou_thrs, nb_det), dtype=torch.bool, device=self.device)

        return {
            "dtMatches": torch.zeros((nb_iou_thrs, nb_det), dtype=torch.bool, device=self.device),
            "gtMatches": torch.zeros((nb_iou_thrs, nb_gt), dtype=torch.bool, device=self.device),
            "dtScores": torch.zeros(nb_det, dtype=torch.float32, device=self.device),
            "gtIgnore": gt_ignore,
            "dtIgnore": det_ignore,
        }

    def __evaluate_image_preds_no_gt(
        self,
        det: Tensor,
        idx: int,
        det_label_mask: Tensor,
        max_det: int,
        area_range: Tuple[int, int],
        nb_iou_thrs: int,
    ) -> Dict[str, Any]:
        """Some predictions but no GT."""
        # GTs
        nb_gt = 0
        gt_ignore = torch.zeros(nb_gt, dtype=torch.bool, device=self.device)

        # Detections
        det = det[det_label_mask]
        scores = self.detection_scores[idx]
        scores_filtered = scores[det_label_mask]
        scores_sorted, dtind = torch.sort(scores_filtered, descending=True)
        det = det[dtind]
        if len(det) > max_det:
            det = det[:max_det]
        nb_det = len(det)
        det_areas = box_area(det).to(self.device)
        det_ignore_area = (det_areas < area_range[0]) | (det_areas > area_range[1])
        ar = det_ignore_area.reshape((1, nb_det))
        det_ignore = torch.repeat_interleave(ar, nb_iou_thrs, 0)

        return {
            "dtMatches": torch.zeros((nb_iou_thrs, nb_det), dtype=torch.bool, device=self.device),
            "gtMatches": torch.zeros((nb_iou_thrs, nb_gt), dtype=torch.bool, device=self.device),
            "dtScores": scores_sorted,
            "gtIgnore": gt_ignore,
            "dtIgnore": det_ignore,
        }

    def _evaluate_image(
        self, idx: int, class_id: int, area_range: Tuple[int, int], max_det: int, ious: dict
    ) -> Optional[dict]:
        """Perform evaluation for single class and image.

        Args:
            idx:
                Image Id, equivalent to the index of supplied samples.
            class_id:
                Class Id of the supplied ground truth and detection labels.
            area_range:
                List of lower and upper bounding box area threshold.
            max_det:
                Maximum number of evaluated detection bounding boxes.
            ious:
                IoU results for image and class.
        """
        gt = self.groundtruth_boxes[idx]
        det = self.detection_boxes[idx]
        gt_label_mask = self.groundtruth_labels[idx] == class_id
        det_label_mask = self.detection_labels[idx] == class_id

        # No Gt and No predictions --> ignore image
        if len(gt_label_mask) == 0 and len(det_label_mask) == 0:
            return None

        nb_iou_thrs = len(self.iou_thresholds)

        # Some GT but no predictions
        if len(gt_label_mask) > 0 and len(det_label_mask) == 0:
            return self.__evaluate_image_gt_no_preds(gt, gt_label_mask, area_range, nb_iou_thrs)

        # Some predictions but no GT
        if len(gt_label_mask) == 0 and len(det_label_mask) >= 0:
            return self.__evaluate_image_preds_no_gt(det, idx, det_label_mask, max_det, area_range, nb_iou_thrs)

        gt = gt[gt_label_mask]
        det = det[det_label_mask]
        if gt.numel() == 0 and det.numel() == 0:
            return None

        areas = box_area(gt)
        ignore_area = (areas < area_range[0]) | (areas > area_range[1])

        # sort dt highest score first, sort gt ignore last
        ignore_area_sorted, gtind = torch.sort(ignore_area.to(torch.uint8))
        # Convert to uint8 temporarily and back to bool, because "Sort currently does not support bool dtype on CUDA"
        ignore_area_sorted = ignore_area_sorted.to(torch.bool)
        gt = gt[gtind]
        scores = self.detection_scores[idx]
        scores_filtered = scores[det_label_mask]
        scores_sorted, dtind = torch.sort(scores_filtered, descending=True)
        det = det[dtind]
        if len(det) > max_det:
            det = det[:max_det]
        # load computed ious
        ious = ious[idx, class_id][:, gtind] if len(ious[idx, class_id]) > 0 else ious[idx, class_id]

        nb_iou_thrs = len(self.iou_thresholds)
        nb_gt = len(gt)
        nb_det = len(det)
        gt_matches = torch.zeros((nb_iou_thrs, nb_gt), dtype=torch.bool, device=gt.device)
        det_matches = torch.zeros((nb_iou_thrs, nb_det), dtype=torch.bool, device=gt.device)
        gt_ignore = ignore_area_sorted
        det_ignore = torch.zeros((nb_iou_thrs, nb_det), dtype=torch.bool, device=gt.device)

        if torch.numel(ious) > 0:
            for idx_iou, t in enumerate(self.iou_thresholds):
                for idx_det, _ in enumerate(det):
                    m = MeanAveragePrecision._find_best_gt_match(t, gt_matches, idx_iou, gt_ignore, ious, idx_det)
                    if m == -1:
                        continue
                    det_ignore[idx_iou, idx_det] = gt_ignore[m]
                    det_matches[idx_iou, idx_det] = 1
                    gt_matches[idx_iou, m] = 1

        # set unmatched detections outside of area range to ignore
        det_areas = box_area(det)
        det_ignore_area = (det_areas < area_range[0]) | (det_areas > area_range[1])
        ar = det_ignore_area.reshape((1, nb_det))
        det_ignore = torch.logical_or(
            det_ignore, torch.logical_and(det_matches == 0, torch.repeat_interleave(ar, nb_iou_thrs, 0))
        )
        return {
            "dtMatches": det_matches.to(self.device),
            "gtMatches": gt_matches.to(self.device),
            "dtScores": scores_sorted.to(self.device),
            "gtIgnore": gt_ignore.to(self.device),
            "dtIgnore": det_ignore.to(self.device),
        }

    @staticmethod
    def _find_best_gt_match(
        thr: int, gt_matches: Tensor, idx_iou: float, gt_ignore: Tensor, ious: Tensor, idx_det: int
    ) -> int:
        """Return id of best ground truth match with current detection.

        Args:
            thr:
                Current threshold value.
            gt_matches:
                Tensor showing if a ground truth matches for threshold ``t`` exists.
            idx_iou:
                Id of threshold ``t``.
            gt_ignore:
                Tensor showing if ground truth should be ignored.
            ious:
                IoUs for all combinations of detection and ground truth.
            idx_det:
                Id of current detection.
        """
        previously_matched = gt_matches[idx_iou]
        # Remove previously matched or ignored gts
        remove_mask = previously_matched | gt_ignore
        gt_ious = ious[idx_det] * ~remove_mask
        match_idx = gt_ious.argmax().item()
        if gt_ious[match_idx] > thr:
            return match_idx
        return -1

    def _summarize(
        self,
        results: Dict,
        avg_prec: bool = True,
        iou_threshold: Optional[float] = None,
        area_range: str = "all",
        max_dets: int = 100,
    ) -> Tensor:
        """Perform evaluation for single class and image.

        Args:
            results:
                Dictionary including precision, recall and scores for all combinations.
            avg_prec:
                Calculate average precision. Else calculate average recall.
            iou_threshold:
                IoU threshold. If set to ``None`` it all values are used. Else results are filtered.
            area_range:
                Bounding box area range key.
            max_dets:
                Maximum detections.
        """
        area_inds = [i for i, k in enumerate(self.bbox_area_ranges.keys()) if k == area_range]
        mdet_inds = [i for i, k in enumerate(self.max_detection_thresholds) if k == max_dets]
        if avg_prec:
            # dimension of precision: [TxRxKxAxM]
            prec = results["precision"]
            # IoU
            if iou_threshold is not None:
                thr = self.iou_thresholds.index(iou_threshold)
                prec = prec[thr, :, :, area_inds, mdet_inds]
            else:
                prec = prec[:, :, :, area_inds, mdet_inds]
        else:
            # dimension of recall: [TxKxAxM]
            prec = results["recall"]
            if iou_threshold is not None:
                thr = self.iou_thresholds.index(iou_threshold)
                prec = prec[thr, :, :, area_inds, mdet_inds]
            else:
                prec = prec[:, :, area_inds, mdet_inds]

        mean_prec = torch.tensor([-1.0]) if len(prec[prec > -1]) == 0 else torch.mean(prec[prec > -1])
        return mean_prec

    def _calculate(self, class_ids: List) -> Tuple[MAPMetricResults, MARMetricResults]:
        """Calculate the precision and recall for all supplied classes to calculate mAP/mAR.

        Args:
            class_ids:
                List of label class Ids.
        """
        img_ids = range(len(self.groundtruth_boxes))
        max_detections = self.max_detection_thresholds[-1]
        area_ranges = self.bbox_area_ranges.values()

        ious = {
            (idx, class_id): self._compute_iou(idx, class_id, max_detections)
            for idx in img_ids
            for class_id in class_ids
        }

        eval_imgs = [
            self._evaluate_image(img_id, class_id, area, max_detections, ious)
            for class_id in class_ids
            for area in area_ranges
            for img_id in img_ids
        ]

        nb_iou_thrs = len(self.iou_thresholds)
        nb_rec_thrs = len(self.rec_thresholds)
        nb_classes = len(class_ids)
        nb_bbox_areas = len(self.bbox_area_ranges)
        nb_max_det_thrs = len(self.max_detection_thresholds)
        nb_imgs = len(img_ids)
        precision = -torch.ones((nb_iou_thrs, nb_rec_thrs, nb_classes, nb_bbox_areas, nb_max_det_thrs))
        recall = -torch.ones((nb_iou_thrs, nb_classes, nb_bbox_areas, nb_max_det_thrs))
        scores = -torch.ones((nb_iou_thrs, nb_rec_thrs, nb_classes, nb_bbox_areas, nb_max_det_thrs))

        # move tensors if necessary
        rec_thresholds_tensor = torch.tensor(self.rec_thresholds)

        # retrieve E at each category, area range, and max number of detections
        for idx_cls, _ in enumerate(class_ids):
            for idx_bbox_area, _ in enumerate(self.bbox_area_ranges):
                for idx_max_det_thrs, max_det in enumerate(self.max_detection_thresholds):
                    recall, precision, scores = MeanAveragePrecision.__calculate_recall_precision_scores(
                        recall,
                        precision,
                        scores,
                        idx_cls=idx_cls,
                        idx_bbox_area=idx_bbox_area,
                        idx_max_det_thrs=idx_max_det_thrs,
                        eval_imgs=eval_imgs,
                        rec_thresholds=rec_thresholds_tensor,
                        max_det=max_det,
                        nb_imgs=nb_imgs,
                        nb_bbox_areas=nb_bbox_areas,
                    )

        return precision, recall

    def _summarize_results(self, precisions: Tensor, recalls: Tensor) -> Tuple[MAPMetricResults, MARMetricResults]:
        """Summarizes the precision and recall values to calculate mAP/mAR.

        Args:
            precisions:
                Precision values for different thresholds
            recalls:
                Recall values for different thresholds
        """
        results = dict(precision=precisions, recall=recalls)
        map_metrics = MAPMetricResults()
        map_metrics.map = self._summarize(results, True)
        last_max_det_thr = self.max_detection_thresholds[-1]
        map_metrics.map_50 = self._summarize(results, True, iou_threshold=0.5, max_dets=last_max_det_thr)
        map_metrics.map_75 = self._summarize(results, True, iou_threshold=0.75, max_dets=last_max_det_thr)
        map_metrics.map_small = self._summarize(results, True, area_range="small", max_dets=last_max_det_thr)
        map_metrics.map_medium = self._summarize(results, True, area_range="medium", max_dets=last_max_det_thr)
        map_metrics.map_large = self._summarize(results, True, area_range="large", max_dets=last_max_det_thr)

        mar_metrics = MARMetricResults()
        for max_det in self.max_detection_thresholds:
            mar_metrics[f"mar_{max_det}"] = self._summarize(results, False, max_dets=max_det)
        mar_metrics.mar_small = self._summarize(results, False, area_range="small", max_dets=last_max_det_thr)
        mar_metrics.mar_medium = self._summarize(results, False, area_range="medium", max_dets=last_max_det_thr)
        mar_metrics.mar_large = self._summarize(results, False, area_range="large", max_dets=last_max_det_thr)

        return map_metrics, mar_metrics

    @staticmethod
    def __calculate_recall_precision_scores(
        recall: Tensor,
        precision: Tensor,
        scores: Tensor,
        idx_cls: int,
        idx_bbox_area: int,
        idx_max_det_thrs: int,
        eval_imgs: list,
        rec_thresholds: Tensor,
        max_det: int,
        nb_imgs: int,
        nb_bbox_areas: int,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        nb_rec_thrs = len(rec_thresholds)
        idx_cls_pointer = idx_cls * nb_bbox_areas * nb_imgs
        idx_bbox_area_pointer = idx_bbox_area * nb_imgs
        # Load all image evals for current class_id and area_range
        img_eval_cls_bbox = [eval_imgs[idx_cls_pointer + idx_bbox_area_pointer + i] for i in range(nb_imgs)]
        img_eval_cls_bbox = [e for e in img_eval_cls_bbox if e is not None]
        if not img_eval_cls_bbox:
            return recall, precision, scores
        det_scores = torch.cat([e["dtScores"][:max_det] for e in img_eval_cls_bbox])

        # different sorting method generates slightly different results.
        # mergesort is used to be consistent as Matlab implementation.
        inds = torch.argsort(det_scores, descending=True)
        det_scores_sorted = det_scores[inds]

        det_matches = torch.cat([e["dtMatches"][:, :max_det] for e in img_eval_cls_bbox], axis=1)[:, inds]
        det_ignore = torch.cat([e["dtIgnore"][:, :max_det] for e in img_eval_cls_bbox], axis=1)[:, inds]
        gt_ignore = torch.cat([e["gtIgnore"] for e in img_eval_cls_bbox])
        npig = torch.count_nonzero(gt_ignore == False)  # noqa: E712
        if npig == 0:
            return recall, precision, scores
        tps = torch.logical_and(det_matches, torch.logical_not(det_ignore))
        fps = torch.logical_and(torch.logical_not(det_matches), torch.logical_not(det_ignore))

        tp_sum = torch.cumsum(tps, axis=1, dtype=torch.float)
        fp_sum = torch.cumsum(fps, axis=1, dtype=torch.float)
        for idx, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
            nd = len(tp)
            rc = tp / npig
            pr = tp / (fp + tp + torch.finfo(torch.float64).eps)
            prec = torch.zeros((nb_rec_thrs,))
            score = torch.zeros((nb_rec_thrs,))

            recall[idx, idx_cls, idx_bbox_area, idx_max_det_thrs] = rc[-1] if nd else 0

            # Remove zigzags for AUC
            diff_zero = torch.zeros((1,), device=pr.device)
            diff = torch.ones((1,), device=pr.device)
            while not torch.all(diff == 0):
                diff = torch.clamp(torch.cat((pr[1:] - pr[:-1], diff_zero), 0), min=0)
                pr += diff

            inds = torch.searchsorted(rc, rec_thresholds.to(rc.device), right=False)
            num_inds = inds.argmax() if inds.max() >= nd else nb_rec_thrs
            inds = inds[:num_inds]
            prec[:num_inds] = pr[inds]
            score[:num_inds] = det_scores_sorted[inds]
            precision[idx, :, idx_cls, idx_bbox_area, idx_max_det_thrs] = prec
            scores[idx, :, idx_cls, idx_bbox_area, idx_max_det_thrs] = score

        return recall, precision, scores

    def compute(self) -> dict:
        """Compute the `Mean-Average-Precision (mAP) and Mean-Average-Recall (mAR)` scores.

        Note:
            ``map`` score is calculated with @[ IoU=self.iou_thresholds | area=all | max_dets=max_detection_thresholds ]

            Caution: If the initialization parameters are changed, dictionary keys for mAR can change as well.
            The default properties are also accessible via fields and will raise an ``AttributeError`` if not available.

        Returns:
            dict containing

            - map: ``torch.Tensor``
            - map_50: ``torch.Tensor``
            - map_75: ``torch.Tensor``
            - map_small: ``torch.Tensor``
            - map_medium: ``torch.Tensor``
            - map_large: ``torch.Tensor``
            - mar_1: ``torch.Tensor``
            - mar_10: ``torch.Tensor``
            - mar_100: ``torch.Tensor``
            - mar_small: ``torch.Tensor``
            - mar_medium: ``torch.Tensor``
            - mar_large: ``torch.Tensor``
            - map_per_class: ``torch.Tensor`` (-1 if class metrics are disabled)
            - mar_100_per_class: ``torch.Tensor`` (-1 if class metrics are disabled)
        """
        classes = self._get_classes()
        precisions, recalls = self._calculate(classes)
        map_val, mar_val = self._summarize_results(precisions, recalls)

        # if class mode is enabled, evaluate metrics per class
        map_per_class_values: Tensor = torch.tensor([-1.0])
        mar_max_dets_per_class_values: Tensor = torch.tensor([-1.0])
        if self.class_metrics:
            map_per_class_list = []
            mar_max_dets_per_class_list = []

            for class_idx, _ in enumerate(classes):
                cls_precisions = precisions[:, :, class_idx].unsqueeze(dim=2)
                cls_recalls = recalls[:, class_idx].unsqueeze(dim=1)
                cls_map, cls_mar = self._summarize_results(cls_precisions, cls_recalls)
                map_per_class_list.append(cls_map.map)
                mar_max_dets_per_class_list.append(cls_mar[f"mar_{self.max_detection_thresholds[-1]}"])

            map_per_class_values = torch.tensor(map_per_class_list, dtype=torch.float)
            mar_max_dets_per_class_values = torch.tensor(mar_max_dets_per_class_list, dtype=torch.float)

        metrics = COCOMetricResults()
        metrics.update(map_val)
        metrics.update(mar_val)
        metrics.map_per_class = map_per_class_values
        metrics[f"mar_{self.max_detection_thresholds[-1]}_per_class"] = mar_max_dets_per_class_values
        return metrics
