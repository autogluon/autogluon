import collections
import copy
import logging
import math
from typing import Dict, List, Optional, Sequence, Tuple, Union

import mmcv
import numpy as np
import pandas as pd
import torch
from mmcv.transforms import BaseTransform
from mmcv.transforms.utils import cache_randomness
from mmdet.structures.bbox import autocast_box_type
from mmdet.utils import log_img_scale
from mmengine.config import Config as MMConfig
from mmengine.dataset import BaseDataset
from numpy import random

from ...constants import GET_ITEM_ERROR_RETRY, MULTI_IMAGE_MIX_DATASET, ROIS
from ..preprocess_dataframe import MultiModalFeaturePreprocessor
from ..utils import apply_data_processor, apply_df_preprocessor, get_per_sample_features

logger = logging.getLogger(__name__)


class MultiImageMixDataset(torch.utils.data.Dataset):
    """
    A Pytorch DataSet class to process a multimodal pd.DataFrame. It first uses a preprocessor to
    produce model-agnostic features. Then, each processor prepares customized data for one modality
    per model. For code simplicity, here we treat ground-truth label as one modality. This class is
    independent of specific data modalities and models.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        preprocessor: List[MultiModalFeaturePreprocessor],
        processors: List[dict],
        model_config: MMConfig,
        id_mappings: Optional[Union[Dict[str, Dict], Dict[str, pd.Series]]] = None,
        is_training: bool = False,
    ):
        """
        Parameters
        ----------
        data
            A pd.DataFrame containing multimodal features.
        preprocessor
            A list of multimodal feature preprocessors generating model-agnostic features.
        processors
            Data processors customizing data for each modality per model.
        id_mappings
            Id-to-content mappings. The contents can be text, image, etc.
            This is used when the dataframe contains the query/response indexes instead of their contents.
        is_training
            Whether in training mode. Some data processing may be different between training
            and validation/testing/prediction, e.g., image data augmentation is used only in
            training.
        model_config
            Model config used to decided dataset type. e.g. if multi_image_mix_dataset is used in detection model,
            MultiImageMixDataset will be used instead of BaseDataset
        """
        super().__init__()
        self.processors = processors
        self.is_training = is_training
        self._consecutive_errors = 0

        mix_config = model_config[MULTI_IMAGE_MIX_DATASET]
        self.mix_data_key = "mmdet_image_image"  # the key of the data to mix, TODO: remove hardcoding
        self.mix_result_key = "mix_results"  # the key of the mix result to store

        self.mix_transforms = []
        self.mix_transforms_types = []  # TODO: remove hardcode
        if "mosaic" in mix_config:
            self.mix_transforms.append(Mosaic(**mix_config["mosaic"]))
            self.mix_transforms_types.append("mosaic")
        if "mixup" in mix_config:
            self.mix_transforms.append(MixUp(**mix_config["mixup"]))
            self.mix_transforms_types.append("mixup")

        self._skip_type_keys = None  # TODO: remove hardcode, we need to disable multi image mix in late epochs
        self.max_refetch = 15  # TODO: remove hardcode (do we need refetch?)

        self.lengths = []

        for i, (per_preprocessor, per_processors_group) in enumerate(zip(preprocessor, processors)):
            modality_features, modality_types, length = apply_df_preprocessor(
                data=data,
                df_preprocessor=per_preprocessor,
                modalities=per_processors_group.keys(),
            )
            self.lengths.append(length)
            setattr(self, f"modality_features_{i}", modality_features)
            setattr(self, f"modality_types_{i}", modality_types)

        assert len(set(self.lengths)) == 1

        self.id_mappings = id_mappings

    def __len__(self):
        """
        Assume that all modalities have the same sample number.

        Returns
        -------
        Sample number in this dataset.
        """
        return self.lengths[0]

    def _load_item(self, idx):
        """
        Get a single item without mix_results.
        Iterate through all data processors to prepare model inputs. The data processors are
        organized first by modalities and then by models.

        Parameters
        ----------
        idx
            Index of sample to process.

        Returns
        -------
        Input data formatted as a dictionary.
        """
        ret = dict()
        try:
            for group_id, per_processors_group in enumerate(self.processors):
                per_sample_features = get_per_sample_features(
                    modality_features=getattr(self, f"modality_features_{group_id}"),
                    modality_types=getattr(self, f"modality_types_{group_id}"),
                    idx=idx,
                    id_mappings=self.id_mappings,
                )
                per_ret = apply_data_processor(
                    per_sample_features=per_sample_features,
                    data_processors=per_processors_group,
                    feature_modalities=getattr(self, f"modality_types_{group_id}"),
                    is_training=self.is_training,
                    load_only=True,
                )
                ret.update(per_ret)
        except Exception as e:
            logger.debug(f"Skipping sample {idx} due to '{e}'")
            self._consecutive_errors += 1
            if self._consecutive_errors < GET_ITEM_ERROR_RETRY:
                return self.__getitem__((idx + 1) % self.__len__())
            else:
                raise e
        self._consecutive_errors = 0

        return ret

    def __getitem__(self, idx):
        """
        Iterate through all data processors to prepare model inputs. The data processors are
        organized first by modalities and then by models.

        Parameters
        ----------
        idx
            Index of sample to process.

        Returns
        -------
        Input data formatted as a dictionary.
        """
        results = copy.deepcopy(self._load_item(idx))

        for transform, transform_type in zip(self.mix_transforms, self.mix_transforms_types):
            assert hasattr(transform, "get_indexes")

            if self._skip_type_keys is not None and transform_type in self._skip_type_keys:
                continue

            for i in range(self.max_refetch):
                # Make sure the results passed the loading pipeline
                # of the original dataset is not None.
                indexes = transform.get_indexes(self)
                if not isinstance(indexes, collections.abc.Sequence):
                    indexes = [indexes]
                mix_results = [copy.deepcopy(self._load_item(index)[self.mix_data_key]) for index in indexes]
                if None not in mix_results:
                    results[self.mix_data_key][self.mix_result_key] = mix_results
                    break
            else:
                raise RuntimeError(
                    "The loading pipeline of the original dataset"
                    " always return None. Please check the correctness "
                    "of the dataset and its pipeline."
                )

            for i in range(self.max_refetch):
                # To confirm the results passed the training pipeline
                # of the wrapper is not None.
                updated_results = transform(copy.deepcopy(results[self.mix_data_key]))
                if updated_results is not None:
                    results[self.mix_data_key] = updated_results
                    break
            else:
                raise RuntimeError(
                    "The training pipeline of the dataset wrapper"
                    " always return None.Please check the correctness "
                    "of the dataset and its pipeline."
                )

            if self.mix_result_key in results[self.mix_data_key]:
                results[self.mix_data_key].pop(self.mix_result_key)

        rois_processor = self.processors[0][ROIS][0]  # TODO: remove hardcode
        results.update(
            rois_processor.process_one_loaded_sample(
                results,
                is_training=True,  # This dataset is used only in training
            )
        )

        return results


class Mosaic(BaseTransform):
    """Mosaic augmentation.

    Given 4 images, mosaic transform combines them into
    one output image. The output image is composed of the parts from each sub-
    image.

    .. code:: text

                        mosaic transform
                        center_x
                +------------------------------+
                |       pad        |  pad      |
                |      +-----------+           |
                |      |           |           |
                |      |  image1   |--------+  |
                |      |           |        |  |
                |      |           | image2 |  |
    center_y   |----+-------------+-----------|
                |    |   cropped   |           |
                |pad |   image3    |  image4   |
                |    |             |           |
                +----|-------------+-----------+
                    |             |
                    +-------------+

    The mosaic transform steps are as follows:

        1. Choose the mosaic center as the intersections of 4 images
        2. Get the left top image according to the index, and randomly
            sample another 3 images from the custom dataset.
        3. Sub image will be cropped if image is larger than mosaic patch

    Required Keys:

    - img
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)
    - gt_bboxes_labels (np.int64) (optional)
    - gt_ignore_flags (bool) (optional)
    - mix_results (List[dict])

    Modified Keys:

    - img
    - img_shape
    - gt_bboxes (optional)
    - gt_bboxes_labels (optional)
    - gt_ignore_flags (optional)

    Args:
        img_scale (Sequence[int]): Image size after mosaic pipeline of single
            image. The shape order should be (width, height).
            Defaults to (640, 640).
        center_ratio_range (Sequence[float]): Center ratio range of mosaic
            output. Defaults to (0.5, 1.5).
        bbox_clip_border (bool, optional): Whether to clip the objects outside
            the border of the image. In some dataset like MOT17, the gt bboxes
            are allowed to cross the border of images. Therefore, we don't
            need to clip the gt bboxes in these cases. Defaults to True.
        pad_val (int): Pad value. Defaults to 114.
        prob (float): Probability of applying this transformation.
            Defaults to 1.0.
    """

    def __init__(
        self,
        img_scale: Tuple[int, int] = (640, 640),
        center_ratio_range: Tuple[float, float] = (0.5, 1.5),
        bbox_clip_border: bool = True,
        pad_val: float = 114.0,
        prob: float = 1.0,
    ) -> None:
        assert isinstance(img_scale, tuple)
        assert 0 <= prob <= 1.0, "The probability should be in range [0,1]. " f"got {prob}."

        log_img_scale(img_scale, skip_square=True, shape_order="wh")
        self.img_scale = img_scale
        self.center_ratio_range = center_ratio_range
        self.bbox_clip_border = bbox_clip_border
        self.pad_val = pad_val
        self.prob = prob

    @cache_randomness
    def get_indexes(self, dataset: BaseDataset) -> int:
        """Call function to collect indexes.

        Args:
            dataset (:obj:`MultiImageMixDataset`): The dataset.

        Returns:
            list: indexes.
        """

        indexes = [random.randint(0, len(dataset)) for _ in range(3)]
        return indexes

    @autocast_box_type()
    def transform(self, results: dict) -> dict:
        """Mosaic transform function.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Updated result dict.
        """
        if random.uniform(0, 1) > self.prob:
            return results

        assert "mix_results" in results
        mosaic_bboxes = []
        mosaic_bboxes_labels = []
        mosaic_ignore_flags = []
        if len(results["img"].shape) == 3:
            mosaic_img = np.full(
                (int(self.img_scale[1] * 2), int(self.img_scale[0] * 2), 3),
                self.pad_val,
                dtype=results["img"].dtype,
            )
        else:
            mosaic_img = np.full(
                (int(self.img_scale[1] * 2), int(self.img_scale[0] * 2)), self.pad_val, dtype=results["img"].dtype
            )

        # mosaic center x, y
        center_x = int(random.uniform(*self.center_ratio_range) * self.img_scale[0])
        center_y = int(random.uniform(*self.center_ratio_range) * self.img_scale[1])
        center_position = (center_x, center_y)

        loc_strs = ("top_left", "top_right", "bottom_left", "bottom_right")
        for i, loc in enumerate(loc_strs):
            if loc == "top_left":
                results_patch = copy.deepcopy(results)
            else:
                results_patch = copy.deepcopy(results["mix_results"][i - 1])

            img_i = results_patch["img"]
            h_i, w_i = img_i.shape[:2]
            # keep_ratio resize
            scale_ratio_i = min(self.img_scale[1] / h_i, self.img_scale[0] / w_i)
            img_i = mmcv.imresize(img_i, (int(w_i * scale_ratio_i), int(h_i * scale_ratio_i)))

            # compute the combine parameters
            paste_coord, crop_coord = self._mosaic_combine(loc, center_position, img_i.shape[:2][::-1])
            x1_p, y1_p, x2_p, y2_p = paste_coord
            x1_c, y1_c, x2_c, y2_c = crop_coord

            # crop and paste image
            mosaic_img[y1_p:y2_p, x1_p:x2_p] = img_i[y1_c:y2_c, x1_c:x2_c]

            # adjust coordinate
            gt_bboxes_i = results_patch["gt_bboxes"]
            gt_bboxes_labels_i = results_patch["gt_bboxes_labels"]
            gt_ignore_flags_i = results_patch["gt_ignore_flags"]

            padw = x1_p - x1_c
            padh = y1_p - y1_c
            gt_bboxes_i.rescale_([scale_ratio_i, scale_ratio_i])
            gt_bboxes_i.translate_([padw, padh])
            mosaic_bboxes.append(gt_bboxes_i)
            mosaic_bboxes_labels.append(gt_bboxes_labels_i)
            mosaic_ignore_flags.append(gt_ignore_flags_i)

        mosaic_bboxes = mosaic_bboxes[0].cat(mosaic_bboxes, 0)
        mosaic_bboxes_labels = np.concatenate(mosaic_bboxes_labels, 0)
        mosaic_ignore_flags = np.concatenate(mosaic_ignore_flags, 0)

        if self.bbox_clip_border:
            mosaic_bboxes.clip_([2 * self.img_scale[1], 2 * self.img_scale[0]])
        # remove outside bboxes
        inside_inds = mosaic_bboxes.is_inside([2 * self.img_scale[1], 2 * self.img_scale[0]]).numpy()
        mosaic_bboxes = mosaic_bboxes[inside_inds]
        mosaic_bboxes_labels = mosaic_bboxes_labels[inside_inds]
        mosaic_ignore_flags = mosaic_ignore_flags[inside_inds]

        results["img"] = mosaic_img
        results["img_shape"] = mosaic_img.shape[:2]
        results["gt_bboxes"] = mosaic_bboxes
        results["gt_bboxes_labels"] = mosaic_bboxes_labels
        results["gt_ignore_flags"] = mosaic_ignore_flags
        return results

    def _mosaic_combine(
        self, loc: str, center_position_xy: Sequence[float], img_shape_wh: Sequence[int]
    ) -> Tuple[Tuple[int], Tuple[int]]:
        """Calculate global coordinate of mosaic image and local coordinate of
        cropped sub-image.

        Args:
            loc (str): Index for the sub-image, loc in ('top_left',
            'top_right', 'bottom_left', 'bottom_right').
            center_position_xy (Sequence[float]): Mixing center for 4 images,
                (x, y).
            img_shape_wh (Sequence[int]): Width and height of sub-image

        Returns:
            tuple[tuple[float]]: Corresponding coordinate of pasting and
                cropping
                - paste_coord (tuple): paste corner coordinate in mosaic image.
                - crop_coord (tuple): crop corner coordinate in mosaic image.
        """
        assert loc in ("top_left", "top_right", "bottom_left", "bottom_right")
        if loc == "top_left":
            # index0 to top left part of image
            x1, y1, x2, y2 = (
                max(center_position_xy[0] - img_shape_wh[0], 0),
                max(center_position_xy[1] - img_shape_wh[1], 0),
                center_position_xy[0],
                center_position_xy[1],
            )
            crop_coord = img_shape_wh[0] - (x2 - x1), img_shape_wh[1] - (y2 - y1), img_shape_wh[0], img_shape_wh[1]

        elif loc == "top_right":
            # index1 to top right part of image
            x1, y1, x2, y2 = (
                center_position_xy[0],
                max(center_position_xy[1] - img_shape_wh[1], 0),
                min(center_position_xy[0] + img_shape_wh[0], self.img_scale[0] * 2),
                center_position_xy[1],
            )
            crop_coord = 0, img_shape_wh[1] - (y2 - y1), min(img_shape_wh[0], x2 - x1), img_shape_wh[1]

        elif loc == "bottom_left":
            # index2 to bottom left part of image
            x1, y1, x2, y2 = (
                max(center_position_xy[0] - img_shape_wh[0], 0),
                center_position_xy[1],
                center_position_xy[0],
                min(self.img_scale[1] * 2, center_position_xy[1] + img_shape_wh[1]),
            )
            crop_coord = img_shape_wh[0] - (x2 - x1), 0, img_shape_wh[0], min(y2 - y1, img_shape_wh[1])

        else:
            # index3 to bottom right part of image
            x1, y1, x2, y2 = (
                center_position_xy[0],
                center_position_xy[1],
                min(center_position_xy[0] + img_shape_wh[0], self.img_scale[0] * 2),
                min(self.img_scale[1] * 2, center_position_xy[1] + img_shape_wh[1]),
            )
            crop_coord = 0, 0, min(img_shape_wh[0], x2 - x1), min(y2 - y1, img_shape_wh[1])

        paste_coord = x1, y1, x2, y2
        return paste_coord, crop_coord

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(img_scale={self.img_scale}, "
        repr_str += f"center_ratio_range={self.center_ratio_range}, "
        repr_str += f"pad_val={self.pad_val}, "
        repr_str += f"prob={self.prob})"
        return repr_str


class MixUp(BaseTransform):
    """MixUp data augmentation.

    .. code:: text

                        mixup transform
                +------------------------------+
                | mixup image   |              |
                |      +--------|--------+     |
                |      |        |        |     |
                |---------------+        |     |
                |      |                 |     |
                |      |      image      |     |
                |      |                 |     |
                |      |                 |     |
                |      |-----------------+     |
                |             pad              |
                +------------------------------+

    The mixup transform steps are as follows:

        1. Another random image is picked by dataset and embedded in
        the top left patch(after padding and resizing)
        2. The target of mixup transform is the weighted average of mixup
        image and origin image.

    Required Keys:

    - img
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)
    - gt_bboxes_labels (np.int64) (optional)
    - gt_ignore_flags (bool) (optional)
    - mix_results (List[dict])


    Modified Keys:

    - img
    - img_shape
    - gt_bboxes (optional)
    - gt_bboxes_labels (optional)
    - gt_ignore_flags (optional)


    Args:
        img_scale (Sequence[int]): Image output size after mixup pipeline.
            The shape order should be (width, height). Defaults to (640, 640).
        ratio_range (Sequence[float]): Scale ratio of mixup image.
            Defaults to (0.5, 1.5).
        flip_ratio (float): Horizontal flip ratio of mixup image.
            Defaults to 0.5.
        pad_val (int): Pad value. Defaults to 114.
        max_iters (int): The maximum number of iterations. If the number of
            iterations is greater than `max_iters`, but gt_bbox is still
            empty, then the iteration is terminated. Defaults to 15.
        bbox_clip_border (bool, optional): Whether to clip the objects outside
            the border of the image. In some dataset like MOT17, the gt bboxes
            are allowed to cross the border of images. Therefore, we don't
            need to clip the gt bboxes in these cases. Defaults to True.
    """

    def __init__(
        self,
        img_scale: Tuple[int, int] = (640, 640),
        ratio_range: Tuple[float, float] = (0.5, 1.5),
        flip_ratio: float = 0.5,
        pad_val: float = 114.0,
        max_iters: int = 15,
        bbox_clip_border: bool = True,
    ) -> None:
        assert isinstance(img_scale, tuple)
        log_img_scale(img_scale, skip_square=True, shape_order="wh")
        self.dynamic_scale = img_scale
        self.ratio_range = ratio_range
        self.flip_ratio = flip_ratio
        self.pad_val = pad_val
        self.max_iters = max_iters
        self.bbox_clip_border = bbox_clip_border

    @cache_randomness
    def get_indexes(self, dataset: BaseDataset) -> int:
        """Call function to collect indexes.

        Args:
            dataset (:obj:`MultiImageMixDataset`): The dataset.

        Returns:
            list: indexes.
        """

        index = [np.random.randint(0, len(dataset)) for _ in range(1)]

        return index

    @autocast_box_type()
    def transform(self, results: dict) -> dict:
        """MixUp transform function.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Updated result dict.
        """

        assert "mix_results" in results
        assert len(results["mix_results"]) == 1, "MixUp only support 2 images now !"

        if results["mix_results"][0]["gt_bboxes"].shape[0] == 0:
            # empty bbox
            return results

        retrieve_results = results["mix_results"][0]
        retrieve_img = retrieve_results["img"]

        jit_factor = random.uniform(*self.ratio_range)
        is_filp = random.uniform(0, 1) > self.flip_ratio

        if len(retrieve_img.shape) == 3:
            out_img = (
                np.ones((self.dynamic_scale[1], self.dynamic_scale[0], 3), dtype=retrieve_img.dtype) * self.pad_val
            )
        else:
            out_img = np.ones(self.dynamic_scale[::-1], dtype=retrieve_img.dtype) * self.pad_val

        # 1. keep_ratio resize
        scale_ratio = min(self.dynamic_scale[1] / retrieve_img.shape[0], self.dynamic_scale[0] / retrieve_img.shape[1])
        retrieve_img = mmcv.imresize(
            retrieve_img, (int(retrieve_img.shape[1] * scale_ratio), int(retrieve_img.shape[0] * scale_ratio))
        )

        # 2. paste
        out_img[: retrieve_img.shape[0], : retrieve_img.shape[1]] = retrieve_img

        # 3. scale jit
        scale_ratio *= jit_factor
        out_img = mmcv.imresize(out_img, (int(out_img.shape[1] * jit_factor), int(out_img.shape[0] * jit_factor)))

        # 4. flip
        if is_filp:
            out_img = out_img[:, ::-1, :]

        # 5. random crop
        ori_img = results["img"]
        origin_h, origin_w = out_img.shape[:2]
        target_h, target_w = ori_img.shape[:2]
        padded_img = np.ones((max(origin_h, target_h), max(origin_w, target_w), 3)) * self.pad_val
        padded_img = padded_img.astype(np.uint8)
        padded_img[:origin_h, :origin_w] = out_img

        x_offset, y_offset = 0, 0
        if padded_img.shape[0] > target_h:
            y_offset = random.randint(0, padded_img.shape[0] - target_h)
        if padded_img.shape[1] > target_w:
            x_offset = random.randint(0, padded_img.shape[1] - target_w)
        padded_cropped_img = padded_img[y_offset : y_offset + target_h, x_offset : x_offset + target_w]

        # 6. adjust bbox
        retrieve_gt_bboxes = retrieve_results["gt_bboxes"]
        retrieve_gt_bboxes.rescale_([scale_ratio, scale_ratio])
        if self.bbox_clip_border:
            retrieve_gt_bboxes.clip_([origin_h, origin_w])

        if is_filp:
            retrieve_gt_bboxes.flip_([origin_h, origin_w], direction="horizontal")

        # 7. filter
        cp_retrieve_gt_bboxes = retrieve_gt_bboxes.clone()
        cp_retrieve_gt_bboxes.translate_([-x_offset, -y_offset])
        if self.bbox_clip_border:
            cp_retrieve_gt_bboxes.clip_([target_h, target_w])

        # 8. mix up
        ori_img = ori_img.astype(np.float32)
        mixup_img = 0.5 * ori_img + 0.5 * padded_cropped_img.astype(np.float32)

        retrieve_gt_bboxes_labels = retrieve_results["gt_bboxes_labels"]
        retrieve_gt_ignore_flags = retrieve_results["gt_ignore_flags"]

        mixup_gt_bboxes = cp_retrieve_gt_bboxes.cat((results["gt_bboxes"], cp_retrieve_gt_bboxes), dim=0)
        mixup_gt_bboxes_labels = np.concatenate((results["gt_bboxes_labels"], retrieve_gt_bboxes_labels), axis=0)
        mixup_gt_ignore_flags = np.concatenate((results["gt_ignore_flags"], retrieve_gt_ignore_flags), axis=0)

        # remove outside bbox
        inside_inds = mixup_gt_bboxes.is_inside([target_h, target_w]).numpy()
        mixup_gt_bboxes = mixup_gt_bboxes[inside_inds]
        mixup_gt_bboxes_labels = mixup_gt_bboxes_labels[inside_inds]
        mixup_gt_ignore_flags = mixup_gt_ignore_flags[inside_inds]

        results["img"] = mixup_img.astype(np.uint8)
        results["img_shape"] = mixup_img.shape[:2]
        results["gt_bboxes"] = mixup_gt_bboxes
        results["gt_bboxes_labels"] = mixup_gt_bboxes_labels
        results["gt_ignore_flags"] = mixup_gt_ignore_flags

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(dynamic_scale={self.dynamic_scale}, "
        repr_str += f"ratio_range={self.ratio_range}, "
        repr_str += f"flip_ratio={self.flip_ratio}, "
        repr_str += f"pad_val={self.pad_val}, "
        repr_str += f"max_iters={self.max_iters}, "
        repr_str += f"bbox_clip_border={self.bbox_clip_border})"
        return repr_str


class RandomAffine(BaseTransform):
    """Random affine transform data augmentation.

    This operation randomly generates affine transform matrix which including
    rotation, translation, shear and scaling transforms.

    Required Keys:

    - img
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)
    - gt_bboxes_labels (np.int64) (optional)
    - gt_ignore_flags (bool) (optional)

    Modified Keys:

    - img
    - img_shape
    - gt_bboxes (optional)
    - gt_bboxes_labels (optional)
    - gt_ignore_flags (optional)

    Args:
        max_rotate_degree (float): Maximum degrees of rotation transform.
            Defaults to 10.
        max_translate_ratio (float): Maximum ratio of translation.
            Defaults to 0.1.
        scaling_ratio_range (tuple[float]): Min and max ratio of
            scaling transform. Defaults to (0.5, 1.5).
        max_shear_degree (float): Maximum degrees of shear
            transform. Defaults to 2.
        border (tuple[int]): Distance from width and height sides of input
            image to adjust output shape. Only used in mosaic dataset.
            Defaults to (0, 0).
        border_val (tuple[int]): Border padding values of 3 channels.
            Defaults to (114, 114, 114).
        bbox_clip_border (bool, optional): Whether to clip the objects outside
            the border of the image. In some dataset like MOT17, the gt bboxes
            are allowed to cross the border of images. Therefore, we don't
            need to clip the gt bboxes in these cases. Defaults to True.
    """

    def __init__(
        self,
        max_rotate_degree: float = 10.0,
        max_translate_ratio: float = 0.1,
        scaling_ratio_range: Tuple[float, float] = (0.5, 1.5),
        max_shear_degree: float = 2.0,
        border: Tuple[int, int] = (0, 0),
        border_val: Tuple[int, int, int] = (114, 114, 114),
        bbox_clip_border: bool = True,
    ) -> None:
        assert 0 <= max_translate_ratio <= 1
        assert scaling_ratio_range[0] <= scaling_ratio_range[1]
        assert scaling_ratio_range[0] > 0
        self.max_rotate_degree = max_rotate_degree
        self.max_translate_ratio = max_translate_ratio
        self.scaling_ratio_range = scaling_ratio_range
        self.max_shear_degree = max_shear_degree
        self.border = border
        self.border_val = border_val
        self.bbox_clip_border = bbox_clip_border

    @cache_randomness
    def _get_random_homography_matrix(self, height, width):
        # Rotation
        rotation_degree = random.uniform(-self.max_rotate_degree, self.max_rotate_degree)
        rotation_matrix = self._get_rotation_matrix(rotation_degree)

        # Scaling
        scaling_ratio = random.uniform(self.scaling_ratio_range[0], self.scaling_ratio_range[1])
        scaling_matrix = self._get_scaling_matrix(scaling_ratio)

        # Shear
        x_degree = random.uniform(-self.max_shear_degree, self.max_shear_degree)
        y_degree = random.uniform(-self.max_shear_degree, self.max_shear_degree)
        shear_matrix = self._get_shear_matrix(x_degree, y_degree)

        # Translation
        trans_x = random.uniform(-self.max_translate_ratio, self.max_translate_ratio) * width
        trans_y = random.uniform(-self.max_translate_ratio, self.max_translate_ratio) * height
        translate_matrix = self._get_translation_matrix(trans_x, trans_y)

        warp_matrix = translate_matrix @ shear_matrix @ rotation_matrix @ scaling_matrix
        return warp_matrix

    @autocast_box_type()
    def transform(self, results: dict) -> dict:
        import cv2  # TODO: support random affine requires cv2

        img = results["img"]
        height = img.shape[0] + self.border[1] * 2
        width = img.shape[1] + self.border[0] * 2

        warp_matrix = self._get_random_homography_matrix(height, width)

        img = cv2.warpPerspective(img, warp_matrix, dsize=(width, height), borderValue=self.border_val)
        results["img"] = img
        results["img_shape"] = img.shape[:2]

        bboxes = results["gt_bboxes"]
        num_bboxes = len(bboxes)
        if num_bboxes:
            bboxes.project_(warp_matrix)
            if self.bbox_clip_border:
                bboxes.clip_([height, width])
            # remove outside bbox
            valid_index = bboxes.is_inside([height, width]).numpy()
            results["gt_bboxes"] = bboxes[valid_index]
            results["gt_bboxes_labels"] = results["gt_bboxes_labels"][valid_index]
            results["gt_ignore_flags"] = results["gt_ignore_flags"][valid_index]

            if "gt_masks" in results:
                raise NotImplementedError("RandomAffine only supports bbox.")
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(max_rotate_degree={self.max_rotate_degree}, "
        repr_str += f"max_translate_ratio={self.max_translate_ratio}, "
        repr_str += f"scaling_ratio_range={self.scaling_ratio_range}, "
        repr_str += f"max_shear_degree={self.max_shear_degree}, "
        repr_str += f"border={self.border}, "
        repr_str += f"border_val={self.border_val}, "
        repr_str += f"bbox_clip_border={self.bbox_clip_border})"
        return repr_str

    @staticmethod
    def _get_rotation_matrix(rotate_degrees: float) -> np.ndarray:
        radian = math.radians(rotate_degrees)
        rotation_matrix = np.array(
            [[np.cos(radian), -np.sin(radian), 0.0], [np.sin(radian), np.cos(radian), 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        return rotation_matrix

    @staticmethod
    def _get_scaling_matrix(scale_ratio: float) -> np.ndarray:
        scaling_matrix = np.array(
            [[scale_ratio, 0.0, 0.0], [0.0, scale_ratio, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
        )
        return scaling_matrix

    @staticmethod
    def _get_shear_matrix(x_shear_degrees: float, y_shear_degrees: float) -> np.ndarray:
        x_radian = math.radians(x_shear_degrees)
        y_radian = math.radians(y_shear_degrees)
        shear_matrix = np.array(
            [[1, np.tan(x_radian), 0.0], [np.tan(y_radian), 1, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
        )
        return shear_matrix

    @staticmethod
    def _get_translation_matrix(x: float, y: float) -> np.ndarray:
        translation_matrix = np.array([[1, 0.0, x], [0.0, 1, y], [0.0, 0.0, 1.0]], dtype=np.float32)
        return translation_matrix
