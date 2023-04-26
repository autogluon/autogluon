import collections
import copy
import logging
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch

from ...constants import AUTOMM, GET_ITEM_ERROR_RETRY, MULTI_IMAGE_MIX_DATASET, ROIS
from ..preprocess_dataframe import MultiModalFeaturePreprocessor
from ..utils import apply_data_processor, apply_df_preprocessor, get_per_sample_features

logger = logging.getLogger(__name__)

try:
    import mmcv
    import mmcv.utils.config.Config as MMCVConfig
    from mmdet.core import find_inside_bboxes
    from mmdet.utils import log_img_scale  # inline import to avoid mmdet uninstall error for other tasks
except:
    MMCVConfig = None


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
        model_config: MMCVConfig,
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

        for (transform, transform_type) in zip(self.mix_transforms, self.mix_transforms_types):
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


class Mosaic:
    """Mosaic augmentation. Code retrieved from mmdetection and modified.

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

    Args:
        img_scale (Sequence[int]): Image size after mosaic pipeline of single
            image. The shape order should be (height, width).
            Default to (640, 640).
        center_ratio_range (Sequence[float]): Center ratio range of mosaic
            output. Default to (0.5, 1.5).
        min_bbox_size (int | float): The minimum pixel for filtering
            invalid bboxes after the mosaic pipeline. Default to 0.
        bbox_clip_border (bool, optional): Whether to clip the objects outside
            the border of the image. In some dataset like MOT17, the gt bboxes
            are allowed to cross the border of images. Therefore, we don't
            need to clip the gt bboxes in these cases. Defaults to True.
        skip_filter (bool): Whether to skip filtering rules. If it
            is True, the filter rule will not be applied, and the
            `min_bbox_size` is invalid. Default to True.
        pad_val (int): Pad value. Default to 114.
        prob (float): Probability of applying this transformation.
            Default to 1.0.
    """

    def __init__(
        self,
        img_scale=(640, 640),
        center_ratio_range=(0.5, 1.5),
        min_bbox_size=0,
        bbox_clip_border=True,
        skip_filter=True,
        pad_val=114,
        prob=1.0,
    ):

        assert isinstance(img_scale, tuple)
        assert 0 <= prob <= 1.0, "The probability should be in range [0,1]. " f"got {prob}."

        log_img_scale(img_scale, skip_square=True)
        self.img_scale = img_scale
        self.center_ratio_range = center_ratio_range
        self.min_bbox_size = min_bbox_size
        self.bbox_clip_border = bbox_clip_border
        self.skip_filter = skip_filter
        self.pad_val = pad_val
        self.prob = prob

    def __call__(self, results):
        """Call function to make a mosaic of image.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Result dict with mosaic transformed.
        """

        if np.random.uniform(0, 1) > self.prob:
            return results

        results = self._mosaic_transform(results)
        return results

    def get_indexes(self, dataset):
        """Call function to collect indexes.

        Args:
            dataset (:obj:`MultiImageMixDataset`): The dataset.

        Returns:
            list: indexes.
        """

        indexes = [np.random.randint(0, len(dataset)) for _ in range(3)]
        return indexes

    def _mosaic_transform(self, results):
        """Mosaic transform function.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Updated result dict.
        """

        assert "mix_results" in results
        mosaic_labels = []
        mosaic_bboxes = []
        img = results["img"]  # CHW in torch tensor to HWC in numpy
        if len(img.shape) == 3:
            mosaic_img = np.full(
                (int(self.img_scale[0] * 2), int(self.img_scale[1] * 2), 3), self.pad_val, dtype=img.dtype
            )
        else:
            mosaic_img = np.full(
                (int(self.img_scale[0] * 2), int(self.img_scale[1] * 2)), self.pad_val, dtype=img.dtype
            )

        # mosaic center x, y
        center_x = int(np.random.uniform(*self.center_ratio_range) * self.img_scale[1])
        center_y = int(np.random.uniform(*self.center_ratio_range) * self.img_scale[0])
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
            scale_ratio_i = min(self.img_scale[0] / h_i, self.img_scale[1] / w_i)
            img_i = mmcv.imresize(img_i, (int(w_i * scale_ratio_i), int(h_i * scale_ratio_i)))

            # compute the combine parameters
            paste_coord, crop_coord = self._mosaic_combine(loc, center_position, img_i.shape[:2][::-1])
            x1_p, y1_p, x2_p, y2_p = paste_coord
            x1_c, y1_c, x2_c, y2_c = crop_coord

            # crop and paste image
            mosaic_img[y1_p:y2_p, x1_p:x2_p] = img_i[y1_c:y2_c, x1_c:x2_c]

            # adjust coordinate
            gt_bboxes_i = results_patch["gt_bboxes"]
            gt_labels_i = results_patch["gt_labels"]

            if gt_bboxes_i.shape[0] > 0:
                padw = x1_p - x1_c
                padh = y1_p - y1_c
                gt_bboxes_i[:, 0::2] = scale_ratio_i * gt_bboxes_i[:, 0::2] + padw
                gt_bboxes_i[:, 1::2] = scale_ratio_i * gt_bboxes_i[:, 1::2] + padh

            mosaic_bboxes.append(gt_bboxes_i)
            mosaic_labels.append(gt_labels_i)

        if len(mosaic_labels) > 0:
            mosaic_bboxes = np.concatenate(mosaic_bboxes, 0)
            mosaic_labels = np.concatenate(mosaic_labels, 0)

            if self.bbox_clip_border:
                mosaic_bboxes[:, 0::2] = np.clip(mosaic_bboxes[:, 0::2], 0, 2 * self.img_scale[1])
                mosaic_bboxes[:, 1::2] = np.clip(mosaic_bboxes[:, 1::2], 0, 2 * self.img_scale[0])

            if not self.skip_filter:
                mosaic_bboxes, mosaic_labels = self._filter_box_candidates(mosaic_bboxes, mosaic_labels)

        # remove outside bboxes
        inside_inds = find_inside_bboxes(mosaic_bboxes, 2 * self.img_scale[0], 2 * self.img_scale[1])
        mosaic_bboxes = mosaic_bboxes[inside_inds]
        mosaic_labels = mosaic_labels[inside_inds]

        results["img"] = mosaic_img
        results["img_shape"] = mosaic_img.shape
        results["gt_bboxes"] = mosaic_bboxes
        results["gt_labels"] = mosaic_labels

        return results

    def _mosaic_combine(self, loc, center_position_xy, img_shape_wh):
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
                min(center_position_xy[0] + img_shape_wh[0], self.img_scale[1] * 2),
                center_position_xy[1],
            )
            crop_coord = 0, img_shape_wh[1] - (y2 - y1), min(img_shape_wh[0], x2 - x1), img_shape_wh[1]

        elif loc == "bottom_left":
            # index2 to bottom left part of image
            x1, y1, x2, y2 = (
                max(center_position_xy[0] - img_shape_wh[0], 0),
                center_position_xy[1],
                center_position_xy[0],
                min(self.img_scale[0] * 2, center_position_xy[1] + img_shape_wh[1]),
            )
            crop_coord = img_shape_wh[0] - (x2 - x1), 0, img_shape_wh[0], min(y2 - y1, img_shape_wh[1])

        else:
            # index3 to bottom right part of image
            x1, y1, x2, y2 = (
                center_position_xy[0],
                center_position_xy[1],
                min(center_position_xy[0] + img_shape_wh[0], self.img_scale[1] * 2),
                min(self.img_scale[0] * 2, center_position_xy[1] + img_shape_wh[1]),
            )
            crop_coord = 0, 0, min(img_shape_wh[0], x2 - x1), min(y2 - y1, img_shape_wh[1])

        paste_coord = x1, y1, x2, y2
        return paste_coord, crop_coord

    def _filter_box_candidates(self, bboxes, labels):
        """Filter out bboxes too small after Mosaic."""
        bbox_w = bboxes[:, 2] - bboxes[:, 0]
        bbox_h = bboxes[:, 3] - bboxes[:, 1]
        valid_inds = (bbox_w > self.min_bbox_size) & (bbox_h > self.min_bbox_size)
        valid_inds = np.nonzero(valid_inds)[0]
        return bboxes[valid_inds], labels[valid_inds]

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"img_scale={self.img_scale}, "
        repr_str += f"center_ratio_range={self.center_ratio_range}, "
        repr_str += f"pad_val={self.pad_val}, "
        repr_str += f"min_bbox_size={self.min_bbox_size}, "
        repr_str += f"skip_filter={self.skip_filter})"
        return repr_str


class MixUp:
    """MixUp data augmentation. Code retrieved from mmdetection and modified.

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

    Args:
        img_scale (Sequence[int]): Image output size after mixup pipeline.
            The shape order should be (height, width). Default: (640, 640).
        ratio_range (Sequence[float]): Scale ratio of mixup image.
            Default: (0.5, 1.5).
        flip_ratio (float): Horizontal flip ratio of mixup image.
            Default: 0.5.
        pad_val (int): Pad value. Default: 114.
        max_iters (int): The maximum number of iterations. If the number of
            iterations is greater than `max_iters`, but gt_bbox is still
            empty, then the iteration is terminated. Default: 15.
        min_bbox_size (float): Width and height threshold to filter bboxes.
            If the height or width of a box is smaller than this value, it
            will be removed. Default: 5.
        min_area_ratio (float): Threshold of area ratio between
            original bboxes and wrapped bboxes. If smaller than this value,
            the box will be removed. Default: 0.2.
        max_aspect_ratio (float): Aspect ratio of width and height
            threshold to filter bboxes. If max(h/w, w/h) larger than this
            value, the box will be removed. Default: 20.
        bbox_clip_border (bool, optional): Whether to clip the objects outside
            the border of the image. In some dataset like MOT17, the gt bboxes
            are allowed to cross the border of images. Therefore, we don't
            need to clip the gt bboxes in these cases. Defaults to True.
        skip_filter (bool): Whether to skip filtering rules. If it
            is True, the filter rule will not be applied, and the
            `min_bbox_size` and `min_area_ratio` and `max_aspect_ratio`
            is invalid. Default to True.
    """

    def __init__(
        self,
        img_scale=(640, 640),
        ratio_range=(0.5, 1.5),
        flip_ratio=0.5,
        pad_val=114,
        max_iters=15,
        min_bbox_size=5,
        min_area_ratio=0.2,
        max_aspect_ratio=20,
        bbox_clip_border=True,
        skip_filter=True,
        prob=1.0,
    ):
        import mmcv
        from mmdet.core import find_inside_bboxes
        from mmdet.utils import log_img_scale  # inline import to avoid mmdet uninstall error for other tasks

        assert isinstance(img_scale, tuple)
        assert 0 <= prob <= 1.0, "The probability should be in range [0,1]. " f"got {prob}."

        log_img_scale(img_scale, skip_square=True)
        self.dynamic_scale = img_scale
        self.ratio_range = ratio_range
        self.flip_ratio = flip_ratio
        self.pad_val = pad_val
        self.max_iters = max_iters
        self.min_bbox_size = min_bbox_size
        self.min_area_ratio = min_area_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.bbox_clip_border = bbox_clip_border
        self.skip_filter = skip_filter
        self.prob = prob

    def __call__(self, results):
        """Call function to make a mixup of image.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Result dict with mixup transformed.
        """
        if np.random.uniform(0, 1) > self.prob:
            return results

        results = self._mixup_transform(results)
        return results

    def get_indexes(self, dataset):
        """Call function to collect indexes.

        Args:
            dataset (:obj:`MultiImageMixDataset`): The dataset.

        Returns:
            list: indexes.
        """

        index = [np.random.randint(0, len(dataset)) for _ in range(1)]

        return index

    def _mixup_transform(self, results):
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

        jit_factor = np.random.uniform(*self.ratio_range)
        is_filp = np.random.uniform(0, 1) < self.flip_ratio

        if len(retrieve_img.shape) == 3:
            out_img = (
                np.ones((self.dynamic_scale[0], self.dynamic_scale[1], 3), dtype=retrieve_img.dtype) * self.pad_val
            )
        else:
            out_img = np.ones(self.dynamic_scale, dtype=retrieve_img.dtype) * self.pad_val

        # 1. keep_ratio resize
        scale_ratio = min(self.dynamic_scale[0] / retrieve_img.shape[0], self.dynamic_scale[1] / retrieve_img.shape[1])
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
        padded_img = np.zeros((max(origin_h, target_h), max(origin_w, target_w), 3)).astype(np.uint8)
        padded_img[:origin_h, :origin_w] = out_img

        x_offset, y_offset = 0, 0
        if padded_img.shape[0] > target_h:
            y_offset = np.random.randint(0, padded_img.shape[0] - target_h)
        if padded_img.shape[1] > target_w:
            x_offset = np.random.randint(0, padded_img.shape[1] - target_w)
        padded_cropped_img = padded_img[y_offset : y_offset + target_h, x_offset : x_offset + target_w]

        # 6. adjust bbox
        retrieve_gt_bboxes = retrieve_results["gt_bboxes"]
        retrieve_gt_bboxes[:, 0::2] = retrieve_gt_bboxes[:, 0::2] * scale_ratio
        retrieve_gt_bboxes[:, 1::2] = retrieve_gt_bboxes[:, 1::2] * scale_ratio
        if self.bbox_clip_border:
            retrieve_gt_bboxes[:, 0::2] = np.clip(retrieve_gt_bboxes[:, 0::2], 0, origin_w)
            retrieve_gt_bboxes[:, 1::2] = np.clip(retrieve_gt_bboxes[:, 1::2], 0, origin_h)

        if is_filp:
            retrieve_gt_bboxes[:, 0::2] = origin_w - retrieve_gt_bboxes[:, 0::2][:, ::-1]

        # 7. filter
        cp_retrieve_gt_bboxes = retrieve_gt_bboxes.copy()
        cp_retrieve_gt_bboxes[:, 0::2] = cp_retrieve_gt_bboxes[:, 0::2] - x_offset
        cp_retrieve_gt_bboxes[:, 1::2] = cp_retrieve_gt_bboxes[:, 1::2] - y_offset
        if self.bbox_clip_border:
            cp_retrieve_gt_bboxes[:, 0::2] = np.clip(cp_retrieve_gt_bboxes[:, 0::2], 0, target_w)
            cp_retrieve_gt_bboxes[:, 1::2] = np.clip(cp_retrieve_gt_bboxes[:, 1::2], 0, target_h)

        # 8. mix up
        ori_img = ori_img.astype(np.float32)
        mixup_img = 0.5 * ori_img + 0.5 * padded_cropped_img.astype(np.float32)

        retrieve_gt_labels = retrieve_results["gt_labels"]
        if not self.skip_filter:
            keep_list = self._filter_box_candidates(retrieve_gt_bboxes.T, cp_retrieve_gt_bboxes.T)

            retrieve_gt_labels = retrieve_gt_labels[keep_list]
            cp_retrieve_gt_bboxes = cp_retrieve_gt_bboxes[keep_list]

        mixup_gt_bboxes = np.concatenate((results["gt_bboxes"], cp_retrieve_gt_bboxes), axis=0)
        mixup_gt_labels = np.concatenate((results["gt_labels"], retrieve_gt_labels), axis=0)

        # remove outside bbox
        inside_inds = find_inside_bboxes(mixup_gt_bboxes, target_h, target_w)
        mixup_gt_bboxes = mixup_gt_bboxes[inside_inds]
        mixup_gt_labels = mixup_gt_labels[inside_inds]

        results["img"] = mixup_img.astype(np.uint8)
        results["img_shape"] = mixup_img.shape
        results["gt_bboxes"] = mixup_gt_bboxes
        results["gt_labels"] = mixup_gt_labels

        return results

    def _filter_box_candidates(self, bbox1, bbox2):
        """Compute candidate boxes which include following 5 things:

        bbox1 before augment, bbox2 after augment, min_bbox_size (pixels),
        min_area_ratio, max_aspect_ratio.
        """

        w1, h1 = bbox1[2] - bbox1[0], bbox1[3] - bbox1[1]
        w2, h2 = bbox2[2] - bbox2[0], bbox2[3] - bbox2[1]
        ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))
        return (
            (w2 > self.min_bbox_size)
            & (h2 > self.min_bbox_size)
            & (w2 * h2 / (w1 * h1 + 1e-16) > self.min_area_ratio)
            & (ar < self.max_aspect_ratio)
        )

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"dynamic_scale={self.dynamic_scale}, "
        repr_str += f"ratio_range={self.ratio_range}, "
        repr_str += f"flip_ratio={self.flip_ratio}, "
        repr_str += f"pad_val={self.pad_val}, "
        repr_str += f"max_iters={self.max_iters}, "
        repr_str += f"min_bbox_size={self.min_bbox_size}, "
        repr_str += f"min_area_ratio={self.min_area_ratio}, "
        repr_str += f"max_aspect_ratio={self.max_aspect_ratio}, "
        repr_str += f"skip_filter={self.skip_filter})"
        return repr_str
