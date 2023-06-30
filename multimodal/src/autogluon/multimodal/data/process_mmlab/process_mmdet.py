import logging
from typing import Dict, List, Optional, Union

import numpy as np
import PIL
from PIL import ImageFile
from torch import nn

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = PIL.Image.BICUBIC

from ..utils import is_rois_input
from .process_mmlab_base import MMLabProcessor

try:
    from mmcv.transforms import Compose
except ImportError as e:
    pass

logger = logging.getLogger(__name__)
ImageFile.LOAD_TRUNCATED_IMAGES = True


class MMDetProcessor(MMLabProcessor):
    """
    Prepare rois data for mmdetection models.
    """

    def __init__(
        self,
        model: nn.Module,
        max_img_num_per_col: Optional[int] = 1,
        missing_value_strategy: Optional[str] = "zero",
        requires_column_info: bool = False,
    ):
        """
        Parameters
        ----------
        model
            The model using this data processor.
        max_img_num_per_col
            The maximum number of images one sample can have.
        missing_value_strategy
            How to deal with a missing image. We now support:
            - skip
                Skip this sample
            -zero
                Use an image with zero pixels.
        requires_column_info
            Whether to require feature column information in dataloader.
        """
        from ...utils import CollateMMDet, check_if_packages_installed

        check_if_packages_installed(package_names=["mmcv", "mmengine", "mmdet"])

        super().__init__(
            model=model,
            collate_func=CollateMMDet,
            max_img_num_per_col=max_img_num_per_col,
            missing_value_strategy=missing_value_strategy,
            requires_column_info=requires_column_info,
        )

        # for yolox we separate loading pipeline to support multi_image_mix_dataset
        if "loading_pipeline" in self.cfg.keys():
            self.load_processor = Compose(self.cfg.loading_pipeline)

    def prepare_one_sample(
        self,
        image_paths: Dict[str, List[str]],
        is_training: bool,
    ):
        mm_data = dict(img_prefix=None, bbox_fields=[])

        for per_col_name, per_col_content in image_paths.items():
            if is_rois_input(per_col_content):
                rois = np.array(per_col_content)
                # https://github.com/open-mmlab/mmdetection/blob/ecac3a77becc63f23d9f6980b2a36f86acd00a8a/mmdet/datasets/transforms/loading.py#L155
                mm_data["instances"] = []
                for roi in rois:
                    mm_data["instances"].append(
                        {
                            "bbox": roi[:4],
                            "bbox_label": roi[4],
                            "ignore_flag": 0,
                        }
                    )
            else:
                with PIL.Image.open(per_col_content[0]) as img:
                    # mm_data["img_info"] = dict(filename=per_col_content[0], height=img.height, width=img.width)
                    mm_data["img_path"] = per_col_content[0]
        if self.requires_column_info:
            pass  # TODO

        mm_data["img_id"] = 0  # TODO: use a non trivial image id for TTA (test time augmentation)

        return mm_data

    def load_one_sample(
        self,
        image_paths: Dict[str, List[str]],
        is_training: bool,
    ) -> Dict:
        """
        Read images, process them, and stack them. One sample can have multiple images,
        resulting in a tensor of (n, 3, size, size), where n <= max_img_num_per_col is the available image number.

        Parameters
        ----------
        image_paths
            One sample may have multiple image columns in a pd.DataFrame and multiple images
            inside each image column.
        is_training
            Whether to process images in the training mode.

        Returns
        -------
        A dictionary containing one sample's images and their number.
        """
        mm_data = self.prepare_one_sample(image_paths=image_paths, is_training=is_training)
        ret = {self.image_key: self.load_processor(mm_data)}
        return ret

    def process_one_loaded_sample(
        self,
        image_paths: Dict[str, List[str]],
        is_training: bool,
    ) -> Dict:
        """
        Read images, process them, and stack them. One sample can have multiple images,
        resulting in a tensor of (n, 3, size, size), where n <= max_img_num_per_col is the available image number.

        Parameters
        ----------
        image_paths
            One sample may have multiple image columns in a pd.DataFrame and multiple images
            inside each image column.
        is_training
            Whether to process images in the training mode.

        Returns
        -------
        A dictionary containing one sample's images and their number.
        """
        assert is_training

        image_paths.update({self.image_key: self.train_processor(image_paths[self.image_key])})

        return image_paths

    def process_one_sample(
        self,
        image_paths: Dict[str, List[str]],
        is_training: bool,
    ) -> Dict:
        """
        Read images, process them, and stack them. One sample can have multiple images,
        resulting in a tensor of (n, 3, size, size), where n <= max_img_num_per_col is the available image number.
        Parameters
        ----------
        image_paths
            One sample may have multiple image columns in a pd.DataFrame and multiple images
            inside each image column.
        is_training
            Whether to process images in the training mode.
        Returns
        -------
        A dictionary containing one sample's images and their number.
        """
        mm_data = self.prepare_one_sample(image_paths=image_paths, is_training=is_training)
        ret = {self.image_key: self.train_processor(mm_data) if is_training else self.val_processor(mm_data)}
        return ret

    def __call__(
        self,
        images: Dict[str, List[str]],
        feature_modalities: Dict[str, Union[int, float, list]],
        is_training: bool,
        load_only: bool = False,
    ) -> Dict:
        """
        Obtain one sample's images and customized them for a specific model.

        Parameters
        ----------
        images
            Images of one sample.
        feature_modalities
            The modality of the feature columns.
        is_training
            Whether to process images in the training mode.
        load_only
            Whether to only load the data. Other processing steps may happen in dataset.__getitem__.

        Returns
        -------
        A dictionary containing one sample's processed images and their number.
        """
        images = {k: [v] if isinstance(v, str) else v for k, v in images.items()}
        if load_only:
            return self.load_one_sample(images, is_training)
        else:
            return self.process_one_sample(images, is_training)
