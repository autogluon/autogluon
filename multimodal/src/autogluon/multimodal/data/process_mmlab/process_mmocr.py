import logging
from typing import Dict, List, Optional

import numpy as np
import PIL
from PIL import ImageFile
from torch import nn

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = PIL.Image.BICUBIC

from ...constants import AUTOMM
from ..utils import is_rois_input
from .process_mmlab_base import MMLabProcessor

logger = logging.getLogger(AUTOMM)
ImageFile.LOAD_TRUNCATED_IMAGES = True


class MMOcrProcessor(MMLabProcessor):
    """
    Prepare data for mmocr models.
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
        from ...utils import CollateMMOcr

        super().__init__(
            model=model,
            collate_func=CollateMMOcr,
            max_img_num_per_col=max_img_num_per_col,
            missing_value_strategy=missing_value_strategy,
            requires_column_info=requires_column_info,
        )

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
        # TODO: modify for MMOCR
        mm_data = dict(img_prefix=None, bbox_fields=[])
        ret = {}

        for per_col_name, per_col_content in image_paths.items():
            if is_rois_input(per_col_content):
                rois = np.array(per_col_content)
                mm_data["ann_info"] = dict(bboxes=rois[:, :4], labels=rois[:, 4])
            else:
                with PIL.Image.open(per_col_content[0]) as img:
                    mm_data["img_info"] = dict(filename=per_col_content[0], height=img.height, width=img.width)
        if self.requires_column_info:
            pass  # TODO

        ret.update({self.image_key: self.train_processor(mm_data) if is_training else self.val_processor(mm_data)})

        return ret
