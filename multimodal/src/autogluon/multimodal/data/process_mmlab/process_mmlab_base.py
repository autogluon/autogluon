import logging
import warnings
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import PIL
from PIL import ImageFile
from torch import nn

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = PIL.Image.BICUBIC

from ...constants import AUTOMM, COLUMN, IMAGE, IMAGE_VALID_NUM, MMDET_IMAGE
from ..collator import StackCollator
from ..utils import is_rois_input

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import mmcv
    from mmcv.transforms import Compose
except ImportError as e:
    mmcv = None

try:
    import mmdet
    from mmdet.datasets.transforms import ImageToTensor
except ImportError as e:
    mmdet = None

try:
    import mmocr
except ImportError:
    mmocr = None


logger = logging.getLogger(__name__)
ImageFile.LOAD_TRUNCATED_IMAGES = True


class MMLabProcessor:
    """
    The base class to prepare data for mmlab models specified by "prefix".
    Child class shall provide its specific collate function in __init__.
    """

    def __init__(
        self,
        model: nn.Module,
        collate_func: Callable,
        max_img_num_per_col: Optional[int] = 1,
        missing_value_strategy: Optional[str] = "zero",
        requires_column_info: bool = False,
    ):
        """
        Parameters
        ----------
        model
            The model using this data processor.
        collate_func
            The collate function to use for this processor
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
        from ...utils import check_if_packages_installed

        check_if_packages_installed(package_names=["mmcv"])

        self.prefix = model.prefix
        self.missing_value_strategy = missing_value_strategy
        self.requires_column_info = requires_column_info
        self.collate_func = collate_func

        self.max_img_num_per_col = max_img_num_per_col
        if max_img_num_per_col <= 0:
            logger.debug(f"max_img_num_per_col {max_img_num_per_col} is reset to 1")
            max_img_num_per_col = 1
        self.max_img_num_per_col = max_img_num_per_col
        logger.debug(f"max_img_num_per_col: {max_img_num_per_col}")

        if self.prefix.lower().startswith(MMDET_IMAGE):
            check_if_packages_installed(package_names=["mmdet"])
        else:
            assert mmocr is not None, "Please install MMOCR by: pip install mmocr."
        self.cfg = model.model.cfg
        self.val_processor = Compose(self.cfg.test_pipeline)
        self.train_processor = Compose(self.cfg.train_pipeline)

    @property
    def image_key(self):
        return f"{self.prefix}_{IMAGE}"

    @property
    def image_valid_num_key(self):
        return f"{self.prefix}_{IMAGE_VALID_NUM}"

    @property
    def image_column_prefix(self):
        return f"{self.image_key}_{COLUMN}"

    def collate_fn(self, image_column_names: Optional[List] = None, per_gpu_batch_size: Optional[int] = None) -> Dict:
        """
        Collate images into a batch. Here it pads images since the image number may
        vary from sample to sample. Samples with less images will be padded zeros.
        The valid image numbers of samples will be stacked into a vector.
        This function will be used when creating Pytorch DataLoader.

        Returns
        -------
        A dictionary containing one model's collator function for image data.
        """

        fn = {}
        if self.requires_column_info:
            assert image_column_names, "Empty image column names."
            for col_name in image_column_names:
                fn[f"{self.image_column_prefix}_{col_name}"] = StackCollator()

        fn.update(
            {
                self.image_key: self.collate_func(samples_per_gpu=per_gpu_batch_size),
            }
        )

        return fn

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
        mm_data = dict(img_prefix=None, bbox_fields=[], mask_fields=[])
        ret = {}

        for per_col_name, per_col_content in image_paths.items():
            if is_rois_input(per_col_content):
                rois = np.array(per_col_content)
                # TODO: add gt masks
                mm_data["ann_info"] = dict(bboxes=rois[:, :4], labels=rois[:, 4], masks=[])
            else:
                with PIL.Image.open(per_col_content[0]) as img:
                    mm_data["img_info"] = dict(filename=per_col_content[0], height=img.height, width=img.width)
        if self.requires_column_info:
            pass  # TODO

        ret.update({self.image_key: self.train_processor(mm_data) if is_training else self.val_processor(mm_data)})

        return ret

    def __call__(
        self,
        images: Dict[str, List[str]],
        feature_modalities: Dict[str, Union[int, float, list]],
        is_training: bool,
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

        Returns
        -------
        A dictionary containing one sample's processed images and their number.
        """
        images = {k: [v] if isinstance(v, str) else v for k, v in images.items()}

        return self.process_one_sample(images, is_training)

    def __getstate__(self):
        odict = self.__dict__.copy()  # get attribute dictionary
        return odict

    def __setstate__(self, state):
        self.__dict__ = state
