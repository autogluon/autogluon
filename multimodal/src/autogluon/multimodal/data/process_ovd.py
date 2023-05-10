import logging
import warnings
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import PIL
from PIL import ImageFile
from torch import nn

from ..constants import COLUMN, IMAGE, IMAGE_META, IMAGE_VALID_NUM, PROMPT, TEXT
from .collator import ListCollator, PadCollator
from .utils import is_rois_input

logger = logging.getLogger(__name__)
ImageFile.LOAD_TRUNCATED_IMAGES = True


class OVDProcessor:
    """
    Prepare data for open vocabulary detection models specified by "prefix".
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

        self.prefix = model.prefix
        self.missing_value_strategy = missing_value_strategy
        self.requires_column_info = requires_column_info

        self.max_img_num_per_col = max_img_num_per_col
        if max_img_num_per_col <= 0:
            logger.debug(f"max_img_num_per_col {max_img_num_per_col} is reset to 1")
            max_img_num_per_col = 1
        self.max_img_num_per_col = max_img_num_per_col
        logger.debug(f"max_img_num_per_col: {max_img_num_per_col}")

    @property
    def image_key(self):
        return f"{self.prefix}_{IMAGE}"

    @property
    def image_meta_key(self):
        return f"{self.prefix}_{IMAGE_META}"

    @property
    def prompt_key(self):
        return f"{self.prefix}_{PROMPT}"

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
            return NotImplementedError(
                f"requires_column_info={self.requires_column_info} not implemented for OVD tasks."
            )

        fn.update(
            {
                self.image_key: PadCollator(pad_val=0),
                self.prompt_key: ListCollator(),
                self.image_meta_key: ListCollator(),
            }
        )
        return fn

    def process_one_sample(
        self,
        texts_and_images: Dict[str, Union[str, List[str]]],
        feature_modalities: Dict[str, List[str]],
        is_training: bool,
    ) -> Dict:
        """
        Read images, process them, and stack them. One sample can have multiple images,
        resulting in a tensor of (n, 3, size, size), where n <= max_img_num_per_col is the available image number.

        Parameters
        ----------
        texts_and_images
            The input could be image or text (image captions) for open vocabulary detection.
        feature_modalities
            What modality each column belongs to.
        is_training
            Whether to process images in the training mode.

        Returns
        -------
        A dictionary containing one sample's images and their number.
        """
        ret = {}
        image_data = {}

        for per_col_name, per_col_content in texts_and_images.items():
            if feature_modalities.get(per_col_name) == TEXT:
                ret[self.prompt_key] = per_col_content[0]
            elif is_rois_input(per_col_content):
                raise NotImplementedError(
                    "Finetuning/Evaluation with ground truth labels are not implemented for OVD yet"
                )  # TODO
            else:
                with PIL.Image.open(per_col_content[0]) as img:
                    image_data[self.image_key] = dict(filename=per_col_content[0], height=img.height, width=img.width)
                    ret[self.image_meta_key] = [img.width, img.height]
        if self.requires_column_info:
            pass  # TODO

        ret.update(
            {self.image_key: self.train_processor(image_data) if is_training else self.val_processor(image_data)}
        )

        return ret

    def train_processor(self, image_data):
        raise NotImplementedError("Training mode not implemented for OVD yet.")

    def val_processor(self, image_data):
        import groundingdino.datasets.transforms as T

        image_path = image_data[self.image_key]["filename"]

        image_pil = PIL.Image.open(image_path).convert("RGB")

        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image, _ = transform(image_pil, None)  # 3, h, w
        return image

    def __call__(
        self,
        texts_and_images: Dict[str, Union[str, List[str]]],
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
        texts_and_images = {k: [v] if isinstance(v, str) else v for k, v in texts_and_images.items()}

        return self.process_one_sample(texts_and_images, feature_modalities, is_training)

    def __getstate__(self):
        odict = self.__dict__.copy()  # get attribute dictionary
        return odict

    def __setstate__(self, state):
        self.__dict__ = state
