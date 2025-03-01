import ast
import copy
import logging
import random
import warnings
from io import BytesIO
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import PIL
import torch
from omegaconf import ListConfig
from PIL import ImageFile
from torch import nn
from torchvision import transforms

from .randaug import RandAugment
from .trivial_augmenter import TrivialAugment

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
    NEAREST = InterpolationMode.NEAREST
except ImportError:
    BICUBIC = PIL.Image.BICUBIC
    NEAREST = PIL.Image.NEAREST

from ..constants import COLUMN, IMAGE, IMAGE_BASE64_STR, IMAGE_BYTEARRAY, IMAGE_VALID_NUM
from .collator import PadCollator, StackCollator

logger = logging.getLogger(__name__)
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageProcessor:
    """
    Prepare image data for the model specified by "prefix". For multiple models requiring image data,
    we need to create a ImageProcessor for each related model so that they will have independent input.
    """

    def __init__(
        self,
        model: nn.Module,
        train_transforms: Union[List[str], Callable, List[Callable]],
        val_transforms: Union[List[str], Callable, List[Callable]],
        max_image_num_per_column: Optional[int] = 1,
        missing_value_strategy: Optional[str] = "zero",
        requires_column_info: Optional[bool] = False,
        dropout: Optional[float] = 0,
    ):
        """
        Parameters
        ----------
        model
            The model for which this processor would be created.
        train_transforms
            A list of image transforms used in training. Note that the transform order matters.
        val_transforms
            A list of image transforms used in validation/test/prediction. Note that the transform order matters.
        max_image_num_per_column
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
        logger.debug(f"initializing image processor for model {model.prefix}")
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        logger.debug(f"image training transforms: {self.train_transforms}")
        logger.debug(f"image validation transforms: {self.val_transforms}")

        self.prefix = model.prefix
        self.missing_value_strategy = missing_value_strategy
        self.requires_column_info = requires_column_info
        assert 0 <= dropout <= 1
        if dropout > 0:
            logger.debug(f"image dropout probability: {dropout}")
        self.dropout = dropout
        self.size = model.image_size
        self.mean = model.image_mean
        self.std = model.image_std

        self.normalization = transforms.Normalize(self.mean, self.std)
        if max_image_num_per_column <= 0:
            logger.debug(f"max_image_num_per_column {max_image_num_per_column} is reset to 1")
            max_image_num_per_column = 1
        self.max_image_num_per_column = max_image_num_per_column
        logger.debug(f"max_image_num_per_column: {max_image_num_per_column}")

        self.train_processor = self.construct_image_processor(
            image_transforms=self.train_transforms, size=self.size, normalization=self.normalization
        )
        self.val_processor = self.construct_image_processor(
            image_transforms=self.val_transforms, size=self.size, normalization=self.normalization
        )

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
        vary from sample to sample. Samples with fewer images will be padded zeros.
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
                self.image_key: PadCollator(pad_val=0),
                self.image_valid_num_key: StackCollator(),
            }
        )

        return fn

    def process_one_sample(
        self,
        images: Dict[str, Union[List[str], List[bytearray]]],
        sub_dtypes: Dict[str, str],
        is_training: bool,
        image_mode: Optional[str] = "RGB",
    ) -> Dict:
        """
        Read images, process them, and stack them. One sample can have multiple images,
        resulting in a tensor of (n, 3, size, size), where n <= max_image_num_per_column is the available image number.

        Parameters
        ----------
        images
            One sample may have multiple image columns in a pd.DataFrame and multiple images
            inside each image column.
        sub_dtypes
            What modality each column belongs to.
        is_training
            Whether to process images in the training mode.
        image_mode
            A string which defines the type and depth of a pixel in the image.
            For example, RGB, RGBA, CMYK, and etc.

        Returns
        -------
        A dictionary containing one sample's images and their number.
        """
        valid_images = []
        zero_images = []
        ret = {}
        column_start = 0

        for per_col_name, per_col_image_raw in images.items():
            for img_raw in per_col_image_raw[: self.max_image_num_per_column]:
                if is_training and self.dropout > 0 and random.uniform(0, 1) <= self.dropout:
                    img = PIL.Image.new(image_mode, (self.size, self.size), color=0)
                    is_zero_img = True
                else:
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore",
                            message=(
                                "Palette images with Transparency expressed in bytes should be converted to RGBA images"
                            ),
                        )
                        is_zero_img = False
                        try:
                            if sub_dtypes.get(per_col_name) in [IMAGE_BYTEARRAY, IMAGE_BASE64_STR]:
                                img_raw = BytesIO(img_raw)

                            with PIL.Image.open(img_raw) as img:
                                img = img.convert(image_mode)
                        except Exception as e:
                            if self.missing_value_strategy.lower() == "zero":
                                logger.debug(f"Using a zero image due to '{e}'")
                                img = PIL.Image.new(image_mode, (self.size, self.size), color=0)
                                is_zero_img = True
                            else:
                                raise e
                if is_training:
                    img = self.train_processor(img)
                else:
                    img = self.val_processor(img)

                if is_zero_img:
                    zero_images.append(img)
                else:
                    valid_images.append(img)

            if self.requires_column_info:
                # only count the valid images since they are put ahead of the zero images in the below returning
                ret[f"{self.image_column_prefix}_{per_col_name}"] = np.array(
                    [column_start, len(valid_images)], dtype=np.int64
                )
                column_start = len(valid_images)

        ret.update(
            {
                self.image_key: torch.tensor([])
                if len(valid_images + zero_images) == 0
                else torch.stack(valid_images + zero_images, dim=0),
                self.image_valid_num_key: len(valid_images),
            }
        )
        return ret

    @staticmethod
    def get_image_transform_funcs(transform_types: Union[List[str], ListConfig, List[Callable]], size: int):
        """
        Parse a list of transform strings into callable objects.

        Parameters
        ----------
        transform_types
            A list of transforms, which can be strings or callable objects.
        size
            Image size.

        Returns
        -------
        A list of transform objects.
        """
        image_transforms = []

        if not transform_types:
            return image_transforms

        if isinstance(transform_types, ListConfig):
            transform_types = list(transform_types)
        elif not isinstance(transform_types, list):
            transform_types = [transform_types]

        if all([isinstance(trans_type, str) for trans_type in transform_types]):
            pass
        elif all([isinstance(trans_type, Callable) for trans_type in transform_types]):
            return copy.copy(transform_types)
        else:
            raise ValueError(
                f"transform_types {transform_types} contain neither all strings nor all callable objects."
            )

        for trans_type in transform_types:
            args = None
            kargs = None
            if "(" in trans_type:
                trans_mode = trans_type[0 : trans_type.find("(")]
                if "{" in trans_type:
                    kargs = ast.literal_eval(trans_type[trans_type.find("{") : trans_type.rfind(")")])
                else:
                    args = ast.literal_eval(trans_type[trans_type.find("(") :])
            else:
                trans_mode = trans_type

            if trans_mode == "resize_to_square":
                image_transforms.append(transforms.Resize((size, size), interpolation=BICUBIC))
            elif trans_mode == "resize_gt_to_square":
                image_transforms.append(transforms.Resize((size, size), interpolation=NEAREST))
            elif trans_mode == "resize_shorter_side":
                image_transforms.append(transforms.Resize(size, interpolation=BICUBIC))
            elif trans_mode == "center_crop":
                image_transforms.append(transforms.CenterCrop(size))
            elif trans_mode == "random_resize_crop":
                image_transforms.append(transforms.RandomResizedCrop(size))
            elif trans_mode == "random_horizontal_flip":
                image_transforms.append(transforms.RandomHorizontalFlip())
            elif trans_mode == "random_vertical_flip":
                image_transforms.append(transforms.RandomVerticalFlip())
            elif trans_mode == "color_jitter":
                if kargs is not None:
                    image_transforms.append(transforms.ColorJitter(**kargs))
                elif args is not None:
                    image_transforms.append(transforms.ColorJitter(*args))
                else:
                    image_transforms.append(transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1))
            elif trans_mode == "affine":
                if kargs is not None:
                    image_transforms.append(transforms.RandomAffine(**kargs))
                elif args is not None:
                    image_transforms.append(transforms.RandomAffine(*args))
                else:
                    image_transforms.append(
                        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1))
                    )
            elif trans_mode == "randaug":
                if kargs is not None:
                    image_transforms.append(RandAugment(**kargs))
                elif args is not None:
                    image_transforms.append(RandAugment(*args))
                else:
                    image_transforms.append(RandAugment(2, 9))
            elif trans_mode == "trivial_augment":
                image_transforms.append(TrivialAugment(IMAGE, 30))
            else:
                raise ValueError(f"unknown transform type: {trans_mode}")

        return image_transforms

    def construct_image_processor(
        self,
        image_transforms: Union[List[Callable], List[str]],
        size: int,
        normalization,
    ) -> transforms.Compose:
        """
        Build up an image processor from the provided list of transform types.

        Parameters
        ----------
        image_transforms
            A list of image transform types.
        size
            Image size.
        normalization
            A transforms.Normalize object. When the image is ground truth image, 'normalization=None' should be specified.

        Returns
        -------
        A transforms.Compose object.
        """
        image_transforms = self.get_image_transform_funcs(transform_types=image_transforms, size=size)
        if not any([isinstance(trans, transforms.ToTensor) for trans in image_transforms]):
            image_transforms.append(transforms.ToTensor())
        if (
            not any([isinstance(trans, transforms.Normalize) for trans in image_transforms])
            and normalization is not None
        ):
            image_transforms.append(normalization)
        return transforms.Compose(image_transforms)

    def __call__(
        self,
        images: Dict[str, List[str]],
        sub_dtypes: Dict[str, str],
        is_training: bool,
    ) -> Dict:
        """
        Obtain one sample's images and customized them for a specific model.

        Parameters
        ----------
        images
            Images of one sample.
        sub_dtypes
            The sub data types of all image columns.
        is_training
            Whether to process images in the training mode.

        Returns
        -------
        A dictionary containing one sample's processed images and their number.
        """
        images = {k: [v] if isinstance(v, str) else v for k, v in images.items()}

        return self.process_one_sample(images=images, sub_dtypes=sub_dtypes, is_training=is_training)

    def __getstate__(self):
        odict = self.__dict__.copy()  # get attribute dictionary
        del odict["train_processor"]  # remove augmenter to support pickle
        return odict

    def __setstate__(self, state):
        self.__dict__ = state
        self.train_processor = self.construct_image_processor(
            image_transforms=self.train_transforms,
            size=self.size,
            normalization=self.normalization,
        )
