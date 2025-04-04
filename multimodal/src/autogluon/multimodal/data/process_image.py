import logging
import warnings
from io import BytesIO
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import PIL
import torch
from PIL import ImageFile
from torch import nn
from torchvision import transforms

from .utils import construct_image_processor, image_mean_std

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = PIL.Image.BICUBIC

from ..constants import CLIP, COLUMN, IMAGE, IMAGE_BASE64_STR, IMAGE_BYTEARRAY, IMAGE_VALID_NUM, TIMM_IMAGE
from ..models.clip import CLIPForImageText
from ..models.timm_image import TimmAutoModelForImagePrediction
from .collator import PadCollator, StackCollator
from .utils import extract_value_from_config

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
        norm_type: Optional[str] = None,
        size: Optional[int] = None,
        max_img_num_per_col: Optional[int] = 1,
        missing_value_strategy: Optional[str] = "zero",
        requires_column_info: bool = False,
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
        norm_type
            How to normalize an image. We now support:
            - inception
                Normalize image by IMAGENET_INCEPTION_MEAN and IMAGENET_INCEPTION_STD from timm
            - imagenet
                Normalize image by IMAGENET_DEFAULT_MEAN and IMAGENET_DEFAULT_STD from timm
            - clip
                Normalize image by mean (0.48145466, 0.4578275, 0.40821073) and
                std (0.26862954, 0.26130258, 0.27577711), used for CLIP.
        size
            The provided width / height of a square image.
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
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        logger.debug(f"image training transforms: {self.train_transforms}")
        logger.debug(f"image validation transforms: {self.val_transforms}")

        self.prefix = model.prefix
        self.missing_value_strategy = missing_value_strategy
        self.requires_column_info = requires_column_info
        self.size = None
        self.mean = None
        self.std = None
        if isinstance(model, CLIPForImageText):
            config = model.model.vision_model.config
        else:
            config = model.config

        if model is not None:
            self.size, self.mean, self.std = self.extract_default(config)
            if isinstance(model, TimmAutoModelForImagePrediction):
                if model.support_variable_input_size() and size is not None:
                    # We have detected that the model supports using an image size that is
                    # different from the pretrained model, e.g., ConvNets with global pooling
                    if size < self.size:
                        logger.warning(
                            f"The provided image size={size} is smaller than the default size "
                            f"of the pretrained backbone, which is {self.size}. "
                            f"Detailed configuration of the backbone is in {config}. "
                            f"You may like to double check your configuration."
                        )
                    self.size = size
            elif size is not None and size != self.size:
                logger.warning(
                    f"The model does not support using an image size that is different from the default size. "
                    f"Provided image size={size}. Default size={self.size}. "
                    f"Detailed model configuration={config}. We have ignored the provided image size."
                )
        if self.size is None:
            if size is not None:
                self.size = size
                logger.debug(f"using provided image size: {self.size}")
            else:
                raise ValueError("image size is missing")
        else:
            logger.debug(f"using detected image size: {self.size}")
        if self.mean is None or self.std is None:
            if norm_type is not None:
                self.mean, self.std = image_mean_std(norm_type)
                logger.debug(f"using provided normalization: {norm_type}")
            else:
                raise ValueError("image normalization mean and std are missing")
        else:
            logger.debug(f"using detected image normalization: {self.mean} and {self.std}")
        self.normalization = transforms.Normalize(self.mean, self.std)
        self.max_img_num_per_col = max_img_num_per_col
        if max_img_num_per_col <= 0:
            logger.debug(f"max_img_num_per_col {max_img_num_per_col} is reset to 1")
            max_img_num_per_col = 1
        self.max_img_num_per_col = max_img_num_per_col
        logger.debug(f"max_img_num_per_col: {max_img_num_per_col}")

        self.train_processor = construct_image_processor(
            image_transforms=self.train_transforms, size=self.size, normalization=self.normalization
        )
        self.val_processor = construct_image_processor(
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
                self.image_key: PadCollator(pad_val=0),
                self.image_valid_num_key: StackCollator(),
            }
        )

        return fn

    def extract_default(self, config=None):
        """
        Extract some default hyper-parameters, e.g., image size, mean, and std,
        from a pre-trained (timm or huggingface) checkpoint.

        Parameters
        ----------
        config
            Config of a pre-trained checkpoint.

        Returns
        -------
        image_size
            Image width/height.
        mean
            Image normalization mean.
        std
            Image normalizaiton std.
        """
        if self.prefix.lower().startswith(TIMM_IMAGE):
            image_size = config["input_size"][-1]
            mean = config["mean"]
            std = config["std"]
        elif self.prefix.lower().startswith(CLIP):
            extracted = extract_value_from_config(
                config=config.to_diff_dict(),
                keys=("image_size",),
            )
            if len(extracted) == 0:
                image_size = None
            elif len(extracted) >= 1:
                image_size = extracted[0]
                if isinstance(image_size, tuple):
                    image_size = image_size[-1]
            else:
                raise ValueError(f" more than one image_size values are detected: {extracted}")
            mean = None
            std = None
        else:
            raise ValueError(f"Unknown image processor prefix: {self.prefix}")
        return image_size, mean, std

    def process_one_sample(
        self,
        image_features: Dict[str, Union[List[str], List[bytearray]]],
        feature_modalities: Dict[str, List[str]],
        is_training: bool,
        image_mode: Optional[str] = "RGB",
    ) -> Dict:
        """
        Read images, process them, and stack them. One sample can have multiple images,
        resulting in a tensor of (n, 3, size, size), where n <= max_img_num_per_col is the available image number.

        Parameters
        ----------
        image_features
            One sample may have multiple image columns in a pd.DataFrame and multiple images
            inside each image column.
        feature_modalities
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
        images = []
        zero_images = []
        ret = {}
        column_start = 0

        for per_col_name, per_col_image_features in image_features.items():
            for img_feature in per_col_image_features[: self.max_img_num_per_col]:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=(
                            "Palette images with Transparency expressed in bytes should be converted to RGBA images"
                        ),
                    )
                    is_zero_img = False
                    try:
                        if feature_modalities.get(per_col_name) in [IMAGE_BYTEARRAY, IMAGE_BASE64_STR]:
                            image_feature = BytesIO(img_feature)
                        else:
                            image_feature = img_feature
                        with PIL.Image.open(image_feature) as img:
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
                    images.append(img)

            if self.requires_column_info:
                # only count the valid images since they are put ahead of the zero images in the below returning
                ret[f"{self.image_column_prefix}_{per_col_name}"] = np.array(
                    [column_start, len(images)], dtype=np.int64
                )
                column_start = len(images)

        ret.update(
            {
                self.image_key: torch.tensor([])
                if len(images + zero_images) == 0
                else torch.stack(images + zero_images, dim=0),
                self.image_valid_num_key: len(images),
            }
        )
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

        return self.process_one_sample(images, feature_modalities, is_training)

    def __getstate__(self):
        odict = self.__dict__.copy()  # get attribute dictionary
        del odict["train_processor"]  # remove augmenter to support pickle
        return odict

    def __setstate__(self, state):
        self.__dict__ = state
        if "train_transform_types" in state:  # backward compatible
            self.train_transforms = list(self.train_transform_types)
        if "val_transform_types" in state:
            self.val_transforms = list(self.val_transform_types)

        self.train_processor = construct_image_processor(
            image_transforms=self.train_transforms,
            size=self.size,
            normalization=self.normalization,
        )
