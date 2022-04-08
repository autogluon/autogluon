import logging
import torch
import warnings
from typing import Optional, List
from torchvision import transforms
import PIL
from .randaug import RandAugment
from timm import create_model
from timm.data.constants import (
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD,
    IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD,
)
from transformers import AutoConfig
from ..constants import (
    IMAGE, IMAGE_VALID_NUM, CLIP_IMAGE_MEAN,
    CLIP_IMAGE_STD, AUTOMM,
)
from .collator import Stack, Pad
from .utils import extract_value_from_config

logger = logging.getLogger(AUTOMM)


class ImageProcessor:
    """
    Prepare image data for the model specified by "prefix". For multiple models requiring image data,
    we need to create a ImageProcessor for each related model so that they will have independent input.
    """

    def __init__(
            self,
            prefix: str,
            train_transform_types: List[str],
            val_transform_types: List[str],
            checkpoint_name: Optional[str] = None,
            norm_type: Optional[str] = None,
            size: Optional[int] = None,
            max_img_num_per_col: Optional[int] = 1,
            missing_value_strategy: Optional[str] = False,
    ):
        """
        Parameters
        ----------
        prefix
            The prefix connecting a processor to its corresponding model.
        train_transform_types
            A list of image transforms used in training. Note that the transform order matters.
        val_transform_types
            A list of image transforms used in validation/test/prediction. Note that the transform order matters.
        checkpoint_name
            Name of a pre-trained checkpoint, which can be from either timm or huggingface.
            It is required to extract some default hyper-parameters.
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
            The width / height of a square image.
        max_img_num_per_col
            The maximum number of images one sample can have.
        missing_value_strategy
            How to deal with a missing image. We now support:
            - skip
                Skip this sample
            -zero
                Use an image with zero pixels.
        """
        self.checkpoint_name = checkpoint_name
        self.prefix = prefix
        self.train_transform_types = train_transform_types
        self.val_transform_types = val_transform_types
        logger.debug(f"image training transform type: {train_transform_types}")
        logger.debug(f"image validation transform type: {val_transform_types}")
        self.missing_value_strategy = missing_value_strategy
        self.size = None
        self.mean = None
        self.std = None

        if checkpoint_name is not None:
            self.size, self.mean, self.std = self.extract_default(checkpoint_name)
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
                self.mean, self.std = self.mean_std(norm_type)
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

        self.train_processor = self.construct_processor(self.train_transform_types)
        self.val_processor = self.construct_processor(self.val_transform_types)

    def collate_fn(self) -> dict:
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
        fn.update({f"{self.prefix}_{IMAGE}": Pad(pad_val=0)})
        fn.update({f"{self.prefix}_{IMAGE_VALID_NUM}": Stack()})
        return fn

    @staticmethod
    def mean_std(norm_type: str):
        """
        Get image normalization mean and std by its name.

        Parameters
        ----------
        norm_type
            Name of image normalization.

        Returns
        -------
        Normalization mean and std.
        """
        if norm_type == "inception":
            return IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
        elif norm_type == "imagenet":
            return IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
        elif norm_type == "clip":
            return CLIP_IMAGE_MEAN, CLIP_IMAGE_STD
        else:
            raise ValueError(f"unknown image normalization: {norm_type}")

    @staticmethod
    def extract_default(checkpoint_name):
        """
        Extract some default hyper-parameters, e.g., image size, mean, and std,
        from a pre-trained (timm or huggingface) checkpoint.

        Parameters
        ----------
        checkpoint_name
            Name of a pre-trained checkpoint.

        Returns
        -------
        image_size
            Image width/height.
        mean
            Image normalization mean.
        std
            Image normalizaiton std.
        """
        try:  # timm checkpoint
            model = create_model(
                checkpoint_name,
                pretrained=True,
                num_classes=0,
            )
            image_size = model.default_cfg["input_size"][-1]
            mean = model.default_cfg["mean"]
            std = model.default_cfg["std"]
        except Exception as exp1:
            try:  # huggingface checkpoint
                config = AutoConfig.from_pretrained(checkpoint_name).to_diff_dict()
                extracted = extract_value_from_config(
                    config=config,
                    keys=("image_size",),
                )
                if len(extracted) == 0:
                    image_size = None
                elif len(extracted) == 1:
                    image_size = extracted[0]
                    if isinstance(image_size, tuple):
                        image_size = image_size[-1]
                else:
                    raise ValueError(
                        f" more than one image_size values are detected: {extracted}"
                    )
                mean = None
                std = None
            except Exception as exp2:
                raise ValueError(f"cann't load checkpoint_name {checkpoint_name}") from exp2

        return image_size, mean, std

    def construct_processor(
            self,
            transform_types: List[str],
    ) -> transforms.Compose:
        """
        Build up an image processor from the provided list of transform types.

        Parameters
        ----------
        transform_types
            A list of image transform types.

        Returns
        -------
        A torchvision transform.
        """
        processor = []
        for trans_type in transform_types:
            if trans_type == "resize_to_square":
                processor.append(transforms.Resize((self.size, self.size)))
            elif trans_type == "resize_shorter_side":
                processor.append(transforms.Resize(self.size))
            elif trans_type == "center_crop":
                processor.append(transforms.CenterCrop(self.size))
            elif trans_type == "horizontal_flip":
                processor.append(transforms.RandomHorizontalFlip())
            elif trans_type == "randaug":
                processor.append(RandAugment(2, 9))
            else:
                raise ValueError(f"unknown transform type: {trans_type}")

        processor.append(transforms.ToTensor())
        processor.append(self.normalization)
        return transforms.Compose(processor)

    def process_one_sample(
            self,
            image_paths: List[List[str]],
            is_training: bool,
    ) -> dict:
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
        images = []
        zero_images = []
        for per_col_image_paths in image_paths:
            for img_path in per_col_image_paths[:self.max_img_num_per_col]:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="Palette images with Transparency "
                                "expressed in bytes should be "
                                "converted to RGBA images"
                    )
                    is_zero_img = False
                    try:
                        img = PIL.Image.open(img_path).convert("RGB")
                    except Exception as e:
                        if self.missing_value_strategy.lower() == "zero":
                            logger.debug(f"Using a zero image due to '{e}'")
                            img = PIL.Image.new("RGB", (self.size, self.size), color=0)
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

        return {
            f"{self.prefix}_{IMAGE}": torch.stack(images+zero_images),
            f"{self.prefix}_{IMAGE_VALID_NUM}": len(images),
        }

    def __call__(
            self,
            all_image_paths: List[List[List[str]]],
            idx: int,
            is_training: bool,
    ) -> dict:
        """
        Obtain one sample's images and customized them for a specific model.

        Parameters
        ----------
        all_image_paths
            Paths of all the images in a dataset.
        idx
            The sample index in a dataset.
        is_training
            Whether to process images in the training mode.

        Returns
        -------
        A dictionary containing one sample's processed images and their number.
        """
        per_sample_paths = [per_column_paths[idx] for per_column_paths in all_image_paths]
        return self.process_one_sample(per_sample_paths, is_training)
