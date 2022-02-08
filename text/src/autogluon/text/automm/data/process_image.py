from typing import Optional, List
import torch
import warnings
from torchvision import transforms
from PIL import Image
from .randaug import RandAugment
from timm import create_model
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, \
    IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from transformers import AutoConfig
from ..constants import IMAGE, IMAGE_VALID_NUM
from .collator import Stack, Pad
from .utils import get_default_config_value


class ImageProcessor:
    def __init__(
            self,
            prefix: str,
            train_transform_types: List[str],
            val_transform_types: List[str],
            checkpoint_name: Optional[str] = None,
            norm_type: Optional[str] = None,
            size: Optional[int] = None,
            max_img_num_per_col: Optional[int] = 1,
    ):
        self.prefix = prefix
        self.train_transform_types = train_transform_types
        self.val_transform_types = val_transform_types
        print(f"image training transform type: {train_transform_types}")
        print(f"image validation transform type: {val_transform_types}")

        if checkpoint_name is not None:
            self.size, self.mean, self.std = self.extract_default(checkpoint_name)
        if self.size is None:
            if size is not None:
                self.size = size
                print(f"using provided image size: {self.size}")
            else:
                raise ValueError("image size is missing")
        else:
            print(f"using detected image size: {self.size}")
        if self.mean is None or self.std is None:
            if norm_type is not None:
                self.mean, self.std = self.mean_std(norm_type)
                print(f"using provided normalization: {norm_type}")
            else:
                raise ValueError("image normalization mean and std are missing")
        else:
            print(f"using detected image normalization: {self.mean} and {self.std}")
        self.normalization = transforms.Normalize(self.mean, self.std)
        if max_img_num_per_col <= 0:
            print(f"max_img_num_per_col {max_img_num_per_col} is reset to 1")
            max_img_num_per_col = 1
        self.max_img_num_per_col = max_img_num_per_col
        print(f"max_img_num_per_col: {max_img_num_per_col}")

        self.train_processor = self.construct_processor(self.train_transform_types)
        self.val_processor = self.construct_processor(self.val_transform_types)

    def collate_fn(self) -> dict:
        fn = {}
        fn.update({f"{self.prefix}_{IMAGE}": Pad(pad_val=0)})
        fn.update({f"{self.prefix}_{IMAGE_VALID_NUM}": Stack()})
        return fn

    @staticmethod
    def mean_std(norm_type: str):
        if norm_type == "inception":
            return IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
        elif norm_type == "imagenet":
            return IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
        elif norm_type == "clip":
            return (0.48145466, 0.4578275, 0.40821073),\
                   (0.26862954, 0.26130258, 0.27577711)
        else:
            raise ValueError(f"unknown image normalization: {norm_type}")

    @staticmethod
    def extract_default(checkpoint_name):
        try:
            model = create_model(
                checkpoint_name,
                pretrained=True,
                num_classes=0,
            )
            image_size = model.default_cfg["input_size"][-1]
            mean = model.default_cfg["mean"]
            std = model.default_cfg["std"]
        except Exception as exp1:
            try:
                config = AutoConfig.from_pretrained(checkpoint_name).to_diff_dict()
                extracted = get_default_config_value(
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
        images = []
        for per_col_image_paths in image_paths:
            for img_path in per_col_image_paths[:self.max_img_num_per_col]:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="Palette images with Transparency "
                                "expressed in bytes should be "
                                "converted to RGBA images"
                    )
                    img = Image.open(img_path).convert("RGB")
                if is_training:
                    img = self.train_processor(img)
                else:
                    img = self.val_processor(img)
                images.append(img)

        return {
            f"{self.prefix}_{IMAGE}": torch.stack(images),
            f"{self.prefix}_{IMAGE_VALID_NUM}": len(images),
        }

    def __call__(
            self,
            all_image_paths: List[List[List[str]]],
            idx: int,
            is_training: bool,
    ) -> dict:

        per_sample_paths = [per_column_paths[idx] for per_column_paths in all_image_paths]
        return self.process_one_sample(per_sample_paths, is_training)



