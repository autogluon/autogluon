import pytest
import PIL
from autogluon.text.automm.data.process_image import ImageProcessor

from datasets import (
    PetFinderDataset,
    HatefulMeMesDataset,
    AEDataset,
)

ALL_DATASETS = {
    "petfinder": PetFinderDataset,
    "hateful_memes": HatefulMeMesDataset,
    "ae": AEDataset,
}

@pytest.mark.parametrize(
    "augmentations",
    [
        {
            "model.timm_image.train_transform_types": ["resize_to_square","center_crop"],
            "model.timm_image.val_transform_types": ["resize_shorter_side","center_crop"]
        },

        {
            "model.timm_image.train_transform_types": ["resize_shorter_side","center_crop"],
            "model.timm_image.val_transform_types": ["resize_shorter_side","center_crop"]
        },

        {
            "model.timm_image.train_transform_types": ["resize_shorter_side","center_crop","horizontal_flip","vertical_flip"],
            "model.timm_image.val_transform_types": ["resize_shorter_side","center_crop"]
        },

        {
            "model.timm_image.train_transform_types": ["resize_shorter_side","center_crop","affine"],
            "model.timm_image.val_transform_types": ["resize_shorter_side","center_crop"]
        },

        {
            "model.timm_image.train_transform_types": ["resize_shorter_side","center_crop","colorjitter"],
            "model.timm_image.val_transform_types": ["resize_shorter_side","center_crop"]
        },

        {
            "model.timm_image.train_transform_types": ["resize_shorter_side","center_crop","randaug"],
            "model.timm_image.val_transform_types": ["resize_shorter_side","center_crop"]
        },
    ]
)

def test_data_process_image(augmentations):
    dataset = ALL_DATASETS["petfinder"]()
    image = PIL.Image.open(dataset.train_df.Images[0]).convert("RGB")

    data_processors=ImageProcessor(
        image_column_names=['train_df.Images'],
        prefix="timm_image",
        train_transform_types=augmentations["model.timm_image.train_transform_types"],
        val_transform_types=augmentations["model.timm_image.val_transform_types"],
        size=224,
        norm_type="imagenet",
    )

    transimage = data_processors.train_processor(image)

    assert len(data_processors.train_processor.transforms) == len(augmentations["model.timm_image.train_transform_types"])+2











