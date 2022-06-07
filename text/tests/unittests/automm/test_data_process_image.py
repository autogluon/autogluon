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
            "model.timm_image.train_transform_types": ["resize_shorter_side","center_crop","affine","color_jitter","randaug"],
            "model.timm_image.val_transform_types": ["resize_shorter_side","center_crop"]
        },

        {
            "model.timm_image.train_transform_types": ["resize_shorter_side","center_crop","affine(15, (0.1, 0.1), (0.9, 1.1))"],
            "model.timm_image.val_transform_types": ["resize_shorter_side","center_crop"]
        },

        {
            "model.timm_image.train_transform_types": ["resize_shorter_side","center_crop","affine({'degrees': 15,'translate': (0.1, 0.1), 'scale': (0.9, 1.1)})"],
            "model.timm_image.val_transform_types": ["resize_shorter_side","center_crop"]
        },

        {
            "model.timm_image.train_transform_types": ["resize_shorter_side","center_crop","color_jitter(0.2, 0.1, 0.1)"],
            "model.timm_image.val_transform_types": ["resize_shorter_side","center_crop"]
        },

        {
            "model.timm_image.train_transform_types": ["resize_shorter_side","center_crop","color_jitter({'brightness': 0.2, 'contrast': 0.1, 'saturation': 0.1})"],
            "model.timm_image.val_transform_types": ["resize_shorter_side","center_crop"]
        },

        {
            "model.timm_image.train_transform_types": ["resize_shorter_side","center_crop","randaug(2, 7)"],
            "model.timm_image.val_transform_types": ["resize_shorter_side","center_crop"]
        },
    ]
)

def test_data_process_image(augmentations):
    dataset = ALL_DATASETS["petfinder"]()
    image = PIL.Image.open(dataset.train_df.Images[0]).convert("RGB")

    image_processor=ImageProcessor(
        image_column_names=['Images'],
        prefix="timm_image",
        train_transform_types=augmentations["model.timm_image.train_transform_types"],
        val_transform_types=augmentations["model.timm_image.val_transform_types"],
        size=224,
        norm_type="imagenet",
    )

    transformed_image = image_processor.train_processor(image)

    assert len(image_processor.train_processor.transforms) == len(augmentations["model.timm_image.train_transform_types"])+2
