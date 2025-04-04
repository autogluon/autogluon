import PIL
import pytest
import torch as th

from autogluon.multimodal.data.process_image import ImageProcessor
from autogluon.multimodal.models.timm_image import TimmAutoModelForImagePrediction
from autogluon.multimodal.utils.misc import shopee_dataset


@pytest.mark.parametrize(
    "checkpoint_name,provided_size,expected_size",
    [
        ("swinv2_tiny_window8_256", 300, 256),  # SwinTransformer
        ("convnext_nano", 300, 300),  # ConvNext
        ("resnet50", 300, 300),  # ResNet
        ("regnety_004", 300, 300),  # RegNet
        ("fbnetv3_b", 300, 300),  # MobileNetV3
        ("tf_efficientnet_b2", 300, 300),
    ],
)  # EfficientNet
def test_variable_input_size_backbone(checkpoint_name, provided_size, expected_size):
    download_dir = "./ag_automm_tutorial_imgcls"
    train_df, test_df = shopee_dataset(download_dir)

    train_transforms = ["resize_shorter_side", "center_crop", "trivial_augment"]
    val_transforms = ["resize_shorter_side", "center_crop"]

    timm_model = TimmAutoModelForImagePrediction(
        prefix="timm_image", checkpoint_name=checkpoint_name, pretrained=False
    )

    image_processor = ImageProcessor(
        model=timm_model,
        train_transforms=train_transforms,
        val_transforms=val_transforms,
        size=provided_size,
    )
    ret = image_processor.process_one_sample(
        {"image": train_df["image"].tolist()}, feature_modalities={"image": "image"}, is_training=True
    )
    assert ret["timm_image_image"].shape[-1] == expected_size
    out = timm_model(
        th.unsqueeze(ret[timm_model.image_key], dim=0),
        th.unsqueeze(th.Tensor(ret[timm_model.image_valid_num_key]), dim=0),
    )


@pytest.mark.parametrize(
    "augmentations",
    [
        {
            "model.timm_image.train_transforms": ["resize_to_square", "center_crop"],
            "model.timm_image.val_transforms": ["resize_shorter_side", "center_crop"],
        },
        {
            "model.timm_image.train_transforms": ["resize_shorter_side", "center_crop"],
            "model.timm_image.val_transforms": ["resize_shorter_side", "center_crop"],
        },
        {
            "model.timm_image.train_transforms": [
                "resize_shorter_side",
                "center_crop",
                "random_horizontal_flip",
                "random_vertical_flip",
            ],
            "model.timm_image.val_transforms": ["resize_shorter_side", "center_crop"],
        },
        {
            "model.timm_image.train_transforms": [
                "resize_shorter_side",
                "center_crop",
                "affine",
                "color_jitter",
                "randaug",
            ],
            "model.timm_image.val_transforms": ["resize_shorter_side", "center_crop"],
        },
        {
            "model.timm_image.train_transforms": [
                "resize_shorter_side",
                "center_crop",
                "affine(15, (0.1, 0.1), (0.9, 1.1))",
            ],
            "model.timm_image.val_transforms": ["resize_shorter_side", "center_crop"],
        },
        {
            "model.timm_image.train_transforms": [
                "resize_shorter_side",
                "center_crop",
                "affine({'degrees': 15,'translate': (0.1, 0.1), 'scale': (0.9, 1.1)})",
            ],
            "model.timm_image.val_transforms": ["resize_shorter_side", "center_crop"],
        },
        {
            "model.timm_image.train_transforms": [
                "resize_shorter_side",
                "center_crop",
                "color_jitter(0.2, 0.1, 0.1)",
            ],
            "model.timm_image.val_transforms": ["resize_shorter_side", "center_crop"],
        },
        {
            "model.timm_image.train_transforms": [
                "resize_shorter_side",
                "center_crop",
                "color_jitter({'brightness': 0.2, 'contrast': 0.1, 'saturation': 0.1})",
            ],
            "model.timm_image.val_transforms": ["resize_shorter_side", "center_crop"],
        },
        {
            "model.timm_image.train_transforms": ["resize_shorter_side", "center_crop", "randaug(2, 7)"],
            "model.timm_image.val_transforms": ["resize_shorter_side", "center_crop"],
        },
    ],
)
def test_data_process_image(augmentations):
    download_dir = "./ag_automm_tutorial_imgcls"
    train_df, test_df = shopee_dataset(download_dir)
    image = PIL.Image.open(train_df["image"][0]).convert("RGB")

    model = TimmAutoModelForImagePrediction(prefix="timm_image", checkpoint_name="swin_tiny_patch4_window7_224")
    image_processor = ImageProcessor(
        model=model,
        train_transforms=augmentations["model.timm_image.train_transforms"],
        val_transforms=augmentations["model.timm_image.val_transforms"],
        size=224,
        norm_type="imagenet",
    )

    transformed_image = image_processor.train_processor(image)

    assert (
        len(image_processor.train_processor.transforms) == len(augmentations["model.timm_image.train_transforms"]) + 2
    )
