import PIL
import pytest
import torch
from torchvision import transforms

from autogluon.multimodal import MultiModalPredictor
from autogluon.multimodal.data.process_image import ImageProcessor
from autogluon.multimodal.models.timm_image import TimmAutoModelForImagePrediction
from autogluon.multimodal.utils.misc import shopee_dataset

from ..utils import IDChangeDetectionDataset


@pytest.mark.parametrize(
    "augmentations",
    [
        {
            "model.timm_image.train_transforms": ["resize_to_square", "center_crop"],
            "model.timm_image.val_transforms": ["resize_shorter_side", "center_crop"],
        },
        {
            "model.timm_image.train_transforms": ["resize_shorter_side", "center_crop", "trivial_augment"],
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
def test_train_transforms(augmentations):
    download_dir = "./ag_automm_tutorial_imgcls"
    train_df, test_df = shopee_dataset(download_dir)
    image = PIL.Image.open(train_df["image"][0]).convert("RGB")

    model = TimmAutoModelForImagePrediction(prefix="timm_image", checkpoint_name="swin_tiny_patch4_window7_224")
    image_processor = ImageProcessor(
        model=model,
        train_transforms=augmentations["model.timm_image.train_transforms"],
        val_transforms=augmentations["model.timm_image.val_transforms"],
    )

    transformed_image = image_processor.train_processor(image)

    assert (
        len(image_processor.train_processor.transforms) == len(augmentations["model.timm_image.train_transforms"]) + 2
    )


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
        prefix="timm_image",
        checkpoint_name=checkpoint_name,
        pretrained=False,
        image_size=provided_size,
    )

    image_processor = ImageProcessor(
        model=timm_model,
        train_transforms=train_transforms,
        val_transforms=val_transforms,
    )
    ret = image_processor.process_one_sample(
        {"image": train_df["image"].tolist()}, sub_dtypes={"image": "image_path"}, is_training=True
    )
    assert ret["timm_image_image"].shape[-1] == expected_size
    out = timm_model(
        torch.unsqueeze(ret[timm_model.image_key], dim=0),
        torch.unsqueeze(torch.Tensor(ret[timm_model.image_valid_num_key]), dim=0),
    )


@pytest.mark.parametrize(
    "train_transforms,val_transforms",
    [
        (
            ["resize_shorter_side", "center_crop", "random_horizontal_flip", "color_jitter"],
            ["resize_shorter_side", "center_crop"],
        ),
        (
            [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip()],
            [transforms.Resize(256), transforms.CenterCrop(224)],
        ),
    ],
)
def test_customize_predictor_image_aug(train_transforms, val_transforms):
    download_dir = "./"
    train_data, test_data = shopee_dataset(download_dir)
    predictor = MultiModalPredictor(label="label", verbosity=4)
    predictor.fit(
        train_data=train_data,
        hyperparameters={
            "model.timm_image.train_transforms": train_transforms,
            "model.timm_image.val_transforms": val_transforms,
        },
        time_limit=10,  # seconds
    )

    assert str(train_transforms) == str(predictor._learner._data_processors["image"][0].train_transforms)
    assert str(val_transforms) == str(predictor._learner._data_processors["image"][0].val_transforms)
    assert len(predictor._learner._data_processors["image"][0].train_processor.transforms) == len(train_transforms) + 2
    assert len(predictor._learner._data_processors["image"][0].val_processor.transforms) == len(val_transforms) + 2


@pytest.mark.parametrize(
    "train_transforms,val_transforms",
    [
        (
            ["resize_shorter_side", "center_crop", "random_horizontal_flip", "color_jitter"],
            ["resize_shorter_side", "center_crop"],
        ),
        (
            [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip()],
            [transforms.Resize(256), transforms.CenterCrop(224)],
        ),
    ],
)
def test_customize_matcher_image_aug(train_transforms, val_transforms):
    dataset = IDChangeDetectionDataset()

    matcher = MultiModalPredictor(
        query="Previous Image",
        response="Current Image",
        problem_type="image_similarity",
        label=dataset.label_columns[0] if dataset.label_columns else None,
        match_label=dataset.match_label,
        eval_metric=dataset.metric,
        hyperparameters={
            "model.timm_image.train_transforms": train_transforms,
            "model.timm_image.val_transforms": val_transforms,
        },
        verbosity=4,
    )

    matcher.fit(
        train_data=dataset.train_df,
        tuning_data=dataset.val_df if hasattr(dataset, "val_df") else None,
        time_limit=10,  # seconds
    )

    assert str(train_transforms) == str(matcher._learner._query_processors["image"][0].train_transforms)
    assert str(train_transforms) == str(matcher._learner._response_processors["image"][0].train_transforms)
    assert str(val_transforms) == str(matcher._learner._query_processors["image"][0].val_transforms)
    assert str(val_transforms) == str(matcher._learner._response_processors["image"][0].val_transforms)
    assert len(matcher._learner._query_processors["image"][0].train_processor.transforms) == len(train_transforms) + 2
    assert len(matcher._learner._query_processors["image"][0].val_processor.transforms) == len(val_transforms) + 2
