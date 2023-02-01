from autogluon.multimodal.models.timm_image import TimmAutoModelForImagePrediction
from autogluon.multimodal.data.process_image import ImageProcessor
from autogluon.multimodal.utils.misc import shopee_dataset
import torch as th
import pytest


@pytest.mark.parametrize("checkpoint_name,provided_size,expected_size",
                         [("swinv2_tiny_window8_256", 300, 256),  # SwinTransformer
                          ("convnext_nano", 300, 300),            # ConvNext
                          ("resnet50", 300, 300),                 # ResNet
                          ("regnety_004", 300, 300),              # RegNet
                          ("fbnetv3_b", 300, 300),                # MobileNetV3
                          ("tf_efficientnet_b2", 300, 300)])      # EfficientNet
def test_variable_input_size_backbone(checkpoint_name, provided_size, expected_size):
    download_dir = './ag_automm_tutorial_imgcls'
    train_df, test_df = shopee_dataset(download_dir)

    train_transform_types = ["resize_shorter_side", "center_crop", "trivial_augment"]
    val_transform_types = ["resize_shorter_side", "center_crop"]

    timm_model = TimmAutoModelForImagePrediction(prefix="timm_image",
                                                 checkpoint_name=checkpoint_name,
                                                 pretrained=False)

    image_processor = ImageProcessor(model=timm_model,
                                     train_transform_types=train_transform_types,
                                     val_transform_types=val_transform_types,
                                     size=provided_size)
    ret = image_processor.process_one_sample({"image": train_df["image"].tolist()},
                                             feature_modalities={"image": "image"},
                                             is_training=True)
    assert ret['timm_image_image'].shape[-1] == expected_size
    out = timm_model(th.unsqueeze(ret[timm_model.image_key], dim=0),
                     th.unsqueeze(th.Tensor(ret[timm_model.image_valid_num_key]), dim=0))
