import uuid

import pytest

from autogluon.multimodal import MultiModalPredictor
from autogluon.multimodal.utils.misc import shopee_dataset


@pytest.mark.parametrize(
    "checkpoint_name",
    [
        "swin_base_patch4_window7_224",
    ],
)
def test_classification_str_list_input(checkpoint_name):
    download_dir = "./ag_automm_tutorial_imgcls"
    train_data, test_data = shopee_dataset(download_dir)

    model_path = f"./tmp/{uuid.uuid4().hex}-automm_shopee"
    predictor = MultiModalPredictor(label="label", path=model_path)
    predictor.fit(
        train_data=train_data,
        time_limit=30,  # seconds
    )  # you can trust the default config, e.g., we use a `swin_base_patch4_window7_224` model

    image_path = test_data.iloc[0]["image"]

    predictions_str = predictor.predict(image_path)
    predictions_list1 = predictor.predict([image_path])
    predictions_list10 = predictor.predict([image_path] * 10)


@pytest.mark.parametrize(
    "checkpoint_name",
    [
        "swin_base_patch4_window7_224",
    ],
)
def test_focal_loss_multiclass(checkpoint_name):
    download_dir = "./ag_automm_tutorial_imgcls"
    train_data, test_data = shopee_dataset(download_dir)

    model_path = f"./tmp/{uuid.uuid4().hex}-automm_shopee"

    predictor = MultiModalPredictor(
        label="label",
        hyperparameters={
            "model.mmdet_image.checkpoint_name": checkpoint_name,
            "env.num_gpus": 1,
            "optimization.loss_function": "focalloss",
            "optimization.focalloss.alpha": [1, 0.25, 0.35, 0.16, 0.8],
            "optimization.focalloss.gamma": 2.5,
            "optimization.focalloss.reduction": "none",
        },
        problem_type="multiclass",
        path=model_path
    )

    # predictor = MultiModalPredictor(label="label", path=model_path)
    predictor.fit(
        train_data=train_data,
        time_limit=30,  # seconds
    )  # you can trust the default config, e.g., we use a `swin_base_patch4_window7_224` model

    image_path = test_data.iloc[0]["image"]

    predictions_str = predictor.predict(image_path)
    predictions_list1 = predictor.predict([image_path])
    predictions_list10 = predictor.predict([image_path] * 10)


if __name__ == "__main__":
    test_focal_loss_multiclass("swin_base_patch4_window7_224")