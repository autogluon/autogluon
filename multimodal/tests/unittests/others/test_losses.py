import os
import shutil

import pytest

from autogluon.multimodal import MultiModalPredictor
from autogluon.multimodal.utils.misc import shopee_dataset


@pytest.mark.parametrize(
    "checkpoint_name",
    [
        "swin_tiny_patch4_window7_224",
    ],
)
def test_focal_loss_multiclass(checkpoint_name):
    download_dir = "./ag_automm_tutorial_imgcls"
    train_data, test_data = shopee_dataset(download_dir)

    save_path = f"./tmp/automm-shopee-focal-loss"
    if os.path.exists(save_path):
        shutil.rmtree(save_path)

    predictor = MultiModalPredictor(label="label", problem_type="multiclass", path=save_path)

    predictor.fit(
        hyperparameters={
            "model.timm_image.checkpoint_name": checkpoint_name,
            "env.num_gpus": -1,
            "optim.loss_func": "focal_loss",
            "optim.focal_loss.alpha": [1, 0.25, 0.35, 0.16],  # shopee dataset has 4 classes.
            "optim.focal_loss.gamma": 2.5,
            "optim.focal_loss.reduction": "mean",
            "optim.max_epochs": 1,
        },
        train_data=train_data,
        time_limit=30,  # seconds
    )  # you can trust the default config, e.g., we use a `swin_base_patch4_window7_224` model

    image_path = test_data.iloc[0]["image"]

    predictions_str = predictor.predict(image_path)
    predictions_list1 = predictor.predict([image_path])
    predictions_list10 = predictor.predict([image_path] * 10)
