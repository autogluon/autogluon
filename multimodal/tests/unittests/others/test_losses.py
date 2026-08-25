import os
import shutil

import numpy as np
import pytest

from autogluon.multimodal import MultiModalPredictor
from autogluon.multimodal.utils.misc import shopee_dataset


@pytest.mark.parametrize(
    "alpha",
    [
        [1, 0.25, 0.35, 0.16],  # Python int/float
        [np.float64(1.0), np.float64(0.25), np.float64(0.35), np.float64(0.16)],  # np.float64
    ],
)
def test_focal_loss_multiclass(alpha):
    download_dir = "./ag_automm_tutorial_imgcls"
    train_data, test_data = shopee_dataset(download_dir)

    save_path = "./tmp/automm-shopee-focal-loss"
    if os.path.exists(save_path):
        shutil.rmtree(save_path)

    predictor = MultiModalPredictor(label="label", problem_type="multiclass", path=save_path)

    predictor.fit(
        train_data=train_data,
        hyperparameters={
            "model.timm_image.checkpoint_name": "swin_tiny_patch4_window7_224",
            "env.num_gpus": -1,
            "optim.loss_func": "focal_loss",
            "optim.focal_loss.alpha": alpha,
            "optim.focal_loss.gamma": 2.5,
            "optim.focal_loss.reduction": "mean",
            "optim.max_epochs": 1,
        },
        time_limit=30,
    )

    image_path = test_data.iloc[0]["image"]

    predictions_str = predictor.predict(image_path)
    predictions_list1 = predictor.predict([image_path])
    predictions_list10 = predictor.predict([image_path] * 10)

    assert len(predictions_str) == 1
    assert len(predictions_list1) == 1
    assert len(predictions_list10) == 10
