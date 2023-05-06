import uuid

import numpy as np
import pytest

from autogluon.multimodal import MultiModalPredictor
from autogluon.multimodal.utils.misc import shopee_dataset


@pytest.mark.parametrize(
    "checkpoint_name",
    [
        "swin_tiny_patch4_window7_224",
    ],
)
def test_learner_image_classification(checkpoint_name):
    download_dir = "./ag_automm_tutorial_imgcls_learner"
    train_data, test_data = shopee_dataset(download_dir)

    model_path_with_learner = f"./tmp/{uuid.uuid4().hex}-automm_shopee"

    predictor_with_learner = MultiModalPredictor(
        label="label",
        problem_type="multiclass",
        path=model_path_with_learner,
        use_learner=True,
    )
    predictor_with_learner.fit(
        hyperparameters={
            "model.mmdet_image.checkpoint_name": checkpoint_name,
            "env.num_gpus": 1,
            "optimization.max_epochs": 5,
        },
        train_data=train_data,
        # time_limit=30,  # seconds
    )  # you can trust the default config, e.g., we use a `swin_base_patch4_window7_224` model

    result_with_learner = predictor_with_learner.evaluate(test_data)
    print("result with learner")
    print(result_with_learner)

    model_path = f"./tmp/{uuid.uuid4().hex}-automm_shopee"
    predictor = MultiModalPredictor(
        label="label",
        problem_type="multiclass",
        path=model_path,
        use_learner=False,
    )
    predictor.fit(
        hyperparameters={
            "model.mmdet_image.checkpoint_name": checkpoint_name,
            "env.num_gpus": 1,
            "optimization.max_epochs": 5,
        },
        train_data=train_data,
        # time_limit=30,  # seconds
    )  # you can trust the default config, e.g., we use a `swin_base_patch4_window7_224` model

    result = predictor.evaluate(test_data)
    print("result without learner")
    print(result)
    assert (np.isclose(result_with_learner["accuracy"], result["accuracy"], atol=0.01)).all()


if __name__ == "__main__":
    test_learner_image_classification("swin_tiny_patch4_window7_224")
