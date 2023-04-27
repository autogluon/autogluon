import uuid

import pytest

from autogluon.multimodal import MultiModalPredictor
from autogluon.multimodal.utils.misc import shopee_dataset


@pytest.mark.parametrize(
    "checkpoint_name",
    [
        "swin_tiny_patch4_window7_224",
    ],
)
def test_predictor_with_learner(checkpoint_name):
    download_dir = "./ag_automm_tutorial_imgcls_learner"
    train_data, test_data = shopee_dataset(download_dir)

    model_path = f"./tmp/{uuid.uuid4().hex}-automm_shopee"

    predictor = MultiModalPredictor(
        label="label",
        problem_type="multiclass",
        path=model_path,
        use_learner=True,
    )
    import ipdb

    ipdb.set_trace()
    predictor.fit(
        hyperparameters={
            "model.mmdet_image.checkpoint_name": checkpoint_name,
            "env.num_gpus": 1,
            "optimization.max_epochs": 2,
        },
        train_data=train_data,
        # time_limit=30,  # seconds
    )  # you can trust the default config, e.g., we use a `swin_base_patch4_window7_224` model

    result = predictor.evaluate(test_data)

    image_path = test_data.iloc[0]["image"]

    predictions_str = predictor.predict(image_path)
    predictions_list1 = predictor.predict([image_path])
    predictions_list10 = predictor.predict([image_path] * 10)


if __name__ == "__main__":
    test_predictor_with_learner("swin_tiny_patch4_window7_224")
