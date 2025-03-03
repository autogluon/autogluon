import os
import shutil
import tempfile

import numpy.testing as npt
import pytest

from autogluon.multimodal import MultiModalPredictor
from autogluon.multimodal.constants import IMAGE_BASE64_STR, IMAGE_BYTEARRAY
from autogluon.multimodal.utils.misc import shopee_dataset

from ..utils import PetFinderDataset, verify_predictor_save_load


@pytest.mark.parametrize("image_type", [IMAGE_BYTEARRAY, IMAGE_BASE64_STR])
def test_image_bytearray_or_base64_str(image_type):
    download_dir = "./"
    train_data_1, test_data_1 = shopee_dataset(download_dir=download_dir)
    if image_type == IMAGE_BYTEARRAY:
        train_data_2, test_data_2 = shopee_dataset(download_dir=download_dir, is_bytearray=True)
    elif image_type == IMAGE_BASE64_STR:
        train_data_2, test_data_2 = shopee_dataset(download_dir=download_dir, is_base64str=True)

    predictor_1 = MultiModalPredictor(
        label="label",
    )
    predictor_2 = MultiModalPredictor(
        label="label",
    )
    model_names = ["timm_image"]
    hyperparameters = {
        "optim.max_epochs": 2,
        "model.names": model_names,
        "model.timm_image.checkpoint_name": "swin_tiny_patch4_window7_224",
    }
    predictor_1.fit(
        train_data=train_data_1,
        hyperparameters=hyperparameters,
        seed=42,
    )
    predictor_2.fit(
        train_data=train_data_2,
        hyperparameters=hyperparameters,
        seed=42,
    )

    score_1 = predictor_1.evaluate(test_data_1)
    score_2 = predictor_2.evaluate(test_data_2)
    # train and predict using different image types
    score_3 = predictor_1.evaluate(test_data_2)
    score_4 = predictor_2.evaluate(test_data_1)

    prediction_1 = predictor_1.predict(test_data_1, as_pandas=False)
    prediction_2 = predictor_2.predict(test_data_2, as_pandas=False)
    prediction_3 = predictor_1.predict(test_data_2, as_pandas=False)
    prediction_4 = predictor_2.predict(test_data_1, as_pandas=False)

    prediction_prob_1 = predictor_1.predict_proba(test_data_1, as_pandas=False)
    prediction_prob_2 = predictor_2.predict_proba(test_data_2, as_pandas=False)
    prediction_prob_3 = predictor_1.predict_proba(test_data_2, as_pandas=False)
    prediction_prob_4 = predictor_1.predict_proba(test_data_1, as_pandas=False)

    npt.assert_array_equal([score_1, score_2, score_3, score_4], [score_1] * 4)
    npt.assert_array_equal([prediction_1, prediction_2, prediction_3, prediction_4], [prediction_1] * 4)
    npt.assert_array_equal(
        [prediction_prob_1, prediction_prob_2, prediction_prob_3, prediction_prob_4], [prediction_prob_1] * 4
    )


def test_predict_with_image_str_or_list():
    download_dir = "./ag_automm_tutorial_imgcls"
    train_data, test_data = shopee_dataset(download_dir)

    save_path = f"./tmp/automm_shopee"
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    predictor = MultiModalPredictor(label="label", path=save_path)
    predictor.fit(
        train_data=train_data,
        time_limit=0,  # seconds
    )  # you can trust the default config, e.g., we use a `swin_base_patch4_window7_224` model

    image_path = test_data.iloc[0]["image"]

    predictions_str = predictor.predict(image_path)
    predictions_list1 = predictor.predict([image_path])
    predictions_list10 = predictor.predict([image_path] * 10)


@pytest.mark.parametrize("invalid_value", [None, "invalid/image/path"])
def test_fit_with_invalid_images(invalid_value):
    dataset = PetFinderDataset()
    train_df = dataset.train_df
    invalid_num = int(0.5 * len(train_df))
    train_df.loc[0:invalid_num, "Images"] = invalid_value

    predictor = MultiModalPredictor(
        label=dataset.label_columns[0],
        problem_type=dataset.problem_type,
        eval_metric=dataset.metric,
    )
    hyperparameters = {
        "model.names": [
            "categorical_mlp",
            "numerical_mlp",
            "timm_image",
            "hf_text",
            "fusion_mlp",
        ],
        "model.hf_text.checkpoint_name": "google/electra-small-discriminator",
        "model.timm_image.checkpoint_name": "swin_small_patch4_window7_224",
        "env.num_workers": 0,
        "env.num_workers_inference": 0,
    }

    with tempfile.TemporaryDirectory() as save_path:
        if os.path.isdir(save_path):
            shutil.rmtree(save_path)

        predictor.fit(
            train_data=train_df,
            time_limit=10,
            save_path=save_path,
            hyperparameters=hyperparameters,
        )

        score = predictor.evaluate(dataset.test_df)
        verify_predictor_save_load(predictor, dataset.test_df)
