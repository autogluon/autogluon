import copy
import os
import pickle
import shutil
import tempfile

import pytest

from autogluon.multimodal import MultiModalPredictor

from ..others.unittest_datasets import PetFinderDataset
from .test_predictor import verify_predictor_save_load


@pytest.mark.parametrize("invalid_value", [None, "invalid/image/path"])
def test_invalid_images(invalid_value):
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
        "env.num_workers": 2,
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
