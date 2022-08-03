import os
import shutil

import pytest
from unittest_datasets import PetFinderDataset

from autogluon.multimodal import MultiModalPredictor


@pytest.mark.parametrize(
    "save_path",
    [
        "an_existing_path",
        "~/an_existing_path",
        os.path.join(os.path.expanduser("~"), "an_existing_path"),
    ],
)
def test_existing_save_path(save_path):
    abs_path = os.path.abspath(os.path.expanduser(save_path))
    os.makedirs(abs_path, exist_ok=True)
    with pytest.raises(ValueError):
        predictor = MultiModalPredictor(path=save_path)

    dataset = PetFinderDataset()
    predictor = MultiModalPredictor(
        label=dataset.label_columns[0],
        problem_type=dataset.problem_type,
        eval_metric=dataset.metric,
    )
    with pytest.raises(ValueError):
        predictor.fit(train_data=dataset.train_df, save_path=save_path)

    shutil.rmtree(abs_path)


def test_continuous_training_save_path():
    save_path = "a_tmp_path"
    abs_path = os.path.abspath(os.path.expanduser(save_path))
    if os.path.exists(abs_path):
        shutil.rmtree(abs_path)

    dataset = PetFinderDataset()
    predictor = MultiModalPredictor(
        label=dataset.label_columns[0],
        problem_type=dataset.problem_type,
        eval_metric=dataset.metric,
    )

    hyperparameters = {
        "env.num_workers": 0,
        "env.num_workers_evaluation": 0,
        "model.names": ["timm_image", "hf_text", "fusion_mlp"],
        "model.hf_text.checkpoint_name": "prajjwal1/bert-tiny",
        "model.timm_image.checkpoint_name": "swin_tiny_patch4_window7_224",
    }
    predictor.fit(train_data=dataset.train_df, save_path=save_path, hyperparameters=hyperparameters, time_limit=10)

    assert predictor.path == abs_path

    with pytest.raises(ValueError):
        predictor.fit(train_data=dataset.train_df, save_path=save_path, time_limit=10)

    # continue training
    predictor.fit(train_data=dataset.train_df, time_limit=10)
    assert predictor.path != abs_path and "AutogluonModels/ag-" in predictor.path

    # load a saved predictor and continue training
    predictor_loaded = MultiModalPredictor.load(save_path)
    predictor_loaded.fit(train_data=dataset.train_df, time_limit=10)
    assert predictor_loaded.path != abs_path and "AutogluonModels/ag-" in predictor.path
