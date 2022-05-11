from omegaconf import OmegaConf
import pytest
import numpy.testing as npt
import tempfile
import copy

from autogluon.text.automm import AutoMMPredictor
from autogluon.text.automm.constants import (
    MODEL,
    DATA,
    OPTIMIZATION,
    ENVIRONMENT,
    BINARY,
    MULTICLASS,
    UNIFORM_SOUP,
    GREEDY_SOUP,
    BEST,
    NORM_FIT,
    BIT_FIT,
)

from datasets import (
    PetFinderDataset,
    HatefulMeMesDataset,
    AEDataset,
)

from test_automm_predictor import verify_predictor_save_load

ALL_DATASETS = {
    "petfinder": PetFinderDataset,
    "hateful_memes": HatefulMeMesDataset,
    "ae": AEDataset,
}

@pytest.mark.parametrize(
    "hyperparameters",
    [
        {
            "model.timm_image.train_transform_types": ["resize_to_square","center_crop"]
        },

        {
            "model.timm_image.train_transform_types": ["resize_shorter_side","center_crop"]
        },

        {
            "model.timm_image.train_transform_types": ["resize_shorter_side","center_crop","horizontal_flip","vertical_flip"]
        },

        {
            "model.timm_image.train_transform_types": ["resize_shorter_side","center_crop","affine"]
        },

        {
            "model.timm_image.train_transform_types": ["resize_shorter_side","center_crop","colorjitter"]
        },

        {
            "model.timm_image.train_transform_types": ["resize_shorter_side","center_crop","randaug"]
        },
    ]
)

def test_data_process_image(hyperparameters):
    dataset = ALL_DATASETS["petfinder"]()
    metric_name = dataset.metric

    predictor = AutoMMPredictor(
        label=dataset.label_columns[0],
        problem_type=dataset.problem_type,
        eval_metric=metric_name,
    )

    hyperparameters.update(
        {
            "optimization.max_epochs": 1,
            "optimization.top_k_average_method": BEST,
            "env.num_workers": 0,
            "env.num_workers_evaluation": 0,
            "data.categorical.convert_to_text": False,
            "data.numerical.convert_to_text": False,
        }
    )

    hyperparameters_gt = copy.deepcopy(hyperparameters)
    if isinstance(hyperparameters_gt["model.timm_image.train_transform_types"], str):
        hyperparameters_gt["model.timm_image.train_transform_types"] = OmegaConf.from_dotlist([f'names={hyperparameters["model.timm_image.train_transform_types"]}']).names

    config = {
        MODEL: f"fusion_mlp_image_text_tabular",
        DATA: "default",
        OPTIMIZATION: "adamw",
        ENVIRONMENT: "default",
    }

    with tempfile.TemporaryDirectory() as save_path:
        predictor.fit(
            train_data=dataset.train_df,
            config=config,
            time_limit=30,
            save_path=save_path,
            hyperparameters=hyperparameters,
        )

        assert sorted(predictor._config.model.timm_image.train_transform_types) == sorted(hyperparameters_gt["model.timm_image.train_transform_types"])
        for per_transform in hyperparameters_gt["model.timm_image.train_transform_types"]:
            assert hasattr(predictor._config.model, per_transform)

        score = predictor.evaluate(dataset.test_df)
        verify_predictor_save_load(predictor, dataset.test_df)



