import copy
import os
import pickle
import shutil
import tempfile

from autogluon.multimodal import MultiModalPredictor
from autogluon.multimodal.constants import (
    BEST,
    BINARY,
    BIT_FIT,
    DATA,
    DISTILLER,
    ENVIRONMENT,
    GREEDY_SOUP,
    LORA,
    LORA_BIAS,
    LORA_NORM,
    MODEL,
    MULTICLASS,
    NORM_FIT,
    OPTIMIZATION,
    UNIFORM_SOUP,
)

from ..predictor.test_predictor import verify_predictor_save_load
from ..utils.unittest_datasets import AEDataset, HatefulMeMesDataset, PetFinderDataset

ALL_DATASETS = {
    "petfinder": PetFinderDataset,
    "hateful_memes": HatefulMeMesDataset,
    "ae": AEDataset,
}


def test_mixup():
    dataset = ALL_DATASETS["petfinder"]()
    metric_name = dataset.metric

    predictor = MultiModalPredictor(
        label=dataset.label_columns[0],
        problem_type=dataset.problem_type,
        eval_metric=metric_name,
    )
    hyperparameters = {
        "optimization.max_epochs": 1,
        "optimization.top_k_average_method": BEST,
        "model.t_few.checkpoint_name": "t5-small",
        "env.num_workers": 0,
        "env.num_workers_evaluation": 0,
        "data.categorical.convert_to_text": False,
        "data.numerical.convert_to_text": False,
        "data.mixup.turn_on": True,
    }

    with tempfile.TemporaryDirectory() as save_path:
        if os.path.isdir(save_path):
            shutil.rmtree(save_path)
        predictor.fit(
            train_data=dataset.train_df,
            time_limit=10,
            save_path=save_path,
            hyperparameters=hyperparameters,
        )

        score = predictor.evaluate(dataset.test_df)
        verify_predictor_save_load(predictor, dataset.test_df)


def test_textagumentor_deepcopy():
    dataset = ALL_DATASETS["ae"]()
    metric_name = dataset.metric

    predictor = MultiModalPredictor(
        label=dataset.label_columns[0],
        problem_type=dataset.problem_type,
        eval_metric=metric_name,
    )
    hyperparameters = {
        "optimization.max_epochs": 1,
        "env.num_workers": 0,
        "env.num_workers_evaluation": 0,
        "data.categorical.convert_to_text": False,
        "data.numerical.convert_to_text": False,
        "model.hf_text.checkpoint_name": "sentence-transformers/all-MiniLM-L6-v2",
        "model.hf_text.text_trivial_aug_maxscale": 0.05,
        "model.hf_text.text_train_augment_types": ["identity"],
        "optimization.top_k_average_method": "uniform_soup",
    }

    with tempfile.TemporaryDirectory() as save_path:
        if os.path.isdir(save_path):
            shutil.rmtree(save_path)
        predictor.fit(
            train_data=dataset.train_df,
            time_limit=10,
            save_path=save_path,
            hyperparameters=hyperparameters,
        )

    # Deepcopy data processors
    predictor._data_processors = copy.deepcopy(predictor._data_processors)

    # Test copied data processors
    predictor.fit(
        train_data=dataset.train_df,
        hyperparameters=hyperparameters,
        time_limit=10,
    )

    # Copy data processors via pickle + load
    predictor._data_processors = pickle.loads(pickle.dumps(predictor._data_processors))

    # Test copied data processors
    predictor.fit(
        train_data=dataset.train_df,
        hyperparameters=hyperparameters,
        time_limit=10,
    )


def test_trivialaugment():
    dataset = ALL_DATASETS["petfinder"]()
    metric_name = dataset.metric

    predictor = MultiModalPredictor(
        label=dataset.label_columns[0],
        problem_type=dataset.problem_type,
        eval_metric=metric_name,
    )
    hyperparameters = {
        "optimization.max_epochs": 1,
        "optimization.top_k_average_method": BEST,
        "env.num_workers": 0,
        "env.num_workers_evaluation": 0,
        "data.categorical.convert_to_text": False,
        "data.numerical.convert_to_text": False,
        "data.mixup.turn_on": True,
        "model.hf_text.checkpoint_name": "sentence-transformers/all-MiniLM-L6-v2",
        "model.hf_text.text_trivial_aug_maxscale": 0.1,
        "model.hf_text.text_aug_detect_length": 10,
        "model.timm_image.checkpoint_name": "mobilenetv3_small_100",
        "model.timm_image.train_transform_types": ["resize_shorter_side", "center_crop", "trivial_augment"],
    }

    with tempfile.TemporaryDirectory() as save_path:
        if os.path.isdir(save_path):
            shutil.rmtree(save_path)
        predictor.fit(
            train_data=dataset.train_df,
            time_limit=30,
            save_path=save_path,
            hyperparameters=hyperparameters,
        )

        score = predictor.evaluate(dataset.test_df)
        verify_predictor_save_load(predictor, dataset.test_df)
