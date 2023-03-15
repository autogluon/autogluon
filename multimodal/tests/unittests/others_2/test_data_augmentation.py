import copy
import os
import pickle
import shutil
import tempfile

import pytest
from torchvision import transforms

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
from autogluon.multimodal.utils.misc import shopee_dataset

from ..predictor.test_predictor import verify_predictor_save_load
from ..utils.unittest_datasets import AEDataset, HatefulMeMesDataset, IDChangeDetectionDataset, PetFinderDataset

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
        "model.timm_image.train_transforms": ["resize_shorter_side", "center_crop", "trivial_augment"],
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


@pytest.mark.parametrize(
    "train_transforms,val_transforms",
    [
        (
            ["resize_shorter_side", "center_crop", "random_horizontal_flip", "color_jitter"],
            ["resize_shorter_side", "center_crop"],
        ),
        (
            [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip()],
            [transforms.Resize(256), transforms.CenterCrop(224)],
        ),
    ],
)
def test_customizing_predictor_image_aug(train_transforms, val_transforms):
    download_dir = "./"
    train_data, test_data = shopee_dataset(download_dir)
    predictor = MultiModalPredictor(label="label", verbosity=4)
    predictor.fit(
        train_data=train_data,
        hyperparameters={
            "model.timm_image.train_transforms": train_transforms,
            "model.timm_image.val_transforms": val_transforms,
        },
        time_limit=10,  # seconds
    )

    assert str(train_transforms) == str(predictor._data_processors["image"][0].train_transforms)
    assert str(val_transforms) == str(predictor._data_processors["image"][0].val_transforms)
    assert len(predictor._data_processors["image"][0].train_processor.transforms) == len(train_transforms) + 2
    assert len(predictor._data_processors["image"][0].val_processor.transforms) == len(val_transforms) + 2


@pytest.mark.parametrize(
    "train_transforms,val_transforms",
    [
        (
            ["resize_shorter_side", "center_crop", "random_horizontal_flip", "color_jitter"],
            ["resize_shorter_side", "center_crop"],
        ),
        (
            [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip()],
            [transforms.Resize(256), transforms.CenterCrop(224)],
        ),
    ],
)
def test_customizing_matcher_image_aug(train_transforms, val_transforms):
    dataset = IDChangeDetectionDataset()

    matcher = MultiModalPredictor(
        query="Previous Image",
        response="Current Image",
        problem_type="image_similarity",
        label=dataset.label_columns[0] if dataset.label_columns else None,
        match_label=dataset.match_label,
        eval_metric=dataset.metric,
        hyperparameters={
            "model.timm_image.train_transforms": train_transforms,
            "model.timm_image.val_transforms": val_transforms,
        },
        verbosity=4,
    )

    matcher.fit(
        train_data=dataset.train_df,
        tuning_data=dataset.val_df if hasattr(dataset, "val_df") else None,
        time_limit=10,  # seconds
    )

    assert str(train_transforms) == str(matcher._matcher._query_processors["image"][0].train_transforms)
    assert str(train_transforms) == str(matcher._matcher._response_processors["image"][0].train_transforms)
    assert str(val_transforms) == str(matcher._matcher._query_processors["image"][0].val_transforms)
    assert str(val_transforms) == str(matcher._matcher._response_processors["image"][0].val_transforms)
    assert len(matcher._matcher._query_processors["image"][0].train_processor.transforms) == len(train_transforms) + 2
    assert len(matcher._matcher._query_processors["image"][0].val_processor.transforms) == len(val_transforms) + 2
