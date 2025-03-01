import os
import shutil

import pytest
from ray import tune
from torch import nn

from autogluon.core.hpo.ray_tune_constants import SCHEDULER_PRESETS, SEARCHER_PRESETS
from autogluon.multimodal import MultiModalPredictor
from autogluon.multimodal.constants import ALL_MODEL_QUALITIES
from autogluon.multimodal.models import modify_duplicate_model_names
from autogluon.multimodal.utils import filter_search_space

from ..utils import (
    IDChangeDetectionDataset,
    PetFinderDataset,
    get_home_dir,
    verify_matcher_save_load,
    verify_predictor_save_load,
)


def predictor_hpo(searcher, scheduler, presets=None):
    dataset = PetFinderDataset()

    hyperparameters = {
        "optim.lr": tune.uniform(0.0001, 0.01),
        "optim.max_epochs": 1,
        "model.names": ["numerical_mlp", "categorical_mlp", "fusion_mlp"],
        "data.categorical.convert_to_text": False,
        "data.numerical.convert_to_text": False,
        "env.num_workers": 0,
        "env.num_workers_inference": 0,
    }

    hyperparameter_tune_kwargs = {
        "searcher": searcher,
        "scheduler": scheduler,
        "num_trials": 2,
    }

    predictor = MultiModalPredictor(
        label=dataset.label_columns[0],
        problem_type=dataset.problem_type,
        eval_metric=dataset.metric,
        presets=presets,
    )

    save_path = os.path.join(get_home_dir(), "outputs", "hpo", f"_{searcher}", f"_{scheduler}")
    if os.path.exists(save_path):
        shutil.rmtree(save_path)

    predictor_hpo = predictor.fit(
        train_data=dataset.train_df,
        hyperparameters=hyperparameters,
        save_path=save_path,
        hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
    )
    assert predictor == predictor_hpo

    score = predictor.evaluate(dataset.test_df)
    verify_predictor_save_load(predictor, dataset.test_df)

    # test for continuous training
    predictor = predictor.fit(
        train_data=dataset.train_df,
        hyperparameters=hyperparameters,
        hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
    )


def matcher_hpo(searcher, scheduler, presets=None):
    dataset = IDChangeDetectionDataset()

    hyperparameters = {
        "optim.lr": tune.uniform(0.0001, 0.001),
        "optim.max_epochs": 1,
        "env.num_workers": 0,
        "env.num_workers_inference": 0,
        "optim.top_k_average_method": "greedy_soup",
        "model.timm_image.checkpoint_name": "swin_tiny_patch4_window7_224",
    }

    hyperparameter_tune_kwargs = {
        "searcher": searcher,
        "scheduler": scheduler,
        "num_trials": 2,
    }

    matcher = MultiModalPredictor(
        query="Previous Image",
        response="Current Image",
        problem_type="image_similarity",
        label=dataset.label_columns[0] if dataset.label_columns else None,
        match_label=dataset.match_label,
        eval_metric=dataset.metric,
        presets=presets,
    )

    save_path = os.path.join(get_home_dir(), "outputs", "hpo", f"_{searcher}", f"_{scheduler}")
    if os.path.exists(save_path):
        shutil.rmtree(save_path)

    matcher_hpo = matcher.fit(
        train_data=dataset.train_df,
        hyperparameters=hyperparameters,
        save_path=save_path,
        hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
    )
    assert matcher_hpo == matcher

    score = matcher.evaluate(dataset.test_df)
    verify_matcher_save_load(matcher, dataset.test_df)

    # test for continuous training
    predictor = matcher.fit(
        train_data=dataset.train_df,
        hyperparameters=hyperparameters,
        hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
    )


@pytest.mark.parametrize(
    "hyperparameters, keys_to_filter, expected",
    [
        ({"model.abc": tune.choice(["a", "b"])}, ["model"], {}),
        ({"model.abc": tune.choice(["a", "b"])}, ["data"], {"model.abc": tune.choice(["a", "b"])}),
        ({"model.abc": "def"}, ["model"], {"model.abc": "def"}),
        (
            {
                "data.abc.def": tune.choice(["a", "b"]),
                "model.abc": "def",
                "env.abc.def": tune.choice(["a", "b"]),
            },
            ["data"],
            {"model.abc": "def", "env.abc.def": tune.choice(["a", "b"])},
        ),
    ],
)
def test_filter_search_space(hyperparameters, keys_to_filter, expected):
    # We test keys here because the object might be copied and hence direct comparison will fail
    assert filter_search_space(hyperparameters, keys_to_filter).keys() == expected.keys()


@pytest.mark.parametrize("hyperparameters, keys_to_filter", [({"model.abc": tune.choice(["a", "b"])}, ["abc"])])
def test_invalid_filter_search_space(hyperparameters, keys_to_filter):
    with pytest.raises(Exception) as e_info:
        filter_search_space(hyperparameters, keys_to_filter)


@pytest.mark.parametrize("searcher", list(SEARCHER_PRESETS.keys()))
@pytest.mark.parametrize("scheduler", list(SCHEDULER_PRESETS.keys()))
def test_predictor_hpo_searchers_schedulers(searcher, scheduler):
    predictor_hpo(searcher, scheduler)


@pytest.mark.parametrize("presets", [f"{quality}_hpo" for quality in ALL_MODEL_QUALITIES])
def test_predictor_hpo_presets(presets):
    predictor_hpo("random", "FIFO", presets)


@pytest.mark.parametrize("searcher", list(SEARCHER_PRESETS.keys()))
@pytest.mark.parametrize("scheduler", list(SCHEDULER_PRESETS.keys()))
def test_matcher_hpo_searchers_schedulers(searcher, scheduler):
    matcher_hpo(searcher, scheduler)


@pytest.mark.parametrize("presets", [f"{quality}_hpo" for quality in ALL_MODEL_QUALITIES])
def test_matcher_hpo_presets(presets):
    matcher_hpo("random", "FIFO", presets)


@pytest.mark.single_gpu
def test_hpo_distillation():
    searcher = "random"
    scheduler = "FIFO"
    dataset = PetFinderDataset()

    hyperparameters = {
        "optim.max_epochs": 1,
        "model.names": ["numerical_mlp", "categorical_mlp", "fusion_mlp"],
        "data.categorical.convert_to_text": False,
        "data.numerical.convert_to_text": False,
        "env.num_workers": 0,
        "env.num_workers_inference": 0,
    }

    hyperparameter_tune_kwargs = {
        "searcher": searcher,
        "scheduler": scheduler,
        "num_trials": 2,
    }

    teacher_predictor = MultiModalPredictor(
        label=dataset.label_columns[0],
        problem_type=dataset.problem_type,
        eval_metric=dataset.metric,
    )

    teacher_save_path = os.path.join(
        get_home_dir(), "outputs", "hpo_distillation_teacher", f"_{searcher}", f"_{scheduler}"
    )
    if os.path.exists(teacher_save_path):
        shutil.rmtree(teacher_save_path)

    teacher_predictor.fit(
        train_data=dataset.train_df,
        hyperparameters=hyperparameters,
        save_path=teacher_save_path,
    )

    hyperparameters = {
        "optim.lr": tune.uniform(0.0001, 0.01),
        "optim.max_epochs": 1,
        "model.names": ["numerical_mlp"],
        "data.numerical.convert_to_text": False,
        "env.num_workers": 0,
        "env.num_workers_inference": 0,
    }

    # test for distillation
    predictor = MultiModalPredictor(
        label=dataset.label_columns[0],
        problem_type=dataset.problem_type,
        eval_metric=dataset.metric,
    )

    student_save_path = os.path.join(
        get_home_dir(), "outputs", "hpo_distillation_student", f"_{searcher}", f"_{scheduler}"
    )
    if os.path.exists(student_save_path):
        shutil.rmtree(student_save_path)

    predictor.fit(
        train_data=dataset.train_df,
        teacher_predictor=teacher_save_path,
        hyperparameters=hyperparameters,
        hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
        save_path=student_save_path,
    )


def test_modifying_duplicate_model_names():
    dataset = PetFinderDataset()
    metric_name = dataset.metric

    teacher_predictor = MultiModalPredictor(
        label=dataset.label_columns[0],
        problem_type=dataset.problem_type,
        eval_metric=metric_name,
    )

    hyperparameters = {
        "optim.max_epochs": 1,
        "model.names": ["numerical_mlp", "categorical_mlp", "timm_image", "hf_text", "fusion_mlp"],
        "model.hf_text.checkpoint_name": "nlpaueb/legal-bert-small-uncased",
        "model.timm_image.checkpoint_name": "swin_tiny_patch4_window7_224",
        "env.num_workers": 0,
        "env.num_workers_inference": 0,
    }

    teacher_predictor.fit(
        train_data=dataset.train_df,
        hyperparameters=hyperparameters,
        time_limit=1,
    )
    student_predictor = MultiModalPredictor(
        label=dataset.label_columns[0],
        problem_type=dataset.problem_type,
        eval_metric=metric_name,
    )
    student_predictor.fit(
        train_data=dataset.train_df,
        time_limit=0,
    )

    teacher_predictor._learner = modify_duplicate_model_names(
        learner=teacher_predictor._learner,
        postfix="teacher",
        blacklist=student_predictor._learner._config.model.names,
    )

    # verify teacher and student have no duplicate model names
    assert all(
        [
            n not in teacher_predictor._learner._config.model.names
            for n in student_predictor._learner._config.model.names
        ]
    ), (
        f"teacher model names {teacher_predictor._learner._config.model.names} and"
        f" student model names {student_predictor._learner._config.model.names} have duplicates."
    )

    # verify each model name prefix is valid
    assert teacher_predictor._learner._model.prefix in teacher_predictor._learner._config.model.names
    if isinstance(teacher_predictor._learner._model.model, nn.ModuleList):
        for per_model in teacher_predictor._learner._model.model:
            assert per_model.prefix in teacher_predictor._learner._config.model.names

    # verify each data processor's prefix is valid
    for per_modality_processors in teacher_predictor._learner._data_processors.values():
        for per_processor in per_modality_processors:
            assert per_processor.prefix in teacher_predictor._learner._config.model.names
