"""Unit tests for trainers"""

import copy
import itertools
import shutil
import sys
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest

import autogluon.core as ag
from autogluon.common import space
from autogluon.common.loaders import load_pkl
from autogluon.timeseries.dataset import TimeSeriesDataFrame
from autogluon.timeseries.models import DeepARModel, ETSModel
from autogluon.timeseries.models.ensemble import GreedyEnsemble, SimpleAverageEnsemble
from autogluon.timeseries.models.ensemble.abstract import AbstractTimeSeriesEnsembleModel
from autogluon.timeseries.models.multi_window.multi_window_model import MultiWindowBacktestingModel
from autogluon.timeseries.trainer import TimeSeriesTrainer
from autogluon.timeseries.trainer.prediction_cache import FileBasedPredictionCache, NoOpPredictionCache

from ..common import (
    DATAFRAME_WITH_COVARIATES,
    DUMMY_TS_DATAFRAME,
    dict_equal_primitive,
    get_data_frame_with_item_index,
    get_data_frame_with_variable_lengths,
)

DUMMY_TRAINER_HYPERPARAMETERS = {"SimpleFeedForward": {"max_epochs": 1}}
TEST_HYPERPARAMETER_SETTINGS = [
    {"SimpleFeedForward": {"max_epochs": 1}},
    {"DeepAR": {"max_epochs": 1}, "ETS": {}},
]
TEST_HYPERPARAMETER_SETTINGS_EXPECTED_LB_LENGTHS = [1, 2]


@pytest.fixture(scope="module")
def trained_trainers():
    trainers = {}
    model_paths = []
    for hp in TEST_HYPERPARAMETER_SETTINGS:
        temp_model_path = tempfile.mkdtemp()
        trainer = TimeSeriesTrainer(
            path=temp_model_path,
            eval_metric="MAPE",
            prediction_length=3,
        )
        trainer.fit(
            train_data=DUMMY_TS_DATAFRAME,
            hyperparameters=hp,
        )
        trainers[repr(hp)] = trainer
        model_paths.append(temp_model_path)

    yield trainers

    for td in model_paths:
        shutil.rmtree(td)


def test_trainer_can_be_initialized(temp_model_path):
    model = TimeSeriesTrainer(path=temp_model_path, prediction_length=24)
    assert isinstance(model, TimeSeriesTrainer)


# smoke test for the short 'happy path'
@pytest.mark.parametrize("hyperparameters", TEST_HYPERPARAMETER_SETTINGS)
def test_when_trainer_called_then_training_is_performed(trained_trainers, hyperparameters):
    assert trained_trainers[repr(hyperparameters)].get_model_names()


@pytest.mark.parametrize("eval_metric", ["MAPE", None])
@pytest.mark.parametrize(
    "hyperparameters, expected_board_length",
    zip(TEST_HYPERPARAMETER_SETTINGS, TEST_HYPERPARAMETER_SETTINGS_EXPECTED_LB_LENGTHS),
)
def test_given_hyperparameters_when_trainer_called_then_leaderboard_is_correct(
    temp_model_path, eval_metric, hyperparameters, expected_board_length
):
    trainer = TimeSeriesTrainer(path=temp_model_path, eval_metric=eval_metric)
    trainer.fit(
        train_data=DUMMY_TS_DATAFRAME,
        hyperparameters=hyperparameters,
    )
    leaderboard = trainer.leaderboard()

    if len(hyperparameters) > 1:
        expected_board_length += int(trainer.enable_ensemble)
    assert len(leaderboard) == expected_board_length
    assert np.all(leaderboard["score_val"] < 0)  # all MAPEs should be negative


@pytest.mark.parametrize(
    "hyperparameters, expected_board_length",
    zip(TEST_HYPERPARAMETER_SETTINGS, TEST_HYPERPARAMETER_SETTINGS_EXPECTED_LB_LENGTHS),
)
def test_given_test_data_when_trainer_called_then_leaderboard_is_correct(
    trained_trainers, hyperparameters, expected_board_length
):
    trainer = trained_trainers[repr(hyperparameters)]
    test_data = get_data_frame_with_item_index(["A", "B", "C"])

    leaderboard = trainer.leaderboard(test_data)

    if len(hyperparameters) > 1:
        expected_board_length += int(trainer.enable_ensemble)

    assert len(leaderboard) == expected_board_length
    assert not np.any(np.isnan(leaderboard["score_test"]))
    assert np.all(leaderboard["score_test"] < 0)  # all MAPEs should be negative


@pytest.mark.parametrize(
    "hyperparameters, expected_board_length",
    zip(TEST_HYPERPARAMETER_SETTINGS, TEST_HYPERPARAMETER_SETTINGS_EXPECTED_LB_LENGTHS),
)
def test_given_hyperparameters_when_trainer_called_then_model_can_predict(
    trained_trainers, hyperparameters, expected_board_length
):
    trainer = trained_trainers[repr(hyperparameters)]
    predictions = trainer.predict(DUMMY_TS_DATAFRAME)

    assert isinstance(predictions, TimeSeriesDataFrame)

    predicted_item_index = predictions.item_ids
    assert all(predicted_item_index == DUMMY_TS_DATAFRAME.item_ids)  # noqa
    assert all(len(predictions.loc[i]) == 3 for i in predicted_item_index)
    assert not np.any(np.isnan(predictions))


@pytest.mark.parametrize(
    "hyperparameters",
    [
        {"DeepAR": {"max_epochs": 4}, "SimpleFeedForward": {"max_epochs": 1}},
        {
            "SimpleFeedForward": {"context_length": 44, "max_epochs": 2},
            "DeepAR": {"max_epochs": 3},
        },
    ],
)
def test_given_hyperparameters_when_get_trainable_base_models_called_then_hyperparameters_set_correctly(
    temp_model_path, hyperparameters
):
    trainer = TimeSeriesTrainer(path=temp_model_path, eval_metric="MAPE")
    models = trainer.get_trainable_base_models(
        hyperparameters=hyperparameters,
    )

    for model in models:
        for k, v in hyperparameters[model.name].items():
            params = model.get_hyperparameters()
            assert params[k] == v


@pytest.mark.parametrize(
    "hyperparameters",
    [
        {"DeepAR": {"max_epochs": 1}, "SimpleFeedForward": {"max_epochs": 1}},
        {
            "SimpleFeedForward": {"context_length": 44, "max_epochs": 1},
            "DeepAR": {"max_epochs": 1},
        },
    ],
)
def test_given_hyperparameters_when_trainer_fit_then_freq_set_correctly(temp_model_path, hyperparameters):
    trainer = TimeSeriesTrainer(path=temp_model_path, eval_metric="MAPE")
    trainer.fit(
        train_data=DUMMY_TS_DATAFRAME,
        hyperparameters=hyperparameters,
    )

    for model_name in trainer.get_model_names():
        model = trainer.load_model(model_name)
        assert model.freq == DUMMY_TS_DATAFRAME.freq


@pytest.mark.skipif(sys.platform.startswith("win"), reason="HPO tests lead to known issues in Windows platform tests")
@pytest.mark.parametrize("model_name", ["DeepAR", "SimpleFeedForward"])
def test_given_hyperparameters_with_spaces_when_trainer_called_then_hpo_is_performed(temp_model_path, model_name):
    hyperparameters = {model_name: {"max_epochs": space.Int(1, 4)}}

    trainer = TimeSeriesTrainer(path=temp_model_path)
    trainer.fit(
        train_data=DUMMY_TS_DATAFRAME,
        hyperparameters=hyperparameters,
        hyperparameter_tune_kwargs={
            "num_trials": 2,
            "searcher": "random",
            "scheduler": "local",
        },
    )
    leaderboard = trainer.leaderboard()

    assert len(leaderboard) == 2 + 1  # include ensemble

    hpo_results_first_model = next(iter(trainer.hpo_results.values()))
    config_history = [result["hyperparameters"] for result in hpo_results_first_model.values()]
    assert len(config_history) == 2
    assert all(1 <= config["max_epochs"] <= 4 for config in config_history)


@pytest.mark.parametrize(
    "hyperparameters, expected_model_names",
    [
        ({"Naive": [{}, {}, {"ag_args": {"name_suffix": "_extra"}}]}, ["Naive", "Naive_2", "Naive_extra"]),
        ({"Naive": [{"ag_args": {"name": "CustomNaive"}}], "SeasonalNaive": {}}, ["CustomNaive", "SeasonalNaive"]),
    ],
)
def test_given_hyperparameters_with_lists_when_trainer_called_then_multiple_models_are_trained(
    temp_model_path, hyperparameters, expected_model_names
):
    trainer = TimeSeriesTrainer(path=temp_model_path, enable_ensemble=False)
    trainer.fit(train_data=DUMMY_TS_DATAFRAME, hyperparameters=hyperparameters)
    leaderboard = trainer.leaderboard()
    assert len(leaderboard) == len(expected_model_names)
    assert all(name in leaderboard["model"].values for name in expected_model_names)


@pytest.mark.parametrize("eval_metric", ["MAPE", None])
@pytest.mark.parametrize(
    "hyperparameters, expected_board_length",
    [
        ({DeepARModel: {"max_epochs": 1}}, 1),
        (
            {
                ETSModel: {},
                DeepARModel: {"max_epochs": 1},
            },
            2,
        ),
    ],
)
def test_given_hyperparameters_and_custom_models_when_trainer_called_then_leaderboard_is_correct(
    temp_model_path, eval_metric, hyperparameters, expected_board_length
):
    trainer = TimeSeriesTrainer(path=temp_model_path, eval_metric=eval_metric)
    trainer.fit(
        train_data=DUMMY_TS_DATAFRAME,
        hyperparameters=hyperparameters,
    )
    leaderboard = trainer.leaderboard()

    if len(hyperparameters) > 1:  # account for ensemble
        expected_board_length += int(trainer.enable_ensemble)
    assert len(leaderboard) == expected_board_length
    assert np.all(leaderboard["score_val"] < 0)  # all MAPEs should be negative


@pytest.mark.parametrize(
    "hyperparameter_list, expected_number_of_unique_names, expected_suffixes",
    [
        (
            [
                {DeepARModel: {"max_epochs": 1}},
                {DeepARModel: {"max_epochs": 1}},
            ],
            3,
            ["AR_2"],
        ),
        (
            [
                {DeepARModel: {"max_epochs": 1}, "ETS": {}},
                {DeepARModel: {"max_epochs": 1}},
                {DeepARModel: {"max_epochs": 1}},
            ],
            7,
            ["AR_2", "AR_3", "Ensemble_2", "Ensemble_3"],
        ),
        (
            [
                {DeepARModel: {"max_epochs": 1}, "DeepAR": {"max_epochs": 1}, "ETS": {}},
                {DeepARModel: {"max_epochs": 1}},
            ],
            6,
            ["AR_2", "AR_3", "Ensemble_2"],
        ),
    ],
)
def test_given_repeating_model_when_trainer_called_incrementally_then_name_collisions_are_prevented(
    temp_model_path,
    hyperparameter_list,
    expected_number_of_unique_names,
    expected_suffixes,
):
    trainer = TimeSeriesTrainer(path=temp_model_path)

    # incrementally train with new hyperparameters
    for hp in hyperparameter_list:
        trainer.fit(
            train_data=DUMMY_TS_DATAFRAME,
            hyperparameters=hp,
        )

    model_names = trainer.get_model_names()

    # account for the ensemble if it should be fitted, and drop ensemble names
    assert len(model_names) == expected_number_of_unique_names
    for suffix in expected_suffixes:
        assert any(name.endswith(suffix) for name in model_names)


@pytest.mark.parametrize(
    "hyperparameters",
    [
        {"DeepAR": {"max_epochs": 1}, "SimpleFeedForward": {"max_epochs": 1}},
        {
            "SimpleFeedForward": {"context_length": 44, "max_epochs": 1},
            "DeepAR": {"max_epochs": 1},
        },
    ],
)
@pytest.mark.parametrize("low_memory", [True, False])
def test_when_trainer_fit_and_deleted_models_load_back_correctly_and_can_predict(
    temp_model_path, hyperparameters, low_memory
):
    trainer = TimeSeriesTrainer(path=temp_model_path, eval_metric="MAPE", prediction_length=2)
    trainer.fit(
        train_data=DUMMY_TS_DATAFRAME,
        hyperparameters=hyperparameters,
    )
    model_names = copy.copy(trainer.get_model_names())
    trainer.save()
    del trainer

    loaded_trainer = TimeSeriesTrainer.load(path=temp_model_path)

    for m in model_names:
        loaded_model = loaded_trainer.load_model(m)
        if isinstance(loaded_model, GreedyEnsemble):
            continue

        predictions = loaded_model.predict(DUMMY_TS_DATAFRAME)

        assert isinstance(predictions, TimeSeriesDataFrame)

        predicted_item_index = predictions.item_ids
        assert all(predicted_item_index == DUMMY_TS_DATAFRAME.item_ids)  # noqa
        assert all(len(predictions.loc[i]) == 2 for i in predicted_item_index)
        assert not np.any(np.isnan(predictions))


def test_when_trainer_fit_and_deleted_then_oof_predictions_can_be_loaded(temp_model_path):
    trainer = TimeSeriesTrainer(path=temp_model_path, eval_metric="MAPE", prediction_length=2)
    trainer.fit(
        train_data=DUMMY_TS_DATAFRAME,
        hyperparameters={
            "Naive": {},
            "ETS": {},
            "DirectTabular": {"model_name": "GBM"},
            "DeepAR": {"max_epochs": 1, "num_batches_per_epoch": 1},
        },
    )
    model_names = copy.copy(trainer.get_model_names())
    trainer.save()
    del trainer

    loaded_trainer = TimeSeriesTrainer.load(path=temp_model_path)

    oof_data = loaded_trainer._get_validation_windows(DUMMY_TS_DATAFRAME, val_data=None)
    for m in model_names:
        if "WeightedEnsemble" not in m:
            oof_predictions = loaded_trainer._get_model_oof_predictions(m)
            for window_idx, oof_pred in enumerate(oof_predictions):
                assert isinstance(oof_pred, TimeSeriesDataFrame)
                loaded_trainer._score_with_predictions(oof_data[window_idx], oof_pred)


def test_when_known_covariates_present_then_all_ensemble_base_models_can_predict(temp_model_path):
    df = DATAFRAME_WITH_COVARIATES.copy()
    prediction_length = 2
    df_train = df.slice_by_timestep(None, -prediction_length)
    df_future = df.slice_by_timestep(-prediction_length, None)
    known_covariates = df_future.drop("target", axis=1)

    trainer = TimeSeriesTrainer(
        path=temp_model_path, prediction_length=prediction_length, enable_ensemble=False, cache_predictions=False
    )
    trainer.fit(
        df_train, hyperparameters={"ETS": {"maxiter": 1}, "DeepAR": {"max_epochs": 1, "num_batches_per_epoch": 1}}
    )

    # Manually add ensemble to ensure that both models have non-zero weight
    ensemble = GreedyEnsemble(name="WeightedEnsemble", path=trainer.path)
    ensemble.model_to_weight = {"DeepAR": 0.5, "ETS": 0.5}
    trainer._add_model(model=ensemble, base_models=["DeepAR", "ETS"])
    trainer.save_model(model=ensemble)
    with mock.patch("autogluon.timeseries.models.ensemble.weighted.greedy.GreedyEnsemble.predict") as mock_predict:
        trainer.predict(df_train, model="WeightedEnsemble", known_covariates=known_covariates)
        inputs = mock_predict.call_args[0][0]
        # No models failed during prediction
        assert inputs["DeepAR"] is not None
        assert inputs["ETS"] is not None


@pytest.fixture(scope="module")
def trained_and_refit_trainers():
    def fit_trainer():
        temp_model_path = tempfile.mkdtemp()
        trainer = TimeSeriesTrainer(
            path=temp_model_path,
            prediction_length=3,
        )
        trainer.fit(
            train_data=DUMMY_TS_DATAFRAME,
            hyperparameters={
                "Naive": {},
                "Theta": {},
                "DeepAR": {"max_epochs": 1, "num_batches_per_epoch": 1},
                "DirectTabular": {},
                "RecursiveTabular": {},
            },
        )
        return trainer

    trainer = fit_trainer()
    refit_trainer = fit_trainer()
    refit_trainer.refit_full("all")

    yield trainer, refit_trainer


def test_when_refit_full_called_then_all_models_are_retrained(trained_and_refit_trainers):
    trainer, refit_trainer = trained_and_refit_trainers
    leaderboard_initial = trainer.leaderboard()
    leaderboard_refit = refit_trainer.leaderboard()

    expected_refit_full_dict = {name: name + ag.constants.REFIT_FULL_SUFFIX for name in trainer.get_model_names()}
    assert dict_equal_primitive(refit_trainer.model_refit_map, expected_refit_full_dict)
    assert len(leaderboard_refit) == len(leaderboard_initial) + len(expected_refit_full_dict)


def test_when_refit_full_called_multiple_times_then_no_new_models_are_trained(trained_and_refit_trainers):
    _, refit_trainer = trained_and_refit_trainers
    model_names_refit = refit_trainer.get_model_names()
    refit_trainer.refit_full("all")
    model_names_second_refit = refit_trainer.get_model_names()
    assert set(model_names_refit) == set(model_names_second_refit)


def test_when_refit_full_called_then_all_models_can_predict(trained_and_refit_trainers):
    _, refit_trainer = trained_and_refit_trainers
    for model in refit_trainer.get_model_names():
        preds = refit_trainer.predict(DUMMY_TS_DATAFRAME, model=model)
        assert isinstance(preds, TimeSeriesDataFrame)
        assert len(preds) == DUMMY_TS_DATAFRAME.num_items * refit_trainer.prediction_length


def test_when_refit_full_called_with_model_name_then_single_model_is_updated(temp_model_path):
    trainer = TimeSeriesTrainer(path=temp_model_path)
    trainer.fit(
        DUMMY_TS_DATAFRAME,
        hyperparameters={
            "DeepAR": {"max_epochs": 1, "num_batches_per_epoch": 1},
            "SimpleFeedForward": {"max_epochs": 1, "num_batches_per_epoch": 1},
        },
    )
    model_refit_map = trainer.refit_full("DeepAR")
    assert list(model_refit_map.values()) == ["DeepAR_FULL"]


def test_given_quantile_levels_is_empty_when_refit_full_is_used_then_all_models_can_predict(temp_model_path):
    trainer = TimeSeriesTrainer(
        path=temp_model_path, ensemble_model_type=SimpleAverageEnsemble, quantile_levels=[], eval_metric="MAE"
    )
    trainer.fit(
        DUMMY_TS_DATAFRAME,
        hyperparameters={
            "Naive": {},
            "RecursiveTabular": {},
        },
    )
    trainer.refit_full(model="all")
    for model in trainer.get_model_names():
        preds = trainer.predict(DUMMY_TS_DATAFRAME, model=model)
        assert isinstance(preds, TimeSeriesDataFrame)
        assert len(preds) == DUMMY_TS_DATAFRAME.num_items * trainer.prediction_length


@pytest.mark.parametrize(
    "hyperparameters, expected_model_names",
    [
        ({"Naive": {}, "SeasonalNaiveModel": {}}, ["Naive", "SeasonalNaive"]),
        ({"Naive": {}, "NaiveModel": {}}, ["Naive", "Naive_2"]),
    ],
)
def test_when_some_models_have_incorrect_suffix_then_correct_model_are_trained(
    temp_model_path, hyperparameters, expected_model_names
):
    trainer = TimeSeriesTrainer(path=temp_model_path, enable_ensemble=False)
    trainer.fit(DUMMY_TS_DATAFRAME, hyperparameters=hyperparameters)
    leaderboard = trainer.leaderboard()
    assert sorted(leaderboard["model"].values) == expected_model_names


@pytest.mark.parametrize("excluded_model_types", [["DeepAR"], ["DeepARModel"]])
def test_when_excluded_model_names_provided_then_excluded_models_are_not_trained(
    temp_model_path, excluded_model_types
):
    trainer = TimeSeriesTrainer(path=temp_model_path)
    trainer.fit(
        DUMMY_TS_DATAFRAME,
        hyperparameters={
            "DeepAR": {"max_epochs": 1, "num_batches_per_epoch": 1},
            "SimpleFeedForward": {"max_epochs": 1, "num_batches_per_epoch": 1},
        },
        excluded_model_types=excluded_model_types,
    )
    leaderboard = trainer.leaderboard()
    assert leaderboard["model"].values == ["SimpleFeedForward"]


@pytest.mark.parametrize("model_names", [["WeightedEnsemble"], ["WeightedEnsemble", "DeepAR"], ["DeepAR"]])
def test_when_get_model_pred_dict_called_then_it_contains_all_required_keys(trained_trainers, model_names):
    trainer = trained_trainers[repr(TEST_HYPERPARAMETER_SETTINGS[1])]
    model_pred_dict, _ = trainer.get_model_pred_dict(model_names=model_names, data=DUMMY_TS_DATAFRAME)
    assert sorted(model_pred_dict.keys()) == sorted(model_names)


@pytest.mark.parametrize("model_names", [["WeightedEnsemble"], ["WeightedEnsemble", "DeepAR"], ["DeepAR"]])
def test_when_get_model_pred_dict_called_then_pred_time_dict_contains_all_required_keys(trained_trainers, model_names):
    trainer = trained_trainers[repr(TEST_HYPERPARAMETER_SETTINGS[1])]
    _, pred_time_dict = trainer.get_model_pred_dict(model_names=model_names, data=DUMMY_TS_DATAFRAME)
    assert sorted(pred_time_dict.keys()) == sorted(model_names)


def test_given_cache_predictions_is_true_when_calling_get_model_pred_dict_then_predictions_are_cached(temp_model_path):
    trainer = TimeSeriesTrainer(path=temp_model_path)
    trainer.fit(DUMMY_TS_DATAFRAME, hyperparameters={"Naive": {}, "SeasonalNaive": {}})

    assert isinstance(trainer.prediction_cache, FileBasedPredictionCache)
    assert not trainer.prediction_cache.path.exists()
    trainer.get_model_pred_dict(trainer.get_model_names(), data=DUMMY_TS_DATAFRAME)

    model_pred_dict, pred_time_dict = trainer.prediction_cache.get(DUMMY_TS_DATAFRAME, known_covariates=None)
    assert pred_time_dict.keys() == model_pred_dict.keys() == set(trainer.get_model_names())
    assert all(isinstance(v, TimeSeriesDataFrame) for v in model_pred_dict.values())
    assert all(isinstance(v, float) for v in pred_time_dict.values())


def test_given_cache_predictions_is_true_when_predicting_multiple_times_then_cached_predictions_are_updated(
    temp_model_path,
):
    trainer = TimeSeriesTrainer(path=temp_model_path)
    trainer.fit(DUMMY_TS_DATAFRAME, hyperparameters={"Naive": {}, "SeasonalNaive": {}})

    trainer.predict(DUMMY_TS_DATAFRAME, model="Naive")
    model_pred_dict, pred_time_dict = trainer.prediction_cache.get(DUMMY_TS_DATAFRAME, None)
    assert sorted(model_pred_dict.keys()) == sorted(pred_time_dict.keys()) == ["Naive"]

    trainer.predict(DUMMY_TS_DATAFRAME, model="SeasonalNaive")
    model_pred_dict, pred_time_dict = trainer.prediction_cache.get(DUMMY_TS_DATAFRAME, None)
    assert sorted(model_pred_dict.keys()) == sorted(pred_time_dict.keys()) == ["Naive", "SeasonalNaive"]


def test_given_cache_predictions_is_true_when_predicting_multiple_times_then_cached_predictions_are_used(
    temp_model_path,
):
    trainer = TimeSeriesTrainer(path=temp_model_path)
    trainer.fit(DUMMY_TS_DATAFRAME, hyperparameters={"Naive": {}})
    mock_predictions = pd.DataFrame(columns=["mean"] + [str(q) for q in trainer.quantile_levels])
    with mock.patch("autogluon.timeseries.models.local.naive.NaiveModel.predict") as naive_predict:
        naive_predict.return_value = mock_predictions
        trainer.predict(DUMMY_TS_DATAFRAME, model="Naive")

    with mock.patch("autogluon.timeseries.models.local.naive.NaiveModel.predict") as naive_predict:
        predictions = trainer.predict(DUMMY_TS_DATAFRAME, model="Naive")
        naive_predict.assert_not_called()
        assert predictions.equals(mock_predictions)


def test_given_cache_predictions_is_false_when_calling_get_model_pred_dict_then_predictions_are_not_cached(
    temp_model_path,
):
    trainer = TimeSeriesTrainer(path=temp_model_path, cache_predictions=False)
    assert isinstance(trainer.prediction_cache, NoOpPredictionCache)

    trainer.fit(DUMMY_TS_DATAFRAME, hyperparameters=DUMMY_TRAINER_HYPERPARAMETERS)
    trainer.get_model_pred_dict(trainer.get_model_names(), data=DUMMY_TS_DATAFRAME)

    assert not Path.exists(Path(temp_model_path) / FileBasedPredictionCache._cached_predictions_filename)


@pytest.mark.parametrize("method_name", ["leaderboard", "predict", "evaluate"])
@pytest.mark.parametrize("use_cache", [True, False])
def test_when_use_cache_is_set_to_false_then_cached_predictions_are_ignored(temp_model_path, use_cache, method_name):
    trainer = TimeSeriesTrainer(path=temp_model_path)
    trainer.fit(DUMMY_TS_DATAFRAME, hyperparameters={"Naive": {}})
    trainer.predict(DUMMY_TS_DATAFRAME)

    with mock.patch.object(trainer, "prediction_cache") as mock_cache:
        mock_cache.get.return_value = {}, {}
        getattr(trainer, method_name)(DUMMY_TS_DATAFRAME, use_cache=use_cache)
        if use_cache:
            mock_cache.get.assert_called()
        else:
            mock_cache.get.assert_not_called()


def test_given_cached_predictions_cannot_be_loaded_when_predict_call_then_new_predictions_are_generated(
    temp_model_path,
):
    trainer = TimeSeriesTrainer(path=temp_model_path)
    trainer.fit(DUMMY_TS_DATAFRAME, hyperparameters={"Naive": {}})
    trainer.predict(DUMMY_TS_DATAFRAME, model="Naive")

    assert isinstance(trainer.prediction_cache, FileBasedPredictionCache)

    # Corrupt the cached predictions file by writing a string into it
    trainer.prediction_cache.path.write_text("foo")

    with mock.patch("autogluon.timeseries.models.local.naive.NaiveModel.predict") as naive_predict:
        naive_predict.return_value = pd.DataFrame()
        trainer.predict(DUMMY_TS_DATAFRAME, model="Naive")
        naive_predict.assert_called()

    # Assert that predictions have been successfully stored
    assert isinstance(load_pkl.load(str(trainer.prediction_cache.path)), dict)


@pytest.mark.parametrize("use_test_data", [True, False])
def test_given_no_models_trained_during_fit_then_empty_leaderboard_returned(use_test_data, temp_model_path):
    trainer = TimeSeriesTrainer(path=temp_model_path)
    with mock.patch("autogluon.timeseries.models.local.naive.NaiveModel.fit") as naive_fit:
        naive_fit.side_effect = RuntimeError()
        trainer.fit(DUMMY_TS_DATAFRAME, hyperparameters={"Naive": {}})
    assert len(trainer.get_model_names()) == 0

    expected_columns = ["model", "score_val", "pred_time_val", "fit_time_marginal", "fit_order"]
    if use_test_data:
        expected_columns += ["score_test", "pred_time_test"]
        test_data = DUMMY_TS_DATAFRAME
    else:
        test_data = None

    leaderboard = trainer.leaderboard(data=test_data)
    assert all(c in leaderboard.columns for c in expected_columns)
    assert len(leaderboard) == 0


@pytest.mark.parametrize("skip_model_selection", [True, False])
def test_given_skip_model_selection_when_trainer_fits_then_val_score_is_not_computed(
    temp_model_path, skip_model_selection
):
    trainer = TimeSeriesTrainer(path=temp_model_path, skip_model_selection=skip_model_selection)
    trainer.fit(DUMMY_TS_DATAFRAME, hyperparameters={"Naive": {}})

    model = trainer.load_model("Naive")
    assert (model.val_score is None) == skip_model_selection


@pytest.mark.parametrize("confidence_level", [0.55, 0.65, 0.95, 0.99])
def test_when_add_ci_to_feature_importance_called_then_confidence_bands_correct(temp_model_path, confidence_level):
    import scipy.stats as sst

    trainer = TimeSeriesTrainer(path=temp_model_path)
    feature_importance = pd.DataFrame(
        {
            "importance": [10.0, 0.1, 0.2, 0.3, -0.5, np.nan, 0.2, 0.1, 55.0],
            "stdev": [0.1, 0.5, 3.0, 1.0, 1.5, 0.1, np.nan, 0.1, 0.1],
            "n": [3, 4, 5, 6, 2, 5, 5, np.nan, 1],
        },
        index=["a", "b", "c", "d", "e", "f", "g", "h", "i"],
    )

    feature_importance = trainer._add_ci_to_feature_importance(feature_importance, confidence_level)
    lower_ci_name, upper_ci_name = [f"p{confidence_level * 100:.0f}_{k}" for k in ["low", "high"]]
    assert lower_ci_name in feature_importance.columns
    assert upper_ci_name in feature_importance.columns

    alpha = 1 - confidence_level

    for i, r in feature_importance.iterrows():
        if np.isnan(r["stdev"]) or np.isnan(r["n"]) or np.isnan(r["importance"]) or r["n"] == 1:  # type: ignore
            assert np.isnan(r[lower_ci_name])
            assert np.isnan(r[upper_ci_name])
        else:
            t_critical = sst.t.ppf(1 - alpha / 2, df=r["n"] - 1)

            expected_lower = r["importance"] - t_critical * r["stdev"] / np.sqrt(r["n"])
            expected_upper = r["importance"] + t_critical * r["stdev"] / np.sqrt(r["n"])

            assert np.isclose(r[lower_ci_name], expected_lower)
            assert np.isclose(r[upper_ci_name], expected_upper)


class TestEnsembleTraining:
    def test_given_multiple_ensemble_hyperparameters_when_trainer_fit_then_multiple_ensembles_created(
        self, tmp_path, patch_models
    ):
        trainer = TimeSeriesTrainer(path=str(tmp_path), prediction_length=3)
        trainer.fit(
            train_data=DUMMY_TS_DATAFRAME,
            hyperparameters={"Naive": {}, "SeasonalNaive": {}},
            ensemble_hyperparameters={
                "GreedyEnsemble": {},
                "PerformanceWeightedEnsemble": {},
                "SimpleAverageEnsemble": {},
            },
        )

        model_names = trainer.get_model_names()
        ensemble_names = [name for name in model_names if "Ensemble" in name]
        expected_names = ["WeightedEnsemble", "PerformanceWeightedEnsemble", "SimpleAverageEnsemble"]

        assert len(ensemble_names) == 3
        assert set(expected_names) == set(ensemble_names)

    def test_given_default_hyperparameters_when_trainer_fit_then_single_ensemble_created(self, tmp_path, patch_models):
        trainer = TimeSeriesTrainer(path=str(tmp_path), prediction_length=3)
        trainer.fit(
            train_data=DUMMY_TS_DATAFRAME,
            hyperparameters={"Naive": {}, "SeasonalNaive": {}},
        )

        model_names = trainer.get_model_names()
        ensemble_names = [name for name in model_names if "Ensemble" in name]
        assert ensemble_names == ["WeightedEnsemble"]

    def test_given_multiple_ensembles_with_mixed_hyperparameters_when_trainer_fit_then_all_ensembles_can_get_hyperparameters(
        self, tmp_path, patch_models
    ):
        trainer = TimeSeriesTrainer(path=str(tmp_path), prediction_length=3)
        trainer.fit(
            train_data=DUMMY_TS_DATAFRAME,
            hyperparameters={"Naive": {}, "SeasonalNaive": {}},
            ensemble_hyperparameters={
                "GreedyEnsemble": {"ensemble_size": 25},
                "PerformanceWeightedEnsemble": {"weight_mode": "sqrt"},
            },
        )

        weighted = trainer.load_model("WeightedEnsemble")
        assert weighted.get_hyperparameters()["ensemble_size"] == 25

        performance_weighted = trainer.load_model("PerformanceWeightedEnsemble")
        assert performance_weighted.get_hyperparameters()["weight_mode"] == "sqrt"

    def test_given_empty_ensemble_hyperparameters_when_trainer_fit_then_ensemble_training_disabled(
        self, tmp_path, patch_models
    ):
        trainer = TimeSeriesTrainer(path=str(tmp_path), prediction_length=3)
        trainer.fit(
            train_data=DUMMY_TS_DATAFRAME,
            hyperparameters={"Naive": {}, "SeasonalNaive": {}},
            ensemble_hyperparameters={},  # Empty dict should disable ensembles
        )

        model_names = trainer.get_model_names()
        ensemble_names = [name for name in model_names if "Ensemble" in name]

        assert len(ensemble_names) == 0
        assert len(model_names) == 2

    def test_given_enable_ensemble_false_when_trainer_initialized_then_ensemble_training_disabled(
        self, tmp_path, patch_models
    ):
        trainer = TimeSeriesTrainer(path=str(tmp_path), prediction_length=3, enable_ensemble=False)
        trainer.fit(
            train_data=DUMMY_TS_DATAFRAME,
            hyperparameters={"Naive": {}, "SeasonalNaive": {}},
        )

        model_names = trainer.get_model_names()
        ensemble_names = [name for name in model_names if "Ensemble" in name]

        assert len(ensemble_names) == 0
        assert len(model_names) == 2


TWO_LAYER_ENSEMBLE_HPS = [
    [
        {"GreedyEnsemble": [{"ensemble_size": 2}, {"ensemble_size": 3}]},
        {"PerformanceWeightedEnsemble": {"weight_mode": "sqrt"}},
    ],
    [
        {"GreedyEnsemble": {"ensemble_size": 2}},
        {"SimpleAverageEnsemble": {}, "GreedyEnsemble": {"ensemble_size": 2}},
    ],
]
TWO_LAYER_NUM_VAL_WINDOWS = [(1, 2), (2, 3)]


class TestMultilayerEnsembleTraining:
    @pytest.fixture(scope="class")
    def train_and_val_data(self):
        train_data = get_data_frame_with_variable_lengths({"A": 50, "B": 40, "1": 100})
        val_data = get_data_frame_with_variable_lengths({"A": 100, "B": 200, "1": 300})
        yield train_data, val_data

    @pytest.fixture(
        params=list(
            itertools.product(
                TWO_LAYER_ENSEMBLE_HPS,
                TWO_LAYER_NUM_VAL_WINDOWS,
                [True, False],  # use_val_data
            )
        )
        + [
            ([{"GreedyEnsemble": {"ensemble_size": 2}}], (1,), True),
            ([{"GreedyEnsemble": {"ensemble_size": 2}}], (1,), False),
        ]
    )
    def trainer_and_params(self, tmp_path_factory, patch_models, request, train_and_val_data):
        ensemble_hyperparameters, num_val_windows, use_val_data = request.param
        train_data, val_data = train_and_val_data
        if use_val_data:
            num_val_windows = num_val_windows[:-1] + (1,)
        else:
            val_data = None

        trainer = TimeSeriesTrainer(
            path=str(tmp_path_factory.mktemp("agts_multilayer_trainer")),
            prediction_length=3,
            num_val_windows=num_val_windows,
        )
        trainer.fit(
            train_data=train_data,
            val_data=val_data,
            hyperparameters={"Naive": {}, "SeasonalNaive": {}},
            ensemble_hyperparameters=ensemble_hyperparameters,
        )

        yield trainer, (num_val_windows, use_val_data)

    def test_when_trainer_fit_then_number_of_ensembles_correct(self, trainer_and_params):
        trainer, (num_val_windows, _) = trainer_and_params

        # Single layer: 2 base + 1 ensemble = 3
        # Two layers: 2 base + 3 ensembles = 5
        expected_models = 3 if len(num_val_windows) == 1 else 5
        assert len(trainer.model_graph) == expected_models

    def test_when_trainer_fit_models_are_not_wrapped_only_when_not_necessary(self, trainer_and_params):
        trainer, (num_val_windows, use_val_data) = trainer_and_params

        if use_val_data:
            assert num_val_windows[-1] == 1
        multi_window_expected = not (use_val_data and sum(num_val_windows[:-1]) == 0)

        all_models = [trainer.load_model(name) for name in trainer.get_model_names()]

        for model in all_models:
            if isinstance(model, AbstractTimeSeriesEnsembleModel):
                continue
            if multi_window_expected:
                assert isinstance(model, MultiWindowBacktestingModel)
            else:
                assert not isinstance(model, MultiWindowBacktestingModel)

    def test_when_trainer_fit_then_base_model_validation_scores_use_last_layer_windows(self, trainer_and_params):
        trainer, (num_val_windows, use_val_data) = trainer_and_params

        all_models = [trainer.load_model(name) for name in trainer.get_model_names()]

        expected_num_windows = num_val_windows[-1]

        for model in all_models:
            if hasattr(model, "info_per_val_window"):
                assert len(model.info_per_val_window) >= expected_num_windows
                expected_val_score = float(
                    np.mean([info["val_score"] for info in model.info_per_val_window[-expected_num_windows:]])
                )
                assert abs(model.val_score - expected_val_score) < 1e-6

            assert model.val_score is not None

    def test_when_trainer_fit_then_last_window_dates_are_correct(self, trainer_and_params, train_and_val_data):
        trainer, (num_val_windows, use_val_data) = trainer_and_params
        train_data, val_data = train_and_val_data

        all_models = [trainer.load_model(name) for name in trainer.get_model_names()]

        for model in all_models:
            last_oof = model.get_oof_predictions()[-1].slice_by_timestep(-3, None)
            if use_val_data:
                assert last_oof.index.equals(val_data.slice_by_timestep(-3, None).index)
            else:
                assert last_oof.index.equals(train_data.slice_by_timestep(-3, None).index)

    def test_when_trainer_fit_then_base_models_have_complete_oof_predictions(self, trainer_and_params):
        trainer, (num_val_windows, use_val_data) = trainer_and_params

        base_model_names = ["Naive", "SeasonalNaive"]

        if use_val_data:
            expected_total_windows = sum(num_val_windows[:-1]) + 1  # +1 for val_data
        else:
            expected_total_windows = sum(num_val_windows)

        for model_name in base_model_names:
            model = trainer.load_model(model_name)

            # Check that model has info_per_val_window (only for MultiWindowBacktestingModel)
            if hasattr(model, "info_per_val_window"):
                # info_per_val_window only has train windows (not val_data)
                expected_info_windows = sum(num_val_windows[:-1]) if use_val_data else sum(num_val_windows)
                assert len(model.info_per_val_window) == expected_info_windows

            # Check that OOF predictions exist and cover all windows (including val_data if provided)
            oof_predictions = trainer._get_model_oof_predictions(model_name)
            assert len(oof_predictions) == expected_total_windows
