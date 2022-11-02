"""Unit tests for trainers"""
import copy
import os
import shutil
import tempfile
from collections import defaultdict
from unittest import mock

import numpy as np
import pytest
from gluonts.model.prophet import PROPHET_IS_INSTALLED

import autogluon.core as ag
from autogluon.timeseries.dataset import TimeSeriesDataFrame
from autogluon.timeseries.models import DeepARModel, ETSModel
from autogluon.timeseries.models.ensemble.greedy_ensemble import TimeSeriesEnsembleWrapper
from autogluon.timeseries.trainer.auto_trainer import AutoTimeSeriesTrainer

from .common import DUMMY_TS_DATAFRAME, get_data_frame_with_item_index

DUMMY_TRAINER_HYPERPARAMETERS = {"SimpleFeedForward": {"epochs": 1}}
TEST_HYPERPARAMETER_SETTINGS = [
    {"SimpleFeedForward": {"epochs": 1}},
    {"DeepAR": {"epochs": 1}, "ETS": {}},
]
TEST_HYPERPARAMETER_SETTINGS_EXPECTED_LB_LENGTHS = [1, 2]


@pytest.fixture(scope="module")
def trained_trainers():
    trainers = {}
    model_paths = []
    for hp in TEST_HYPERPARAMETER_SETTINGS:
        temp_model_path = tempfile.mkdtemp()
        trainer = AutoTimeSeriesTrainer(
            path=temp_model_path + os.path.sep,
            eval_metric="MAPE",
            prediction_length=3,
        )
        trainer.fit(
            train_data=DUMMY_TS_DATAFRAME,
            val_data=DUMMY_TS_DATAFRAME,
            hyperparameters=hp,
        )
        trainers[repr(hp)] = trainer
        model_paths.append(temp_model_path)

    yield trainers

    for td in model_paths:
        shutil.rmtree(td)


def test_trainer_can_be_initialized(temp_model_path):
    model = AutoTimeSeriesTrainer(path=temp_model_path, prediction_length=24)
    assert isinstance(model, AutoTimeSeriesTrainer)


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
    trainer = AutoTimeSeriesTrainer(path=temp_model_path, eval_metric=eval_metric)
    trainer.fit(
        train_data=DUMMY_TS_DATAFRAME,
        hyperparameters=hyperparameters,
        val_data=DUMMY_TS_DATAFRAME,
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
        {"DeepAR": {"epochs": 4}, "SimpleFeedForward": {"epochs": 1}},
        {
            "SimpleFeedForward": {"context_length": 44, "epochs": 2},
            "DeepAR": {"epochs": 3},
        },
    ],
)
def test_given_hyperparameters_when_trainer_model_templates_called_then_hyperparameters_set_correctly(
    temp_model_path, hyperparameters
):
    trainer = AutoTimeSeriesTrainer(path=temp_model_path, eval_metric="MAPE")
    models = trainer.construct_model_templates(
        hyperparameters=hyperparameters,
    )

    for model in models:
        for k, v in hyperparameters[model.name].items():
            assert model._user_params[k] == v


@pytest.mark.parametrize(
    "hyperparameters",
    [
        {"DeepAR": {"epochs": 1}, "SimpleFeedForward": {"epochs": 1}},
        {
            "SimpleFeedForward": {"context_length": 44, "epochs": 1},
            "DeepAR": {"epochs": 1},
        },
    ],
)
def test_given_hyperparameters_when_trainer_fit_then_freq_set_correctly(temp_model_path, hyperparameters):
    trainer = AutoTimeSeriesTrainer(path=temp_model_path, eval_metric="MAPE")
    trainer.fit(
        train_data=DUMMY_TS_DATAFRAME,
        hyperparameters=hyperparameters,
        val_data=DUMMY_TS_DATAFRAME,
    )

    for model_name in trainer.get_model_names():
        model = trainer.load_model(model_name)
        assert model.freq == DUMMY_TS_DATAFRAME.freq


@pytest.mark.parametrize("model_name", ["DeepAR", "SimpleFeedForward"])
def test_given_hyperparameters_with_spaces_when_trainer_called_then_hpo_is_performed(temp_model_path, model_name):
    hyperparameters = {model_name: {"epochs": ag.Int(1, 4)}}
    # mock the default hps factory to prevent preset hyperparameter configurations from
    # creeping into the test case
    with mock.patch("autogluon.timeseries.models.presets.get_default_hps") as default_hps_mock:
        default_hps_mock.return_value = defaultdict(dict)
        trainer = AutoTimeSeriesTrainer(path=temp_model_path)
        trainer.fit(
            train_data=DUMMY_TS_DATAFRAME,
            hyperparameters=hyperparameters,
            val_data=DUMMY_TS_DATAFRAME,
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
    assert all(1 <= config["epochs"] <= 4 for config in config_history)


@pytest.mark.skipif(not PROPHET_IS_INSTALLED, reason="Prophet is not installed.")
@pytest.mark.parametrize("eval_metric", ["MAPE", None])
@pytest.mark.parametrize(
    "hyperparameters, expected_board_length",
    [({"Prophet": {}, "DeepAR": {"epochs": 1}}, 2), ({"Prophet": {}}, 1)],
)
def test_given_hyperparameters_to_prophet_when_trainer_called_then_leaderboard_is_correct(
    temp_model_path, eval_metric, hyperparameters, expected_board_length
):
    trainer = AutoTimeSeriesTrainer(path=temp_model_path, eval_metric=eval_metric)
    trainer.fit(
        train_data=DUMMY_TS_DATAFRAME,
        hyperparameters=hyperparameters,
        val_data=DUMMY_TS_DATAFRAME,
    )
    leaderboard = trainer.leaderboard()

    expected_board_length += int(trainer.enable_ensemble)
    assert len(leaderboard) == expected_board_length
    assert np.all(leaderboard["score_val"] < 0)  # all MAPEs should be negative


@pytest.mark.skipif(not PROPHET_IS_INSTALLED, reason="Prophet is not installed.")
@pytest.mark.parametrize(
    "hyperparameters",
    [
        {"Prophet": {"n_changepoints": 4}, "SimpleFeedForward": {"epochs": 1}},
        {"Prophet": {"mcmc_samples": 44}, "DeepAR": {"epochs": 3}},
    ],
)
def test_given_hyperparameters_to_prophet_when_trainer_model_templates_called_then_hyperparameters_set_correctly(
    temp_model_path, hyperparameters
):
    trainer = AutoTimeSeriesTrainer(path=temp_model_path, eval_metric="MAPE")
    models = trainer.construct_model_templates(
        hyperparameters=hyperparameters,
    )

    for model in models:
        for k, v in hyperparameters[model.name].items():
            assert model._user_params[k] == v


@pytest.mark.skipif(not PROPHET_IS_INSTALLED, reason="Prophet is not installed.")
def test_given_hyperparameters_with_spaces_to_prophet_when_trainer_called_then_hpo_is_performed(
    temp_model_path,
):
    hyperparameters = {"Prophet": {"n_changepoints": ag.Int(1, 4)}}
    # mock the default hps factory to prevent preset hyperparameter configurations from
    # creeping into the test case
    with mock.patch("autogluon.timeseries.models.presets.get_default_hps") as default_hps_mock:
        default_hps_mock.return_value = defaultdict(dict)
        trainer = AutoTimeSeriesTrainer(path=temp_model_path)
        trainer.fit(
            train_data=DUMMY_TS_DATAFRAME,
            hyperparameters=hyperparameters,
            val_data=DUMMY_TS_DATAFRAME,
            hyperparameter_tune_kwargs={
                "num_trials": 2,
                "searcher": "random",
                "scheduler": "local",
            },
        )
        leaderboard = trainer.leaderboard()

    assert len(leaderboard) == 2 + 1  # include ensemble
    assert all([1 <= v["params"]["epochs"] < 5 for k, v in trainer.model_graph.nodes.items()])


@pytest.mark.parametrize("eval_metric", ["MAPE", None])
@pytest.mark.parametrize(
    "hyperparameters, expected_board_length",
    [
        ({DeepARModel: {"epochs": 1}}, 1),
        (
            {
                ETSModel: {},
                DeepARModel: {"epochs": 1},
            },
            2,
        ),
    ],
)
def test_given_hyperparameters_and_custom_models_when_trainer_called_then_leaderboard_is_correct(
    temp_model_path, eval_metric, hyperparameters, expected_board_length
):
    trainer = AutoTimeSeriesTrainer(path=temp_model_path, eval_metric=eval_metric)
    trainer.fit(
        train_data=DUMMY_TS_DATAFRAME,
        hyperparameters=hyperparameters,
        val_data=DUMMY_TS_DATAFRAME,
    )
    leaderboard = trainer.leaderboard()

    if len(hyperparameters) > 1:  # account for ensemble
        expected_board_length += int(trainer.enable_ensemble)
    assert len(leaderboard) == expected_board_length
    assert np.all(leaderboard["score_val"] < 0)  # all MAPEs should be negative


@pytest.mark.parametrize(
    "hyperparameter_list, expected_number_of_unique_names, expected_suffixes",
    [
        ([{DeepARModel: {"epochs": 1}}], 1, []),
        (
            [
                {DeepARModel: {"epochs": 1}},
                {DeepARModel: {"epochs": 1}},
            ],
            2,
            ["AR_2"],
        ),
        (
            [
                {DeepARModel: {"epochs": 1}},
                {DeepARModel: {"epochs": 1}},
                {DeepARModel: {"epochs": 1}},
            ],
            3,
            ["AR_2", "AR_3"],
        ),
        # FIXME: model name collision prevention is broken
        # (
        #     [
        #         {DeepARModel: {"epochs": 1}},
        #         {DeepARModel: {"epochs": 1}},
        #         {
        #             DeepARModel: {"epochs": 1},
        #             DeepARModel: {"epochs": 1},
        #         },
        #     ],
        #     4,
        #     ["AR_2", "AR_3", "AR_4"],
        # ),
        # (
        #     [
        #         {
        #             DeepARModel: {"epochs": 1},
        #             DeepARModel: {"epochs": 1},
        #             DeepARModel: {"epochs": 1},
        #         },
        #         {
        #             DeepARModel: {"epochs": 1},
        #             DeepARModel: {"epochs": 1},
        #         },
        #     ],
        #     5,
        #     ["AR_2", "AR_3", "AR_4", "AR_5"],
        # ),
    ],
)
def test_given_repeating_model_when_trainer_called_incrementally_then_name_collisions_are_prevented(
    temp_model_path,
    hyperparameter_list,
    expected_number_of_unique_names,
    expected_suffixes,
):
    trainer = AutoTimeSeriesTrainer(path=temp_model_path)

    # incrementally train with new hyperparameters
    for hp in hyperparameter_list:
        trainer.fit(
            train_data=DUMMY_TS_DATAFRAME,
            hyperparameters=hp,
            val_data=DUMMY_TS_DATAFRAME,
        )

    model_names = trainer.get_model_names()

    # account for the ensemble if it should be fitted, and drop ensemble names
    if trainer.enable_ensemble and sum(len(hp) for hp in hyperparameter_list) > 1:
        model_names = [n for n in model_names if "WeightedEnsemble" not in n]
    assert len(model_names) == expected_number_of_unique_names
    for suffix in expected_suffixes:
        assert any(name.endswith(suffix) for name in model_names)

    if not trainer.enable_ensemble:
        # there should be no edges in the model graph without ensembling
        assert not trainer.model_graph.edges


@pytest.mark.parametrize(
    "hyperparameters",
    [
        {"DeepAR": {"epochs": 1}, "SimpleFeedForward": {"epochs": 1}},
        {
            "SimpleFeedForward": {"context_length": 44, "epochs": 1},
            "DeepAR": {"epochs": 1},
        },
    ],
)
@pytest.mark.parametrize("low_memory", [True, False])
def test_when_trainer_fit_and_deleted_models_load_back_correctly_and_can_predict(
    temp_model_path, hyperparameters, low_memory
):
    trainer = AutoTimeSeriesTrainer(path=temp_model_path, eval_metric="MAPE", prediction_length=2)
    trainer.fit(
        train_data=DUMMY_TS_DATAFRAME,
        hyperparameters=hyperparameters,
        val_data=DUMMY_TS_DATAFRAME,
    )
    model_names = copy.copy(trainer.get_model_names())
    trainer.save()
    del trainer

    loaded_trainer = AutoTimeSeriesTrainer.load(path=temp_model_path)

    for m in model_names:
        loaded_model = loaded_trainer.load_model(m)
        if isinstance(loaded_model, TimeSeriesEnsembleWrapper):
            continue

        predictions = loaded_model.predict(DUMMY_TS_DATAFRAME)

        assert isinstance(predictions, TimeSeriesDataFrame)

        predicted_item_index = predictions.item_ids
        assert all(predicted_item_index == DUMMY_TS_DATAFRAME.item_ids)  # noqa
        assert all(len(predictions.loc[i]) == 2 for i in predicted_item_index)
        assert not np.any(np.isnan(predictions))


@pytest.mark.parametrize("failing_model", ["NaiveModel", "SeasonalNaiveModel"])
def test_given_base_model_fails_when_trainer_predicts_then_weighted_ensemble_can_predict(
    temp_model_path, failing_model
):
    trainer = AutoTimeSeriesTrainer(path=temp_model_path, enable_ensemble=False)
    trainer.fit(train_data=DUMMY_TS_DATAFRAME, hyperparameters={"Naive": {}, "SeasonalNaive": {}})
    ensemble = TimeSeriesEnsembleWrapper(weights={"Naive": 0.5, "SeasonalNaive": 0.5}, name="WeightedEnsemble")
    trainer._add_model(ensemble, base_models=["Naive", "SeasonalNaive"])

    with mock.patch(f"autogluon.timeseries.models.local.models.{failing_model}.predict") as fail_predict:
        fail_predict.side_effect = RuntimeError("Numerical error")
        preds = trainer.predict(DUMMY_TS_DATAFRAME, model="WeightedEnsemble")
        fail_predict.assert_called()
        assert isinstance(preds, TimeSeriesDataFrame)


@pytest.mark.parametrize("failing_model", ["NaiveModel", "SeasonalNaiveModel"])
def test_given_base_model_fails_when_trainer_scores_then_weighted_ensemble_can_score(temp_model_path, failing_model):
    trainer = AutoTimeSeriesTrainer(path=temp_model_path, enable_ensemble=False)
    trainer.fit(train_data=DUMMY_TS_DATAFRAME, hyperparameters={"Naive": {}, "SeasonalNaive": {}})
    ensemble = TimeSeriesEnsembleWrapper(weights={"Naive": 0.5, "SeasonalNaive": 0.5}, name="WeightedEnsemble")
    trainer._add_model(ensemble, base_models=["Naive", "SeasonalNaive"])

    with mock.patch(f"autogluon.timeseries.models.local.models.{failing_model}.predict") as fail_predict:
        fail_predict.side_effect = RuntimeError("Numerical error")
        score = trainer.score(DUMMY_TS_DATAFRAME, model="WeightedEnsemble")
        fail_predict.assert_called()
        assert isinstance(score, float)
