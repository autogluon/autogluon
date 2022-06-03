"""Unit tests for trainers"""
import copy
from collections import defaultdict
from unittest import mock

import numpy as np
import pytest
from gluonts.model.prophet import PROPHET_IS_INSTALLED
from gluonts.model.seq2seq import MQRNNEstimator

import autogluon.core as ag
from autogluon.timeseries.dataset import TimeSeriesDataFrame
from autogluon.timeseries.models import DeepARModel
from autogluon.timeseries.models.gluonts import GenericGluonTSModel
from autogluon.timeseries.models.gluonts.models import GenericGluonTSModelFactory
from autogluon.timeseries.models.presets import get_default_hps
from autogluon.timeseries.trainer.auto_trainer import AutoTimeSeriesTrainer

from .common import DUMMY_TS_DATAFRAME

DUMMY_TRAINER_HYPERPARAMETERS = {"SimpleFeedForward": {"epochs": 1}}
TEST_HYPERPARAMETER_SETTINGS = [
    "toy",
    DUMMY_TRAINER_HYPERPARAMETERS,
    {"DeepAR": {"epochs": 1}, "SimpleFeedForward": {"epochs": 1}},
]


def test_trainer_can_be_initialized(temp_model_path):
    model = AutoTimeSeriesTrainer(path=temp_model_path, prediction_length=24)
    assert isinstance(model, AutoTimeSeriesTrainer)


# smoke test for the short 'happy path'
def test_when_trainer_called_then_training_is_performed(temp_model_path):
    trainer = AutoTimeSeriesTrainer(path=temp_model_path)
    trainer.fit(
        train_data=DUMMY_TS_DATAFRAME, hyperparameters=DUMMY_TRAINER_HYPERPARAMETERS
    )

    assert "SimpleFeedForward" in trainer.get_model_names()


def test_given_validation_data_when_trainer_called_then_training_is_performed(
    temp_model_path,
):
    trainer = AutoTimeSeriesTrainer(path=temp_model_path)
    trainer.fit(
        train_data=DUMMY_TS_DATAFRAME,
        hyperparameters=DUMMY_TRAINER_HYPERPARAMETERS,
        val_data=DUMMY_TS_DATAFRAME,
    )

    assert "SimpleFeedForward" in trainer.get_model_names()
    val_score = trainer.get_model_attribute("SimpleFeedForward", "val_score")
    assert val_score is not None


@pytest.mark.parametrize("eval_metric", ["MAPE", None])
@pytest.mark.parametrize(
    "hyperparameters, expected_board_length",
    zip(TEST_HYPERPARAMETER_SETTINGS, [len(get_default_hps("toy", 1)), 1, 2]),
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

    assert len(leaderboard) == expected_board_length
    assert np.all(leaderboard["score_val"] < 0)  # all MAPEs should be negative


@pytest.mark.parametrize(
    "hyperparameters, expected_board_length",
    zip(TEST_HYPERPARAMETER_SETTINGS, [len(get_default_hps("toy", 1)), 1, 2]),
)
def test_given_hyperparameters_when_trainer_called_then_model_can_predict(
    temp_model_path, hyperparameters, expected_board_length
):
    trainer = AutoTimeSeriesTrainer(
        path=temp_model_path,
        prediction_length=3,
    )
    trainer.fit(
        train_data=DUMMY_TS_DATAFRAME,
        hyperparameters=hyperparameters,
        val_data=DUMMY_TS_DATAFRAME,
    )
    predictions = trainer.predict(DUMMY_TS_DATAFRAME)

    assert isinstance(predictions, TimeSeriesDataFrame)

    predicted_item_index = predictions.index.levels[0]
    assert all(predicted_item_index == DUMMY_TS_DATAFRAME.index.levels[0])  # noqa
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
def test_given_hyperparameters_when_trainer_fit_then_freq_set_correctly(
    temp_model_path, hyperparameters
):
    trainer = AutoTimeSeriesTrainer(path=temp_model_path, eval_metric="MAPE")
    trainer.fit(
        train_data=DUMMY_TS_DATAFRAME,
        hyperparameters=hyperparameters,
        val_data=DUMMY_TS_DATAFRAME,
    )

    for model_name in trainer.get_model_names():
        model = trainer.load_model(model_name)
        assert model.freq == DUMMY_TS_DATAFRAME.freq


@pytest.mark.parametrize("model_name", ["DeepAR", "SimpleFeedForward", "MQCNN"])
def test_given_hyperparameters_with_spaces_when_trainer_called_then_hpo_is_performed(
    temp_model_path, model_name
):
    hyperparameters = {model_name: {"epochs": ag.Int(1, 4)}}
    # mock the default hps factory to prevent preset hyperparameter configurations from
    # creeping into the test case
    with mock.patch(
        "autogluon.timeseries.models.presets.get_default_hps"
    ) as default_hps_mock:
        default_hps_mock.return_value = defaultdict(dict)
        trainer = AutoTimeSeriesTrainer(path=temp_model_path)
        trainer.fit(
            train_data=DUMMY_TS_DATAFRAME,
            hyperparameters=hyperparameters,
            val_data=DUMMY_TS_DATAFRAME,
            hyperparameter_tune=True,
        )
        leaderboard = trainer.leaderboard()

    assert len(leaderboard) == 4

    config_history = trainer.hpo_results[model_name]["config_history"]
    assert len(config_history) == 4
    assert all(1 <= model["epochs"] <= 4 for model in config_history.values())


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
    with mock.patch(
        "autogluon.timeseries.models.presets.get_default_hps"
    ) as default_hps_mock:
        default_hps_mock.return_value = defaultdict(dict)
        trainer = AutoTimeSeriesTrainer(path=temp_model_path)
        trainer.fit(
            train_data=DUMMY_TS_DATAFRAME,
            hyperparameters=hyperparameters,
            val_data=DUMMY_TS_DATAFRAME,
            hyperparameter_tune=True,
        )
        leaderboard = trainer.leaderboard()

    assert len(leaderboard) == 4

    config_history = trainer.hpo_results["Prophet"]["config_history"]
    assert len(config_history) == 4
    assert all(1 <= model["n_changepoints"] <= 4 for model in config_history.values())


@pytest.mark.parametrize("eval_metric", ["MAPE", None])
@pytest.mark.parametrize(
    "hyperparameters, expected_board_length",
    [
        ({DeepARModel: {"epochs": 1}}, 1),
        (
            {
                GenericGluonTSModelFactory(MQRNNEstimator): {"epochs": 1},
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

    assert len(leaderboard) == expected_board_length
    assert np.all(leaderboard["score_val"] < 0)  # all MAPEs should be negative


@pytest.mark.parametrize(
    "hyperparameter_list, expected_number_of_unique_names, expected_suffixes",
    [
        ([{DeepARModel: {"epochs": 1}}], 1, []),
        (
            [
                {
                    GenericGluonTSModelFactory(MQRNNEstimator): {"epochs": 1},
                    GenericGluonTSModelFactory(MQRNNEstimator): {"epochs": 1},
                }
            ],
            2,
            ["RNN_2"],
        ),
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
                {
                    GenericGluonTSModelFactory(MQRNNEstimator): {"epochs": 1},
                    GenericGluonTSModelFactory(MQRNNEstimator): {"epochs": 1},
                    GenericGluonTSModelFactory(MQRNNEstimator): {"epochs": 1},
                }
            ],
            3,
            ["RNN_2", "RNN_3"],
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
        (
            [
                {GenericGluonTSModelFactory(MQRNNEstimator): {"epochs": 1}},
                {GenericGluonTSModelFactory(MQRNNEstimator): {"epochs": 1}},
                {
                    GenericGluonTSModelFactory(MQRNNEstimator): {"epochs": 1},
                    GenericGluonTSModelFactory(MQRNNEstimator): {"epochs": 1},
                },
            ],
            4,
            ["RNN_2", "RNN_3", "RNN_4"],
        ),
        (
            [
                {
                    GenericGluonTSModelFactory(MQRNNEstimator): {"epochs": 1},
                    GenericGluonTSModelFactory(MQRNNEstimator): {"epochs": 1},
                    GenericGluonTSModelFactory(MQRNNEstimator): {"epochs": 1},
                },
                {
                    GenericGluonTSModelFactory(MQRNNEstimator): {"epochs": 1},
                    GenericGluonTSModelFactory(MQRNNEstimator): {"epochs": 1},
                },
            ],
            5,
            ["RNN_2", "RNN_3", "RNN_4", "RNN_5"],
        ),
        (
            [
                {
                    GenericGluonTSModelFactory(MQRNNEstimator): {"epochs": 1},
                    GenericGluonTSModelFactory(MQRNNEstimator): {"epochs": 1},
                    GenericGluonTSModelFactory(MQRNNEstimator, name="MQRNN_2"): {
                        "epochs": 1
                    },
                },
            ],
            3,
            ["RNN_2", "RNN_2_2"],
        ),
        (
            [
                {
                    GenericGluonTSModelFactory(MQRNNEstimator): {"epochs": 1},
                    GenericGluonTSModelFactory(MQRNNEstimator): {"epochs": 1},
                },
                {
                    GenericGluonTSModelFactory(MQRNNEstimator): {"epochs": 1},
                    GenericGluonTSModelFactory(MQRNNEstimator): {"epochs": 1},
                    GenericGluonTSModelFactory(MQRNNEstimator, name="MQRNN_2"): {
                        "epochs": 1
                    },
                },
                {
                    GenericGluonTSModelFactory(MQRNNEstimator): {"epochs": 1},
                    GenericGluonTSModelFactory(MQRNNEstimator, name="MQRNN_2"): {
                        "epochs": 1
                    },
                },
            ],
            7,
            ["RNN", "RNN_2", "RNN_3", "RNN_4", "RNN_5", "RNN_2_2", "RNN_2_3"],
        ),
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

    assert len(model_names) == expected_number_of_unique_names
    for suffix in expected_suffixes:
        assert any(name.endswith(suffix) for name in model_names)

    # there should be no edges in the model graph without ensembling
    assert not trainer.model_graph.edges


@pytest.mark.parametrize(
    "hyperparameters",
    [
        {
            GenericGluonTSModelFactory(MQRNNEstimator): {
                "context_length": 4,
                "epochs": 1,
            },
            "SimpleFeedForward": {"epochs": 1},
        },
        {
            GenericGluonTSModelFactory(MQRNNEstimator): {"epochs": 2},
            "DeepAR": {"epochs": 2},
        },
    ],
)
def test_given_hyperparameters_and_custom_models_when_trainer_model_templates_called_then_hyperparameters_set_correctly(
    temp_model_path, hyperparameters
):
    trainer = AutoTimeSeriesTrainer(path=temp_model_path, eval_metric="MAPE")
    models = trainer.construct_model_templates(
        hyperparameters=hyperparameters,
    )

    for model in models:
        if isinstance(model, GenericGluonTSModel):
            model_hyperparam = next(
                hyperparameters[m]
                for m in hyperparameters
                if isinstance(m, GenericGluonTSModelFactory)
            )
        else:
            model_hyperparam = hyperparameters[model.name]

        for k, v in model_hyperparam.items():
            assert model._user_params[k] == v


def test_given_hyperparameters_with_spaces_and_custom_model_when_trainer_called_then_hpo_is_performed(
    temp_model_path,
):
    hyperparameters = {
        GenericGluonTSModelFactory(MQRNNEstimator): {"epochs": ag.Int(1, 4)}
    }
    # mock the default hps factory to prevent preset hyperparameter configurations from
    # creeping into the test case
    with mock.patch(
        "autogluon.timeseries.models.presets.get_default_hps"
    ) as default_hps_mock:
        default_hps_mock.return_value = defaultdict(dict)
        trainer = AutoTimeSeriesTrainer(path=temp_model_path)
        trainer.fit(
            train_data=DUMMY_TS_DATAFRAME,
            hyperparameters=hyperparameters,
            val_data=DUMMY_TS_DATAFRAME,
            hyperparameter_tune=True,
        )
        leaderboard = trainer.leaderboard()

    assert len(leaderboard) == 4

    config_history = next(iter(trainer.hpo_results.values()))["config_history"]
    assert len(config_history) == 4
    assert all(1 <= model["epochs"] <= 4 for model in config_history.values())


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
    trainer = AutoTimeSeriesTrainer(
        path=temp_model_path, eval_metric="MAPE", prediction_length=2
    )
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
        predictions = loaded_model.predict(DUMMY_TS_DATAFRAME)

        assert isinstance(predictions, TimeSeriesDataFrame)

        predicted_item_index = predictions.index.levels[0]
        assert all(predicted_item_index == DUMMY_TS_DATAFRAME.index.levels[0])  # noqa
        assert all(len(predictions.loc[i]) == 2 for i in predicted_item_index)
        assert not np.any(np.isnan(predictions))
