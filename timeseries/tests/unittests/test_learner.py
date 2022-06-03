"""Unit tests for learners"""
from collections import defaultdict
from unittest import mock

import numpy as np
import pytest
from gluonts.model.seq2seq import MQRNNEstimator

import autogluon.core as ag
from autogluon.timeseries.dataset import TimeSeriesDataFrame
from autogluon.timeseries.learner import TimeSeriesLearner
from autogluon.timeseries.models import DeepARModel
from autogluon.timeseries.models.gluonts.models import GenericGluonTSModelFactory
from autogluon.timeseries.models.presets import get_default_hps

from .common import DUMMY_TS_DATAFRAME

TEST_HYPERPARAMETER_SETTINGS = [
    "toy",
    {"SimpleFeedForward": {"epochs": 1}},
    {"DeepAR": {"epochs": 1}, "SimpleFeedForward": {"epochs": 1}},
]


def test_learner_can_be_initialized(temp_model_path):
    learner = TimeSeriesLearner(path_context=temp_model_path)
    assert isinstance(learner, TimeSeriesLearner)


# smoke test for the short 'happy path'
def test_when_learner_called_then_training_is_performed(temp_model_path):
    learner = TimeSeriesLearner(path_context=temp_model_path, eval_metric="MAPE")
    learner.fit(
        train_data=DUMMY_TS_DATAFRAME,
        hyperparameters={"SimpleFeedForward": {"epochs": 1}},
        val_data=DUMMY_TS_DATAFRAME,
    )
    assert "SimpleFeedForward" in learner.load_trainer().get_model_names()


@pytest.mark.parametrize("eval_metric", ["MAPE", None])
@pytest.mark.parametrize(
    "hyperparameters, expected_board_length",
    zip(TEST_HYPERPARAMETER_SETTINGS, [len(get_default_hps("toy", 1)), 1, 2]),
)
def test_given_hyperparameters_when_learner_called_then_leaderboard_is_correct(
    temp_model_path, eval_metric, hyperparameters, expected_board_length
):
    print(temp_model_path)
    learner = TimeSeriesLearner(path_context=temp_model_path, eval_metric="MAPE")
    learner.fit(
        train_data=DUMMY_TS_DATAFRAME,
        hyperparameters=hyperparameters,
        val_data=DUMMY_TS_DATAFRAME,
    )
    leaderboard = learner.leaderboard()
    print(temp_model_path)
    assert len(leaderboard) == expected_board_length
    assert np.all(leaderboard["score_val"] < 0)  # all MAPEs should be negative


@pytest.mark.parametrize(
    "hyperparameters, expected_board_length",
    zip(TEST_HYPERPARAMETER_SETTINGS, [len(get_default_hps("toy", 1)), 1, 2]),
)
def test_given_hyperparameters_when_learner_called_then_model_can_predict(
    temp_model_path, hyperparameters, expected_board_length
):
    learner = TimeSeriesLearner(path_context=temp_model_path, eval_metric="MAPE", prediction_length=3)
    learner.fit(
        train_data=DUMMY_TS_DATAFRAME,
        hyperparameters=hyperparameters,
        val_data=DUMMY_TS_DATAFRAME,
    )
    predictions = learner.predict(DUMMY_TS_DATAFRAME)

    assert isinstance(predictions, TimeSeriesDataFrame)

    predicted_item_index = predictions.index.levels[0]
    assert all(predicted_item_index == DUMMY_TS_DATAFRAME.index.levels[0])  # noqa
    assert all(len(predictions.loc[i]) == 3 for i in predicted_item_index)
    assert not np.any(np.isnan(predictions))


@pytest.mark.parametrize("model_name", ["DeepAR", "SimpleFeedForward", "MQCNN"])
def test_given_hyperparameters_with_spaces_when_learner_called_then_hpo_is_performed(
    temp_model_path, model_name
):
    hyperparameters = {model_name: {"epochs": ag.Int(1, 3)}}
    # mock the default hps factory to prevent preset hyperparameter configurations from
    # creeping into the test case
    with mock.patch(
        "autogluon.timeseries.models.presets.get_default_hps"
    ) as default_hps_mock:
        default_hps_mock.return_value = defaultdict(dict)
        learner = TimeSeriesLearner(path_context=temp_model_path, eval_metric="MAPE")
        learner.fit(
            train_data=DUMMY_TS_DATAFRAME,
            hyperparameters=hyperparameters,
            val_data=DUMMY_TS_DATAFRAME,
            hyperparameter_tune=True,
        )

        leaderboard = learner.leaderboard()

    assert len(leaderboard) == 3

    config_history = learner.load_trainer().hpo_results[model_name]["config_history"]
    assert len(config_history) == 3
    assert all(1 <= model["epochs"] <= 3 for model in config_history.values())


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
def test_given_hyperparameters_and_custom_models_when_learner_called_then_leaderboard_is_correct(
    temp_model_path, eval_metric, hyperparameters, expected_board_length
):
    learner = TimeSeriesLearner(path_context=temp_model_path, eval_metric=eval_metric)
    learner.fit(
        train_data=DUMMY_TS_DATAFRAME,
        hyperparameters=hyperparameters,
        val_data=DUMMY_TS_DATAFRAME,
    )
    leaderboard = learner.leaderboard()

    assert len(leaderboard) == expected_board_length
    assert np.all(leaderboard["score_val"] < 0)  # all MAPEs should be negative


@pytest.mark.parametrize(
    "hyperparameters, expected_board_length",
    zip(TEST_HYPERPARAMETER_SETTINGS, [len(get_default_hps("toy", 1)), 1, 2]),
)
def test_given_hyperparameters_when_learner_called_and_loaded_back_then_all_models_can_predict(
    temp_model_path, hyperparameters, expected_board_length
):
    learner = TimeSeriesLearner(path_context=temp_model_path, eval_metric="MAPE", prediction_length=2)
    learner.fit(
        train_data=DUMMY_TS_DATAFRAME,
        hyperparameters=hyperparameters,
        val_data=DUMMY_TS_DATAFRAME,
    )
    learner.save()
    del learner

    loaded_learner = TimeSeriesLearner.load(temp_model_path)

    for model_name in loaded_learner.load_trainer().get_model_names():
        predictions = loaded_learner.predict(DUMMY_TS_DATAFRAME, model=model_name)

        assert isinstance(predictions, TimeSeriesDataFrame)

        predicted_item_index = predictions.index.levels[0]
        assert all(predicted_item_index == DUMMY_TS_DATAFRAME.index.levels[0])  # noqa
        assert all(len(predictions.loc[i]) == 2 for i in predicted_item_index)
        assert not np.any(np.isnan(predictions))


@pytest.mark.parametrize("random_seed", [None, 12, 23, 34])
def test_given_random_seed_when_learner_called_then_random_seed_set_correctly(
    temp_model_path, random_seed
):
    init_kwargs = dict(path_context=temp_model_path, eval_metric="MAPE")
    if random_seed is not None:
        init_kwargs["random_state"] = random_seed

    learner = TimeSeriesLearner(**init_kwargs)
    learner.fit(
        train_data=DUMMY_TS_DATAFRAME,
        hyperparameters="toy",
        val_data=DUMMY_TS_DATAFRAME,
    )
    if random_seed is None:
        random_seed = learner.random_state
        assert random_seed is not None
    learner.save()
    del learner

    loaded_learner = TimeSeriesLearner.load(temp_model_path)
    assert random_seed == loaded_learner.random_state
