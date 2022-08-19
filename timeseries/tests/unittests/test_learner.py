"""Unit tests for learners"""
import os
import shutil
import tempfile
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

from .common import DUMMY_TS_DATAFRAME

TEST_HYPERPARAMETER_SETTINGS = [
    {"SimpleFeedForward": {"epochs": 1}},
    {"DeepAR": {"epochs": 1}, "AutoETS": {}},
]
TEST_HYPERPARAMETER_SETTINGS_EXPECTED_LB_LENGTHS = [1, 2]


@pytest.fixture(scope="module")
def trained_learners():
    learners = {}
    model_paths = []
    for hp in TEST_HYPERPARAMETER_SETTINGS:
        temp_model_path = tempfile.mkdtemp()
        learner = TimeSeriesLearner(
            path_context=temp_model_path + os.path.sep,
            eval_metric="MASE",
            prediction_length=3,
        )
        learner.fit(
            train_data=DUMMY_TS_DATAFRAME,
            hyperparameters=hp,
            val_data=DUMMY_TS_DATAFRAME,
        )
        learners[repr(hp)] = learner
        model_paths.append(temp_model_path)

    yield learners

    for td in model_paths:
        shutil.rmtree(td)


def test_learner_can_be_initialized(temp_model_path):
    learner = TimeSeriesLearner(path_context=temp_model_path)
    assert isinstance(learner, TimeSeriesLearner)


# smoke test for the short 'happy path'
@pytest.mark.parametrize("hyperparameters", TEST_HYPERPARAMETER_SETTINGS)
def test_when_learner_called_then_training_is_performed(hyperparameters, trained_learners):
    learner = trained_learners[repr(hyperparameters)]
    assert learner.load_trainer().get_model_names()


@pytest.mark.parametrize(
    "hyperparameters, expected_board_length",
    zip(TEST_HYPERPARAMETER_SETTINGS, TEST_HYPERPARAMETER_SETTINGS_EXPECTED_LB_LENGTHS),
)
def test_given_hyperparameters_when_learner_called_then_leaderboard_is_correct(
    trained_learners, hyperparameters, expected_board_length
):
    learner = trained_learners[repr(hyperparameters)]
    leaderboard = learner.leaderboard()

    if learner.load_trainer().enable_ensemble and len(hyperparameters) > 1:
        expected_board_length += 1

    assert len(leaderboard) == expected_board_length
    assert np.all(leaderboard["score_val"] < 0)  # all MAPEs should be negative


@pytest.mark.parametrize(
    "hyperparameters, expected_board_length",
    zip(TEST_HYPERPARAMETER_SETTINGS, TEST_HYPERPARAMETER_SETTINGS_EXPECTED_LB_LENGTHS),
)
def test_given_hyperparameters_when_learner_called_then_model_can_predict(
    trained_learners, hyperparameters, expected_board_length
):
    learner = trained_learners[repr(hyperparameters)]
    predictions = learner.predict(DUMMY_TS_DATAFRAME)

    assert isinstance(predictions, TimeSeriesDataFrame)

    predicted_item_index = predictions.index.levels[0]
    assert all(predicted_item_index == DUMMY_TS_DATAFRAME.index.levels[0])  # noqa
    assert all(len(predictions.loc[i]) == 3 for i in predicted_item_index)
    assert not np.any(np.isnan(predictions))


@pytest.mark.parametrize("model_name", ["DeepAR", "SimpleFeedForward"])
def test_given_hyperparameters_with_spaces_when_learner_called_then_hpo_is_performed(temp_model_path, model_name):
    hyperparameters = {model_name: {"epochs": ag.Int(1, 3)}}
    # mock the default hps factory to prevent preset hyperparameter configurations from
    # creeping into the test case
    with mock.patch("autogluon.timeseries.models.presets.get_default_hps") as default_hps_mock:
        default_hps_mock.return_value = defaultdict(dict)
        learner = TimeSeriesLearner(path_context=temp_model_path, eval_metric="MAPE")
        learner.fit(
            train_data=DUMMY_TS_DATAFRAME,
            hyperparameters=hyperparameters,
            val_data=DUMMY_TS_DATAFRAME,
            hyperparameter_tune_kwargs={
                "searcher": "random",
                "scheduler": "local",
                "num_trials": 2,
            },
        )

        leaderboard = learner.leaderboard()

    assert len(leaderboard) == 2 + 1  # include ensemble

    config_history = learner.load_trainer().hpo_results[model_name]["config_history"]
    assert len(config_history) == 2
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

    if learner.load_trainer().enable_ensemble and len(hyperparameters) > 1:
        expected_board_length += 1

    assert len(leaderboard) == expected_board_length
    assert np.all(leaderboard["score_val"] < 0)  # all MAPEs should be negative


@pytest.mark.parametrize(
    "hyperparameters, expected_board_length",
    zip(TEST_HYPERPARAMETER_SETTINGS, TEST_HYPERPARAMETER_SETTINGS_EXPECTED_LB_LENGTHS),
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


@pytest.mark.parametrize("random_seed", [None, 1616])
def test_given_random_seed_when_learner_called_then_random_seed_set_correctly(temp_model_path, random_seed):
    init_kwargs = dict(path_context=temp_model_path, eval_metric="MAPE")
    if random_seed is not None:
        init_kwargs["random_state"] = random_seed

    learner = TimeSeriesLearner(**init_kwargs)
    learner.fit(
        train_data=DUMMY_TS_DATAFRAME,
        hyperparameters="toy",
        val_data=DUMMY_TS_DATAFRAME,
        time_limit=2,
    )
    if random_seed is None:
        random_seed = learner.random_state
        assert random_seed is not None
    learner.save()
    del learner

    loaded_learner = TimeSeriesLearner.load(temp_model_path)
    assert random_seed == loaded_learner.random_state
