"""Unit tests for learners"""
from collections import defaultdict
from unittest import mock

import numpy as np
import pytest
from gluonts.model.seq2seq import MQRNNEstimator

import autogluon.core as ag
from autogluon.forecasting.dataset import TimeSeriesDataFrame
from autogluon.forecasting.learner import ForecastingLearner
from autogluon.forecasting.models import DeepARModel
from autogluon.forecasting.models.gluonts.models import GenericGluonTSModelFactory
from autogluon.forecasting.models.presets import get_default_hps
from autogluon.forecasting.predictor import ForecastingPredictor

from .common import DUMMY_TS_DATAFRAME

TEST_HYPERPARAMETER_SETTINGS = [
    "toy",
    {"SimpleFeedForward": {"epochs": 1}},
    {"DeepAR": {"epochs": 1}, "SimpleFeedForward": {"epochs": 1}},
]


def test_predictor_can_be_initialized(temp_model_path):
    predictor = ForecastingPredictor(path=temp_model_path)
    assert isinstance(predictor, ForecastingPredictor)


# smoke test for the short 'happy path'
def test_when_predictor_called_then_training_is_performed(temp_model_path):
    predictor = ForecastingPredictor(path=temp_model_path, eval_metric="MAPE")
    predictor.fit(
        train_data=DUMMY_TS_DATAFRAME,
        hyperparameters={"SimpleFeedForward": {"epochs": 1}},
        val_data=DUMMY_TS_DATAFRAME,
        prediction_length=1,
    )
    assert "SimpleFeedForward" in predictor.get_model_names()


@pytest.mark.parametrize("hyperparameters", TEST_HYPERPARAMETER_SETTINGS)
def test_given_hyperparameters_when_predictor_called_then_model_can_predict(
    temp_model_path, hyperparameters
):
    predictor = ForecastingPredictor(path=temp_model_path, eval_metric="MAPE")
    predictor.fit(
        train_data=DUMMY_TS_DATAFRAME,
        hyperparameters=hyperparameters,
        val_data=DUMMY_TS_DATAFRAME,
        prediction_length=3,
    )
    predictions = predictor.predict(DUMMY_TS_DATAFRAME)

    assert isinstance(predictions, TimeSeriesDataFrame)

    predicted_item_index = predictions.index.levels[0]
    assert all(predicted_item_index == DUMMY_TS_DATAFRAME.index.levels[0])  # noqa
    assert all(len(predictions.loc[i]) == 3 for i in predicted_item_index)
    assert not np.any(np.isnan(predictions))


@pytest.mark.parametrize("hyperparameters", TEST_HYPERPARAMETER_SETTINGS)
@pytest.mark.parametrize("quantile_kwarg_name", ["quantiles", "quantile_levels"])
def test_given_hyperparameters_and_quantiles_when_predictor_called_then_model_can_predict(
    temp_model_path, hyperparameters, quantile_kwarg_name
):
    predictor = ForecastingPredictor(path=temp_model_path, eval_metric="MAPE")
    predictor_fit_kwargs = dict(
        train_data=DUMMY_TS_DATAFRAME,
        hyperparameters=hyperparameters,
        val_data=DUMMY_TS_DATAFRAME,
        prediction_length=3,
    )
    predictor_fit_kwargs[quantile_kwarg_name] = [0.1, 0.4, 0.9]
    predictor.fit(**predictor_fit_kwargs)
    predictions = predictor.predict(DUMMY_TS_DATAFRAME)

    assert tuple(predictions.columns) == ("mean", "0.1", "0.4", "0.9")


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
def test_given_hyperparameters_and_custom_models_when_predictor_called_then_leaderboard_is_correct(
    temp_model_path, eval_metric, hyperparameters, expected_board_length
):
    predictor = ForecastingPredictor(path=temp_model_path)
    predictor.fit(
        train_data=DUMMY_TS_DATAFRAME,
        hyperparameters=hyperparameters,
        val_data=DUMMY_TS_DATAFRAME,
        prediction_length=1,
    )
    leaderboard = predictor.leaderboard()

    assert len(leaderboard) == expected_board_length
    assert np.all(leaderboard["val_score"] < 0)  # all MAPEs should be negative


@pytest.mark.parametrize("hyperparameters", TEST_HYPERPARAMETER_SETTINGS)
def test_given_hyperparameters_when_predictor_called_and_loaded_back_then_all_models_can_predict(
    temp_model_path, hyperparameters
):
    predictor = ForecastingPredictor(path=temp_model_path)
    predictor.fit(
        train_data=DUMMY_TS_DATAFRAME,
        hyperparameters=hyperparameters,
        val_data=DUMMY_TS_DATAFRAME,
        prediction_length=2,
    )
    predictor.save()
    del predictor

    loaded_predictor = ForecastingPredictor.load(temp_model_path)

    for model_name in loaded_predictor.get_model_names():
        predictions = loaded_predictor.predict(DUMMY_TS_DATAFRAME, model=model_name)

        assert isinstance(predictions, TimeSeriesDataFrame)

        predicted_item_index = predictions.index.levels[0]
        assert all(predicted_item_index == DUMMY_TS_DATAFRAME.index.levels[0])  # noqa
        assert all(len(predictions.loc[i]) == 2 for i in predicted_item_index)
        assert not np.any(np.isnan(predictions))


@pytest.mark.parametrize("hyperparameters", TEST_HYPERPARAMETER_SETTINGS)
def test_given_hyperparameters_when_predictor_called_and_loaded_back_then_loaded_learner_can_predict(
    temp_model_path, hyperparameters
):
    predictor = ForecastingPredictor(path=temp_model_path)
    predictor.fit(
        train_data=DUMMY_TS_DATAFRAME,
        hyperparameters=hyperparameters,
        val_data=DUMMY_TS_DATAFRAME,
        prediction_length=2,
    )
    predictor.save()
    del predictor

    loaded_predictor = ForecastingPredictor.load(temp_model_path)

    predictions = loaded_predictor._learner.predict(DUMMY_TS_DATAFRAME)

    assert isinstance(predictions, TimeSeriesDataFrame)

    predicted_item_index = predictions.index.levels[0]
    assert all(predicted_item_index == DUMMY_TS_DATAFRAME.index.levels[0])  # noqa
    assert all(len(predictions.loc[i]) == 2 for i in predicted_item_index)
    assert not np.any(np.isnan(predictions))
