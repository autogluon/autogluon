"""Unit tests for learners"""
import copy

import numpy as np
import pytest
from gluonts.model.seq2seq import MQRNNEstimator

import autogluon.core as ag
from autogluon.timeseries.dataset import TimeSeriesDataFrame
from autogluon.timeseries.models import DeepARModel
from autogluon.timeseries.models.gluonts.models import GenericGluonTSModelFactory
from autogluon.timeseries.predictor import TimeSeriesPredictor

from .common import DUMMY_TS_DATAFRAME

TEST_HYPERPARAMETER_SETTINGS = [
    {"SimpleFeedForward": {"epochs": 1}},
    {"AutoETS": {}, "SimpleFeedForward": {"epochs": 1}},
]


def test_predictor_can_be_initialized(temp_model_path):
    predictor = TimeSeriesPredictor(path=temp_model_path)
    assert isinstance(predictor, TimeSeriesPredictor)


# smoke test for the short 'happy path'
def test_when_predictor_called_then_training_is_performed(temp_model_path):
    predictor = TimeSeriesPredictor(path=temp_model_path, eval_metric="MAPE")
    predictor.fit(
        train_data=DUMMY_TS_DATAFRAME,
        hyperparameters={"SimpleFeedForward": {"epochs": 1}},
        tuning_data=DUMMY_TS_DATAFRAME,
    )
    assert "SimpleFeedForward" in predictor.get_model_names()


@pytest.mark.parametrize("hyperparameters", TEST_HYPERPARAMETER_SETTINGS + ["toy"])
def test_given_hyperparameters_when_predictor_called_then_model_can_predict(
    temp_model_path, hyperparameters
):
    predictor = TimeSeriesPredictor(
        path=temp_model_path, eval_metric="MAPE", prediction_length=3
    )
    predictor.fit(
        train_data=DUMMY_TS_DATAFRAME,
        hyperparameters=hyperparameters,
        tuning_data=DUMMY_TS_DATAFRAME,
    )
    predictions = predictor.predict(DUMMY_TS_DATAFRAME)

    assert isinstance(predictions, TimeSeriesDataFrame)

    predicted_item_index = predictions.index.levels[0]
    assert all(predicted_item_index == DUMMY_TS_DATAFRAME.index.levels[0])  # noqa
    assert all(len(predictions.loc[i]) == 3 for i in predicted_item_index)
    assert not np.any(np.isnan(predictions))


@pytest.mark.parametrize("hyperparameters", TEST_HYPERPARAMETER_SETTINGS + ["toy"])
def test_given_different_target_name_when_predictor_called_then_model_can_predict(
    temp_model_path, hyperparameters
):
    df = TimeSeriesDataFrame(copy.copy(DUMMY_TS_DATAFRAME))
    df.rename(columns={"target": "mytarget"}, inplace=True)

    predictor = TimeSeriesPredictor(
        target="mytarget",
        path=temp_model_path,
        eval_metric="MAPE",
        prediction_length=3,
    )
    predictor.fit(
        train_data=df,
        hyperparameters=hyperparameters,
    )
    predictions = predictor.predict(df)

    assert isinstance(predictions, TimeSeriesDataFrame)

    predicted_item_index = predictions.index.levels[0]
    assert all(predicted_item_index == DUMMY_TS_DATAFRAME.index.levels[0])  # noqa
    assert all(len(predictions.loc[i]) == 3 for i in predicted_item_index)
    assert not np.any(np.isnan(predictions))


@pytest.mark.parametrize("hyperparameters", TEST_HYPERPARAMETER_SETTINGS)
def test_given_no_tuning_data_when_predictor_called_then_model_can_predict(
    temp_model_path, hyperparameters
):
    predictor = TimeSeriesPredictor(
        path=temp_model_path, eval_metric="MAPE", prediction_length=3
    )
    predictor.fit(
        train_data=DUMMY_TS_DATAFRAME,
        hyperparameters=hyperparameters,
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
    predictor_init_kwargs = dict(
        path=temp_model_path, eval_metric="MAPE", prediction_length=3
    )
    predictor_init_kwargs[quantile_kwarg_name] = [0.1, 0.4, 0.9]
    predictor = TimeSeriesPredictor(**predictor_init_kwargs)

    predictor.fit(
        train_data=DUMMY_TS_DATAFRAME,
        hyperparameters=hyperparameters,
        tuning_data=DUMMY_TS_DATAFRAME,
    )
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
    predictor = TimeSeriesPredictor(path=temp_model_path)
    predictor.fit(
        train_data=DUMMY_TS_DATAFRAME,
        hyperparameters=hyperparameters,
        tuning_data=DUMMY_TS_DATAFRAME,
    )
    leaderboard = predictor.leaderboard()

    if predictor._trainer.enable_ensemble:
        expected_board_length += 1

    assert len(leaderboard) == expected_board_length
    assert np.all(leaderboard["score_val"] < 0)  # all MAPEs should be negative


@pytest.mark.parametrize("hyperparameters", TEST_HYPERPARAMETER_SETTINGS)
def test_given_hyperparameters_when_predictor_called_and_loaded_back_then_all_models_can_predict(
    temp_model_path, hyperparameters
):
    predictor = TimeSeriesPredictor(path=temp_model_path, prediction_length=2)
    predictor.fit(
        train_data=DUMMY_TS_DATAFRAME,
        hyperparameters=hyperparameters,
        tuning_data=DUMMY_TS_DATAFRAME,
    )
    predictor.save()
    del predictor

    loaded_predictor = TimeSeriesPredictor.load(temp_model_path)

    for model_name in loaded_predictor.get_model_names():
        predictions = loaded_predictor.predict(DUMMY_TS_DATAFRAME, model=model_name)

        assert isinstance(predictions, TimeSeriesDataFrame)

        predicted_item_index = predictions.index.levels[0]
        assert all(predicted_item_index == DUMMY_TS_DATAFRAME.index.levels[0])  # noqa
        assert all(len(predictions.loc[i]) == 2 for i in predicted_item_index)
        assert not np.any(np.isnan(predictions))


@pytest.mark.parametrize("target_column", ["target", "custom"])
@pytest.mark.parametrize(
    "hyperparameters",
    [
        {"AutoETS": {}, "SimpleFeedForward": {"epochs": 1}},
        {"AutoETS": {}, "SimpleFeedForward": {"epochs": ag.Int(1, 3)}},
    ],
)
def test_given_hp_spaces_and_custom_target_when_predictor_called_predictor_can_predict(
    temp_model_path, hyperparameters, target_column
):
    df = DUMMY_TS_DATAFRAME.rename(columns={"target": target_column})

    fit_kwargs = dict(
        train_data=df,
        hyperparameters=hyperparameters,
        tuning_data=df,
    )
    init_kwargs = dict(path=temp_model_path, prediction_length=2)
    if target_column != "target":
        init_kwargs.update({"target": target_column})

    for hps in hyperparameters.values():
        if any(isinstance(v, ag.Space) for v in hps.values()):
            fit_kwargs.update(
                {
                    "hyperparameter_tune_kwargs": {
                        "scheduler": "local",
                        "searcher": "random",
                        "num_trials": 2,
                    },
                }
            )
            break

    predictor = TimeSeriesPredictor(**init_kwargs)
    predictor.fit(**fit_kwargs)

    assert predictor.get_model_names()

    for model_name in predictor.get_model_names():
        predictions = predictor.predict(df, model=model_name)

        assert isinstance(predictions, TimeSeriesDataFrame)

        predicted_item_index = predictions.index.levels[0]
        assert all(predicted_item_index == df.index.levels[0])  # noqa
        assert all(len(predictions.loc[i]) == 2 for i in predicted_item_index)
        assert not np.any(np.isnan(predictions))


@pytest.mark.parametrize("hyperparameters", TEST_HYPERPARAMETER_SETTINGS)
def test_given_hyperparameters_when_predictor_called_and_loaded_back_then_loaded_learner_can_predict(
    temp_model_path, hyperparameters
):
    predictor = TimeSeriesPredictor(path=temp_model_path, prediction_length=2)
    predictor.fit(
        train_data=DUMMY_TS_DATAFRAME,
        hyperparameters=hyperparameters,
        tuning_data=DUMMY_TS_DATAFRAME,
    )
    predictor.save()
    del predictor

    loaded_predictor = TimeSeriesPredictor.load(temp_model_path)

    predictions = loaded_predictor._learner.predict(DUMMY_TS_DATAFRAME)

    assert isinstance(predictions, TimeSeriesDataFrame)

    predicted_item_index = predictions.index.levels[0]
    assert all(predicted_item_index == DUMMY_TS_DATAFRAME.index.levels[0])  # noqa
    assert all(len(predictions.loc[i]) == 2 for i in predicted_item_index)
    assert not np.any(np.isnan(predictions))
