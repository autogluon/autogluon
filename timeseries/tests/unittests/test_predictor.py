"""Unit tests for predictors"""
import copy
from unittest import mock

import numpy as np
import pandas as pd
import pytest

import autogluon.core as ag
from autogluon.timeseries.dataset import TimeSeriesDataFrame
from autogluon.timeseries.dataset.ts_dataframe import ITEMID, TIMESTAMP
from autogluon.timeseries.models import DeepARModel
from autogluon.timeseries.models.gluonts.models import GenericGluonTSModelFactory, MQRNNEstimator
from autogluon.timeseries.predictor import TimeSeriesPredictor
from autogluon.timeseries.splitter import LastWindowSplitter, MultiWindowSplitter

from .common import DUMMY_TS_DATAFRAME

TEST_HYPERPARAMETER_SETTINGS = [
    {"SimpleFeedForward": {"epochs": 1}},
    {"ETS": {"maxiter": 1}, "SimpleFeedForward": {"epochs": 1}},
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


@pytest.mark.parametrize("hyperparameters", TEST_HYPERPARAMETER_SETTINGS + ["local_only"])  # noqa
def test_given_hyperparameters_when_predictor_called_then_model_can_predict(temp_model_path, hyperparameters):
    predictor = TimeSeriesPredictor(path=temp_model_path, eval_metric="MAPE", prediction_length=3)
    predictor.fit(
        train_data=DUMMY_TS_DATAFRAME,
        hyperparameters=hyperparameters,
        tuning_data=DUMMY_TS_DATAFRAME,
    )
    predictions = predictor.predict(DUMMY_TS_DATAFRAME)

    assert isinstance(predictions, TimeSeriesDataFrame)

    predicted_item_index = predictions.item_ids
    assert all(predicted_item_index == DUMMY_TS_DATAFRAME.item_ids)  # noqa
    assert all(len(predictions.loc[i]) == 3 for i in predicted_item_index)
    assert not np.any(np.isnan(predictions))


@pytest.mark.parametrize("hyperparameters", TEST_HYPERPARAMETER_SETTINGS + ["local_only"])  # noqa
def test_given_different_target_name_when_predictor_called_then_model_can_predict(temp_model_path, hyperparameters):
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

    predicted_item_index = predictions.item_ids
    assert all(predicted_item_index == DUMMY_TS_DATAFRAME.item_ids)  # noqa
    assert all(len(predictions.loc[i]) == 3 for i in predicted_item_index)
    assert not np.any(np.isnan(predictions))


@pytest.mark.parametrize("hyperparameters", TEST_HYPERPARAMETER_SETTINGS)
def test_given_no_tuning_data_when_predictor_called_then_model_can_predict(temp_model_path, hyperparameters):
    predictor = TimeSeriesPredictor(path=temp_model_path, eval_metric="MAPE", prediction_length=3)
    predictor.fit(
        train_data=DUMMY_TS_DATAFRAME,
        hyperparameters=hyperparameters,
    )
    predictions = predictor.predict(DUMMY_TS_DATAFRAME)

    assert isinstance(predictions, TimeSeriesDataFrame)

    predicted_item_index = predictions.item_ids
    assert all(predicted_item_index == DUMMY_TS_DATAFRAME.item_ids)  # noqa
    assert all(len(predictions.loc[i]) == 3 for i in predicted_item_index)
    assert not np.any(np.isnan(predictions))


@pytest.mark.parametrize("hyperparameters", TEST_HYPERPARAMETER_SETTINGS)
@pytest.mark.parametrize("quantile_kwarg_name", ["quantiles", "quantile_levels"])
def test_given_hyperparameters_and_quantiles_when_predictor_called_then_model_can_predict(
    temp_model_path, hyperparameters, quantile_kwarg_name
):
    predictor_init_kwargs = dict(path=temp_model_path, eval_metric="MAPE", prediction_length=3)
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
    predictor = TimeSeriesPredictor(path=temp_model_path, eval_metric=eval_metric)
    predictor.fit(
        train_data=DUMMY_TS_DATAFRAME,
        hyperparameters=hyperparameters,
        tuning_data=DUMMY_TS_DATAFRAME,
    )
    leaderboard = predictor.leaderboard()

    if predictor._trainer.enable_ensemble and len(hyperparameters) > 1:
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

        predicted_item_index = predictions.item_ids
        assert all(predicted_item_index == DUMMY_TS_DATAFRAME.item_ids)  # noqa
        assert all(len(predictions.loc[i]) == 2 for i in predicted_item_index)
        assert not np.any(np.isnan(predictions))


@pytest.mark.parametrize("target_column", ["target", "custom"])
@pytest.mark.parametrize(
    "hyperparameters",
    [
        {"ETS": {"maxiter": 1}, "SimpleFeedForward": {"epochs": 1}},
        {"ETS": {"maxiter": 1}, "SimpleFeedForward": {"epochs": ag.space.Int(1, 3)}},
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

        predicted_item_index = predictions.item_ids
        assert all(predicted_item_index == df.item_ids)  # noqa
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

    predicted_item_index = predictions.item_ids
    assert all(predicted_item_index == DUMMY_TS_DATAFRAME.item_ids)  # noqa
    assert all(len(predictions.loc[i]) == 2 for i in predicted_item_index)
    assert not np.any(np.isnan(predictions))


@pytest.fixture(
    scope="module",
    params=[
        [
            ["2020-01-01 00:00:00", "2020-01-02 00:00:00", "2020-01-03 00:01:00"],
        ],
        [
            ["2020-01-01 00:00:00", "2020-01-02 00:00:00", "2020-01-03 00:00:00"],
            ["2020-01-01 00:00:00", "2020-01-02 00:00:00", "2020-01-03 00:00:01"],
        ],
        [
            ["2020-01-01 00:00:00", "2020-01-02 00:00:00", "2020-01-03 00:00:00"],
            ["2020-01-01 00:00:00", "2020-01-02 00:00:00", "2020-01-04 00:00:00"],
        ],
        [
            ["2020-01-01 00:00:00", "2020-01-02 00:00:00", "2020-01-03 00:01:00"],
            ["2020-01-01 00:00:00", "2020-01-02 00:00:00", "2020-01-03 00:01:00"],
            ["2020-01-01 00:00:00", "2020-01-02 00:00:00", "2020-01-03 00:01:00"],
        ],
    ],
)
def irregular_timestamp_data_frame(request):
    df_tuples = []
    for i, ts in enumerate(request.param):
        for t in ts:
            df_tuples.append((i, pd.Timestamp(t), np.random.rand()))
    return TimeSeriesDataFrame.from_data_frame(pd.DataFrame(df_tuples, columns=[ITEMID, TIMESTAMP, "target"]))


def test_given_irregular_time_series_when_predictor_called_with_ignore_then_training_is_performed(
    temp_model_path, irregular_timestamp_data_frame
):
    df = irregular_timestamp_data_frame
    predictor = TimeSeriesPredictor(
        path=temp_model_path,
        eval_metric="MAPE",
        ignore_time_index=True,
    )
    predictor.fit(
        train_data=df,
        hyperparameters={"DeepAR": {"epochs": 1}},
        tuning_data=df,
    )
    assert "DeepAR" in predictor.get_model_names()


def test_given_irregular_time_series_and_no_tuning_when_predictor_called_with_ignore_then_training_is_performed(
    temp_model_path, irregular_timestamp_data_frame
):
    df = irregular_timestamp_data_frame
    predictor = TimeSeriesPredictor(
        path=temp_model_path,
        eval_metric="MAPE",
        ignore_time_index=True,
    )
    predictor.fit(
        train_data=df,
        hyperparameters={"SimpleFeedForward": {"epochs": 1}},
    )
    assert "SimpleFeedForward" in predictor.get_model_names()


def test_given_irregular_time_series_when_predictor_called_without_ignore_then_training_fails(
    temp_model_path, irregular_timestamp_data_frame
):
    df = irregular_timestamp_data_frame
    predictor = TimeSeriesPredictor(
        path=temp_model_path,
        eval_metric="MAPE",
        ignore_time_index=False,
    )
    with pytest.raises(ValueError):
        predictor.fit(
            train_data=df,
            hyperparameters={"SimpleFeedForward": {"epochs": 1}},
        )


def test_given_irregular_time_series_when_predictor_called_with_ignore_then_predictor_can_predict(
    temp_model_path, irregular_timestamp_data_frame
):
    df = irregular_timestamp_data_frame
    predictor = TimeSeriesPredictor(
        path=temp_model_path,
        eval_metric="MAPE",
        ignore_time_index=True,
    )
    predictor.fit(
        train_data=df,
        hyperparameters={"SimpleFeedForward": {"epochs": 1}},
    )
    predictions = predictor.predict(df)

    assert isinstance(predictions, TimeSeriesDataFrame)

    predicted_item_index = predictions.item_ids
    assert all(predicted_item_index == df.item_ids)  # noqa
    assert all(len(predictions.loc[i]) == 1 for i in predicted_item_index)
    assert not np.any(np.isnan(predictions))


def test_given_irregular_time_series_when_predictor_called_without_ignore_then_predictor_cannot_predict(
    temp_model_path, irregular_timestamp_data_frame
):
    df = irregular_timestamp_data_frame
    predictor = TimeSeriesPredictor(
        path=temp_model_path,
        eval_metric="MAPE",
        ignore_time_index=False,
    )
    predictor.fit(
        train_data=DUMMY_TS_DATAFRAME.get_reindexed_view(),
        hyperparameters={"SimpleFeedForward": {"epochs": 1}},
    )
    with pytest.raises(ValueError, match="irregularly sampled"):
        _ = predictor.predict(df)


@pytest.mark.parametrize("ignore_time_index", [True, False])
def test_when_predictor_called_and_loaded_back_then_ignore_time_index_persists(temp_model_path, ignore_time_index):
    predictor = TimeSeriesPredictor(
        path=temp_model_path,
        prediction_length=2,
        ignore_time_index=ignore_time_index,
    )
    predictor.fit(
        train_data=DUMMY_TS_DATAFRAME,
        hyperparameters={"SimpleFeedForward": {"epochs": 1}},
    )
    predictor.save()
    del predictor

    loaded_predictor = TimeSeriesPredictor.load(temp_model_path)
    assert loaded_predictor.ignore_time_index == ignore_time_index


@pytest.mark.parametrize(
    "splitter_string, expected_splitter_class",
    [("last_window", LastWindowSplitter), ("multi_window", MultiWindowSplitter)],
)
def test_when_passing_magic_string_as_validation_splitter_then_correct_splitter_object_is_created(
    splitter_string, expected_splitter_class
):
    predictor = TimeSeriesPredictor(validation_splitter=splitter_string)
    assert isinstance(predictor.validation_splitter, expected_splitter_class)


def test_given_enable_ensemble_true_when_predictor_called_then_ensemble_is_fitted(temp_model_path):
    predictor = TimeSeriesPredictor(
        path=temp_model_path,
        eval_metric="MAPE",
        enable_ensemble=True,
    )
    predictor.fit(
        train_data=DUMMY_TS_DATAFRAME,
        hyperparameters={
            "SimpleFeedForward": {"epochs": 1},
            "DeepAR": {"epochs": 1},
        },
    )
    assert any("ensemble" in n.lower() for n in predictor.get_model_names())


def test_given_enable_ensemble_true_and_only_one_model_when_predictor_called_then_ensemble_is_not_fitted(
    temp_model_path,
):
    predictor = TimeSeriesPredictor(
        path=temp_model_path,
        eval_metric="MAPE",
        enable_ensemble=True,
    )
    predictor.fit(
        train_data=DUMMY_TS_DATAFRAME,
        hyperparameters={"SimpleFeedForward": {"epochs": 1}},
    )
    assert not any("ensemble" in n.lower() for n in predictor.get_model_names())


def test_given_enable_ensemble_false_when_predictor_called_then_ensemble_is_not_fitted(temp_model_path):
    predictor = TimeSeriesPredictor(
        path=temp_model_path,
        eval_metric="MAPE",
        enable_ensemble=False,
    )
    predictor.fit(
        train_data=DUMMY_TS_DATAFRAME,
        hyperparameters={"SimpleFeedForward": {"epochs": 1}},
    )
    assert not any("ensemble" in n.lower() for n in predictor.get_model_names())


def test_given_model_fails_when_predictor_predicts_then_exception_is_caught_by_learner(temp_model_path):
    predictor = TimeSeriesPredictor(
        path=temp_model_path,
        eval_metric="MAPE",
        enable_ensemble=False,
    )
    predictor.fit(
        train_data=DUMMY_TS_DATAFRAME,
        hyperparameters={"ARIMA": {"maxiter": 1, "seasonal_period": 1, "seasonal_order": (0, 0, 0)}},
    )
    with mock.patch("autogluon.timeseries.models.statsmodels.models.ARIMAModel.predict") as arima_predict:
        arima_predict.side_effect = RuntimeError("Numerical error")
        with pytest.raises(RuntimeError, match="Prediction failed, please provide a different model to"):
            predictor.predict(DUMMY_TS_DATAFRAME)


def test_given_no_searchspace_and_hyperparameter_tune_kwargs_when_predictor_fits_then_exception_is_raised(
    temp_model_path,
):
    predictor = TimeSeriesPredictor(path=temp_model_path, enable_ensemble=False)
    with pytest.raises(ValueError, match="not a single model contains a hyperparameter search space"):
        predictor.fit(
            train_data=DUMMY_TS_DATAFRAME,
            hyperparameters={"SimpleFeedForward": {"epochs": 1}},
            hyperparameter_tune_kwargs="random",
        )


def test_given_searchspace_and_no_hyperparameter_tune_kwargs_when_predictor_fits_then_exception_is_raised(
    temp_model_path,
):
    predictor = TimeSeriesPredictor(path=temp_model_path, enable_ensemble=False)
    with pytest.raises(ValueError, match="Hyperparameter tuning not specified, so hyperparameters must have fixed values"):
        predictor.fit(
            train_data=DUMMY_TS_DATAFRAME,
            hyperparameters={"SimpleFeedForward": {"epochs": ag.space.Categorical(1, 2)}},
        )


def test_given_mixed_searchspace_and_hyperparameter_tune_kwargs_when_predictor_fits_then_no_exception_is_raised(
    temp_model_path,
):
    predictor = TimeSeriesPredictor(path=temp_model_path, enable_ensemble=False)
    predictor.fit(
        train_data=DUMMY_TS_DATAFRAME,
        hyperparameters={"SimpleFeedForward": {"epochs": ag.space.Categorical(1, 2), "ETS": {}}},
        hyperparameter_tune_kwargs={
            "scheduler": "local",
            "searcher": "random",
            "num_trials": 2,
        },
    )
