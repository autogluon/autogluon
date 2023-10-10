import logging

import numpy as np
import pandas as pd
import pytest

from autogluon.timeseries import TimeSeriesDataFrame
from autogluon.timeseries.models.local import (
    AutoARIMAModel,
    AutoETSModel,
    AverageModel,
    DynamicOptimizedThetaModel,
    ETSModel,
    NaiveModel,
    NPTSModel,
    SeasonalAverageModel,
    SeasonalNaiveModel,
    ThetaModel,
)

from ..common import (
    DUMMY_TS_DATAFRAME,
    DUMMY_VARIABLE_LENGTH_TS_DATAFRAME,
    dict_equal_primitive,
    get_data_frame_with_item_index,
)

TESTABLE_MODELS = [
    AutoARIMAModel,
    AutoETSModel,
    AverageModel,
    DynamicOptimizedThetaModel,
    ETSModel,
    ThetaModel,
    NaiveModel,
    NPTSModel,
    SeasonalAverageModel,
    SeasonalNaiveModel,
]


# Restrict to single core for faster training on small datasets
DEFAULT_HYPERPARAMETERS = {"n_jobs": 1, "use_fallback_model": False}


@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
def test_when_local_model_is_saved_and_loaded_then_model_can_predict(model_class, temp_model_path):
    model = model_class(path=temp_model_path, hyperparameters=DEFAULT_HYPERPARAMETERS, freq=DUMMY_TS_DATAFRAME.freq)
    model.fit(train_data=DUMMY_TS_DATAFRAME)
    model.save()
    loaded_model = model.__class__.load(path=model.path)
    loaded_model.predict(data=DUMMY_TS_DATAFRAME)


@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
@pytest.mark.parametrize(
    "hyperparameters", [{}, {"seasonal_period": 5}, {"seasonal_period": 5, "dummy_argument": "a"}]
)
def test_when_local_model_saved_then_local_model_args_are_saved(model_class, hyperparameters, temp_model_path):
    model = model_class(path=temp_model_path, hyperparameters=hyperparameters)
    model.fit(train_data=DUMMY_TS_DATAFRAME)
    model.save()

    loaded_model = model.__class__.load(path=model.path)
    assert dict_equal_primitive(model._local_model_args, loaded_model._local_model_args)


@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
@pytest.mark.parametrize("prediction_length", [1, 3, 10])
def test_when_local_model_predicts_then_time_index_is_correct(model_class, prediction_length, temp_model_path):
    data = DUMMY_VARIABLE_LENGTH_TS_DATAFRAME
    model = model_class(
        path=temp_model_path,
        prediction_length=prediction_length,
        hyperparameters=DEFAULT_HYPERPARAMETERS,
        freq=data.freq,
    )
    model.fit(train_data=data)
    predictions = model.predict(data=data)
    for item_id in data.item_ids:
        cutoff = data.loc[item_id].index[-1]
        start = cutoff + pd.tseries.frequencies.to_offset(data.freq)
        expected_timestamps = pd.date_range(start, periods=prediction_length, freq=data.freq)
        assert (predictions.loc[item_id].index == expected_timestamps).all()


def get_seasonal_period_from_fitted_local_model(model):
    if model.name in ["ARIMA", "AutoETS", "AutoARIMA", "DynamicOptimizedTheta", "ETS", "Theta"]:
        return model._local_model_args["season_length"]
    else:
        return model._local_model_args["seasonal_period"]


@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
@pytest.mark.parametrize(
    "hyperparameters", [{**DEFAULT_HYPERPARAMETERS, "seasonal_period": None}, DEFAULT_HYPERPARAMETERS]
)
@pytest.mark.parametrize(
    "freqstr, ts_length, expected_seasonal_period",
    [
        ("H", 100, 24),
        ("2H", 100, 12),
        ("B", 100, 5),
        ("D", 100, 7),
        ("M", 100, 12),
    ],
)
def test_when_seasonal_period_is_set_to_none_then_inferred_period_is_used(
    model_class,
    hyperparameters,
    temp_model_path,
    freqstr,
    ts_length,
    expected_seasonal_period,
):
    train_data = get_data_frame_with_item_index(["A", "B", "C"], data_length=ts_length, freq=freqstr)
    model = model_class(path=temp_model_path, prediction_length=3, hyperparameters=hyperparameters)

    model.fit(train_data=train_data)
    assert get_seasonal_period_from_fitted_local_model(model) == expected_seasonal_period


@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
@pytest.mark.parametrize(
    "freqstr, ts_length, provided_seasonal_period",
    [
        ("H", 100, 12),
        ("2H", 100, 5),
        ("B", 100, 10),
        ("D", 100, 8),
        ("M", 100, 24),
    ],
)
def test_when_seasonal_period_is_provided_then_inferred_period_is_overridden(
    model_class,
    temp_model_path,
    freqstr,
    ts_length,
    provided_seasonal_period,
):
    train_data = get_data_frame_with_item_index(["A", "B", "C"], data_length=ts_length, freq=freqstr)
    model = model_class(
        path=temp_model_path,
        prediction_length=3,
        hyperparameters={"seasonal_period": provided_seasonal_period, **DEFAULT_HYPERPARAMETERS},
    )

    model.fit(train_data=train_data)
    assert get_seasonal_period_from_fitted_local_model(model) == provided_seasonal_period


@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
def test_when_invalid_model_arguments_provided_then_model_ignores_them(model_class, temp_model_path, caplog):
    model = model_class(
        path=temp_model_path,
        prediction_length=3,
        hyperparameters={"bad_argument": 33, "n_jobs": 1},
    )
    with caplog.at_level(logging.WARNING):
        model.fit(train_data=DUMMY_TS_DATAFRAME)
        assert "bad_argument" not in model._local_model_args


@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
@pytest.mark.parametrize("n_jobs", [0.5, 3])
def test_when_local_model_saved_then_n_jobs_is_saved(model_class, n_jobs, temp_model_path):
    model = model_class(path=temp_model_path, hyperparameters={"n_jobs": n_jobs})
    model.save()

    loaded_model = model.__class__.load(path=model.path)
    assert model.n_jobs == loaded_model.n_jobs


def failing_predict(*args, **kwargs):
    raise RuntimeError("Custom error message")


@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
def test_when_fallback_model_disabled_and_model_fails_then_exception_is_raised(temp_model_path, model_class):
    model = model_class(temp_model_path, hyperparameters={"use_fallback_model": False, "n_jobs": 1})
    model.fit(train_data=DUMMY_TS_DATAFRAME)
    model._predict_with_local_model = failing_predict
    with pytest.raises(RuntimeError, match="Custom error message"):
        model.predict(DUMMY_TS_DATAFRAME)


@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
def test_when_fallback_model_enabled_and_model_fails_then_no_exception_is_raised(temp_model_path, model_class):
    model = model_class(temp_model_path, hyperparameters={"use_fallback_model": True, "n_jobs": 1})
    model.fit(train_data=DUMMY_TS_DATAFRAME)
    model._predict_with_local_model = failing_predict
    predictions = model.predict(DUMMY_TS_DATAFRAME)
    assert isinstance(predictions, TimeSeriesDataFrame)


@pytest.mark.parametrize("seasonal_period, should_match", [(1, True), (2, False)])
def test_when_seasonal_period_equals_one_then_average_and_seasonal_average_are_equivalent(
    seasonal_period, should_match
):
    avg = AverageModel(
        hyperparameters=DEFAULT_HYPERPARAMETERS,
        prediction_length=3,
    )
    avg.fit(train_data=DUMMY_TS_DATAFRAME)
    predictions_avg = avg.predict(DUMMY_TS_DATAFRAME)

    seasonal_avg = SeasonalAverageModel(
        hyperparameters={"seasonal_period": seasonal_period, **DEFAULT_HYPERPARAMETERS},
        prediction_length=3,
    )
    seasonal_avg.fit(train_data=DUMMY_TS_DATAFRAME)
    predictions_seasonal_avg = seasonal_avg.predict(DUMMY_TS_DATAFRAME)

    allclose = np.allclose(predictions_avg.values, predictions_seasonal_avg.values)
    if should_match:
        assert allclose
    else:
        assert not allclose


def test_when_data_shorter_than_seasonal_period_then_average_forecast_is_used():
    prediction_length = 20
    seasonal_period = DUMMY_TS_DATAFRAME.num_timesteps_per_item().max() + prediction_length

    avg = AverageModel(
        hyperparameters=DEFAULT_HYPERPARAMETERS,
        prediction_length=prediction_length,
    )
    avg.fit(train_data=DUMMY_TS_DATAFRAME)
    predictions_avg = avg.predict(DUMMY_TS_DATAFRAME)

    seasonal_avg = SeasonalAverageModel(
        hyperparameters={"seasonal_period": seasonal_period, "n_jobs": 1},
        prediction_length=prediction_length,
    )
    seasonal_avg.fit(train_data=DUMMY_TS_DATAFRAME)
    predictions_seasonal_avg = seasonal_avg.predict(DUMMY_TS_DATAFRAME)

    assert np.allclose(predictions_avg.values, predictions_seasonal_avg.values)


@pytest.mark.parametrize("freq", ["H", "W", "D", "T", "S", "B", "Q", "M", "A"])
def test_when_npts_fit_with_default_seasonal_features_then_predictions_match_gluonts(freq):
    from gluonts.model.npts import NPTSPredictor

    item_id = "A"
    prediction_length = 9
    data = get_data_frame_with_item_index([item_id], freq=freq, data_length=100)

    npts_ag = NPTSModel(
        freq=freq,
        prediction_length=prediction_length,
        hyperparameters={"n_jobs": 1, "use_fallback_model": False},
    )
    npts_gts = NPTSPredictor(freq=freq, prediction_length=prediction_length)
    npts_ag.fit(train_data=data)

    np.random.seed(123)
    pred_ag = npts_ag.predict(data)

    np.random.seed(123)
    ts = data.loc[item_id]["target"]
    ts.index = ts.index.to_period(freq=freq)
    pred_gts = npts_gts.predict_time_series(ts, num_samples=100)

    assert (pred_gts.mean == pred_ag["mean"]).all()
    for q in npts_ag.quantile_levels:
        assert (pred_gts.quantile(str(q)) == pred_ag[str(q)]).all()
