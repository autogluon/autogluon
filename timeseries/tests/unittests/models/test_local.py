import logging

import pandas as pd
import pytest

from autogluon.timeseries.models.local import ARIMAModel, ETSModel, NaiveModel, SeasonalNaiveModel, ThetaModel
from autogluon.timeseries.models.statsforecast import (
    AutoARIMAStatsForecastModel,
    AutoETSStatsForecastModel,
    DynamicOptimizedThetaStatsForecastModel,
)

from ..common import (
    DUMMY_TS_DATAFRAME,
    DUMMY_VARIABLE_LENGTH_TS_DATAFRAME,
    dict_equal_primitive,
    get_data_frame_with_item_index,
)

TESTABLE_MODELS = [
    AutoARIMAStatsForecastModel,
    AutoETSStatsForecastModel,
    DynamicOptimizedThetaStatsForecastModel,
    ARIMAModel,
    ETSModel,
    ThetaModel,
    NaiveModel,
    SeasonalNaiveModel,
]


# Restrict to single core for faster training on small datasets
DEFAULT_HYPERPARAMETERS = {"n_jobs": 1}


@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
def test_when_local_model_saved_then_cached_predictions_can_be_loaded(model_class, temp_model_path):
    model = model_class(path=temp_model_path, hyperparameters=DEFAULT_HYPERPARAMETERS)
    model.fit(train_data=DUMMY_TS_DATAFRAME)
    model.save()

    loaded_model = model.__class__.load(path=model.path)
    for ts_hash, pred in model._cached_predictions.items():
        assert ts_hash in loaded_model._cached_predictions
        assert (loaded_model._cached_predictions[ts_hash] == pred).all()


@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
def test_when_local_model_is_saved_and_loaded_then_model_can_predict(model_class, temp_model_path):
    model = model_class(path=temp_model_path, hyperparameters=DEFAULT_HYPERPARAMETERS)
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
        path=temp_model_path, prediction_length=prediction_length, hyperparameters=DEFAULT_HYPERPARAMETERS
    )
    model.fit(train_data=data)
    predictions = model.predict(data=data)
    for item_id in data.item_ids:
        cutoff = data.loc[item_id].index[-1]
        start = cutoff + pd.tseries.frequencies.to_offset(data.freq)
        expected_timestamps = pd.date_range(start, periods=prediction_length, freq=data.freq)
        assert (predictions.loc[item_id].index == expected_timestamps).all()


def get_seasonal_period_from_fitted_local_model(model):
    if model.name == "ARIMA":
        return model._local_model_args["seasonal_order"][-1]
    elif model.name == "ETS":
        return model._local_model_args["seasonal_periods"]
    elif model.name == "Theta":
        return model._local_model_args["period"]
    elif "StatsForecast" in model.name:
        return model._local_model_args["season_length"]
    else:
        return model._local_model_args["seasonal_period"]


@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
@pytest.mark.parametrize("hyperparameters", [{"seasonal_period": None, "n_jobs": 1}, {"n_jobs": 1}])
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
def test_when_seasonal_period_is_provided_then_inferred_period_is_overriden(
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
        hyperparameters={"seasonal_period": provided_seasonal_period, "n_jobs": 1},
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
def test_when_train_and_test_data_have_different_freq_then_exception_is_raised(model_class, temp_model_path):
    model = model_class(
        path=temp_model_path,
        prediction_length=3,
        hyperparameters=DEFAULT_HYPERPARAMETERS,
    )
    train_data = get_data_frame_with_item_index([1, 2, 3], freq="H")
    test_data = get_data_frame_with_item_index([1, 2, 3], freq="D")

    model.fit(train_data=train_data)
    with pytest.raises(RuntimeError, match="doesn't match the frequency"):
        model.predict(test_data)


@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
@pytest.mark.parametrize("n_jobs", [0.5, 3])
def test_when_local_model_saved_then_n_jobs_is_saved(model_class, n_jobs, temp_model_path):
    model = model_class(path=temp_model_path, hyperparameters={"n_jobs": n_jobs})
    model.save()

    loaded_model = model.__class__.load(path=model.path)
    assert model.n_jobs == loaded_model.n_jobs
