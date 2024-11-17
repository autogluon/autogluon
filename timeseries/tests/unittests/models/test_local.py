import logging
from typing import Dict

import numpy as np
import pandas as pd
import pytest
from statsforecast.models import CrostonClassic, CrostonOptimized, CrostonSBA

from autogluon.timeseries import TimeSeriesDataFrame
from autogluon.timeseries.models.local import (
    ADIDAModel,
    ARIMAModel,
    AutoARIMAModel,
    AutoCESModel,
    AutoETSModel,
    AverageModel,
    CrostonModel,
    DynamicOptimizedThetaModel,
    ETSModel,
    IMAPAModel,
    NaiveModel,
    NPTSModel,
    SeasonalAverageModel,
    SeasonalNaiveModel,
    ThetaModel,
    ZeroModel,
)
from autogluon.timeseries.models.local.statsforecast import AbstractConformalizedStatsForecastModel

from ..common import (
    DUMMY_TS_DATAFRAME,
    DUMMY_VARIABLE_LENGTH_TS_DATAFRAME,
    dict_equal_primitive,
    get_data_frame_with_item_index,
    to_supported_pandas_freq,
)

# models accepting seasonal_period
SEASONAL_TESTABLE_MODELS = [
    AutoARIMAModel,
    AutoETSModel,
    AverageModel,
    DynamicOptimizedThetaModel,
    NaiveModel,
    NPTSModel,
    SeasonalAverageModel,
    SeasonalNaiveModel,
]
# these models will only be tested in local tests, and will not be exported
# to model tests to decrease test running time
SEASONAL_TESTABLE_MODELS_LOCAL_ONLY = [
    AutoCESModel,
    ThetaModel,
    ETSModel,
    ARIMAModel,
]
# intermittent demand models do not accept seasonal_period
NONSEASONAL_TESTABLE_MODELS = [
    ADIDAModel,
    ZeroModel,
    CrostonModel,
    IMAPAModel,
]
TESTABLE_MODELS_LOCAL = SEASONAL_TESTABLE_MODELS + SEASONAL_TESTABLE_MODELS_LOCAL_ONLY + NONSEASONAL_TESTABLE_MODELS
SEASONAL_TESTABLE_MODELS_LOCAL = SEASONAL_TESTABLE_MODELS + SEASONAL_TESTABLE_MODELS_LOCAL_ONLY
TESTABLE_MODELS = SEASONAL_TESTABLE_MODELS + NONSEASONAL_TESTABLE_MODELS

# Restrict to single core for faster training on small datasets
DEFAULT_HYPERPARAMETERS = {"n_jobs": 1, "use_fallback_model": False}


@pytest.mark.parametrize("model_class", TESTABLE_MODELS_LOCAL)
def test_when_local_model_is_saved_and_loaded_then_model_can_predict(model_class, temp_model_path):
    model = model_class(path=temp_model_path, hyperparameters=DEFAULT_HYPERPARAMETERS, freq=DUMMY_TS_DATAFRAME.freq)
    model.fit(train_data=DUMMY_TS_DATAFRAME)
    model.save()
    loaded_model = model.__class__.load(path=model.path)
    loaded_model.predict(data=DUMMY_TS_DATAFRAME)


@pytest.mark.parametrize("model_class", TESTABLE_MODELS_LOCAL)
@pytest.mark.parametrize(
    "hyperparameters", [{}, {"seasonal_period": 5}, {"seasonal_period": 5, "dummy_argument": "a"}]
)
def test_when_local_model_saved_then_local_model_args_are_saved(model_class, hyperparameters, temp_model_path):
    model = model_class(path=temp_model_path, hyperparameters=hyperparameters)
    model.fit(train_data=DUMMY_TS_DATAFRAME)
    model.save()

    loaded_model = model.__class__.load(path=model.path)
    assert dict_equal_primitive(model._local_model_args, loaded_model._local_model_args)


def get_seasonal_period_from_fitted_local_model(model):
    if model.name in ["ARIMA", "AutoETS", "AutoARIMA", "AutoCES", "DynamicOptimizedTheta", "ETS", "Theta"]:
        return model._local_model_args["season_length"]
    else:
        return model._local_model_args["seasonal_period"]


@pytest.mark.parametrize("model_class", SEASONAL_TESTABLE_MODELS_LOCAL)
@pytest.mark.parametrize(
    "hyperparameters", [{**DEFAULT_HYPERPARAMETERS, "seasonal_period": None}, DEFAULT_HYPERPARAMETERS]
)
@pytest.mark.parametrize(
    "freqstr, ts_length, expected_seasonal_period",
    [
        ("h", 100, 24),
        ("2h", 100, 12),
        ("B", 100, 5),
        ("D", 100, 7),
        ("ME", 100, 12),
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


@pytest.mark.parametrize("model_class", SEASONAL_TESTABLE_MODELS_LOCAL)
@pytest.mark.parametrize(
    "freqstr, ts_length, provided_seasonal_period",
    [
        ("h", 100, 12),
        ("2h", 100, 5),
        ("B", 100, 10),
        ("D", 100, 8),
        ("ME", 100, 24),
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


@pytest.mark.parametrize("model_class", TESTABLE_MODELS_LOCAL)
def test_when_invalid_model_arguments_provided_then_model_ignores_them(model_class, temp_model_path, caplog):
    model = model_class(
        path=temp_model_path,
        prediction_length=3,
        hyperparameters={"bad_argument": 33, "n_jobs": 1},
    )
    with caplog.at_level(logging.WARNING):
        model.fit(train_data=DUMMY_TS_DATAFRAME)
        assert "bad_argument" not in model._local_model_args


@pytest.mark.parametrize("model_class", TESTABLE_MODELS_LOCAL)
@pytest.mark.parametrize("n_jobs", [0.5, 3])
def test_when_local_model_saved_then_n_jobs_is_saved(model_class, n_jobs, temp_model_path):
    model = model_class(path=temp_model_path, hyperparameters={"n_jobs": n_jobs})
    model.save()

    loaded_model = model.__class__.load(path=model.path)
    assert model.n_jobs == loaded_model.n_jobs


def failing_predict(*args, **kwargs):
    raise RuntimeError("Custom error message")


@pytest.mark.parametrize("model_class", TESTABLE_MODELS_LOCAL)
def test_when_fallback_model_disabled_and_model_fails_then_exception_is_raised(temp_model_path, model_class):
    model = model_class(
        path=temp_model_path, hyperparameters={"use_fallback_model": False, "n_jobs": 1}, freq=DUMMY_TS_DATAFRAME.freq
    )
    model.fit(train_data=DUMMY_TS_DATAFRAME)
    model._predict_with_local_model = failing_predict
    with pytest.raises(RuntimeError, match="Custom error message"):
        model.predict(DUMMY_TS_DATAFRAME)


@pytest.mark.parametrize("model_class", TESTABLE_MODELS_LOCAL)
def test_when_fallback_model_enabled_and_model_fails_then_no_exception_is_raised(temp_model_path, model_class):
    model = model_class(
        path=temp_model_path, hyperparameters={"use_fallback_model": True, "n_jobs": 1}, freq=DUMMY_TS_DATAFRAME.freq
    )
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


@pytest.mark.parametrize("freq", ["h", "W", "D", "min", "s", "B", "QE", "ME", "YE"])
def test_when_npts_fit_with_default_seasonal_features_then_predictions_match_gluonts(freq):
    from gluonts.model.npts import NPTSPredictor

    freq = to_supported_pandas_freq(freq)
    item_id = "A"
    prediction_length = 9
    data = get_data_frame_with_item_index([item_id], freq=freq, data_length=100)

    npts_ag = NPTSModel(
        freq=freq,
        prediction_length=prediction_length,
        hyperparameters={"n_jobs": 1, "use_fallback_model": False},
    )
    npts_gts = NPTSPredictor(prediction_length=prediction_length)
    npts_ag.fit(train_data=data)

    np.random.seed(123)
    pred_ag = npts_ag.predict(data)

    np.random.seed(123)
    ts = data.loc[item_id]["target"]
    freq_for_period = {"ME": "M", "YE": "Y", "QE": "Q"}.get(freq, freq)
    ts.index = ts.index.to_period(freq=freq_for_period)
    pred_gts = npts_gts.predict_time_series(ts, num_samples=100)

    assert (pred_gts.mean == pred_ag["mean"]).all()
    for q in npts_ag.quantile_levels:
        assert (pred_gts.quantile(str(q)) == pred_ag[str(q)]).all()


class MockConformalModel(AbstractConformalizedStatsForecastModel):
    def _get_point_forecast(self, time_series: pd.Series, local_model_args: Dict):
        return np.ones(self.prediction_length)

    def _get_nonconformity_scores(self, time_series: pd.Series, local_model_args: Dict):
        scores = super()._get_nonconformity_scores(time_series, local_model_args)
        self.returned_nonconformity_scores = scores
        return scores


@pytest.mark.parametrize(
    "prediction_length, time_series_length, expected_num_windows",
    [
        (1, 100, 5),
        (1, 10, 5),
        (3, 100, 5),
        (3, 10, 3),
        (10, 40, 3),
        (10, 41, 4),
        (10, 100, 5),
        (10, 11, 1),
        (10, 10, 1),
        (3, 3, 1),
        (1, 1, 1),
    ],
)
def test_when_conformalized_model_called_then_nonconformity_score_shapes_correct(
    prediction_length, time_series_length, expected_num_windows
):
    model = MockConformalModel(
        prediction_length=prediction_length, hyperparameters={"n_jobs": 1, "use_fallback_model": False}
    )

    data = get_data_frame_with_item_index(["A", "B", "C"], data_length=time_series_length)

    model.fit(train_data=data)
    _ = model.predict(data)

    assert model.returned_nonconformity_scores.shape == (expected_num_windows, prediction_length)


@pytest.mark.parametrize(
    "prediction_length, time_series_length, expected_num_windows",
    [
        (1, 100, 5),
        (1, 10, 5),
        (3, 100, 5),
        (3, 10, 3),
        (10, 40, 3),
        (10, 41, 4),
        (10, 100, 5),
        (10, 11, 1),
        (10, 10, 1),
        (3, 3, 1),
        (1, 1, 1),
    ],
)
def test_when_conformalized_model_called_then_nonconformity_score_values_correct(
    prediction_length, time_series_length, expected_num_windows
):
    model = MockConformalModel(
        prediction_length=prediction_length, hyperparameters={"n_jobs": 1, "use_fallback_model": False}
    )

    data = get_data_frame_with_item_index(["A"], data_length=time_series_length)
    data["target"] = np.arange(time_series_length)

    model.fit(train_data=data)
    _ = model.predict(data)

    test_length = expected_num_windows * prediction_length

    expected_scores = np.abs(data.values.ravel()[-test_length:] - 1)
    if time_series_length == prediction_length:
        # conformalization will fall back to naive
        expected_scores = np.abs(data.values.ravel()[-test_length + 1 :] - data.values.ravel()[0])
        expected_scores = np.r_[expected_scores, expected_scores[-1]]
    expected_scores = np.sort(expected_scores)

    returned_scores = np.sort(model.returned_nonconformity_scores.ravel())

    assert np.allclose(expected_scores, returned_scores)


@pytest.mark.parametrize("model_class", NONSEASONAL_TESTABLE_MODELS)
@pytest.mark.parametrize("prediction_length", [1, 3, 10])
@pytest.mark.parametrize("positive_only", [True, False])
def test_when_intermittent_models_fit_then_values_are_lower_bounded(
    model_class, prediction_length, positive_only, temp_model_path
):
    data = DUMMY_VARIABLE_LENGTH_TS_DATAFRAME
    if positive_only:
        data[data < 0] = 0.0
    else:
        # make sure there are some negative values
        for c in data.columns:
            data[c] *= np.random.randn(*data[c].values.shape)

    model = model_class(
        path=temp_model_path,
        prediction_length=prediction_length,
        hyperparameters=DEFAULT_HYPERPARAMETERS,
        freq=data.freq,
    )
    model.fit(train_data=data)
    predictions = model.predict(data=data)

    for item_id in data.index.levels[0]:
        if positive_only:
            predictions.loc[item_id].values.min() >= 0
        else:
            predictions.loc[item_id].values.min() >= data.loc[item_id].values.min()


@pytest.mark.parametrize("model_class", TESTABLE_MODELS_LOCAL)
@pytest.mark.parametrize("prediction_length", [1, 3])
def test_when_local_models_fit_then_quantiles_are_present_and_ranked(model_class, prediction_length, temp_model_path):
    data = get_data_frame_with_item_index(["B", "A", "X"])
    model = model_class(
        path=temp_model_path,
        prediction_length=prediction_length,
        hyperparameters=DEFAULT_HYPERPARAMETERS,
        freq=data.freq,
    )
    model.fit(train_data=data)
    predictions = model.predict(data=data)

    quantile_columns = sorted(list(set(predictions.columns) - {"mean"}))

    assert set(model.quantile_levels) == set(float(q) for q in quantile_columns)
    assert np.diff(predictions[quantile_columns].values, axis=1).min() >= 0


def test_when_leading_nans_are_present_then_seasonal_naive_can_forecast(temp_model_path):
    data = get_data_frame_with_item_index(item_list=["A"], data_length=30, freq="D")
    data.iloc[:-3] = float("nan")
    model = SeasonalNaiveModel(
        path=temp_model_path, prediction_length=7, hyperparameters={**DEFAULT_HYPERPARAMETERS, "seasonal_period": 7}
    )
    model.fit(train_data=data)
    predictions = model.predict(data)

    assert not pd.isna(predictions).any(axis=None)


@pytest.mark.parametrize(
    "hyperparameters, expected_cls",
    [
        ({}, CrostonSBA),
        ({"variant": "SBA"}, CrostonSBA),
        ({"variant": "Classic"}, CrostonClassic),
        ({"variant": "Optimized"}, CrostonOptimized),
    ],
)
def test_when_variant_hyperparameter_provided_to_croston_model_then_correct_model_class_is_created(
    hyperparameters, expected_cls
):
    data = DUMMY_TS_DATAFRAME.copy()
    model = CrostonModel(freq=data.freq, hyperparameters=hyperparameters)
    model.fit(train_data=data)
    model_cls = model._get_model_type(model._local_model_args.get("variant"))
    assert model_cls is expected_cls
