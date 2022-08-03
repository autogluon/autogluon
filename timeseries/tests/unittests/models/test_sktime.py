from unittest import mock

import pandas as pd
import pytest

from autogluon.timeseries import TimeSeriesDataFrame

try:
    from sktime.forecasting.arima import ARIMA, AutoARIMA
    from sktime.forecasting.ets import AutoETS
    from sktime.forecasting.tbats import TBATS
    from sktime.forecasting.theta import ThetaForecaster
except ImportError:
    pytest.skip("sktime not available", allow_module_level=True)

from autogluon.timeseries.models.sktime import (
    AbstractSktimeModel,
    ARIMAModel,
    AutoARIMAModel,
    AutoETSModel,
    TBATSModel,
    ThetaModel,
)

from ..common import DUMMY_TS_DATAFRAME, get_data_frame_with_item_index

TESTABLE_MODELS = [
    ARIMAModel,
    # AutoARIMAModel,
    AutoETSModel,
    # TBATSModel,
    ThetaModel,
]


def test_when_sktime_converts_dataframe_then_data_not_duplicated_and_index_correct():
    model = AutoETSModel()

    df = DUMMY_TS_DATAFRAME.copy(deep=True)
    sktime_df = model._to_sktime_data_frame(df)

    assert isinstance(sktime_df, pd.DataFrame)
    assert not isinstance(sktime_df, TimeSeriesDataFrame)

    assert len(df) == len(sktime_df)
    assert isinstance(sktime_df.index.levels[-1], pd.PeriodIndex)
    assert (
        a.to_timestamp() == b
        for a, b in zip(
            sktime_df.index.get_level_values(-1),
            df.index.get_level_values(-1),
        )
    )

    # data is not copied
    assert df.values.base is sktime_df.values.base


def test_when_sktime_converts_from_dataframe_then_data_not_duplicated_and_index_correct():
    model = AutoETSModel()

    sktime_df = model._to_sktime_data_frame(DUMMY_TS_DATAFRAME.copy(deep=True))
    df = model._to_time_series_data_frame(sktime_df)

    assert isinstance(df, TimeSeriesDataFrame)
    assert isinstance(df.index.levels[-1], pd.DatetimeIndex)
    assert (
        a.to_timestamp() == b
        for a, b in zip(
            sktime_df.index.get_level_values(-1),
            df.index.get_level_values(-1),
        )
    )

    # data is not copied
    assert df.values.base is sktime_df.values.base


@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
def test_when_sktime_models_saved_then_forecasters_can_be_loaded(model_class, temp_model_path):
    model = model_class()
    model.fit(train_data=DUMMY_TS_DATAFRAME)
    model.save()

    loaded_model = model.__class__.load(path=model.path)

    assert isinstance(model.sktime_forecaster, loaded_model.sktime_forecaster.__class__)
    assert repr(loaded_model.sktime_forecaster) == repr(model.sktime_forecaster)


@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
@pytest.mark.parametrize(
    "train_data, test_data",
    [
        (
            get_data_frame_with_item_index([0, 1, 2, 3]),
            get_data_frame_with_item_index([2, 3, 4, 5]),
        ),
        (
            get_data_frame_with_item_index(["A", "B", "C"]),
            get_data_frame_with_item_index(["A", "B", "D"]),
        ),
        (
            get_data_frame_with_item_index(["A", "B"]),
            get_data_frame_with_item_index(["A", "B", "C"]),
        ),
        (
            get_data_frame_with_item_index(["A", "B", "C"]),
            get_data_frame_with_item_index(["A", "B"]),
        ),
    ],
)
def test_when_predict_called_with_test_data_then_predictor_inference_correct(
    model_class, temp_model_path, train_data, test_data
):
    model = model_class(
        path=temp_model_path,
        prediction_length=3,
    )

    model.fit(train_data=train_data)
    with mock.patch.object(AbstractSktimeModel, "_fit") as mock_fit:

        _ = model.predict(test_data)
        mock_fit.assert_called_with(test_data)


@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
@pytest.mark.parametrize(
    "freqstr, ts_length, expected_sp",
    [
        ("H", 100, 24),
        ("2H", 100, 12),
        ("B", 100, 5),
        ("M", 100, 12),
        ("H", 5, 1),
        ("2H", 5, 1),
        ("B", 5, 1),
        ("M", 5, 1),
    ],
)
def test_when_fit_called_with_then_seasonality_period_set_correctly(
    model_class,
    temp_model_path,
    freqstr,
    ts_length,
    expected_sp,
):
    if "sp" not in model_class.sktime_allowed_init_args:
        return

    model = model_class(
        path=temp_model_path,
        prediction_length=3,
    )

    train_data = get_data_frame_with_item_index(
        ["A", "B", "C"],
        data_length=ts_length,
        freq=freqstr,
    )

    model.fit(train_data=train_data)

    assert model.sktime_forecaster.sp == expected_sp
