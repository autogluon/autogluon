from unittest import mock

import pandas as pd
import pytest

from autogluon.timeseries import TimeSeriesDataFrame

try:
    from sktime.forecasting.ets import AutoETS
    from sktime.forecasting.arima import AutoARIMA, ARIMA
    from sktime.forecasting.tbats import TBATS
    from sktime.forecasting.theta import ThetaForecaster
except ImportError:
    pytest.skip("sktime not available", allow_module_level=True)

from autogluon.timeseries.models.sktime import (
    ARIMAModel,
    AutoARIMAModel,
    AutoETSModel,
    TBATSModel,
    ThetaModel,
    AbstractSktimeModel,
)

from ..common import DUMMY_TS_DATAFRAME, get_data_frame_with_item_index

TESTABLE_MODELS = [
    ARIMAModel,
    # AutoARIMAModel,
    AutoETSModel,
    # TBATSModel,
    ThetaModel,
]


@pytest.mark.parametrize(
    "model_class, sktime_forecaster_class, good_params, bad_params",
    [
        (
            ThetaModel,
            ThetaForecaster,
            dict(sp=2),
            dict(some_bad_param="A"),
        ),
        (
            AutoARIMAModel,
            AutoARIMA,
            dict(sp=2, max_q=4),
            dict(some_bad_param="A", some_other_bad_param=10),
        ),
        (
            ARIMAModel,
            ARIMA,
            dict(order=(1, 1, 1)),
            dict(some_bad_param="A", some_other_bad_param=10),
        ),
        (
            TBATSModel,
            TBATS,
            dict(use_box_cox=False),
            dict(some_bad_param="A", some_other_bad_param=10),
        ),
        (
            AutoETSModel,
            AutoETS,
            dict(error="mul", trend=None),
            dict(some_bad_param="A"),
        ),
    ],
)
def test_when_sktime_models_fitted_then_allowed_hyperparameters_are_passed_to_sktime_forecasters(
    model_class, sktime_forecaster_class, good_params, bad_params
):
    hyperparameters = good_params.copy()
    hyperparameters.update(bad_params)
    model = model_class(hyperparameters=hyperparameters)

    with mock.patch.object(target=sktime_forecaster_class, attribute="__init__") as const_mock:
        try:
            model.fit(train_data=DUMMY_TS_DATAFRAME)
        except TypeError:
            pass
        finally:
            const_mock.assert_called_with(**good_params)


def test_when_sktime_converts_dataframe_then_data_not_duplicated_and_index_correct():
    model = AutoETSModel()

    df = DUMMY_TS_DATAFRAME.copy(deep=True)
    skt_df = model._to_skt_data_frame(df)

    assert isinstance(skt_df, pd.DataFrame)
    assert not isinstance(skt_df, TimeSeriesDataFrame)

    assert len(df) == len(skt_df)
    assert isinstance(skt_df.index.levels[-1], pd.PeriodIndex)
    assert (
        a.to_timestamp() == b
        for a, b in zip(
            skt_df.index.get_level_values(-1),
            df.index.get_level_values(-1),
        )
    )

    # data is not copied
    assert df.values.base is skt_df.values.base


def test_when_sktime_converts_from_dataframe_then_data_not_duplicated_and_index_correct():
    model = AutoETSModel()

    skt_df = model._to_skt_data_frame(DUMMY_TS_DATAFRAME.copy(deep=True))
    df = model._to_time_series_data_frame(skt_df)

    assert isinstance(df, TimeSeriesDataFrame)
    assert isinstance(df.index.levels[-1], pd.DatetimeIndex)
    assert (
        a.to_timestamp() == b
        for a, b in zip(
            skt_df.index.get_level_values(-1),
            df.index.get_level_values(-1),
        )
    )

    # data is not copied
    assert df.values.base is skt_df.values.base


@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
def test_when_skt_models_saved_then_forecasters_can_be_loaded(model_class, temp_model_path):
    model = model_class()
    model.fit(train_data=DUMMY_TS_DATAFRAME)
    model.save()

    loaded_model = model.__class__.load(path=model.path)

    assert isinstance(model.skt_forecaster, loaded_model.skt_forecaster.__class__)
    assert repr(loaded_model.skt_forecaster) == repr(model.skt_forecaster)


@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
@pytest.mark.parametrize(
    "train_data, test_data",
    [
        (get_data_frame_with_item_index([0, 1, 2, 3]), get_data_frame_with_item_index([2, 3, 4, 5])),
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
        freq="H",
        prediction_length=3,
    )

    model.fit(train_data=train_data)
    with mock.patch.object(AbstractSktimeModel, "_fit") as mock_fit:

        _ = model.predict(test_data)
        mock_fit.assert_called_with(test_data)
