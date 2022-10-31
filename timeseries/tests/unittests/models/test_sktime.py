import logging
from unittest import mock

import pandas as pd
import pytest

from autogluon.timeseries import TimeSeriesDataFrame, SKTIME_INSTALLED
if not SKTIME_INSTALLED:
    pytest.skip()

from autogluon.timeseries.models.sktime import (  # AutoARIMAModel,; TBATSModel,
    AbstractSktimeModel,
    ARIMASktimeModel,
    AutoETSSktimeModel,
    ThetaSktimeModel,
)

from ..common import DUMMY_TS_DATAFRAME, get_data_frame_with_item_index

TESTABLE_MODELS = [
    ARIMASktimeModel,
    AutoETSSktimeModel,
    ThetaSktimeModel,
]


def test_when_sktime_converts_dataframe_then_data_not_duplicated_and_index_correct():
    model = AutoETSSktimeModel()

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
    model = AutoETSSktimeModel()

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


@pytest.mark.skip("Skip for now because of the logging changes.")
@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
@pytest.mark.parametrize("hyperparameters", [{"seasonal_period": None}, {}])
@pytest.mark.parametrize(
    "freqstr, ts_length, expected_sp, should_warn",
    [
        ("H", 100, 24, False),
        ("2H", 100, 12, False),
        ("B", 100, 5, False),
        ("M", 100, 12, False),
        ("H", 5, 1, True),
        ("2H", 5, 1, True),
        ("B", 5, 1, True),
        ("M", 5, 1, True),
    ],
)
def test_when_seasonal_period_is_set_to_none_then_inferred_period_is_used(
    model_class,
    hyperparameters,
    temp_model_path,
    freqstr,
    ts_length,
    expected_sp,
    should_warn,
    caplog,
):
    if "sp" not in model_class.sktime_allowed_init_args:
        return

    model = model_class(
        path=temp_model_path,
        prediction_length=3,
        hyperparameters=hyperparameters,
    )

    train_data = get_data_frame_with_item_index(
        ["A", "B", "C"],
        data_length=ts_length,
        freq=freqstr,
    )

    with caplog.at_level(logging.WARNING):
        model.fit(train_data=train_data)
        if should_warn:
            assert (
                "requires training series of length at least 2 * seasonal_period" in caplog.text
            ), "Model should raise a warning since fail_if_misconfigured = False by default"

    assert model.sktime_forecaster.sp == expected_sp


@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
@pytest.mark.parametrize(
    "freqstr, ts_length, provided_seasonal_period",
    [
        ("H", 100, 12),
        ("2H", 100, 5),
        ("B", 100, 10),
        ("M", 100, 24),
        ("H", 5, 1),
    ],
)
def test_when_seasonal_period_is_provided_then_inferred_period_is_overriden(
    model_class,
    temp_model_path,
    freqstr,
    ts_length,
    provided_seasonal_period,
):
    if "sp" not in model_class.sktime_allowed_init_args:
        return

    model = model_class(
        path=temp_model_path,
        prediction_length=3,
        hyperparameters={"seasonal_period": provided_seasonal_period},
    )

    train_data = get_data_frame_with_item_index(
        ["A", "B", "C"],
        data_length=ts_length,
        freq=freqstr,
    )

    model.fit(train_data=train_data)

    assert model.sktime_forecaster.sp == provided_seasonal_period


@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
def test_when_fail_if_misconfigured_is_true_then_seasonality_fails_on_short_sequences(model_class, temp_model_path):
    if "sp" not in model_class.sktime_allowed_init_args:
        return

    model = model_class(
        path=temp_model_path,
        prediction_length=3,
        hyperparameters={"fail_if_misconfigured": True},
    )

    train_data = get_data_frame_with_item_index(
        ["A", "B", "C"],
        data_length=40,
        freq="H",
    )

    with pytest.raises(ValueError):
        model.fit(train_data=train_data)
        pytest.fail("Model should have failed because train_data too short and fail_if_misconfigured = True")


@pytest.mark.skip("Skip for now because of the logging changes.")
@pytest.mark.parametrize("model_class", TESTABLE_MODELS)
def test_when_invalid_model_arguments_provided_then_sktime_ignores_them(model_class, temp_model_path, caplog):
    model = model_class(
        path=temp_model_path,
        prediction_length=3,
        hyperparameters={"bad_argument": 33},
    )
    with caplog.at_level(logging.WARNING):
        model.fit(train_data=DUMMY_TS_DATAFRAME)
        assert "ignores following arguments: ['bad_argument']" in caplog.text
        assert "bad_argument" not in model.sktime_forecaster.get_params().keys()
