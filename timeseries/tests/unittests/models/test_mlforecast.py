from unittest import mock

import numpy as np
import pandas as pd
import pytest

from autogluon.timeseries import TimeSeriesDataFrame
from autogluon.timeseries.models.autogluon_tabular.mlforecast import DirectTabularModel, RecursiveTabularModel
from autogluon.timeseries.utils.features import TimeSeriesFeatureGenerator
from autogluon.timeseries.utils.forecast import get_forecast_horizon_index_ts_dataframe

from ..common import (
    DATAFRAME_WITH_COVARIATES,
    DATAFRAME_WITH_STATIC,
    DUMMY_TS_DATAFRAME,
    DUMMY_VARIABLE_LENGTH_TS_DATAFRAME,
    get_data_frame_with_variable_lengths,
)

TESTABLE_MODELS = [
    DirectTabularModel,
    RecursiveTabularModel,
]


@pytest.mark.parametrize("model_type", TESTABLE_MODELS)
@pytest.mark.parametrize("prediction_length", [1, 5])
@pytest.mark.parametrize("known_covariates_names", [["known_1", "known_2"], []])
@pytest.mark.parametrize("static_features_names", [["cat_1"], []])
@pytest.mark.parametrize("differences", [[2, 3], []])
@pytest.mark.parametrize("lags", [[1, 2, 5], [4]])
def test_when_covariates_and_features_present_then_train_and_val_dfs_have_correct_shape(
    temp_model_path, model_type, prediction_length, known_covariates_names, static_features_names, differences, lags
):
    item_id_to_length = {1: 30, 5: 40, 2: 25}
    data = get_data_frame_with_variable_lengths(item_id_to_length, covariates_names=known_covariates_names)
    if static_features_names:
        columns = {k: np.random.normal(size=len(item_id_to_length)) for k in static_features_names}
        data.static_features = pd.DataFrame(columns, index=data.item_ids)

    feat_gen = TimeSeriesFeatureGenerator(target="target", known_covariates_names=known_covariates_names)
    data = feat_gen.fit_transform(data)
    model = model_type(
        freq=data.freq,
        path=temp_model_path,
        prediction_length=prediction_length,
        metadata=feat_gen.covariate_metadata,
        hyperparameters={"differences": differences, "lags": lags, "tabular_hyperparameters": {"DUMMY": {}}},
    )
    # Initialize model._target_lags and model._date_features from freq
    model.fit(train_data=data, time_limit=3)
    train_df, val_df = model._generate_train_val_dfs(data)
    expected_num_features = (
        len(lags)
        + len(known_covariates_names)
        + len(model.metadata.known_covariates_real)  # item-normalized version of each real covariate
        + len(static_features_names)
        + len(model._date_features)
        + 2  # target, item_id
    )
    # sum(differences) rows  dropped per item, prediction_length rows are reserved for validation
    expected_num_train_rows = len(data) - (sum(differences) + model.prediction_length) * data.num_items
    expected_num_val_rows = data.num_items * model.prediction_length
    assert train_df.shape == (expected_num_train_rows, expected_num_features)
    assert val_df.shape == (expected_num_val_rows, expected_num_features)


@pytest.mark.parametrize("model_type", TESTABLE_MODELS)
@pytest.mark.parametrize("data", [DATAFRAME_WITH_STATIC, DATAFRAME_WITH_COVARIATES])
def test_when_covariates_and_features_present_then_model_can_predict(temp_model_path, model_type, data):
    prediction_length = 3
    known_covariates_names = data.columns.drop("target")
    data_train, known_covariates = data.get_model_inputs_for_scoring(
        prediction_length, known_covariates_names=known_covariates_names
    )

    feat_gen = TimeSeriesFeatureGenerator(target="target", known_covariates_names=known_covariates_names)
    data_train = feat_gen.fit_transform(data_train)

    model = model_type(
        path=temp_model_path, prediction_length=prediction_length, freq=data.freq, metadata=feat_gen.covariate_metadata
    )
    model.fit(train_data=data_train, time_limit=10)
    predictions = model.predict(data_train, known_covariates=known_covariates)
    assert isinstance(predictions, TimeSeriesDataFrame)
    assert len(predictions) == data.num_items * model.prediction_length


@pytest.mark.parametrize("model_type", TESTABLE_MODELS)
@pytest.mark.parametrize("eval_metric", ["RMSE", "WQL", "MAPE", None])
def test_when_eval_metric_is_changed_then_model_can_predict(temp_model_path, model_type, eval_metric):
    data = DUMMY_VARIABLE_LENGTH_TS_DATAFRAME.copy()
    model = model_type(path=temp_model_path, eval_metric=eval_metric, freq=data.freq)
    model.fit(train_data=data)
    predictions = model.predict(data)
    assert len(predictions) == data.num_items * model.prediction_length


@pytest.mark.parametrize("model_type", TESTABLE_MODELS)
@pytest.mark.parametrize("scaler", ["standard", "mean_abs"])
def test_when_scaler_used_during_fit_then_scales_are_stored(temp_model_path, model_type, scaler):
    data = DUMMY_VARIABLE_LENGTH_TS_DATAFRAME.copy()
    model = model_type(
        path=temp_model_path,
        freq=data.freq,
        hyperparameters={"scaler": scaler, "tabular_hyperparameters": {"DUMMY": {}}},
    )
    model.fit(train_data=data)
    scale_per_item = model._get_scale_per_item(data.item_ids)
    assert model._scaler is not None
    assert scale_per_item.index.equals(data.item_ids)


@pytest.mark.parametrize("model_type", TESTABLE_MODELS)
@pytest.mark.parametrize("differences", [[], [14]])
def test_given_long_time_series_passed_to_model_then_preprocess_receives_shortened_time_series(
    temp_model_path, model_type, differences
):
    max_num_samples = 1000
    prediction_length = 17
    data = get_data_frame_with_variable_lengths({"A": 1_000_000}, freq="T")
    model = model_type(
        path=temp_model_path,
        freq=data.freq,
        hyperparameters={"max_num_samples": max_num_samples, "differences": differences},
        prediction_length=prediction_length,
    )
    with mock.patch("mlforecast.MLForecast.preprocess") as mock_preprocess:
        try:
            model.fit(train_data=data)
        # using mock leads to AssertionError
        except AssertionError:
            pass
        received_mlforecast_df = mock_preprocess.call_args[0][0]
        assert len(received_mlforecast_df) == max_num_samples + prediction_length + sum(differences)


@pytest.mark.parametrize("model_type", TESTABLE_MODELS)
@pytest.mark.parametrize("differences", [[5], [15]])
@pytest.mark.parametrize("eval_metric", ["WQL", "MAPE"])
def test_given_some_time_series_are_too_short_then_forecast_doesnt_contain_nans_and_index_correct(
    temp_model_path, model_type, differences, eval_metric
):
    data = DUMMY_VARIABLE_LENGTH_TS_DATAFRAME
    prediction_length = 5
    model = model_type(
        path=temp_model_path,
        freq=data.freq,
        hyperparameters={"differences": differences},
        prediction_length=prediction_length,
        eval_metric=eval_metric,
    )
    model.fit(train_data=data)

    df_with_short = get_data_frame_with_variable_lengths(
        {"A": sum(differences), "B": sum(differences) + 5, "C": sum(differences) + 100}, freq=model.freq
    )
    expected_forecast_index = get_forecast_horizon_index_ts_dataframe(df_with_short, prediction_length)

    predictions = model.predict(df_with_short)
    assert not predictions.isna().values.any()
    assert (predictions.index == expected_forecast_index).all()


@pytest.mark.parametrize("model_type", TESTABLE_MODELS)
@pytest.mark.parametrize("differences", [[5], [15]])
@pytest.mark.parametrize("eval_metric", ["WQL", "MAPE"])
def test_given_some_time_series_are_too_short_then_seasonal_naive_forecast_is_used(
    temp_model_path,
    model_type,
    differences,
    eval_metric,
):
    data = get_data_frame_with_variable_lengths({"A": 50, "B": 60})
    prediction_length = 5
    model = model_type(
        path=temp_model_path,
        freq=data.freq,
        hyperparameters={"differences": differences},
        prediction_length=prediction_length,
        eval_metric=eval_metric,
    )
    model.fit(train_data=data)

    df_with_short = get_data_frame_with_variable_lengths(
        {"A": sum(differences), "B": sum(differences) + 5, "C": sum(differences) + 100}, freq=model.freq
    )
    with mock.patch("autogluon.timeseries.models.local.naive.SeasonalNaiveModel.predict") as snaive_predict:
        try:
            model.predict(df_with_short)
        except TypeError:
            pass
        assert snaive_predict.call_args[0][0].equals(df_with_short.loc[["A"]])


@pytest.mark.parametrize("model_type", TESTABLE_MODELS)
def test_when_point_forecast_metric_is_used_then_per_item_residuals_are_used_for_prediction(
    temp_model_path, model_type
):
    data = get_data_frame_with_variable_lengths({"A": 20, "B": 30, "C": 15})
    prediction_length = 5
    model = model_type(
        path=temp_model_path,
        freq=data.freq,
        prediction_length=prediction_length,
        eval_metric="MASE",
    )
    model.fit(train_data=data, time_limit=15)
    assert (model._residuals_std_per_item.index == sorted(data.item_ids)).all()

    # Remove _avg_residuals_std to ensure that it's not used to impute missing values
    model._avg_residuals_std = None

    predictions = model.predict(data)
    expected_forecast_index = get_forecast_horizon_index_ts_dataframe(data, prediction_length)
    assert not predictions.isna().values.any()
    assert (predictions.index == expected_forecast_index).all()


@pytest.mark.parametrize(
    "model_type, eval_metric",
    [(RecursiveTabularModel, "WQL"), (DirectTabularModel, "WQL"), (DirectTabularModel, "MASE")],
)
def test_when_mlf_model_is_used_then_predictions_have_correct_scale(temp_model_path, model_type, eval_metric):
    prediction_length = 5
    value = 2e6
    data = TimeSeriesDataFrame.from_iterable_dataset(
        [{"start": pd.Period("2020-01-01", freq="D"), "target": np.random.normal(loc=value, scale=10, size=[30])}]
    )
    model = model_type(
        path=temp_model_path,
        freq=data.freq,
        eval_metric=eval_metric,
        quantile_levels=[0.1, 0.5, 0.9],
        prediction_length=prediction_length,
    )
    model.fit(train_data=data)
    predictions = model.predict(data)
    assert np.all(np.abs(predictions.values - value) < value)


@pytest.mark.parametrize("model_type", TESTABLE_MODELS)
def test_given_train_data_has_nans_when_fit_called_then_nan_rows_removed_from_train_df(temp_model_path, model_type):
    data = DUMMY_TS_DATAFRAME.copy()
    model = model_type(
        path=temp_model_path,
        freq=data.freq,
        eval_metric="WAPE",
        prediction_length=3,
        hyperparameters={"differences": []},
    )
    model.fit(train_data=data)
    train_df, val_df = model._generate_train_val_dfs(model.preprocess(data, is_train=True))
    assert len(train_df) + len(val_df) == len(data.dropna())
