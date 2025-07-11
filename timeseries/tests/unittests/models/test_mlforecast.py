import os
import shutil
import tempfile
from unittest import mock

import numpy as np
import pandas as pd
import pytest
from mlforecast.lag_transforms import RollingMean

from autogluon.timeseries import TimeSeriesDataFrame
from autogluon.timeseries.models.autogluon_tabular import DirectTabularModel, RecursiveTabularModel
from autogluon.timeseries.transforms.target_scaler import LocalMinMaxScaler, LocalStandardScaler
from autogluon.timeseries.utils.features import TimeSeriesFeatureGenerator

from ..common import (
    DATAFRAME_WITH_COVARIATES,
    DATAFRAME_WITH_STATIC,
    DUMMY_TS_DATAFRAME,
    DUMMY_VARIABLE_LENGTH_TS_DATAFRAME,
    get_data_frame_with_variable_lengths,
)


@pytest.mark.parametrize("prediction_length", [1, 5])
@pytest.mark.parametrize("known_covariates_names", [["known_1", "known_2"], []])
@pytest.mark.parametrize("static_features_names", [["cat_1"], []])
@pytest.mark.parametrize("differences", [[2, 3], []])
@pytest.mark.parametrize("lags", [[1, 2, 5], [4]])
def test_when_covariates_and_features_present_then_train_and_val_dfs_have_correct_shape(
    temp_model_path,
    mlforecast_model_class,
    prediction_length,
    known_covariates_names,
    static_features_names,
    differences,
    lags,
):
    item_id_to_length = {1: 30, 5: 40, 2: 25}
    data = get_data_frame_with_variable_lengths(item_id_to_length, covariates_names=known_covariates_names)
    if static_features_names:
        columns = {k: np.random.normal(size=len(item_id_to_length)) for k in static_features_names}
        data.static_features = pd.DataFrame(columns, index=data.item_ids)

    feat_gen = TimeSeriesFeatureGenerator(target="target", known_covariates_names=known_covariates_names)
    data = feat_gen.fit_transform(data)
    model = mlforecast_model_class(
        freq=data.freq,
        path=temp_model_path,
        prediction_length=prediction_length,
        covariate_metadata=feat_gen.covariate_metadata,
        hyperparameters={"differences": differences, "lags": lags},
    )
    # Initialize model._target_lags and model._date_features from freq
    model.fit(train_data=data, time_limit=3)
    train_df, val_df = model._generate_train_val_dfs(data)
    expected_num_features = (
        len(lags)
        + len(known_covariates_names)
        + len(model.covariate_metadata.known_covariates_real)  # item-normalized version of each real covariate
        + len(static_features_names)
        + len(model._date_features)
        + 2  # target, item_id
    )
    # sum(differences) rows  dropped per item, prediction_length rows are reserved for validation
    expected_num_train_rows = len(data) - (sum(differences) + model.prediction_length) * data.num_items
    expected_num_val_rows = data.num_items * model.prediction_length
    assert train_df.shape == (expected_num_train_rows, expected_num_features)
    assert val_df.shape == (expected_num_val_rows, expected_num_features)


@pytest.mark.parametrize("prediction_length", [1, 5])
@pytest.mark.parametrize("use_past_covariates", [True, False])
@pytest.mark.parametrize("use_known_covariates", [True, False])
@pytest.mark.parametrize("use_static_features", [True, False])
@pytest.mark.parametrize("eval_metric", ["WQL", "MASE"])
def test_when_covariates_and_features_are_varied_and_metric_provided_then_models_can_predict(
    temp_model_path,
    mlforecast_model_class,
    prediction_length,
    use_past_covariates,
    use_known_covariates,
    use_static_features,
    eval_metric,
):
    item_id_to_length = {1: 30, 5: 40, 2: 25}
    covariates_names = []
    known_covariates_names = []
    if use_known_covariates:
        known_covariates_names = ["known_1", "known_2"]
        covariates_names += known_covariates_names
    if use_past_covariates:
        covariates_names += ["past_1", "past_2"]

    data = get_data_frame_with_variable_lengths(item_id_to_length, covariates_names=known_covariates_names)

    if use_static_features:
        columns = {k: np.random.normal(size=len(item_id_to_length)) for k in ["static_cont_1", "static_cont_2"]} | {
            k: np.random.choice(["a", "b", "c"], size=len(item_id_to_length)) for k in ["static_cat_1", "static_cat_2"]
        }
        data.static_features = pd.DataFrame(columns, index=data.item_ids)

    feat_gen = TimeSeriesFeatureGenerator(target="target", known_covariates_names=known_covariates_names)
    data = feat_gen.fit_transform(data)
    model = mlforecast_model_class(
        freq=data.freq,
        path=temp_model_path,
        prediction_length=prediction_length,
        covariate_metadata=feat_gen.covariate_metadata,
        eval_metric=eval_metric,
    )
    # Initialize model._target_lags and model._date_features from freq
    model.fit(train_data=data, time_limit=3)

    train_data, known_covariates = data.get_model_inputs_for_scoring(prediction_length, known_covariates_names)
    predictions = model.predict(train_data, known_covariates=known_covariates)
    assert isinstance(predictions, TimeSeriesDataFrame)
    assert len(predictions) == train_data.num_items * model.prediction_length


@pytest.mark.parametrize("data", [DATAFRAME_WITH_STATIC, DATAFRAME_WITH_COVARIATES])
def test_when_covariates_and_features_present_then_model_can_predict(temp_model_path, mlforecast_model_class, data):
    prediction_length = 3
    known_covariates_names = data.columns.drop("target")
    data_train, known_covariates = data.get_model_inputs_for_scoring(
        prediction_length, known_covariates_names=known_covariates_names
    )

    feat_gen = TimeSeriesFeatureGenerator(target="target", known_covariates_names=known_covariates_names)
    data_train = feat_gen.fit_transform(data_train)

    model = mlforecast_model_class(
        path=temp_model_path,
        prediction_length=prediction_length,
        freq=data.freq,
        covariate_metadata=feat_gen.covariate_metadata,
    )
    model.fit(train_data=data_train, time_limit=10)
    predictions = model.predict(data_train, known_covariates=known_covariates)
    assert isinstance(predictions, TimeSeriesDataFrame)
    assert len(predictions) == data.num_items * model.prediction_length


@pytest.mark.parametrize("eval_metric", ["RMSE", "WQL", "MAPE", None])
def test_when_eval_metric_is_changed_then_model_can_predict(temp_model_path, mlforecast_model_class, eval_metric):
    data = DUMMY_VARIABLE_LENGTH_TS_DATAFRAME.copy()
    model = mlforecast_model_class(path=temp_model_path, eval_metric=eval_metric, freq=data.freq)
    model.fit(train_data=data)
    predictions = model.predict(data)
    assert len(predictions) == data.num_items * model.prediction_length


@pytest.mark.parametrize("differences", [[], [14]])
def test_given_long_time_series_passed_to_model_then_preprocess_receives_shortened_time_series(
    temp_model_path, mlforecast_model_class, differences
):
    max_num_samples = 1000
    prediction_length = 17
    data = get_data_frame_with_variable_lengths({"A": 1_000_000}, freq="min")
    model = mlforecast_model_class(
        path=temp_model_path,
        freq=data.freq,
        hyperparameters={"max_num_samples": max_num_samples, "differences": differences},
        prediction_length=prediction_length,
    )
    with mock.patch("mlforecast.MLForecast.preprocess") as mock_preprocess:
        try:
            model.fit(train_data=data)
        # using mock leads to ZeroDivisionError
        except ZeroDivisionError:
            pass
        received_mlforecast_df = mock_preprocess.call_args[0][0]
        assert len(received_mlforecast_df) == max_num_samples + prediction_length + sum(differences)


@pytest.mark.parametrize("differences", [[5], [15]])
@pytest.mark.parametrize("eval_metric", ["WQL", "MAPE"])
def test_given_some_time_series_are_too_short_then_forecast_doesnt_contain_nans_and_index_correct(
    temp_model_path, mlforecast_model_class, differences, eval_metric
):
    data = DUMMY_VARIABLE_LENGTH_TS_DATAFRAME
    prediction_length = 5
    model = mlforecast_model_class(
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
    expected_forecast_index = model.get_forecast_horizon_index(df_with_short)

    predictions = model.predict(df_with_short)
    assert not predictions.isna().values.any()
    assert (predictions.index == expected_forecast_index).all()


@pytest.mark.parametrize("differences", [[5], [15]])
@pytest.mark.parametrize("eval_metric", ["WQL", "MAPE"])
def test_given_some_time_series_are_too_short_then_seasonal_naive_forecast_is_used(
    temp_model_path,
    mlforecast_model_class,
    differences,
    eval_metric,
):
    data = get_data_frame_with_variable_lengths({"A": 50, "B": 60})
    prediction_length = 5
    model = mlforecast_model_class(
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


def test_when_point_forecast_metric_is_used_then_per_item_residuals_are_used_for_prediction(
    temp_model_path, mlforecast_model_class
):
    data = get_data_frame_with_variable_lengths({"A": 20, "B": 30, "C": 15})
    prediction_length = 5
    model = mlforecast_model_class(
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
    expected_forecast_index = model.get_forecast_horizon_index(data)
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


def test_given_train_data_has_nans_when_fit_called_then_nan_rows_removed_from_train_df(
    temp_model_path, mlforecast_model_class
):
    data = DUMMY_TS_DATAFRAME.copy()
    model = mlforecast_model_class(
        path=temp_model_path,
        freq=data.freq,
        eval_metric="WAPE",
        prediction_length=3,
        hyperparameters={"differences": []},
    )
    model.fit(train_data=data)
    train_df, val_df = model._generate_train_val_dfs(model.preprocess(data, is_train=True)[0])
    assert len(train_df) + len(val_df) == len(data.dropna())


@pytest.mark.parametrize("eval_metric", ["WAPE", "WQL"])
def test_when_trained_model_moved_to_different_folder_then_loaded_model_can_predict(
    mlforecast_model_class, eval_metric
):
    data = DUMMY_TS_DATAFRAME.copy().sort_index()
    old_model_dir = tempfile.mkdtemp()
    model = mlforecast_model_class(
        path=old_model_dir,
        freq=data.freq,
        eval_metric=eval_metric,
        quantile_levels=[0.1, 0.5, 0.9],
        prediction_length=3,
        hyperparameters={"differences": []},
    )
    model.fit(train_data=data)
    model.save()
    new_model_dir = tempfile.mkdtemp()
    shutil.move(model.path, new_model_dir)
    loaded_model = model.__class__.load(os.path.join(new_model_dir, model.name))
    predictions = loaded_model.predict(data)
    assert isinstance(predictions, TimeSeriesDataFrame)


@pytest.mark.parametrize("eval_metric", ["WAPE", "WQL"])
def test_when_target_transform_provided_then_scaler_is_used_inside_mlforecast(mlforecast_model_class, eval_metric):
    data = DUMMY_TS_DATAFRAME.copy().sort_index()
    model = mlforecast_model_class(
        freq=data.freq,
        eval_metric=eval_metric,
        quantile_levels=[0.1, 0.5, 0.9],
        prediction_length=3,
        hyperparameters={"target_scaler": "robust"},
    )
    model.fit(train_data=data)
    assert model.target_scaler is None
    assert model._scaler is not None


@pytest.mark.parametrize(
    "scaler_hp, expected_ag_scaler_type",
    [("min_max", LocalMinMaxScaler), ("standard", LocalStandardScaler), (None, type(None))],
)
def test_when_deprecated_scaler_hyperparameter_is_provided_then_correct_scaler_is_created(
    mlforecast_model_class, scaler_hp, expected_ag_scaler_type
):
    data = DUMMY_TS_DATAFRAME.copy().sort_index()
    model = mlforecast_model_class(
        freq=data.freq,
        hyperparameters={"scaler": scaler_hp, "model_name": "DUMMY"},
    )
    model.fit(train_data=data)
    assert model.target_scaler is None
    if scaler_hp is None:
        assert model._scaler is None
    else:
        assert isinstance(model._scaler.ag_scaler, expected_ag_scaler_type)


# TODO: Remove in v1.5 after 'tabular_hyperparameters' is removed
@pytest.mark.parametrize(
    "hparams_with_deprecated, model_name, model_hyperparameters",
    [
        ({"tabular_hyperparameters": {"CAT": {"iterations": 2}}}, "CAT", {"iterations": 2}),
        ({"tabular_hyperparameters": {"DUMMY": {}}}, "DUMMY", {}),
    ],
)
def test_when_deprecated_tabular_hyperparameters_are_provided_then_model_can_predict(
    mlforecast_model_class, hparams_with_deprecated, model_name, model_hyperparameters
):
    data = DUMMY_TS_DATAFRAME.copy()
    model = mlforecast_model_class(
        freq=data.freq,
        prediction_length=2,
        hyperparameters=hparams_with_deprecated,
    )
    model.fit(train_data=data, time_limit=3)
    tabular_model = model.get_tabular_model().model
    assert tabular_model.ag_key == model_name
    assert tabular_model._user_params == model_hyperparameters
    predictions = model.predict(data)
    assert isinstance(predictions, TimeSeriesDataFrame)


@pytest.mark.parametrize(
    "invalid_hparams_with_deprecated",
    [
        {"tabular_hyperparameters": {"CAT": {"iterations": 2}, "DUMMY": {}}},
        {"tabular_hyperparameters": {}},
    ],
)
def test_when_invalid_deprecated_tabular_hyperparameters_are_provided_then_exception_is_raised(
    mlforecast_model_class, invalid_hparams_with_deprecated
):
    data = DUMMY_TS_DATAFRAME.copy()
    model = mlforecast_model_class(
        freq=data.freq,
        prediction_length=2,
        hyperparameters=invalid_hparams_with_deprecated,
    )
    with pytest.raises(ValueError, match="cannot be automatically converted"):
        model.fit(train_data=data)


@pytest.mark.parametrize(
    "lag_transforms",
    [
        {1: [RollingMean(3)]},
        {2: [RollingMean(3), RollingMean(4)], 3: [RollingMean(5)]},
    ],
)
@pytest.mark.parametrize("hyperparameters", [{"differences": []}, {"differences": [1]}, {"differences": [1, 3]}])
def test_when_lag_transforms_provided_then_model_can_fit_and_predict(
    df_with_covariates_and_metadata, hyperparameters, lag_transforms
):
    data, covariate_metadata = df_with_covariates_and_metadata
    prediction_length = 4
    train_data, known_covariates = data.get_model_inputs_for_scoring(
        prediction_length, covariate_metadata.known_covariates
    )
    model = RecursiveTabularModel(
        freq=train_data.freq,
        prediction_length=prediction_length,
        covariate_metadata=covariate_metadata,
        hyperparameters={**hyperparameters, "lag_transforms": lag_transforms},
    )
    model.fit(train_data=train_data)
    predictions = model.predict(train_data, known_covariates)
    assert isinstance(predictions, TimeSeriesDataFrame)
    assert not predictions.isna().any(axis=None)
    assert predictions.index.equals(model.get_forecast_horizon_index(train_data))
