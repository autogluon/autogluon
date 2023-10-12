import numpy as np
import pandas as pd
import pytest

from autogluon.timeseries import TimeSeriesDataFrame
from autogluon.timeseries.models.autogluon_tabular.mlforecast import DirectTabularModel, RecursiveTabularModel
from autogluon.timeseries.utils.features import TimeSeriesFeatureGenerator

from ..common import (
    DATAFRAME_WITH_COVARIATES,
    DATAFRAME_WITH_STATIC,
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
        len(lags) + len(known_covariates_names) + len(static_features_names) + len(model._date_features) + 1
    )
    # sum(differences) rows  dropped per item, prediction_length rows are reserved for validation
    expected_num_train_rows = len(data) - (sum(differences) + model.prediction_length) * data.num_items
    expected_num_val_rows = data.num_items * model.prediction_length
    assert train_df.shape == (expected_num_train_rows, expected_num_features)
    assert val_df.shape == (expected_num_val_rows, expected_num_features)


@pytest.mark.parametrize("model_type", TESTABLE_MODELS)
@pytest.mark.parametrize("data", [DATAFRAME_WITH_STATIC, DATAFRAME_WITH_COVARIATES])
def test_when_covariates_and_features_present_then_model_can_predict(temp_model_path, model_type, data):
    data = data.copy()
    model = model_type(path=temp_model_path, freq=data.freq)
    model.fit(train_data=data, time_limit=10)
    predictions = model.predict(data)
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
