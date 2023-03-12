from unittest import mock

import numpy as np
import pandas as pd
import pytest

from autogluon.timeseries.models.autogluon_tabular import AutoGluonTabularModel
from autogluon.timeseries.utils.features import TimeSeriesFeatureGenerator

from ..common import DATAFRAME_WITH_STATIC, DUMMY_VARIABLE_LENGTH_TS_DATAFRAME, get_data_frame_with_variable_lengths

TESTABLE_MODELS = [
    AutoGluonTabularModel,
]


@pytest.mark.parametrize(
    "data, max_rows_per_item, expected_length",
    [
        (DUMMY_VARIABLE_LENGTH_TS_DATAFRAME, None, DUMMY_VARIABLE_LENGTH_TS_DATAFRAME.shape[0]),
        (DUMMY_VARIABLE_LENGTH_TS_DATAFRAME, 1, DUMMY_VARIABLE_LENGTH_TS_DATAFRAME.num_items),
        (DUMMY_VARIABLE_LENGTH_TS_DATAFRAME, 3, 3 * DUMMY_VARIABLE_LENGTH_TS_DATAFRAME.num_items),
    ],
)
def test_when_feature_df_is_constructed_then_shape_is_correct(
    data, max_rows_per_item, expected_length, temp_model_path
):
    model = AutoGluonTabularModel(path=temp_model_path)
    # Initialize model._lag_indices and model._time_features from freq
    model.fit(train_data=data, time_limit=2)
    df = model._get_features_dataframe(data, max_rows_per_item=max_rows_per_item)
    expected_num_features = len(model._target_lag_indices) + len(model._time_features) + 1
    assert df.shape == (expected_length, expected_num_features)


@pytest.mark.parametrize("prediction_length", [1, 5])
def test_when_predict_is_called_then_get_features_dataframe_receives_correct_input_with_dummy(
    temp_model_path, prediction_length
):
    model = AutoGluonTabularModel(path=temp_model_path, prediction_length=prediction_length)
    model.fit(train_data=DUMMY_VARIABLE_LENGTH_TS_DATAFRAME, time_limit=2)
    with mock.patch.object(model, "_get_features_dataframe") as patch_method:
        try:
            model.predict(DUMMY_VARIABLE_LENGTH_TS_DATAFRAME)
        except ValueError as e:
            if "No objects to concatenate" in str(e):
                df_with_dummy = patch_method.call_args.args[0]
                for item_id in DUMMY_VARIABLE_LENGTH_TS_DATAFRAME.item_ids:
                    original_timestamps = DUMMY_VARIABLE_LENGTH_TS_DATAFRAME.loc[item_id].index
                    new_timestamps = df_with_dummy.loc[item_id].index
                    assert len(new_timestamps.difference(original_timestamps)) == prediction_length


@pytest.mark.parametrize("known_covariates_names", [["known_1", "known_2"], []])
@pytest.mark.parametrize("past_covariates_names", [["past_1", "past_2", "past_3"], []])
@pytest.mark.parametrize("static_features_names", [["cat_1"], []])
def test_when_covariates_and_features_present_then_feature_df_shape_is_correct(
    temp_model_path, known_covariates_names, past_covariates_names, static_features_names
):
    item_id_to_length = {1: 10, 5: 20, 2: 30}
    data = get_data_frame_with_variable_lengths(
        item_id_to_length, covariates_names=known_covariates_names + past_covariates_names
    )
    if static_features_names:
        columns = {k: np.random.normal(size=len(item_id_to_length)) for k in static_features_names}
        data.static_features = pd.DataFrame(columns, index=data.item_ids)

    feat_gen = TimeSeriesFeatureGenerator(target="target", known_covariates_names=known_covariates_names)
    data = feat_gen.fit_transform(data)
    # Initialize model._target_lag_indices and model._time_features from freq
    model = AutoGluonTabularModel(path=temp_model_path, metadata=feat_gen.covariate_metadata)
    model.fit(train_data=data, time_limit=2)
    df = model._get_features_dataframe(data)
    expected_num_features = (
        len(model._time_features)
        + 1
        + len(model._target_lag_indices)
        + len(known_covariates_names) * len(model._known_covariates_lag_indices)
        + len(past_covariates_names) * len(model._past_covariates_lag_indices)
        + len(static_features_names)
    )
    assert len(df.columns) == expected_num_features
    assert all(col in df.columns for col in static_features_names)


def test_when_static_features_present_then_prediction_works(temp_model_path):
    data = DATAFRAME_WITH_STATIC.copy()
    model = AutoGluonTabularModel(path=temp_model_path)
    model.fit(train_data=data, time_limit=2)
    model.predict(data)
