from unittest import mock

import pytest

from autogluon.timeseries.models.autogluon_tabular import AutoGluonTabularModel

from ..common import DATAFRAME_WITH_STATIC, DUMMY_VARIABLE_LENGTH_TS_DATAFRAME

TESTABLE_MODELS = [
    AutoGluonTabularModel,
]


@pytest.mark.parametrize(
    "data, last_k_values, expected_length",
    [
        (DUMMY_VARIABLE_LENGTH_TS_DATAFRAME, None, DUMMY_VARIABLE_LENGTH_TS_DATAFRAME.shape[0]),
        (DUMMY_VARIABLE_LENGTH_TS_DATAFRAME, 1, DUMMY_VARIABLE_LENGTH_TS_DATAFRAME.num_items),
        (DUMMY_VARIABLE_LENGTH_TS_DATAFRAME, 3, 3 * DUMMY_VARIABLE_LENGTH_TS_DATAFRAME.num_items),
    ],
)
def test_when_feature_df_is_constructed_then_shape_is_correct(data, last_k_values, expected_length, temp_model_path):
    model = AutoGluonTabularModel(path=temp_model_path)
    # Initialize model._lag_indices and model._time_features from freq
    model.fit(train_data=data, time_limit=2)
    df = model._get_features_dataframe(data, last_k_values=last_k_values)
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


def test_when_static_features_present_then_shape_is_correct(temp_model_path):
    data = DATAFRAME_WITH_STATIC.copy()
    model = AutoGluonTabularModel(path=temp_model_path)
    # Initialize model._target_lag_indices and model._time_features from freq
    model.fit(train_data=data, time_limit=2)
    df = model._get_features_dataframe(data)
    expected_num_features = (
        len(model._target_lag_indices) + len(model._time_features) + 1 + len(data.static_features.columns)
    )
    assert len(df.columns) == expected_num_features
    assert all(col in df.columns for col in data.static_features.columns)


def test_when_static_features_present_then_prediction_works(temp_model_path):
    data = DATAFRAME_WITH_STATIC.copy()
    model = AutoGluonTabularModel(path=temp_model_path)
    model.fit(train_data=data, time_limit=2)
    model.predict(data)
