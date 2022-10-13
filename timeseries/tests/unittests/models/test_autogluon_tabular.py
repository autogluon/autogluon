import pytest
from unittest import mock

from autogluon.timeseries.models.autogluon_tabular import AutoGluonTabularModel

from ..common import DUMMY_VARIABLE_LENGTH_TS_DATAFRAME

TESTABLE_MODELS = [
    AutoGluonTabularModel,
]


@pytest.mark.parametrize(
    "df, last_k_values, expected_length",
    [
        (DUMMY_VARIABLE_LENGTH_TS_DATAFRAME, None, DUMMY_VARIABLE_LENGTH_TS_DATAFRAME.shape[0]),
        (DUMMY_VARIABLE_LENGTH_TS_DATAFRAME, 1, DUMMY_VARIABLE_LENGTH_TS_DATAFRAME.num_items),
        (DUMMY_VARIABLE_LENGTH_TS_DATAFRAME, 3, 3 * DUMMY_VARIABLE_LENGTH_TS_DATAFRAME.num_items),
    ],
)
def test_when_feature_df_is_constructed_then_shape_is_correct(df, last_k_values, expected_length):
    model = AutoGluonTabularModel()
    # Initialize model._lag_indices and model._time_features from freq
    model.fit(train_data=DUMMY_VARIABLE_LENGTH_TS_DATAFRAME, time_limit=2)
    df = model._get_features_dataframe(DUMMY_VARIABLE_LENGTH_TS_DATAFRAME, last_k_values=last_k_values)
    expected_num_features = len(model._lag_indices) + len(model._time_features) + 1
    assert df.shape == (expected_length, expected_num_features)


def test_when_predict_is_called_then_get_features_dataframe_receives_correct_input_with_dummy():
    model = AutoGluonTabularModel(prediction_length=1)
    model.fit(train_data=DUMMY_VARIABLE_LENGTH_TS_DATAFRAME, time_limit=2)
    with mock.patch.object(model, "_get_features_dataframe") as patch_method:
        try:
            model.predict(DUMMY_VARIABLE_LENGTH_TS_DATAFRAME)
        except TypeError as e:
            if "data must be TabularDataset or pandas.DataFrame" in str(e):
                df_with_dummy = patch_method.call_args.args[0]
                for item_id in DUMMY_VARIABLE_LENGTH_TS_DATAFRAME.item_ids:
                    original_timestamps = DUMMY_VARIABLE_LENGTH_TS_DATAFRAME.loc[item_id].index
                    new_timestamps = df_with_dummy.loc[item_id].index
                    assert len(new_timestamps.difference(original_timestamps)) == 1
