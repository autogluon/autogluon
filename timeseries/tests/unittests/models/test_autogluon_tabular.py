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
