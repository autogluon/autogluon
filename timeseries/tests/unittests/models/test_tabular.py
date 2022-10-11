import pytest

from autogluon.timeseries.models.tabular import TabularModel

from ..common import DUMMY_VARIABLE_LENGTH_TS_DATAFRAME

TESTABLE_MODELS = [
    TabularModel,
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
    model = TabularModel()
    model._generate_features_from_freq(DUMMY_VARIABLE_LENGTH_TS_DATAFRAME.freq)
    df = model._get_features_dataframe(DUMMY_VARIABLE_LENGTH_TS_DATAFRAME, last_k_values=last_k_values)
    expected_num_features = len(model._lag_indices) + len(model._time_features) + 1
    assert df.shape == (expected_length, expected_num_features)
