from ast import literal_eval

import pytest

from autogluon.timeseries.splitter import MultiWindowSplitter, append_suffix_to_item_id

from .common import (
    DATAFRAME_WITH_STATIC,
    DUMMY_VARIABLE_LENGTH_TS_DATAFRAME,
    get_data_frame_with_variable_lengths,
    DATAFRAME_WITH_COVARIATES,
)


def get_original_item_id_and_slice(tuning_item_id: str):
    """Extract information from tuning set item_id that has format f"{item_id}_[{start}:{end}]"."""
    item_id, slice_info = tuning_item_id.rsplit("_", maxsplit=1)
    start, end = slice_info.strip("[]").split(":")
    return item_id, literal_eval(start), literal_eval(end)


@pytest.mark.parametrize("item_id_to_length", [{"A": 22, "B": 50, "C": 10}, {"A": 23}])
@pytest.mark.parametrize("prediction_length, num_windows", [(5, 2), (2, 5), (8, 1)])
def test_when_multi_window_splitter_splits_then_train_items_have_correct_length(
    item_id_to_length, prediction_length, num_windows
):
    splitter = MultiWindowSplitter(num_windows=num_windows)
    ts_dataframe = get_data_frame_with_variable_lengths(item_id_to_length=item_id_to_length)
    train_data, _ = splitter.split(ts_dataframe=ts_dataframe, prediction_length=prediction_length)
    original_lengths = ts_dataframe.num_timesteps_per_item()

    for item_id, length in original_lengths.iteritems():
        num_windows_from_this_item = min(num_windows, max((length - 1) // prediction_length - 1, 0))
        expected_length = length - num_windows_from_this_item * prediction_length
        assert expected_length == len(train_data.loc[item_id])


@pytest.mark.parametrize("item_id_to_length", [{"A": 22, "B": 50, "C": 10}, {"A": 23}])
@pytest.mark.parametrize("prediction_length, num_windows", [(5, 2), (2, 5), (8, 1), (10, 2)])
def test_when_multi_window_splitter_splits_then_val_item_ids_correctly_represent_length(
    item_id_to_length, prediction_length, num_windows
):
    splitter = MultiWindowSplitter(num_windows=num_windows)
    ts_dataframe = get_data_frame_with_variable_lengths(item_id_to_length=item_id_to_length)

    _, val_data = splitter.split(ts_dataframe=ts_dataframe, prediction_length=prediction_length)
    for new_item_id in val_data.item_ids:
        old_item_id, start, end = get_original_item_id_and_slice(new_item_id)
        new_length = len(val_data.loc[new_item_id])
        expected_length = len(ts_dataframe.loc[old_item_id][start:end])
        assert expected_length == new_length


def test_when_multi_window_splitter_splits_then_cached_freq_is_preserved():
    splitter = MultiWindowSplitter()
    prediction_length = 10
    train_data, val_data = splitter.split(
        ts_dataframe=DUMMY_VARIABLE_LENGTH_TS_DATAFRAME, prediction_length=prediction_length
    )
    assert DUMMY_VARIABLE_LENGTH_TS_DATAFRAME._cached_freq == train_data._cached_freq == val_data._cached_freq


def test_when_all_series_too_short_then_multi_window_splitter_raises_value_error():
    splitter = MultiWindowSplitter(num_windows=5)
    prediction_length = DUMMY_VARIABLE_LENGTH_TS_DATAFRAME.num_timesteps_per_item().max() // 2 + 1
    with pytest.raises(ValueError, match="all training time series are too short"):
        splitter.split(ts_dataframe=DUMMY_VARIABLE_LENGTH_TS_DATAFRAME, prediction_length=prediction_length)


def test_when_splitter_adds_suffix_to_index_then_data_is_not_copied():
    ts_df = DUMMY_VARIABLE_LENGTH_TS_DATAFRAME.copy()
    ts_df_with_suffix = append_suffix_to_item_id(ts_dataframe=ts_df, suffix="_[None:None]")
    assert ts_df.values.base is ts_df_with_suffix.values.base


def test_when_static_features_are_present_then_splitter_correctly_splits_them():
    original_df = DATAFRAME_WITH_STATIC.copy()
    splitter = MultiWindowSplitter()
    prediction_length = 7
    train_data, val_data = splitter.split(ts_dataframe=original_df, prediction_length=prediction_length)

    for item_id in train_data.item_ids:
        assert (train_data.static_features.loc[item_id] == original_df.static_features.loc[item_id]).all()

    for item_id in val_data.item_ids:
        original_item_id, _, _ = get_original_item_id_and_slice(item_id)
        assert (val_data.static_features.loc[item_id] == original_df.static_features.loc[original_item_id]).all()


def test_when_covariates_are_present_then_splitter_correctly_splits_them():
    original_df = DATAFRAME_WITH_COVARIATES.copy()
    splitter = MultiWindowSplitter()
    prediction_length = 7
    train_data, val_data = splitter.split(ts_dataframe=original_df, prediction_length=prediction_length)

    for column in original_df.drop("target", axis=1).columns:
        for item_id in train_data.item_ids:
            train_series = train_data[column].loc[item_id]
            assert (train_series == original_df[column].loc[item_id][: len(train_series)]).all()

        for item_id in val_data.item_ids:
            original_item_id, _, _ = get_original_item_id_and_slice(item_id)
            val_series = val_data[column].loc[item_id]
            assert (val_series == original_df[column].loc[original_item_id][: len(val_series)]).all()
