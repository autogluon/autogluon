import logging
from ast import literal_eval

import pytest

from autogluon.timeseries.dataset.ts_dataframe import ITEMID
from autogluon.timeseries.splitter import LastWindowSplitter, MultiWindowSplitter, append_suffix_to_item_id

from .common import DUMMY_VARIABLE_LENGTH_TS_DATAFRAME, get_data_frame_with_variable_lengths

SPLITTERS = [
    LastWindowSplitter(),
    MultiWindowSplitter(num_windows=2, overlap=0),
    MultiWindowSplitter(num_windows=5, overlap=2),
]


def get_original_item_id_and_slice(tuning_item_id: str):
    """Extract information from tuning set item_id that has format f"{item_id}_[{start}:{end}]"."""
    item_id, slice_info = tuning_item_id.rsplit("_", maxsplit=1)
    start, end = slice_info.strip("[]").split(":")
    return item_id, literal_eval(start), literal_eval(end)


@pytest.mark.parametrize("item_id_to_length", ({"A": 22, "B": 50, "C": 10}, {"A": 23}))
@pytest.mark.parametrize("prediction_length, num_windows, overlap", [(5, 2, 0), (2, 5, 1), (8, 1, 0)])
def test_when_multi_window_splitter_splits_then_train_lengths_are_correct(
    item_id_to_length, prediction_length, num_windows, overlap
):
    splitter = MultiWindowSplitter(num_windows=num_windows, overlap=overlap)
    ts_dataframe = get_data_frame_with_variable_lengths(item_id_to_length=item_id_to_length)
    original_lengths = ts_dataframe.index.get_level_values(0).value_counts(sort=False)

    train_data, val_data = splitter.split(ts_dataframe=ts_dataframe, prediction_length=prediction_length)
    num_total_validation_steps = num_windows * (prediction_length - overlap) + overlap

    for item_id in train_data.iter_items():
        new_length = len(train_data.loc[item_id])
        expected_length = original_lengths.loc[item_id] - num_total_validation_steps
        assert expected_length == new_length


@pytest.mark.parametrize("item_id_to_length", ({"A": 22, "B": 50, "C": 10}, {"A": 23}))
@pytest.mark.parametrize("prediction_length, num_windows, overlap", [(5, 2, 0), (2, 5, 1), (8, 1, 0)])
def test_when_multi_window_splitter_splits_then_val_index_and_lengths_are_correct(
    item_id_to_length, prediction_length, num_windows, overlap
):
    splitter = MultiWindowSplitter(num_windows=num_windows, overlap=overlap)
    ts_dataframe = get_data_frame_with_variable_lengths(item_id_to_length=item_id_to_length)

    train_data, val_data = splitter.split(ts_dataframe=ts_dataframe, prediction_length=prediction_length)

    for new_item_id in val_data.iter_items():
        old_item_id, start, end = get_original_item_id_and_slice(new_item_id)
        new_length = len(val_data.loc[new_item_id])
        expected_length = len(ts_dataframe.loc[old_item_id][start:end])
        assert expected_length == new_length


@pytest.mark.parametrize("splitter", SPLITTERS)
def test_when_some_series_too_short_then_they_disappear_from_train_data(splitter):
    prediction_length = 10
    train_data, val_data = splitter.split(
        ts_dataframe=DUMMY_VARIABLE_LENGTH_TS_DATAFRAME, prediction_length=prediction_length
    )
    remaining_items = train_data.index.get_level_values(ITEMID)

    original_lengths = DUMMY_VARIABLE_LENGTH_TS_DATAFRAME.index.get_level_values(0).value_counts(sort=False)
    num_total_validation_steps = splitter.num_windows * (prediction_length - splitter.overlap) + splitter.overlap
    should_be_missing = original_lengths.index[original_lengths <= num_total_validation_steps]

    for item_id in should_be_missing:
        assert item_id not in remaining_items

    should_be_present = original_lengths.index[original_lengths > num_total_validation_steps]
    for item_id in should_be_present:
        assert item_id in remaining_items


@pytest.mark.parametrize("splitter", SPLITTERS)
def test_when_multi_window_splitter_splits_then_cached_freq_is_preserved(splitter):
    prediction_length = 10
    train_data, val_data = splitter.split(
        ts_dataframe=DUMMY_VARIABLE_LENGTH_TS_DATAFRAME, prediction_length=prediction_length
    )
    assert DUMMY_VARIABLE_LENGTH_TS_DATAFRAME._cached_freq == train_data._cached_freq == val_data._cached_freq


@pytest.mark.parametrize("splitter", SPLITTERS)
def test_when_some_series_too_short_then_warning_is_raised(splitter, caplog):
    with caplog.at_level(logging.WARNING):
        splitter.split(ts_dataframe=DUMMY_VARIABLE_LENGTH_TS_DATAFRAME, prediction_length=10)
        assert "are too short and won't appear in the training set" in caplog.text


def test_when_all_series_too_short_then_sliding_window_splitter_raises_exception():
    splitter = MultiWindowSplitter(num_windows=5, overlap=0)
    with pytest.raises(ValueError):
        splitter.split(ts_dataframe=DUMMY_VARIABLE_LENGTH_TS_DATAFRAME, prediction_length=10)
        pytest.fail(f"{splitter.name} should raise ValueError since the training set is empty")


def test_when_splitter_adds_suffix_to_index_then_data_is_not_copied():
    ts_df = DUMMY_VARIABLE_LENGTH_TS_DATAFRAME.copy()
    ts_df_with_suffix = append_suffix_to_item_id(ts_dataframe=ts_df, suffix="_[None:None]")
    assert ts_df.values.base is ts_df_with_suffix.values.base
