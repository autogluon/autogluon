import copy

import pandas as pd
import numpy as np
import pytest

from autogluon.forecasting.dataset.ts_dataframe import (
    TimeSeriesDataFrame,
    TimeSeriesListDataset,
)


START_TIMESTAMP = pd.Timestamp("01-01-2019", freq="D")
END_TIMESTAMP = pd.Timestamp("01-02-2019")
ITEM_IDS = (0, 1, 2)
TARGETS = list(range(9))
DATETIME_INDEX = tuple(pd.date_range(START_TIMESTAMP, periods=3))
EMPTY_ITEM_IDS = np.array([], dtype=np.int64)
EMPTY_DATETIME_INDEX = np.array([], dtype=np.dtype("datetime64[ns]"))
EMPTY_TARGETS = np.array([], dtype=np.int64)


SAMPLE_LIST_DATASET = TimeSeriesListDataset(
    data_iter=[
        {"target": [0, 1, 2], "start": START_TIMESTAMP},
        {"target": [3, 4, 5], "start": START_TIMESTAMP},
        {"target": [6, 7, 8], "start": START_TIMESTAMP},
    ],
    freq="D",
)


def get_sample_ts_dataframe():
    """Returns a TimeSeriesDataFrame as following:
                            target
    item_id timestamp
    0       2019-01-01       0
            2019-01-02       1
            2019-01-03       2
    1       2019-01-01       3
            2019-01-02       4
            2019-01-03       5
    2       2019-01-01       6
            2019-01-02       7
            2019-01-03       8
    """

    target = list(range(9))
    datetime_index = tuple(pd.date_range(START_TIMESTAMP, periods=3))
    item_ids = (0, 1, 2)
    multi_inds = pd.MultiIndex.from_product(
        [item_ids, datetime_index], names=["item_id", "timestamp"]
    )
    ts_df = pd.Series(target, name="target", index=multi_inds).to_frame()
    return TimeSeriesDataFrame(ts_df)


SAMPLE_TS_DATAFRAME = get_sample_ts_dataframe()


def _build_ts_dataframe(item_ids, datetime_index, target):
    multi_inds = pd.MultiIndex.from_product(
        [item_ids, datetime_index], names=["item_id", "timestamp"]
    )
    return TimeSeriesDataFrame(
        pd.Series(target, name="target", index=multi_inds).to_frame()
    )


def test_validate_data_farme():
    TimeSeriesDataFrame._validate_data_frame(SAMPLE_TS_DATAFRAME)

    target = list(range(4))
    item_ids = (1, 2, 3, 4)

    with pytest.raises(ValueError):
        TimeSeriesDataFrame(np.array([item_ids, target]).T, freq="D")

    ts_df = pd.Series(target, name="target", index=item_ids).to_frame()
    with pytest.raises(ValueError):
        TimeSeriesDataFrame(ts_df, freq="D")


def test_from_ts_list_dataset():
    tsdf_from_list_dataset = TimeSeriesDataFrame(SAMPLE_LIST_DATASET)
    pd.testing.assert_frame_equal(
        tsdf_from_list_dataset, SAMPLE_TS_DATAFRAME, check_dtype=False
    )


def test_validate_list_dataset():
    with pytest.raises(ValueError):
        TimeSeriesDataFrame(SAMPLE_LIST_DATASET.data_iter)

    empty_list = TimeSeriesListDataset(data_iter=[], freq="D")
    with pytest.raises(ValueError):
        TimeSeriesDataFrame(empty_list, freq="D")

    list_dataset = copy.deepcopy(SAMPLE_LIST_DATASET)
    list_dataset.data_iter[1] = {}
    with pytest.raises(ValueError):
        TimeSeriesDataFrame(list_dataset, freq="D")


@pytest.mark.parametrize(
    "split_item_id, left_items, left_datetimes, left_targets, right_items, right_datetimes, right_targets",
    [
        (
            2,
            (0, 1),
            DATETIME_INDEX,
            [0, 1, 2, 3, 4, 5],
            (2,),
            DATETIME_INDEX,
            [6, 7, 8],
        ),
        (
            0,
            EMPTY_ITEM_IDS,
            EMPTY_DATETIME_INDEX,
            EMPTY_TARGETS,
            ITEM_IDS,
            DATETIME_INDEX,
            TARGETS,
        ),
        (
            6,
            ITEM_IDS,
            DATETIME_INDEX,
            TARGETS,
            EMPTY_ITEM_IDS,
            EMPTY_DATETIME_INDEX,
            EMPTY_TARGETS,
        ),
    ],
)
def test_split_by_item(
    split_item_id,
    left_items,
    left_datetimes,
    left_targets,
    right_items,
    right_datetimes,
    right_targets,
):
    left, right = SAMPLE_TS_DATAFRAME.split_by_item(split_item_id)
    left_true = _build_ts_dataframe(left_items, left_datetimes, left_targets)
    right_true = _build_ts_dataframe(right_items, right_datetimes, right_targets)
    pd.testing.assert_frame_equal(left, left_true)
    pd.testing.assert_frame_equal(right, right_true)


@pytest.mark.parametrize(
    "split_time_stamp, left_items, left_datetimes, left_targets, right_items, right_datetimes, right_targets",
    [
        (
            pd.Timestamp("01-03-2019"),
            ITEM_IDS,
            tuple(pd.date_range(START_TIMESTAMP, periods=2)),
            [0, 1, 3, 4, 6, 7],
            ITEM_IDS,
            tuple(pd.date_range(pd.Timestamp("01-03-2019"), periods=1)),
            [2, 5, 8],
        ),
        (
            pd.Timestamp("01-01-2019"),
            EMPTY_ITEM_IDS,
            EMPTY_DATETIME_INDEX,
            EMPTY_TARGETS,
            ITEM_IDS,
            DATETIME_INDEX,
            TARGETS,
        ),
        (
            pd.Timestamp("01-04-2019"),
            ITEM_IDS,
            DATETIME_INDEX,
            TARGETS,
            EMPTY_ITEM_IDS,
            EMPTY_DATETIME_INDEX,
            EMPTY_TARGETS,
        ),
    ],
)
def test_split_by_time(
    split_time_stamp,
    left_items,
    left_datetimes,
    left_targets,
    right_items,
    right_datetimes,
    right_targets,
):
    left, right = SAMPLE_TS_DATAFRAME.split_by_time(split_time_stamp)
    left_true = _build_ts_dataframe(left_items, left_datetimes, left_targets)
    right_true = _build_ts_dataframe(right_items, right_datetimes, right_targets)
    pd.testing.assert_frame_equal(left, left_true)
    pd.testing.assert_frame_equal(right, right_true)


@pytest.mark.parametrize(
    "start_timestamp, end_timestamp, item_ids, datetimes, targets",
    [
        (
            START_TIMESTAMP,
            END_TIMESTAMP,
            ITEM_IDS,
            tuple(pd.date_range(START_TIMESTAMP, periods=1)),
            [0, 3, 6],
        ),
        (
            pd.Timestamp("12-31-2018"),
            END_TIMESTAMP,
            ITEM_IDS,
            tuple(pd.date_range(START_TIMESTAMP, periods=1)),
            [0, 3, 6],
        ),
        (
            START_TIMESTAMP,
            START_TIMESTAMP,
            EMPTY_ITEM_IDS,
            EMPTY_DATETIME_INDEX,
            EMPTY_TARGETS,
        ),
        (
            pd.Timestamp("01-04-2019"),
            pd.Timestamp("01-05-2019"),
            EMPTY_ITEM_IDS,
            EMPTY_DATETIME_INDEX,
            EMPTY_TARGETS,
        ),
    ],
)
def test_subsequence(start_timestamp, end_timestamp, item_ids, datetimes, targets):
    new_tsdf = SAMPLE_TS_DATAFRAME.subsequence(start_timestamp, end_timestamp)
    ts_df = _build_ts_dataframe(item_ids, datetimes, targets)
    pd.testing.assert_frame_equal(new_tsdf, ts_df)
