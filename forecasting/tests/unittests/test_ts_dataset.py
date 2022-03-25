import copy

import pandas as pd
import numpy as np
import pytest

from autogluon.forecasting.dataset.ts_dataframe import TimeSeriesDataFrame, TimeSeriesListDataset


START_TIMESTAMP = pd.Timestamp("01-01-2019", freq="D")


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
        TimeSeriesDataFrame(np.array([item_ids, target]).T, freq='D')

    ts_df = pd.Series(target, name="target", index=item_ids).to_frame()
    with pytest.raises(ValueError):
        TimeSeriesDataFrame(ts_df, freq='D')


def test_from_ts_list_dataset():
    tsdf_from_list_dataset = TimeSeriesDataFrame(SAMPLE_LIST_DATASET)
    pd.testing.assert_frame_equal(
        tsdf_from_list_dataset, SAMPLE_TS_DATAFRAME, check_dtype=False
    )


def test_validate_list_dataset():
    with pytest.raises(ValueError):
        TimeSeriesDataFrame(SAMPLE_LIST_DATASET.data_iter)

    empty_list = TimeSeriesListDataset(data_iter=[], freq='D')
    with pytest.raises(ValueError):
        TimeSeriesDataFrame(empty_list, freq='D')

    list_dataset = copy.deepcopy(SAMPLE_LIST_DATASET)
    list_dataset.data_iter[1] = {}
    with pytest.raises(ValueError):
        TimeSeriesDataFrame(list_dataset, freq='D')


def test_split_by_item():
    left, right = SAMPLE_TS_DATAFRAME.split_by_item(2)

    # build ground truth for left data frame
    target = [0, 1, 2, 3, 4, 5]
    datetime_index = tuple(pd.date_range(START_TIMESTAMP, periods=3))
    item_ids = (0, 1)
    ts_df = _build_ts_dataframe(item_ids, datetime_index, target)
    pd.testing.assert_frame_equal(left, ts_df)

    # build ground truth for right data frame
    target = [6, 7, 8]
    datetime_index = tuple(pd.date_range(START_TIMESTAMP, periods=3))
    item_ids = (2,)
    ts_df = _build_ts_dataframe(item_ids, datetime_index, target)
    pd.testing.assert_frame_equal(right, ts_df)

    # test corner case
    left, right = SAMPLE_TS_DATAFRAME.split_by_item(0)
    assert len(left) == 0
    pd.testing.assert_frame_equal(right, SAMPLE_TS_DATAFRAME)

    left, right = SAMPLE_TS_DATAFRAME.split_by_item(6)
    assert len(right) == 0
    pd.testing.assert_frame_equal(left, SAMPLE_TS_DATAFRAME)


def test_split_by_time():
    cutoff = pd.Timestamp("01-03-2019")
    left, right = SAMPLE_TS_DATAFRAME.split_by_time(cutoff)

    # build ground truth for left data frame
    target = [0, 1, 3, 4, 6, 7]
    datetime_index = tuple(pd.date_range(START_TIMESTAMP, periods=2))
    item_ids = (0, 1, 2)
    ts_df = _build_ts_dataframe(item_ids, datetime_index, target)
    pd.testing.assert_frame_equal(left, ts_df)

    # build ground truth for right data frame
    target = [2, 5, 8]
    start_timestamp = pd.Timestamp("01-03-2019")
    datetime_index = tuple(pd.date_range(start_timestamp, periods=1))
    item_ids = (0, 1, 2)
    ts_df = _build_ts_dataframe(item_ids, datetime_index, target)
    pd.testing.assert_frame_equal(right, ts_df)

    # test corner case
    left, right = SAMPLE_TS_DATAFRAME.split_by_time(pd.Timestamp("01-01-2019"))
    pd.testing.assert_frame_equal(right, SAMPLE_TS_DATAFRAME)

    left, right = SAMPLE_TS_DATAFRAME.split_by_time(pd.Timestamp("01-04-2019"))
    pd.testing.assert_frame_equal(left, SAMPLE_TS_DATAFRAME)


def test_subsequence():
    end_timestamp = pd.Timestamp("01-02-2019")
    new_tsdf = SAMPLE_TS_DATAFRAME.subsequence(START_TIMESTAMP, end_timestamp)

    target = [0, 3, 6]
    datetime_index = tuple(pd.date_range(START_TIMESTAMP, periods=1))
    item_ids = (0, 1, 2)
    ts_df = _build_ts_dataframe(item_ids, datetime_index, target)
    pd.testing.assert_frame_equal(new_tsdf, ts_df)

    # test corner case
    new_tsdf = SAMPLE_TS_DATAFRAME.subsequence(pd.Timestamp("12-31-2018"), end_timestamp)
    pd.testing.assert_frame_equal(new_tsdf, ts_df)

    new_tsdf = SAMPLE_TS_DATAFRAME.subsequence(START_TIMESTAMP, START_TIMESTAMP)
    assert len(new_tsdf) == 0

    new_tsdf = SAMPLE_TS_DATAFRAME.subsequence(pd.Timestamp("01-04-2019"), pd.Timestamp("01-05-2019"))
    assert len(new_tsdf) == 0
