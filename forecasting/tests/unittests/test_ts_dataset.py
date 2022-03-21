import pandas as pd
import numpy as np

from autogluon.forecasting.dataset.ts_dataframe import TimeSeriesDataFrame
from gluonts.dataset.common import ListDataset


START_TIMESTAMP = pd.Timestamp("01-01-2019", freq='D')


SAMPLE_LIST_DATASET = ListDataset(
    [
        {'target': [0, 1, 2], 'start': START_TIMESTAMP},
        {'target': [3, 4, 5], 'start': START_TIMESTAMP},
        {'target': [6, 7, 8], 'start': START_TIMESTAMP},
    ],
    freq='D'
)


def get_sample_ts_dataframe():
    target = list(range(9))
    datetime_index = tuple(pd.date_range(START_TIMESTAMP, periods=3))
    item_ids = (0, 1, 2)
    multi_inds = pd.MultiIndex.from_product([item_ids, datetime_index], names=['item_id', 'timestamp'])
    ts_df = pd.Series(target, name='target', index=multi_inds).to_frame()
    return TimeSeriesDataFrame(ts_df)


SAMPLE_TS_DATAFRAME = get_sample_ts_dataframe()


def _build_ts_dataframe(item_ids, datetime_index, target):
    multi_inds = pd.MultiIndex.from_product([item_ids, datetime_index], names=['item_id', 'timestamp'])
    return TimeSeriesDataFrame(pd.Series(target, name='target', index=multi_inds).to_frame())


def test_validate():
    target = list(range(4))
    datetime_index = tuple(pd.date_range(START_TIMESTAMP, periods=2))
    item_ids = (3, 4)
    multi_index = pd.MultiIndex.from_product([item_ids, datetime_index], names=['item_id', 'timestamp'])
    ts_df = pd.Series(target, name='target', index=multi_index).to_frame()
    TimeSeriesDataFrame.validate(ts_df)
    TimeSeriesDataFrame(SAMPLE_TS_DATAFRAME)


def test_from_gluonts():
    tsdf_from_gluonts = TimeSeriesDataFrame(SAMPLE_LIST_DATASET)
    pd.testing.assert_frame_equal(tsdf_from_gluonts, SAMPLE_TS_DATAFRAME, check_dtype=False)


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
    item_ids = (2, )
    ts_df = _build_ts_dataframe(item_ids, datetime_index, target)
    pd.testing.assert_frame_equal(right, ts_df)

    # test corner case
    left, right = SAMPLE_TS_DATAFRAME.split_by_item(0)
    assert len(left) == 0
    pd.testing.assert_frame_equal(right, SAMPLE_TS_DATAFRAME)


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


def test_subsequence():
    end_timestamp = pd.Timestamp("01-02-2019")
    new_tsdf = SAMPLE_TS_DATAFRAME.subsequence(START_TIMESTAMP, end_timestamp)

    target = [0, 3, 6]
    datetime_index = tuple(pd.date_range(START_TIMESTAMP, periods=1))
    item_ids = (0, 1, 2)
    ts_df = _build_ts_dataframe(item_ids, datetime_index, target)
    pd.testing.assert_frame_equal(new_tsdf, ts_df)

    new_tsdf = SAMPLE_TS_DATAFRAME.subsequence(START_TIMESTAMP, START_TIMESTAMP)
    assert len(new_tsdf) == 0
