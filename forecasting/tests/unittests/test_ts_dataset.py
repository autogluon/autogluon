import pandas as pd
import numpy as np

from autogluon.forecasting.dataset.ts_dataframe import TimeSeriesDataFrame
from gluonts.dataset.common import ListDataset


START_TIMESTAMP = pd.Timestamp("01-01-2019", freq='D')


def get_sample_gluonts_dataset():
    ds_gluonts = ListDataset(
        [
            {'target': [0, 1, 2], 'start': START_TIMESTAMP},
            {'target': [3, 4, 5], 'start': START_TIMESTAMP},
            {'target': [6, 7, 8], 'start': START_TIMESTAMP},
        ],
        freq='D'
    )
    return ds_gluonts


def get_sample_ts_dataframe():
    target = list(range(9))
    datetime_index = tuple(pd.date_range(START_TIMESTAMP, periods=3))
    item_ids = (0, 1, 2)
    multi_inds = pd.MultiIndex.from_product([item_ids, datetime_index], names=['item_id', 'timestamp'])
    ts_df = pd.Series(target, name='target', index=multi_inds).to_frame()
    return TimeSeriesDataFrame(ts_df)


def get_train_test_ds():
    N = 10  # number of time series
    T = 100  # number of timesteps
    prediction_length = 24
    freq = "1H"
    custom_dataset = np.random.normal(size=(N, T))

    train_ds = ListDataset(
        [{'target': x, 'start': START_TIMESTAMP} for x in custom_dataset[:, :-prediction_length]],
        freq=freq
    )
    # test dataset: use the whole dataset, add "target" and "start" fields
    test_ds = ListDataset(
        [{'target': x, 'start': START_TIMESTAMP} for x in custom_dataset],
        freq=freq
    )
    return train_ds, test_ds


def _build_ts_dataframe(item_ids, datetime_index, target):
    multi_inds = pd.MultiIndex.from_product([item_ids, datetime_index], names=['item_id', 'timestamp'])
    return TimeSeriesDataFrame(pd.Series(target, name='target', index=multi_inds).to_frame())


def test_from_gluonts():
    ds = get_sample_gluonts_dataset()
    tsdf_from_gluonts = TimeSeriesDataFrame(ds)
    tsdf_true = get_sample_ts_dataframe()
    pd.testing.assert_frame_equal(tsdf_from_gluonts, tsdf_true, check_dtype=False)


def test_split_by_item():
    sample_tsdf = get_sample_ts_dataframe()
    left, right = sample_tsdf.split_by_item(2)

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
    left, right = sample_tsdf.split_by_item(0)
    assert len(left) == 0
    pd.testing.assert_frame_equal(right, sample_tsdf)


def test_split_by_time():
    tsdf = get_sample_ts_dataframe()
    cutoff = pd.Timestamp("01-03-2019")
    left, right = tsdf.split_by_time(cutoff)

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
    sample_tsdf = get_sample_ts_dataframe()
    end_timestamp = pd.Timestamp("01-02-2019")
    new_tsdf = sample_tsdf.subsequence(START_TIMESTAMP, end_timestamp)

    target = [0, 3, 6]
    datetime_index = tuple(pd.date_range(START_TIMESTAMP, periods=1))
    item_ids = (0, 1, 2)
    ts_df = _build_ts_dataframe(item_ids, datetime_index, target)
    pd.testing.assert_frame_equal(new_tsdf, ts_df)

    new_tsdf = sample_tsdf.subsequence(START_TIMESTAMP, START_TIMESTAMP)
    assert len(new_tsdf) == 0


if __name__ == "__main__":
    test_from_gluonts()
    test_split_by_time()
    test_split_by_item()
    test_subsequence()