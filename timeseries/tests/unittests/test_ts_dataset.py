import copy
import tempfile
from pathlib import Path

import pandas as pd
import numpy as np
import pytest

from gluonts.dataset.common import ListDataset
from autogluon.timeseries.dataset.ts_dataframe import TimeSeriesDataFrame, ITEMID, TIMESTAMP


START_TIMESTAMP = pd.Timestamp("01-01-2019", freq="D")  # type: ignore
END_TIMESTAMP = pd.Timestamp("01-02-2019", freq="D")  # type: ignore
ITEM_IDS = (0, 1, 2)
TARGETS = np.arange(9)
DATETIME_INDEX = tuple(pd.date_range(START_TIMESTAMP, periods=3))
EMPTY_ITEM_IDS = np.array([], dtype=np.int64)
EMPTY_DATETIME_INDEX = np.array([], dtype=np.dtype("datetime64[ns]"))  # type: ignore
EMPTY_TARGETS = np.array([], dtype=np.int64)


def _build_ts_dataframe(item_ids, datetime_index, target):
    multi_inds = pd.MultiIndex.from_product(
        [item_ids, datetime_index], names=["item_id", "timestamp"]
    )
    return TimeSeriesDataFrame(
        pd.Series(target, name="target", index=multi_inds).to_frame()
    )


SAMPLE_TS_DATAFRAME = _build_ts_dataframe(ITEM_IDS, DATETIME_INDEX, TARGETS)
SAMPLE_TS_DATAFRAME_EMPTY = _build_ts_dataframe(EMPTY_ITEM_IDS, EMPTY_DATETIME_INDEX, EMPTY_TARGETS)
SAMPLE_DATAFRAME = pd.DataFrame(SAMPLE_TS_DATAFRAME).reset_index()


SAMPLE_ITERABLE = [
    {"target": [0, 1, 2], "start": pd.Timestamp("01-01-2019", freq="D")},  # type: ignore
    {"target": [3, 4, 5], "start": pd.Timestamp("01-01-2019", freq="D")},  # type: ignore
    {"target": [6, 7, 8], "start": pd.Timestamp("01-01-2019", freq="D")},  # type: ignore
]


def test_from_iterable():
    ts_df = TimeSeriesDataFrame(SAMPLE_ITERABLE)
    pd.testing.assert_frame_equal(ts_df, SAMPLE_TS_DATAFRAME, check_dtype=True)

    with pytest.raises(ValueError):
        TimeSeriesDataFrame([])

    sample_iter = [{"target": [0, 1, 2]}]
    with pytest.raises(ValueError):
        TimeSeriesDataFrame(sample_iter)

    sample_iter = [{"target": [0, 1, 2], "start": pd.Timestamp("01-01-2019")}]  # type: ignore
    with pytest.raises(ValueError):
        TimeSeriesDataFrame(sample_iter)


def test_validate_data_frame():
    item_ids = pd.Series(np.repeat(ITEM_IDS, 3))
    datetimes = pd.Series(np.tile(DATETIME_INDEX, 3))
    targets = pd.Series(TARGETS)
    df = pd.concat([item_ids, datetimes, targets], axis=1)

    with pytest.raises(ValueError):
        TimeSeriesDataFrame(df)

    df.columns = ["item_id", "timestamp", "target"]
    TimeSeriesDataFrame(df)


def test_validate_multi_index_data_frame():
    TimeSeriesDataFrame(SAMPLE_TS_DATAFRAME)

    target = list(range(4))
    item_ids = (1, 2, 3, 4)

    with pytest.raises(ValueError):
        TimeSeriesDataFrame(np.array([item_ids, target]).T, freq="D")

    ts_df = pd.Series(target, name="target", index=item_ids).to_frame()
    with pytest.raises(ValueError):
        TimeSeriesDataFrame(ts_df, freq="D")


def test_from_gluonts_list_dataset():
    number_of_ts = 10  # number of time series
    ts_length = 100  # number of timesteps
    prediction_length = 24
    freq = "D"
    custom_dataset = np.random.normal(size=(number_of_ts, ts_length))
    start = pd.Timestamp("01-01-2019", freq=freq)  # type: ignore

    gluonts_list_dataset = ListDataset(
        [{"target": x, "start": start} for x in custom_dataset[:, :-prediction_length]],
        freq=freq,
    )
    TimeSeriesDataFrame(gluonts_list_dataset)

    ts_df = TimeSeriesDataFrame(ListDataset(SAMPLE_ITERABLE, freq=freq))
    pd.testing.assert_frame_equal(ts_df, SAMPLE_TS_DATAFRAME, check_dtype=False)

    empty_list_dataset = ListDataset([], freq=freq)
    with pytest.raises(ValueError):
        TimeSeriesDataFrame(empty_list_dataset)


def test_from_data_frame():
    tsdf_from_data_frame = TimeSeriesDataFrame(SAMPLE_DATAFRAME)
    pd.testing.assert_frame_equal(
        tsdf_from_data_frame, SAMPLE_TS_DATAFRAME, check_dtype=True
    )


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
            pd.Timestamp("01-03-2019"),  # type: ignore
            ITEM_IDS,
            tuple(pd.date_range(START_TIMESTAMP, periods=2)),
            [0, 1, 3, 4, 6, 7],
            ITEM_IDS,
            tuple(pd.date_range(pd.Timestamp("01-03-2019"), periods=1)),  # type: ignore
            [2, 5, 8],
        ),
        (
            pd.Timestamp("01-01-2019"),  # type: ignore
            EMPTY_ITEM_IDS,
            EMPTY_DATETIME_INDEX,
            EMPTY_TARGETS,
            ITEM_IDS,
            DATETIME_INDEX,
            TARGETS,
        ),
        (
            pd.Timestamp("01-04-2019"),  # type: ignore
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
            pd.Timestamp("12-31-2018"),  # type: ignore
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
            pd.Timestamp("01-04-2019"),  # type: ignore
            pd.Timestamp("01-05-2019"),  # type: ignore
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


@pytest.mark.parametrize(
    "timestamps, expected_freq",
    [
        (["2020-01-01 00:00:00", "2020-01-02 00:00:00", "2020-01-03 00:00:00"], "D"),
        (["2020-01-01 00:00:00", "2020-01-03 00:00:00", "2020-01-05 00:00:00"], "2D"),
        (["2020-01-01 00:00:00", "2020-01-01 00:01:00", "2020-01-01 00:02:00"], "T"),
        (["2020-01-01 00:00:00", "2020-01-01 01:00:00", "2020-01-01 02:00:00"], "H"),
    ],
)
def test_when_dataset_constructed_from_dataframe_without_freq_then_freq_is_inferred(
    timestamps, expected_freq
):
    df = pd.DataFrame(
        {
            "item_id": [0, 0, 0],
            "target": [1, 2, 3],
            "timestamp": map(pd.Timestamp, timestamps),  # noqa
        }
    )

    ts_df = TimeSeriesDataFrame.from_data_frame(df)
    assert ts_df.freq == expected_freq


@pytest.mark.parametrize(
    "start_time, freq",
    [
        ("2020-01-01 00:00:00", "D"),
        ("2020-01-01 00:00:00", "2D"),
        ("2020-01-01 00:00:00", "T"),
        ("2020-01-01 00:00:00", "H"),
    ],
)
def test_when_dataset_constructed_from_iterable_with_freq_then_freq_is_inferred(
    start_time, freq
):
    item_list = ListDataset(
        [{"target": [1, 2, 3], "start": pd.Timestamp(start_time)} for _ in range(3)],  # type: ignore
        freq=freq,
    )

    ts_df = TimeSeriesDataFrame.from_iterable_dataset(item_list)

    assert ts_df.freq == freq


@pytest.mark.parametrize("list_of_timestamps", [
    [
        ["2020-01-01 00:00:00", "2020-01-02 00:00:00", "2020-01-03 00:01:00"],
    ],
    [
        ["2020-01-01 00:00:00", "2020-01-02 00:00:00", "2020-01-03 00:00:00"],
        ["2020-01-01 00:00:00", "2020-01-02 00:00:00", "2020-01-03 00:00:01"],
    ],
    [
        ["2020-01-01 00:00:00", "2020-01-02 00:00:00", "2020-01-03 00:00:00"],
        ["2020-01-01 00:00:00", "2020-01-02 00:00:00", "2020-01-04 00:00:00"],
    ],
    [
        ["2020-01-01 00:00:00", "2020-01-02 00:00:00", "2020-01-03 00:01:00"],
        ["2020-01-01 00:00:00", "2020-01-02 00:00:00", "2020-01-03 00:01:00"],
        ["2020-01-01 00:00:00", "2020-01-02 00:00:00", "2020-01-03 00:01:00"],
    ]
])
def test_when_dataset_constructed_with_irregular_timestamps_then_constructor_raises(
    list_of_timestamps
):
    df_tuples = []
    for i, ts in enumerate(list_of_timestamps):
        for t in ts:
            df_tuples.append((i, pd.Timestamp(t), np.random.rand()))

    df = pd.DataFrame(df_tuples, columns=[ITEMID, TIMESTAMP, "target"])

    with pytest.raises(ValueError, match="uniformly sampled"):
        TimeSeriesDataFrame.from_data_frame(df)


SAMPLE_ITERABLE_2 = [
    {"target": [0, 1, 2, 3], "start": pd.Timestamp("2019-01-01", freq="D")},  # type: ignore
    {"target": [3, 4, 5, 4], "start": pd.Timestamp("2019-01-02", freq="D")},  # type: ignore
    {"target": [6, 7, 8, 5], "start": pd.Timestamp("2019-01-03", freq="D")},  # type: ignore
]


@pytest.mark.parametrize(
    "input_iterable, input_slice, expected_times, expected_values",
    [
        (
            SAMPLE_ITERABLE,
            slice(None, 2),
            [
                "2019-01-01",
                "2019-01-02",
                "2019-01-01",
                "2019-01-02",
                "2019-01-01",
                "2019-01-02",
            ],
            [0, 1, 3, 4, 6, 7],
        ),
        (
            SAMPLE_ITERABLE,
            slice(1, 2),
            ["2019-01-02", "2019-01-02", "2019-01-02"],
            [1, 4, 7],
        ),
        (
            SAMPLE_ITERABLE_2,
            slice(None, 2),
            [
                "2019-01-01",
                "2019-01-02",
                "2019-01-02",
                "2019-01-03",
                "2019-01-03",
                "2019-01-04",
            ],
            [0, 1, 3, 4, 6, 7],
        ),
        (
            SAMPLE_ITERABLE_2,
            slice(-2, None),
            [
                "2019-01-03",
                "2019-01-04",
                "2019-01-04",
                "2019-01-05",
                "2019-01-05",
                "2019-01-06",
            ],
            [2, 3, 5, 4, 8, 5],
        ),
        (
            SAMPLE_ITERABLE_2,
            slice(-1000, 2),
            [
                "2019-01-01",
                "2019-01-02",
                "2019-01-02",
                "2019-01-03",
                "2019-01-03",
                "2019-01-04",
            ],
            [0, 1, 3, 4, 6, 7],
        ),
        (
            SAMPLE_ITERABLE_2,
            slice(1000, 1002),
            [],
            [],
        ),
    ],
)
def test_when_dataset_sliced_by_step_then_output_times_and_values_correct(
    input_iterable, input_slice, expected_times, expected_values
):
    df = TimeSeriesDataFrame.from_iterable_dataset(input_iterable)
    dfv = df.slice_by_timestep(input_slice)

    if not expected_times:
        assert len(dfv) == 0

    assert np.allclose(dfv["target"], expected_values)
    assert isinstance(dfv, TimeSeriesDataFrame)

    assert all(ixval[1] == pd.Timestamp(expected_times[i]) for i, ixval in enumerate(dfv.index.values))  # type: ignore


@pytest.mark.parametrize("input_df", [
    SAMPLE_TS_DATAFRAME, SAMPLE_TS_DATAFRAME_EMPTY
])
def test_when_dataframe_copy_called_on_instance_then_output_correct(input_df):
    copied_df = input_df.copy()

    assert isinstance(copied_df, TimeSeriesDataFrame)
    assert copied_df._data is not input_df._data


@pytest.mark.parametrize("input_df", [
    SAMPLE_TS_DATAFRAME, SAMPLE_TS_DATAFRAME_EMPTY
])
def test_when_dataframe_stdlib_copy_called_then_output_correct(input_df):
    copied_df = copy.deepcopy(input_df)

    assert isinstance(copied_df, TimeSeriesDataFrame)
    assert copied_df._data is not input_df._data


@pytest.mark.parametrize("input_df", [
    SAMPLE_TS_DATAFRAME, SAMPLE_TS_DATAFRAME_EMPTY
])
def test_when_dataframe_class_copy_called_then_output_correct(input_df):
    copied_df = TimeSeriesDataFrame.copy(input_df, deep=True)

    assert isinstance(copied_df, TimeSeriesDataFrame)
    assert copied_df._data is not input_df._data


@pytest.mark.parametrize("input_df", [
    SAMPLE_TS_DATAFRAME, SAMPLE_TS_DATAFRAME_EMPTY
])
@pytest.mark.parametrize("inplace", [True, False])
def test_when_dataframe_class_rename_called_then_output_correct(input_df, inplace):
    renamed_df = TimeSeriesDataFrame.rename(input_df, columns={"target": "mytarget"}, inplace=inplace)
    if inplace:
        renamed_df = input_df

    assert isinstance(renamed_df, TimeSeriesDataFrame)
    assert "mytarget" in renamed_df.columns
    assert "target" not in renamed_df.columns
    if inplace:
        assert renamed_df._data is input_df._data


@pytest.mark.parametrize("input_df", [
    SAMPLE_TS_DATAFRAME, SAMPLE_TS_DATAFRAME_EMPTY
])
@pytest.mark.parametrize("inplace", [True, False])
def test_when_dataframe_instance_rename_called_then_output_correct(input_df, inplace):
    renamed_df = input_df.rename(columns={"target": "mytarget"}, inplace=inplace)
    if inplace:
        renamed_df = input_df

    assert isinstance(renamed_df, TimeSeriesDataFrame)
    assert "mytarget" in renamed_df.columns
    assert "target" not in renamed_df.columns
    if inplace:
        assert renamed_df._data is input_df._data


@pytest.mark.parametrize("input_df", [
    SAMPLE_TS_DATAFRAME, SAMPLE_TS_DATAFRAME_EMPTY
])
@pytest.mark.parametrize("read_fn", [pd.read_pickle, TimeSeriesDataFrame.from_pickle])
def test_when_dataframe_read_pickle_called_then_output_correct(input_df, read_fn):

    with tempfile.TemporaryDirectory() as td:
        pkl_filename = Path(td) / "temp_pickle.pkl"
        input_df.to_pickle(str(pkl_filename))

        read_df = read_fn(pkl_filename)

    assert isinstance(read_df, TimeSeriesDataFrame)
    assert np.allclose(read_df, input_df)
