import copy
import datetime
import tempfile
import traceback
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pytest
from gluonts.dataset.common import ListDataset

from autogluon.timeseries.dataset.ts_dataframe import (
    IRREGULAR_TIME_INDEX_FREQSTR,
    ITEMID,
    TIMESTAMP,
    TimeSeriesDataFrame,
)

from .common import get_data_frame_with_variable_lengths, to_supported_pandas_freq

START_TIMESTAMP = pd.Timestamp("01-01-2019")  # type: ignore
END_TIMESTAMP = pd.Timestamp("01-02-2019")  # type: ignore
ITEM_IDS = np.array([0, 1, 2], dtype=int)
TARGETS = np.arange(9, dtype=np.float64)
DATETIME_INDEX = tuple(pd.date_range(START_TIMESTAMP, periods=3, freq="D"))
EMPTY_ITEM_IDS = np.array([], dtype=int)
EMPTY_DATETIME_INDEX = np.array([], dtype=np.dtype("datetime64[ns]"))  # type: ignore
EMPTY_TARGETS = np.array([], dtype=np.float64)


def _build_ts_dataframe(item_ids, datetime_index, target, static_features=None):
    multi_inds = pd.MultiIndex.from_product([item_ids, datetime_index], names=["item_id", "timestamp"])
    return TimeSeriesDataFrame(
        pd.Series(target, name="target", index=multi_inds).to_frame(),
        static_features=static_features,
    )


SAMPLE_TS_DATAFRAME = _build_ts_dataframe(ITEM_IDS, DATETIME_INDEX, TARGETS)
SAMPLE_TS_DATAFRAME_EMPTY = _build_ts_dataframe(EMPTY_ITEM_IDS, EMPTY_DATETIME_INDEX, EMPTY_TARGETS)
SAMPLE_TS_DATAFRAME_STATIC = _build_ts_dataframe(
    item_ids=ITEM_IDS,
    datetime_index=DATETIME_INDEX,
    target=TARGETS,
    static_features=pd.DataFrame(
        {
            "feat1": np.random.choice(["A", "B", "C"], size=len(ITEM_IDS)),
            "feat2": np.random.rand(len(ITEM_IDS)),
        },
        index=ITEM_IDS,
    ),
)
SAMPLE_DATAFRAME = pd.DataFrame(SAMPLE_TS_DATAFRAME).reset_index()
SAMPLE_STATIC_DATAFRAME = SAMPLE_TS_DATAFRAME_STATIC.static_features.reset_index()


SAMPLE_ITERABLE = [
    {"target": [0.0, 1.0, 2.0], "start": pd.Period("01-01-2019", freq="D")},  # type: ignore
    {"target": [3.0, 4.0, 5.0], "start": pd.Period("01-01-2019", freq="D")},  # type: ignore
    {"target": [6.0, 7.0, 8.0], "start": pd.Period("01-01-2019", freq="D")},  # type: ignore
]


def test_from_iterable():
    ts_df = TimeSeriesDataFrame(SAMPLE_ITERABLE)
    pd.testing.assert_frame_equal(ts_df, SAMPLE_TS_DATAFRAME, check_dtype=True, check_index_type=False)

    with pytest.raises(ValueError):
        TimeSeriesDataFrame([])

    sample_iter = [{"target": [0, 1, 2]}]
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
    start = pd.Period("01-01-2019", freq=freq)  # type: ignore

    gluonts_list_dataset = ListDataset(
        [{"target": x, "start": start} for x in custom_dataset[:, :-prediction_length]],
        freq=freq,
    )
    TimeSeriesDataFrame(gluonts_list_dataset)

    ts_df = TimeSeriesDataFrame(ListDataset(SAMPLE_ITERABLE, freq=freq))
    pd.testing.assert_frame_equal(ts_df, SAMPLE_TS_DATAFRAME, check_dtype=False, check_index_type=False)

    empty_list_dataset = ListDataset([], freq=freq)
    with pytest.raises(ValueError):
        TimeSeriesDataFrame(empty_list_dataset)


def test_from_data_frame():
    tsdf_from_data_frame = TimeSeriesDataFrame(SAMPLE_DATAFRAME)
    pd.testing.assert_frame_equal(tsdf_from_data_frame, SAMPLE_TS_DATAFRAME, check_dtype=True)


@pytest.mark.parametrize(
    "split_time_stamp, left_items, left_datetimes, left_targets, right_items, right_datetimes, right_targets",
    [
        (
            pd.Timestamp("01-03-2019"),  # type: ignore
            ITEM_IDS,
            tuple(pd.date_range(START_TIMESTAMP, periods=2)),
            [0.0, 1.0, 3.0, 4.0, 6.0, 7.0],
            ITEM_IDS,
            tuple(pd.date_range(pd.Timestamp("01-03-2019"), periods=1)),  # type: ignore
            [2.0, 5.0, 8.0],
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
            [0.0, 3.0, 6.0],
        ),
        (
            pd.Timestamp("12-31-2018"),  # type: ignore
            END_TIMESTAMP,
            ITEM_IDS,
            tuple(pd.date_range(START_TIMESTAMP, periods=1)),
            [0.0, 3.0, 6.0],
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
def test_slice_by_time(start_timestamp, end_timestamp, item_ids, datetimes, targets):
    new_tsdf = SAMPLE_TS_DATAFRAME.slice_by_time(start_timestamp, end_timestamp)
    ts_df = _build_ts_dataframe(item_ids, datetimes, targets)
    pd.testing.assert_frame_equal(new_tsdf, ts_df)


@pytest.mark.parametrize(
    "timestamps, expected_freq",
    [
        (["2020-01-01 00:00:00", "2020-01-02 00:00:00", "2020-01-03 00:00:00"], "D"),
        (["2020-01-01 00:00:00", "2020-01-03 00:00:00", "2020-01-05 00:00:00"], "2D"),
        (["2020-01-01 00:00:00", "2020-01-01 00:01:00", "2020-01-01 00:02:00"], "min"),
        (["2020-01-01 00:00:00", "2020-01-01 01:00:00", "2020-01-01 02:00:00"], "h"),
    ],
)
def test_when_dataset_constructed_from_dataframe_without_freq_then_freq_is_inferred(timestamps, expected_freq):
    df = pd.DataFrame(
        {
            "item_id": [0, 0, 0],
            "target": [1, 2, 3],
            "timestamp": map(pd.Timestamp, timestamps),  # noqa
        }
    )

    ts_df = TimeSeriesDataFrame.from_data_frame(df)
    assert ts_df.freq == to_supported_pandas_freq(expected_freq)


FREQ_TEST_CASES = [
    ("2020-01-01 00:00:00", "D"),
    ("2020-01-01", "D"),
    ("2020-01-01 00:00:00", "2D"),
    ("2020-01-01 00:00:00", "min"),
    ("2020-01-01 00:00:00", "h"),
    ("2020-01-31 00:00:00", "ME"),
    ("2020-01-31", "ME"),
]


@pytest.mark.parametrize("start_time, freq", FREQ_TEST_CASES)
def test_when_dataset_constructed_from_iterable_with_freq_then_freq_is_inferred(start_time, freq):
    freq = to_supported_pandas_freq(freq)
    item_list = ListDataset(
        [{"target": [1, 2, 3], "start": pd.Timestamp(start_time)} for _ in range(3)],  # type: ignore
        freq=freq,
    )

    ts_df = TimeSeriesDataFrame.from_iterable_dataset(item_list)

    assert ts_df.freq == freq


@pytest.mark.parametrize("start_time, freq", FREQ_TEST_CASES)
def test_when_dataset_constructed_via_constructor_with_freq_then_freq_is_inferred(start_time, freq):
    freq = to_supported_pandas_freq(freq)
    # Period requires freq=M for ME frequency
    start_period = pd.Period(start_time, freq={"ME": "M"}.get(freq))
    item_list = ListDataset(
        [{"target": [1, 2, 3], "start": start_period} for _ in range(3)],  # type: ignore
        freq=freq,
    )

    ts_df = TimeSeriesDataFrame(item_list)

    assert ts_df.freq == freq


IRREGULAR_TIME_INDEXES = [
    [
        ["2020-01-01 00:00:00", "2020-01-01 00:01:00"],
    ],
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
        ["2020-01-01 00:00:00", "2020-01-02 00:00:00", "2020-01-03 00:00:00"],
        ["2020-01-01 00:00:00", "2020-02-01 00:00:00", "2020-03-01 00:00:00"],
    ],
    [
        ["2020-01-01 00:00:00", "2020-01-02 00:00:00", "2020-01-03 00:01:00"],
        ["2020-01-01 00:00:00", "2020-01-02 00:00:00", "2020-01-03 00:01:00"],
        ["2020-01-01 00:00:00", "2020-01-02 00:00:00", "2020-01-03 00:01:00"],
    ],
]


@pytest.mark.parametrize("irregular_index", IRREGULAR_TIME_INDEXES)
def test_when_dataset_constructed_with_irregular_timestamps_then_freq_call_returns_none(
    irregular_index,
):
    df_tuples = []
    for i, ts in enumerate(irregular_index):
        for t in ts:
            df_tuples.append((i, pd.Timestamp(t), np.random.rand()))

    df = pd.DataFrame(df_tuples, columns=[ITEMID, TIMESTAMP, "target"])

    tsdf = TimeSeriesDataFrame.from_data_frame(df)
    assert tsdf.freq is None


@pytest.mark.parametrize("irregular_index", IRREGULAR_TIME_INDEXES)
def test_when_dataset_constructed_with_irregular_timestamps_then_irregular_freqstr_is_inferred(
    irregular_index,
):
    df_tuples = []
    for i, ts in enumerate(irregular_index):
        for t in ts:
            df_tuples.append((i, pd.Timestamp(t), np.random.rand()))

    df = pd.DataFrame(df_tuples, columns=[ITEMID, TIMESTAMP, "target"])

    tsdf = TimeSeriesDataFrame.from_data_frame(df)
    assert tsdf.infer_frequency() == IRREGULAR_TIME_INDEX_FREQSTR


@pytest.mark.parametrize("irregular_index", IRREGULAR_TIME_INDEXES)
def test_given_raise_if_irregular_is_true_when_frequency_inferred_then_error_is_raised(irregular_index):
    df_tuples = []
    for i, ts in enumerate(irregular_index):
        for t in ts:
            df_tuples.append((i, pd.Timestamp(t), np.random.rand()))

    df = pd.DataFrame(df_tuples, columns=[ITEMID, TIMESTAMP, "target"])

    tsdf = TimeSeriesDataFrame.from_data_frame(df)
    with pytest.raises(ValueError, match="Cannot infer frequency"):
        tsdf.infer_frequency(raise_if_irregular=True)


SAMPLE_ITERABLE_2 = [
    {"target": [0, 1, 2, 3], "start": pd.Period("2019-01-01", freq="D")},  # type: ignore
    {"target": [3, 4, 5, 4], "start": pd.Period("2019-01-02", freq="D")},  # type: ignore
    {"target": [6, 7, 8, 5], "start": pd.Period("2019-01-03", freq="D")},  # type: ignore
]


@pytest.mark.parametrize(
    "input_iterable, start_index, end_index, expected_times, expected_values",
    [
        (
            SAMPLE_ITERABLE,
            None,
            2,
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
            1,
            2,
            ["2019-01-02", "2019-01-02", "2019-01-02"],
            [1, 4, 7],
        ),
        (
            SAMPLE_ITERABLE_2,
            None,
            2,
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
            -2,
            None,
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
            -1000,
            2,
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
            1000,
            1002,
            [],
            [],
        ),
    ],
)
def test_when_dataset_sliced_by_step_then_output_times_and_values_correct(
    input_iterable, start_index, end_index, expected_times, expected_values
):
    df = TimeSeriesDataFrame.from_iterable_dataset(input_iterable)
    dfv = df.slice_by_timestep(start_index, end_index)

    if not expected_times:
        assert len(dfv) == 0

    assert np.allclose(dfv["target"], expected_values)
    assert isinstance(dfv, TimeSeriesDataFrame)

    assert all(ixval[1] == pd.Timestamp(expected_times[i]) for i, ixval in enumerate(dfv.index.values))  # type: ignore


@pytest.mark.parametrize(
    "start_index, end_index",
    [
        # None cases
        (None, None),
        (None, 2),
        (None, -1),
        (1, None),
        (4, None),
        (-2, None),
        # Positive indices
        (0, 1),
        (1, 3),
        (2, 4),
        # Negative indices
        (-3, -1),
        (-1, None),
        # Mixed positive/negative
        (1, -1),
        (-3, 5),
        # Out of bounds cases
        (10, None),
        (1, 100),
        (-100, 2),
        (None, -100),
        (4, 20),
        (4, -1),
        # Edge cases
        (0, 0),
        (3, 1),
        (2, 2),
        (-2, 2),
    ],
)
def test_when_slice_by_timestep_used_with_different_inputs_then_output_selects_correct_indices(start_index, end_index):
    df = get_data_frame_with_variable_lengths({"A": 5, "B": 3, "C": 4})
    result = df.slice_by_timestep(start_index, end_index)
    expected = df.groupby(ITEMID).nth(slice(start_index, end_index))
    pd.testing.assert_frame_equal(result, expected)


@pytest.mark.parametrize("input_df", [SAMPLE_TS_DATAFRAME, SAMPLE_TS_DATAFRAME_EMPTY])
def test_when_dataframe_copy_called_on_instance_then_output_correct(input_df):
    copied_df = input_df.copy()

    assert isinstance(copied_df, TimeSeriesDataFrame)
    assert copied_df._mgr is not input_df._mgr


@pytest.mark.parametrize("input_df", [SAMPLE_TS_DATAFRAME, SAMPLE_TS_DATAFRAME_EMPTY])
def test_when_dataframe_stdlib_copy_called_then_output_correct(input_df):
    copied_df = copy.deepcopy(input_df)

    assert isinstance(copied_df, TimeSeriesDataFrame)
    assert copied_df._mgr is not input_df._mgr


@pytest.mark.parametrize("input_df", [SAMPLE_TS_DATAFRAME, SAMPLE_TS_DATAFRAME_EMPTY])
def test_when_dataframe_class_copy_called_then_output_correct(input_df):
    copied_df = TimeSeriesDataFrame.copy(input_df, deep=True)

    assert isinstance(copied_df, TimeSeriesDataFrame)
    assert copied_df._mgr is not input_df._mgr


@pytest.mark.parametrize("input_df", [SAMPLE_TS_DATAFRAME, SAMPLE_TS_DATAFRAME_EMPTY])
@pytest.mark.parametrize("inplace", [True, False])
def test_when_dataframe_class_rename_called_then_output_correct(input_df, inplace):
    renamed_df = TimeSeriesDataFrame.rename(input_df, columns={"target": "mytarget"}, inplace=inplace)
    if inplace:
        renamed_df = input_df

    assert isinstance(renamed_df, TimeSeriesDataFrame)
    assert "mytarget" in renamed_df.columns
    assert "target" not in renamed_df.columns
    if inplace:
        assert renamed_df._mgr is input_df._mgr


@pytest.mark.parametrize("input_df", [SAMPLE_TS_DATAFRAME, SAMPLE_TS_DATAFRAME_EMPTY])
@pytest.mark.parametrize("inplace", [True, False])
def test_when_dataframe_instance_rename_called_then_output_correct(input_df, inplace):
    renamed_df = input_df.rename(columns={"target": "mytarget"}, inplace=inplace)
    if inplace:
        renamed_df = input_df

    assert isinstance(renamed_df, TimeSeriesDataFrame)
    assert "mytarget" in renamed_df.columns
    assert "target" not in renamed_df.columns
    if inplace:
        assert renamed_df._mgr is input_df._mgr


@pytest.mark.parametrize("input_df", [SAMPLE_TS_DATAFRAME, SAMPLE_TS_DATAFRAME_EMPTY])
@pytest.mark.parametrize("read_fn", [pd.read_pickle, TimeSeriesDataFrame.from_pickle])
def test_when_dataframe_read_pickle_called_then_output_correct(input_df, read_fn):
    with tempfile.TemporaryDirectory() as td:
        pkl_filename = Path(td) / "temp_pickle.pkl"
        input_df.to_pickle(str(pkl_filename))

        read_df = read_fn(pkl_filename)

    assert isinstance(read_df, TimeSeriesDataFrame)
    assert np.allclose(read_df, input_df)
    assert read_df.static_features is None


@pytest.mark.parametrize("read_fn", [pd.read_pickle, TimeSeriesDataFrame.from_pickle])
def test_when_dataframe_read_pickle_called_then_static_features_are_correct(read_fn):
    input_df = SAMPLE_TS_DATAFRAME_STATIC

    with tempfile.TemporaryDirectory() as td:
        pkl_filename = Path(td) / "temp_pickle.pkl"
        input_df.to_pickle(str(pkl_filename))

        read_df = read_fn(pkl_filename)

    assert isinstance(read_df, TimeSeriesDataFrame)
    assert np.allclose(read_df, input_df)
    assert read_df.static_features.equals(input_df.static_features)


def test_when_dataframe_copy_called_on_instance_then_static_features_are_correct():
    input_df = SAMPLE_TS_DATAFRAME_STATIC
    copied_df = input_df.copy()

    assert input_df.static_features.equals(copied_df.static_features)
    assert input_df.static_features is not copied_df.static_features


def test_when_dataframe_stdlib_copy_called_then_static_features_are_correct():
    input_df = SAMPLE_TS_DATAFRAME_STATIC
    copied_df = copy.deepcopy(input_df)

    assert input_df.static_features.equals(copied_df.static_features)
    assert copied_df._mgr is not input_df._mgr


@pytest.mark.parametrize("inplace", [True, False])
def test_when_dataframe_class_rename_called_then_static_features_are_correct(inplace):
    input_df = SAMPLE_TS_DATAFRAME_STATIC
    renamed_df = TimeSeriesDataFrame.rename(input_df, columns={"target": "mytarget"}, inplace=inplace)
    if inplace:
        renamed_df = input_df

    assert isinstance(renamed_df, TimeSeriesDataFrame)
    assert "mytarget" in renamed_df.columns
    assert "target" not in renamed_df.columns
    if inplace:
        assert renamed_df._mgr is input_df._mgr
    assert renamed_df.static_features.equals(input_df.static_features)


@pytest.mark.parametrize("inplace", [True, False])
def test_when_dataframe_instance_rename_called_then_static_features_are_correct(
    inplace,
):
    input_df = SAMPLE_TS_DATAFRAME_STATIC
    renamed_df = input_df.rename(columns={"target": "mytarget"}, inplace=inplace)
    if inplace:
        renamed_df = input_df

    assert isinstance(renamed_df, TimeSeriesDataFrame)
    assert "mytarget" in renamed_df.columns
    assert "target" not in renamed_df.columns
    if inplace:
        assert renamed_df._mgr is input_df._mgr
    assert renamed_df.static_features.equals(input_df.static_features)


def test_when_dataset_sliced_by_step_then_static_features_are_correct():
    df = SAMPLE_TS_DATAFRAME_STATIC
    dfv = df.slice_by_timestep(-2, None)

    assert isinstance(dfv, TimeSeriesDataFrame)
    assert len(dfv) == 2 * len(dfv.item_ids)

    assert dfv.static_features.equals(df.static_features)


def test_when_static_features_index_has_wrong_name_then_its_renamed_to_item_id():
    original_df = SAMPLE_TS_DATAFRAME.copy()
    item_ids = original_df.item_ids
    static_features = pd.DataFrame({"feat1": np.zeros_like(item_ids)}, index=item_ids.rename("wrong_index_name"))
    original_df.static_features = static_features
    assert static_features.index.name != ITEMID
    assert original_df.static_features.index.name == ITEMID


def test_when_dataset_sliced_by_time_then_static_features_are_correct():
    df = SAMPLE_TS_DATAFRAME_STATIC
    dfv = df.slice_by_time(START_TIMESTAMP, START_TIMESTAMP + datetime.timedelta(days=1))

    assert isinstance(dfv, TimeSeriesDataFrame)
    assert len(dfv) == 1 * len(dfv.item_ids)

    assert dfv.static_features.equals(df.static_features)


def test_when_dataset_split_by_time_then_static_features_are_correct():
    left, right = SAMPLE_TS_DATAFRAME_STATIC.split_by_time(START_TIMESTAMP + datetime.timedelta(days=1))

    assert len(left) == 1 * len(SAMPLE_TS_DATAFRAME_STATIC.item_ids)
    assert len(right) == 2 * len(SAMPLE_TS_DATAFRAME_STATIC.item_ids)

    assert left.static_features.equals(SAMPLE_TS_DATAFRAME_STATIC.static_features)
    assert right.static_features.equals(SAMPLE_TS_DATAFRAME_STATIC.static_features)


@pytest.mark.parametrize("static_feature_index", [ITEM_IDS])
def test_given_correct_static_feature_index_when_constructing_data_frame_then_error_not_raised(
    static_feature_index: Iterable[Any],
):
    static_features = pd.DataFrame(
        {
            "feat1": np.random.choice(["A", "B", "C"], size=len(static_feature_index)),  # noqa
            "feat2": np.random.rand(len(static_feature_index)),  # noqa
        },
        index=static_feature_index,  # noqa
    )
    try:
        TimeSeriesDataFrame(data=SAMPLE_DATAFRAME, static_features=static_features)
    except Exception as e:  # noqa
        pytest.fail(f"Exception raised: {str(e)} \n Traceback:\n {traceback.format_exc()}")


@pytest.mark.parametrize(
    "static_feature_index",
    [
        (1, 2, 3, 4),
        (1, 2),
        (6, 7),
        (),
        ("A", "B"),
    ],
)
def test_given_wrong_static_feature_index_when_constructing_data_frame_then_error_raised(
    static_feature_index,
):
    static_features = pd.DataFrame(
        {
            "feat1": np.random.choice(["A", "B", "C"], size=len(static_feature_index)),  # noqa
            "feat2": np.random.rand(len(static_feature_index)),  # noqa
        },
        index=static_feature_index,  # noqa
    )
    with pytest.raises(ValueError, match="are missing from the index of static_features"):
        TimeSeriesDataFrame(data=SAMPLE_DATAFRAME, static_features=static_features)


@pytest.mark.parametrize(
    "left_index, right_index",
    [
        ([0, 1], [2]),
        ([0], [1, 2]),
        ([], [0, 1, 2]),
    ],
)
def test_when_dataframe_sliced_by_item_array_then_static_features_stay_consistent(left_index, right_index):
    df = SAMPLE_TS_DATAFRAME_STATIC
    left, right = df.loc[left_index], df.loc[right_index]

    assert set(left.static_features.index) == set(left_index)
    assert set(right.static_features.index) == set(right_index)


def test_given_wrong_ids_stored_in_item_id_column_when_constructing_tsdf_then_exception_is_raised():
    df = SAMPLE_DATAFRAME.copy()
    static_df = SAMPLE_STATIC_DATAFRAME.copy()
    static_df.index = ["B", "C", "A"]
    static_df["item_id"] = ["B", "C", "A"]
    with pytest.raises(ValueError, match="ids are missing from the index"):
        TimeSeriesDataFrame(df, static_features=static_df)


def test_given_static_features_have_multiindex_when_constructing_tsdf_then_exception_is_raised():
    df = SAMPLE_DATAFRAME.copy()
    static_df = SAMPLE_STATIC_DATAFRAME.copy()
    static_df.index = pd.MultiIndex.from_arrays([ITEM_IDS, ["B", "C", "A"]], names=["item_id", "extra_level"])
    with pytest.raises(ValueError, match="cannot have a MultiIndex"):
        TimeSeriesDataFrame(df, static_features=static_df)


def test_given_item_id_is_stored_as_column_and_not_index_in_static_features_then_tsdf_is_constructed_correctly():
    df = SAMPLE_DATAFRAME.copy()
    static_df = SAMPLE_STATIC_DATAFRAME.copy()
    static_df.index = ["B", "C", "A"]
    static_df["item_id"] = ITEM_IDS
    ts_df = TimeSeriesDataFrame(df, static_features=static_df)
    assert ts_df.static_features.equals(SAMPLE_TS_DATAFRAME_STATIC.static_features)


def test_given_item_id_stored_in_both_index_and_column_when_constructing_tsdf_then_values_in_index_are_used():
    df = SAMPLE_DATAFRAME.copy()
    static_df = SAMPLE_STATIC_DATAFRAME.copy()
    static_df = static_df.set_index(ITEMID)
    static_df[ITEMID] = ["B", "C", "A"]  # these shouldn't be used; static_df.index should be used instead
    ts_df = TimeSeriesDataFrame(df)
    ts_df.static_features = static_df
    assert (ts_df.static_features[ITEMID] == ["B", "C", "A"]).all()


SAMPLE_DATAFRAME_WITH_MIXED_INDEX = pd.DataFrame(
    [
        {ITEMID: 2, TIMESTAMP: pd.Timestamp("2020-01-01"), "target": 2.5},
        {ITEMID: 2, TIMESTAMP: pd.Timestamp("2020-01-02"), "target": 3.5},
        {ITEMID: "a", TIMESTAMP: pd.Timestamp("2020-01-01"), "target": 2.5},
        {ITEMID: "a", TIMESTAMP: pd.Timestamp("2020-01-02"), "target": 3.5},
    ]
)


@pytest.mark.parametrize(
    "input_df",
    [
        SAMPLE_DATAFRAME_WITH_MIXED_INDEX,
        SAMPLE_DATAFRAME_WITH_MIXED_INDEX.set_index([ITEMID, TIMESTAMP]),
    ],
)
def test_when_item_id_index_has_mixed_dtype_then_value_error_is_raised(input_df):
    with pytest.raises(ValueError, match="must be of integer or string dtype"):
        TimeSeriesDataFrame(input_df)


def test_when_static_features_are_modified_on_shallow_copy_then_original_df_doesnt_change():
    old_df = SAMPLE_TS_DATAFRAME_STATIC
    new_df = old_df.copy(deep=False)
    new_df.static_features = None
    assert old_df.static_features is not None


@pytest.mark.parametrize("timestamp_column", ["timestamp", None, "custom_ts_column"])
def test_when_dataset_constructed_from_dataframe_then_timestamp_column_is_converted_to_datetime(timestamp_column):
    timestamps = ["2020-01-01", "2020-01-02", "2020-01-03"]
    df = pd.DataFrame(
        {
            "item_id": np.ones(len(timestamps), dtype=np.int64),
            timestamp_column or "timestamp": timestamps,
            "target": np.ones(len(timestamps)),
        }
    )
    ts_df = TimeSeriesDataFrame.from_data_frame(df, timestamp_column=timestamp_column)
    assert ts_df.index.get_level_values(level=TIMESTAMP).dtype == "datetime64[ns]"


def test_when_path_is_given_to_constructor_then_tsdf_is_constructed_correctly():
    df = SAMPLE_TS_DATAFRAME.reset_index()
    with TemporaryDirectory() as temp_dir:
        temp_file = str(Path(temp_dir) / "temp.csv")
        df.to_csv(temp_file)

        ts_df = TimeSeriesDataFrame(temp_file)
        assert isinstance(ts_df.index, pd.MultiIndex)
        assert ts_df.index.names == [ITEMID, TIMESTAMP]
        assert len(ts_df) == len(SAMPLE_TS_DATAFRAME)


def test_given_custom_id_column_when_data_and_static_are_loaded_from_path_them_tsdf_is_constructed_correctly():
    df = pd.DataFrame(SAMPLE_TS_DATAFRAME_STATIC).reset_index()
    static_df = SAMPLE_TS_DATAFRAME_STATIC.static_features.reset_index()

    df = df.rename(columns={ITEMID: "custom_item_id"})
    static_df = static_df.rename(columns={ITEMID: "custom_item_id"})

    with TemporaryDirectory() as temp_dir:
        temp_file = Path(temp_dir) / "data.csv"
        df.to_csv(temp_file, index=False)

        temp_static_file = Path(temp_dir) / "static.csv"
        static_df.to_csv(temp_static_file, index=False)

        ts_df = TimeSeriesDataFrame.from_path(
            temp_file, id_column="custom_item_id", static_features_path=temp_static_file
        )
    assert isinstance(ts_df, TimeSeriesDataFrame)
    assert isinstance(ts_df.index, pd.MultiIndex)
    assert ts_df.index.names == [ITEMID, TIMESTAMP]
    assert len(ts_df) == len(SAMPLE_TS_DATAFRAME_STATIC)

    assert ts_df.static_features.index.equals(SAMPLE_TS_DATAFRAME_STATIC.static_features.index)
    assert ts_df.static_features.columns.equals(SAMPLE_TS_DATAFRAME_STATIC.static_features.columns)


def test_given_static_features_are_missing_when_loading_from_path_then_tsdf_can_be_constructed():
    df = SAMPLE_DATAFRAME.copy()
    with TemporaryDirectory() as temp_dir:
        temp_file = Path(temp_dir) / "data.csv"
        df.to_csv(temp_file, index=False)

        ts_df = TimeSeriesDataFrame.from_path(temp_file, id_column=ITEMID, static_features_path=None)
    assert isinstance(ts_df, TimeSeriesDataFrame)
    assert ts_df.static_features is None


FILL_METHODS = ["auto", "ffill", "pad", "backfill", "bfill", "interpolate", "constant"]


@pytest.mark.parametrize("method", FILL_METHODS)
def test_when_fill_missing_values_called_then_gaps_are_filled_and_index_is_unchanged(method):
    df = get_data_frame_with_variable_lengths({"B": 15, "A": 20})
    df.iloc[[1, 5, 10, 22]] = np.nan
    df_filled = df.fill_missing_values(method=method)
    assert not df_filled.isna().any().any()
    assert df_filled.index.equals(df.index)


@pytest.mark.parametrize("method", FILL_METHODS)
def test_when_fill_missing_values_called_then_leading_nans_are_filled_and_index_is_unchanged(method):
    if method in ["ffill", "pad", "interpolate"]:
        pytest.skip(f"{method} doesn't fill leading NaNs")
    df = get_data_frame_with_variable_lengths({"B": 15, "A": 20})
    df.iloc[[0, 1, 2, 15, 16]] = np.nan
    df_filled = df.fill_missing_values(method=method)
    assert not df_filled.isna().any().any()
    assert df_filled.index.equals(df.index)


@pytest.mark.parametrize("method", FILL_METHODS)
def test_when_fill_missing_values_called_then_trailing_nans_are_filled_and_index_is_unchanged(method):
    if method in ["bfill", "backfill"]:
        pytest.skip(f"{method} doesn't fill trailing NaNs")
    df = get_data_frame_with_variable_lengths({"B": 15, "A": 20})
    df.iloc[[13, 14, 34]] = np.nan
    df_filled = df.fill_missing_values(method=method)
    assert not df_filled.isna().any().any()
    assert df_filled.index.equals(df.index)


def test_when_dropna_called_then_missing_values_are_dropped():
    df = get_data_frame_with_variable_lengths({"B": 15, "A": 20})
    df.iloc[[1, 5, 10, 14, 22]] = np.nan
    df_dropped = df.dropna()
    assert not df_dropped.isna().any().any()


def test_given_static_features_dont_contain_custom_id_column_when_from_data_frame_called_then_exception_is_raised():
    df = SAMPLE_DATAFRAME.copy()
    df = df.rename(columns={ITEMID: "custom_id"})
    static_df = SAMPLE_STATIC_DATAFRAME.copy()
    with pytest.raises(AssertionError, match="id' not found in static"):
        TimeSeriesDataFrame.from_data_frame(df, id_column="custom_id", static_features_df=static_df)


def test_when_data_contains_item_id_column_that_is_unused_then_column_is_renamed():
    df = SAMPLE_DATAFRAME.copy()
    df["custom_id"] = df[ITEMID]
    ts_df = TimeSeriesDataFrame.from_data_frame(df, id_column="custom_id")
    assert f"__{ITEMID}" in ts_df.columns


def test_when_static_features_contain_item_id_column_that_is_unused_then_column_is_renamed():
    df = SAMPLE_DATAFRAME.copy()
    df["custom_id"] = df[ITEMID]

    static_df = SAMPLE_STATIC_DATAFRAME.copy()
    static_df["custom_id"] = static_df[ITEMID]

    ts_df = TimeSeriesDataFrame.from_data_frame(df, id_column="custom_id", static_features_df=static_df)
    assert f"__{ITEMID}" in ts_df.static_features.columns


def test_when_data_contains_timestamp_column_that_is_unused_then_column_is_renamed():
    df = SAMPLE_DATAFRAME.copy()
    df["custom_timestamp"] = df[TIMESTAMP]
    ts_df = TimeSeriesDataFrame.from_data_frame(df, timestamp_column="custom_timestamp")
    assert f"__{TIMESTAMP}" in ts_df.columns


@pytest.mark.parametrize("freq", ["D", "W", "ME", "QE", "YE", "h", "min", "s", "30min", "2h", "17s"])
def test_given_index_is_irregular_when_convert_frequency_called_then_result_has_regular_index(freq):
    freq = to_supported_pandas_freq(freq)
    df_original = get_data_frame_with_variable_lengths({"B": 15, "A": 20}, freq=freq, covariates_names=["Y", "X"])

    # Select random rows & reset cached freq
    df_irregular = df_original.iloc[[2, 5, 7, 10, 14, 15, 16, 33]]
    df_regular = df_irregular.convert_frequency(freq=freq)
    for idx, value in df_regular.iterrows():
        if idx in df_irregular.index:
            assert (value == df_original.loc[idx]).all()
        else:
            assert value.isna().all()


@pytest.mark.parametrize("freq", ["D", "W", "ME", "QE", "YE", "h", "min", "s", "30min", "2h", "17s"])
def test_given_index_is_irregular_when_convert_frequency_called_then_new_index_has_desired_frequency(freq):
    freq = to_supported_pandas_freq(freq)
    df_original = get_data_frame_with_variable_lengths(
        {"B": 15, "A": 20, "C": 2}, freq=freq, covariates_names=["Y", "X"]
    )

    # [35, 36] covers the edge case where only 2 timestamps are present which prevents pandas from inferring freq
    df_irregular = df_original.iloc[[2, 5, 7, 10, 14, 15, 16, 33, 35, 36]]
    assert df_irregular.freq is None
    df_regular = df_irregular.convert_frequency(freq=freq)
    assert df_regular.freq == pd.tseries.frequencies.to_offset(freq).freqstr


def test_given_index_is_regular_when_convert_frequency_called_the_df_doesnt_change():
    df = SAMPLE_TS_DATAFRAME.copy()
    df_resampled = df.convert_frequency(freq=df.freq)
    assert df.equals(df_resampled)


def test_when_convert_frequency_called_with_different_freq_then_original_df_is_not_modified():
    df = SAMPLE_TS_DATAFRAME.copy()
    original_freq = df.freq
    df_resampled = df.convert_frequency(freq="h")
    assert df_resampled.freq != original_freq
    assert df.equals(SAMPLE_TS_DATAFRAME)
    assert df.freq == original_freq


def test_when_convert_frequency_called_then_static_features_are_kept():
    df = SAMPLE_TS_DATAFRAME_STATIC.copy()
    df_resampled = df.convert_frequency("W")
    assert df_resampled.static_features is not None
    assert df_resampled.static_features.equals(df.static_features)


@pytest.mark.parametrize("freq", ["D", "ME", "6h"])
def test_given_index_is_regular_when_convert_frequency_is_called_then_new_index_has_desired_frequency(freq):
    freq = to_supported_pandas_freq(freq)
    start = "2020-05-01"
    end = "2020-07-31"
    timestamps_original = pd.date_range(start=start, end=end, freq="D")
    timestamps_resampled = pd.date_range(start=start, end=end, freq=freq)
    df = pd.DataFrame(
        {
            ITEMID: [0] * len(timestamps_original),
            TIMESTAMP: timestamps_original,
            "target": np.random.rand(len(timestamps_original)),
        }
    )
    ts_df = TimeSeriesDataFrame(df)
    ts_df_resampled = ts_df.convert_frequency(freq=freq)
    assert (ts_df_resampled.index.get_level_values(TIMESTAMP) == timestamps_resampled).all()
    assert pd.tseries.frequencies.to_offset(ts_df_resampled.freq) == pd.tseries.frequencies.to_offset(freq)


@pytest.mark.parametrize(
    "agg_method, values_after_aggregation",
    [
        ("mean", [1.5, 3.0, 4.5, 6.0]),
        ("min", [1.0, 3.0, 4.0, 6.0]),
        ("max", [2.0, 3.0, 5.0, 6.0]),
        ("first", [1.0, 3.0, 4.0, 6.0]),
        ("last", [2.0, 3.0, 5.0, 6.0]),
        ("sum", [3.0, 3.0, 9.0, 6.0]),
    ],
)
def test_when_aggregation_method_is_changed_then_aggregated_result_is_correct(agg_method, values_after_aggregation):
    ts_df = TimeSeriesDataFrame(
        pd.DataFrame(
            {
                ITEMID: ["A", "A", "A", "A", "A", "B"],
                TIMESTAMP: ["2022-01-01", "2022-01-02", "2022-01-05", "2022-01-10", "2022-01-11", "2020-01-01"],
                "target": np.arange(1, 7),
            }
        )
    )
    aggregated = ts_df.convert_frequency(freq="W", agg_numeric=agg_method)
    assert np.all(aggregated.values.ravel() == np.array(values_after_aggregation))


@pytest.mark.parametrize("freq", ["D", "W", "ME", "QE", "YE", "h", "min", "s", "30min", "2h", "17s"])
def test_when_convert_frequency_called_then_categorical_columns_are_preserved(freq):
    freq = to_supported_pandas_freq(freq)
    df_original = get_data_frame_with_variable_lengths({"B": 15, "A": 20}, freq=freq, covariates_names=["Y", "X"])
    cat_columns = ["cat_1", "cat_2"]
    for col in cat_columns:
        df_original[col] = np.random.choice(["foo", "bar", "baz"], size=len(df_original))
    # Select random rows & reset cached freq
    df_irregular = df_original.iloc[[2, 5, 7, 10, 14, 15, 16, 33]]
    df_regular = df_irregular.convert_frequency(freq=freq)
    assert all(col in df_regular.columns for col in cat_columns)
    assert df_regular.freq == pd.tseries.frequencies.to_offset(freq).freqstr


@pytest.mark.parametrize("dtype", ["datetime64[ns]", "datetime64[us]", "datetime64[ms]", "datetime64[s]"])
def test_when_timestamps_have_datetime64_type_then_tsdf_can_be_constructed(dtype):
    df = SAMPLE_DATAFRAME.copy()
    df[TIMESTAMP] = df[TIMESTAMP].astype(dtype)
    assert df[TIMESTAMP].dtype == dtype
    TimeSeriesDataFrame.from_data_frame(df)


def test_when_to_data_frame_called_then_return_values_is_a_pandas_df():
    tsdf = SAMPLE_TS_DATAFRAME.copy()
    df = tsdf.to_data_frame()
    assert isinstance(df, pd.DataFrame)
    assert not isinstance(df, TimeSeriesDataFrame)


@pytest.mark.parametrize("unit", ["s", "ms", "ns"])
def test_when_resampling_timestamps_with_different_dtypes_then_no_nat_values_in_index(unit):
    df = pd.DataFrame(
        [
            ["H1", "2023-01-15", 42],
            ["H1", "2023-03-10", 33],
            ["H2", "2023-02-20", 78],
            ["H2", "2023-04-05", 91],
        ],
        columns=["item_id", "timestamp", "target"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"]).astype(f"datetime64[{unit}]")
    df_converted = TimeSeriesDataFrame(df).convert_frequency("D")
    assert not df_converted.index.to_frame().isna().any().any()
