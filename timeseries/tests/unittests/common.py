"""Common utils and data for all model tests"""

import random
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from packaging.version import Version

from autogluon.timeseries.dataset.ts_dataframe import ITEMID, TIMESTAMP, TimeSeriesDataFrame
from autogluon.timeseries.metrics import TimeSeriesScorer
from autogluon.timeseries.utils.forecast import get_forecast_horizon_index_ts_dataframe

# TODO: add larger unit test data sets to S3


# List of all non-deprecated pandas frequencies, based on https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
def get_all_pandas_frequencies():
    if Version(pd.__version__) >= Version("2.2"):
        return {
            "B",
            "C",
            "D",
            "W",
            "ME",
            "SME",
            "BME",
            "CBME",
            "MS",
            "SMS",
            "BMS",
            "CBMS",
            "QE",
            "BQE",
            "QS",
            "BQS",
            "YE",
            "BYE",
            "YS",
            "BYS",
            "h",
            "bh",
            "cbh",
            "min",
            "s",
            "ms",
            "us",
            "ns",
        }
    else:
        return {
            "B",
            "C",
            "D",
            "W",
            "M",
            "SM",
            "BM",
            "CBM",
            "MS",
            "SMS",
            "BMS",
            "CBMS",
            "Q",
            "BQ",
            "QS",
            "BQS",
            "A",
            "Y",
            "BA",
            "BY",
            "AS",
            "YS",
            "BAS",
            "BYS",
            "BH",
            "H",
            "T",
            "min",
            "S",
            "L",
            "ms",
            "U",
            "us",
            "N",
        }


ALL_PANDAS_FREQUENCIES = get_all_pandas_frequencies()


def to_supported_pandas_freq(freq: str) -> str:
    """If necessary, convert pandas 2.2+ freq strings to an alias supported by currently installed pandas version."""
    if Version(pd.__version__) < Version("2.2"):
        return {"ME": "M", "QE": "Q", "YE": "Y", "SME": "SM", "h": "H", "min": "T"}.get(freq, freq)
    else:
        return freq


def get_data_frame_with_item_index(
    item_list: List[Union[str, int]],
    data_length: int = 20,
    freq: str = "h",
    start_date: str = "2022-01-01",
    columns: List[str] = ["target"],
    data_generation: str = "random",
):
    assert data_generation in ["random", "sequential"]
    if data_generation == "random":
        data = [random.random() for _ in range(len(item_list) * data_length)]
    elif data_generation == "sequential":
        data = [e for e in range(len(item_list) * data_length)]

    return TimeSeriesDataFrame(
        pd.DataFrame(
            index=pd.MultiIndex.from_product(
                [
                    item_list,
                    pd.date_range(
                        pd.Timestamp(start_date),  # noqa
                        freq=to_supported_pandas_freq(freq),
                        periods=data_length,
                    ),
                ],
                names=(ITEMID, TIMESTAMP),
            ),
            data=data,
            columns=columns,
        )
    )


def mask_entries(data: TimeSeriesDataFrame) -> TimeSeriesDataFrame:
    """Replace some values in a TimeSeriesDataFrame with NaNs"""
    data = data.copy()
    # Mask all but the first entry for item #1
    data.iloc[1 : data.num_timesteps_per_item()[data.item_ids[0]]] = float("nan")
    # Completely mask item #2
    data.loc[data.item_ids[1]] = float("nan")
    # Mask random indices for item #3
    nan_idx = [42, 53, 58, 59][: len(data)]
    data.iloc[nan_idx] = float("nan")
    return data


DUMMY_TS_DATAFRAME = mask_entries(get_data_frame_with_item_index(["10", "A", "2", "1"]))


def get_data_frame_with_variable_lengths(
    item_id_to_length: Dict[str, int],
    static_features: Optional[pd.DataFrame] = None,
    covariates_names: Optional[List[str]] = None,
    freq: str = "D",
):
    tuples = []
    for item_id, length in item_id_to_length.items():
        for ts in pd.date_range(pd.Timestamp("2022-01-01"), periods=length, freq=freq):
            tuples.append((item_id, ts))
    index = pd.MultiIndex.from_tuples(tuples, names=[ITEMID, TIMESTAMP])
    df = TimeSeriesDataFrame(
        pd.DataFrame(
            index=index,
            data=[random.random() for _ in index],
            columns=["target"],
        )
    )
    df.freq  # compute _cached_freq
    df.static_features = static_features
    if covariates_names is not None:
        for i, name in enumerate(covariates_names):
            # Make every second feature categorical
            if i % 2:
                df[name] = np.random.normal(size=len(df))
            else:
                df[name] = np.random.choice(["foo", "bar"], size=len(df))
    return df


ITEM_ID_TO_LENGTH = {"D": 22, "A": 50, "C": 10, "B": 17}
DUMMY_VARIABLE_LENGTH_TS_DATAFRAME = mask_entries(get_data_frame_with_variable_lengths(ITEM_ID_TO_LENGTH))


def get_static_features(item_ids: List[Union[str, int]], feature_names: List[str]):
    features = {}
    for idx, feat_name in enumerate(feature_names):
        if idx % 2 == 0:
            values = np.random.rand(len(item_ids))
        else:
            values = np.random.choice(["X", "Y", "Z", "1"], size=len(item_ids)).astype(object)
        features[feat_name] = values
    df = pd.DataFrame(features, index=list(item_ids))
    df.index.name = ITEMID
    return df


DATAFRAME_WITH_STATIC = get_data_frame_with_variable_lengths(
    ITEM_ID_TO_LENGTH, static_features=get_static_features(ITEM_ID_TO_LENGTH.keys(), ["feat1", "feat2", "feat3"])
)

DATAFRAME_WITH_COVARIATES = get_data_frame_with_variable_lengths(
    ITEM_ID_TO_LENGTH, covariates_names=["cov1", "cov2", "cov3"]
)

DATAFRAME_WITH_STATIC_AND_COVARIATES = get_data_frame_with_variable_lengths(
    ITEM_ID_TO_LENGTH,
    covariates_names=["cov1", "cov2", "cov3"],
    static_features=get_static_features(ITEM_ID_TO_LENGTH.keys(), ["feat1", "feat2", "feat3"]),
)


def dict_equal_primitive(this, that):
    """Compare two dictionaries but consider only primitive values"""
    if not this.keys() == that.keys():
        return False

    equal_fields = []
    for k, v in this.items():
        if isinstance(v, (int, float, bool, str)):
            equal_fields.append(v == that[k])
        if isinstance(v, dict):
            equal_fields.append(dict_equal_primitive(v, that[k]))
        if isinstance(v, list):
            equal_fields.append(dict_equal_primitive(dict(enumerate(v)), dict(enumerate(that[k]))))

    return all(equal_fields)


class CustomMetric(TimeSeriesScorer):
    def save_past_metrics(self, data_past: TimeSeriesDataFrame, target: str = "target", **kwargs) -> None:
        self._past_target_mean = 1.0 + data_past[target].abs().mean()

    def compute_metric(
        self, data_future: TimeSeriesDataFrame, predictions: TimeSeriesDataFrame, target: str = "target", **kwargs
    ) -> float:
        return ((data_future[target] - predictions["mean"]) / self._past_target_mean).mean()

    def clear_past_metrics(self) -> None:
        del self._past_target_mean


def get_prediction_for_df(data, prediction_length=5):
    forecast_index = get_forecast_horizon_index_ts_dataframe(data, prediction_length=prediction_length)
    columns = ["mean", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"]
    return TimeSeriesDataFrame(
        pd.DataFrame(np.random.normal(size=[len(forecast_index), len(columns)]), index=forecast_index, columns=columns)
    )


PREDICTIONS_FOR_DUMMY_TS_DATAFRAME = get_prediction_for_df(DUMMY_TS_DATAFRAME)
