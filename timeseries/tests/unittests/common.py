"""Common utils and data for all model tests"""
import random
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from gluonts.dataset.common import ListDataset

from autogluon.timeseries.dataset import TimeSeriesDataFrame
from autogluon.timeseries.dataset.ts_dataframe import ITEMID, TIMESTAMP

# TODO: add larger unit test data sets to S3

DUMMY_DATASET = ListDataset(
    [
        {
            "target": [random.random() for _ in range(10)],
            "start": pd.Timestamp("2022-01-01 00:00:00"),  # noqa
            "item_id": 0,
        },
        {
            "target": [random.random() for _ in range(10)],
            "start": pd.Timestamp("2022-01-01 00:00:00"),  # noqa
            "item_id": 1,
        },
    ],
    freq="H",
)


def get_data_frame_with_item_index(
    item_list: List[Union[str, int]],
    data_length: int = 20,
    freq: str = "H",
):
    return TimeSeriesDataFrame(
        pd.DataFrame(
            index=pd.MultiIndex.from_product(
                [
                    item_list,
                    pd.date_range(
                        pd.Timestamp("2022-01-01"),  # noqa
                        freq=freq,
                        periods=data_length,
                    ),
                ],
                names=(ITEMID, TIMESTAMP),
            ),
            data=[random.random() for _ in range(len(item_list) * data_length)],
            columns=["target"],
        )
    )


DUMMY_TS_DATAFRAME = get_data_frame_with_item_index(["10", "A", "2", "1"])


def get_data_frame_with_variable_lengths(
    item_id_to_length: Dict[str, int],
    static_features: Optional[pd.DataFrame] = None,
    known_covariates_names: Optional[List[str]] = None,
):
    tuples = []
    for item_id, length in item_id_to_length.items():
        for ts in pd.date_range(pd.Timestamp("2022-01-01"), periods=length, freq="D"):
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
    if known_covariates_names is not None:
        for name in known_covariates_names:
            df[name] = np.random.normal(size=len(df))
    return df


ITEM_ID_TO_LENGTH = {"D": 22, "A": 50, "C": 10, "B": 17}
DUMMY_VARIABLE_LENGTH_TS_DATAFRAME = get_data_frame_with_variable_lengths(ITEM_ID_TO_LENGTH)


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
    ITEM_ID_TO_LENGTH, known_covariates_names=["cov1", "cov2", "cov3"]
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
