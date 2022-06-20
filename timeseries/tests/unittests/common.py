"""Common utils and data for all model tests"""
import random
from typing import List, Union

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


def get_data_frame_with_item_index(item_list: List[Union[str, int]], data_length: int = 20):
    return TimeSeriesDataFrame(
        pd.DataFrame(
            index=pd.MultiIndex.from_product(
                [
                    item_list,
                    pd.date_range(pd.Timestamp("2022-01-01"), freq="H", periods=data_length),  # noqa
                ],
                names=(ITEMID, TIMESTAMP),
            ),
            data=[random.random() for _ in range(len(item_list) * data_length)],
            columns=["target"],
        )
    )


DUMMY_TS_DATAFRAME = get_data_frame_with_item_index(["A", "B", "C", "D"])


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
            equal_fields.append(
                dict_equal_primitive(dict(enumerate(v)), dict(enumerate(that[k])))
            )

    return all(equal_fields)
