"""Common utils and data for all model tests"""
import random
from typing import Dict, List, Union

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


def get_data_frame_with_variable_lengths(item_id_to_length: Dict[str, int]):
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
    return df


DUMMY_VARIABLE_LENGTH_TS_DATAFRAME = get_data_frame_with_variable_lengths(
    item_id_to_length={"A": 22, "B": 50, "C": 10, "D": 17}
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
