"""Common utils and data for all model tests"""
import random

import pandas as pd
from gluonts.dataset.common import ListDataset

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
