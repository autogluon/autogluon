import re

import numpy as np
import pandas as pd
from pandas import DataFrame


def make_date(df: DataFrame, date_field: str):
    """Make sure `df[field_name]` is of the right date type."""
    field_dtype = df[date_field].dtype
    if isinstance(field_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
        field_dtype = np.datetime64
    if not np.issubdtype(field_dtype, np.datetime64):
        df[date_field] = pd.to_datetime(df[date_field], infer_datetime_format=True)


def add_datepart(df: DataFrame, field_name: str, prefix: str = None, drop: bool = True, time: bool = False):
    """Helper function that adds columns relevant to a date in the column `field_name` of `df`."""
    make_date(df, field_name)
    field = df[field_name]
    prefix = ifnone(prefix, re.sub('[Dd]ate$', '', field_name))
    attr = [
        'Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear', 'Is_month_end', 'Is_month_start',
        'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start'
    ]
    if time:
        attr = attr + ['Hour', 'Minute', 'Second']
    for n in attr:
        df[prefix + n] = getattr(field.dt, n.lower())
    df[prefix + 'Elapsed'] = field.astype(np.int64) // 10 ** 9
    if drop:
        df.drop(field_name, axis=1, inplace=True)
    return df


def ifnone(a, b):
    """`a` if `a` is not None, otherwise `b`."""
    return b if a is None else a
