"""
Generate time features based on frequency string. Adapted from gluonts.time_feature.time_feature.
"""
from typing import Callable, List

import numpy as np
import pandas as pd

from .base import norm_freq_str


def _normalize(values, num: float):
    """Scale values of ``values`` to [-0.5, 0.5]."""
    return np.asarray(values, dtype=np.float32) / (num - 1) - 0.5


def quarter_of_year(index: pd.DatetimeIndex) -> np.ndarray:
    return _normalize(index.quarter, num=4)


def month_of_year(index: pd.DatetimeIndex) -> np.ndarray:
    return _normalize(index.month - 1, num=12)


def week_of_year(index: pd.DatetimeIndex) -> np.ndarray:
    try:
        week = index.isocalendar().week
    except AttributeError:
        week = index.week

    return _normalize(week - 1, num=53)


def day_of_month(index: pd.DatetimeIndex) -> np.ndarray:
    return _normalize(index.day - 1, num=31)


def day_of_year(index: pd.DatetimeIndex) -> np.ndarray:
    return _normalize(index.dayofyear - 1, num=366)


def day_of_week(index: pd.DatetimeIndex) -> np.ndarray:
    return _normalize(index.dayofweek, num=7)


def hour_of_day(index: pd.DatetimeIndex) -> np.ndarray:
    return _normalize(index.hour, num=24)


def minute_of_hour(index: pd.DatetimeIndex) -> np.ndarray:
    return _normalize(index.minute, num=60)


def second_of_minute(index: pd.DatetimeIndex) -> np.ndarray:
    return _normalize(index.second, num=60)


def get_time_features_for_frequency(freq) -> List[Callable]:
    features_by_offset_name = {
        "A": [],
        "Q": [quarter_of_year],
        "M": [month_of_year],
        "SM": [day_of_month, month_of_year],
        "W": [day_of_month, week_of_year],
        "D": [day_of_week, day_of_month, day_of_year],
        "B": [day_of_week, day_of_month, day_of_year],
        "BH": [hour_of_day, day_of_week, day_of_month, day_of_year],
        "H": [hour_of_day, day_of_week, day_of_month, day_of_year],
        "T": [minute_of_hour, hour_of_day, day_of_week, day_of_month, day_of_year],
        "S": [second_of_minute, minute_of_hour, hour_of_day, day_of_week, day_of_month, day_of_year],
        "L": [second_of_minute, minute_of_hour, hour_of_day, day_of_week, day_of_month, day_of_year],
        "U": [second_of_minute, minute_of_hour, hour_of_day, day_of_week, day_of_month, day_of_year],
        "N": [second_of_minute, minute_of_hour, hour_of_day, day_of_week, day_of_month, day_of_year],
    }
    offset = pd.tseries.frequencies.to_offset(freq)
    offset_name = norm_freq_str(offset)
    return features_by_offset_name[offset_name]
