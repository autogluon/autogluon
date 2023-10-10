"""
Generate lag indices based on frequency string. Adapted from gluonts.time_feature.lag.
"""
from typing import List, Optional

import numpy as np
import pandas as pd

from .base import norm_freq_str


def _make_lags(middle: int, delta: int) -> np.ndarray:
    """
    Create a set of lags around a middle point including +/- delta.
    """
    return np.arange(middle - delta, middle + delta + 1).tolist()


# Lags are target values at the same `season` (+/- delta) but in the previous cycle.
def _make_lags_for_second(multiple, num_cycles=3):
    # We use previous ``num_cycles`` hours to generate lags
    return [_make_lags(k * 60 // multiple, 2) for k in range(1, num_cycles + 1)]


def _make_lags_for_minute(multiple, num_cycles=3):
    # We use previous ``num_cycles`` hours to generate lags
    return [_make_lags(k * 60 // multiple, 2) for k in range(1, num_cycles + 1)]


def _make_lags_for_hour(multiple, num_cycles=7):
    # We use previous ``num_cycles`` days to generate lags
    return [_make_lags(k * 24 // multiple, 1) for k in range(1, num_cycles + 1)]


def _make_lags_for_business_hour(multiple, num_cycles=7):
    return [_make_lags(k * 9 // multiple, 1) for k in range(1, num_cycles + 1)]


def _make_lags_for_day(multiple, num_cycles=4, days_in_week=7, days_in_month=30):
    # We use previous ``num_cycles`` weeks to generate lags
    # We use the last month (in addition to 4 weeks) to generate lag.
    return [_make_lags(k * days_in_week // multiple, 1) for k in range(1, num_cycles + 1)] + [
        _make_lags(days_in_month // multiple, 1)
    ]


def _make_lags_for_week(multiple, num_cycles=3):
    # We use previous ``num_cycles`` years to generate lags
    # Additionally, we use previous 4, 8, 12 weeks
    return [_make_lags(k * 52 // multiple, 1) for k in range(1, num_cycles + 1)] + [
        [4 // multiple, 8 // multiple, 12 // multiple]
    ]


def _make_lags_for_month(multiple, num_cycles=3):
    # We use previous ``num_cycles`` years to generate lags
    return [_make_lags(k * 12 // multiple, 1) for k in range(1, num_cycles + 1)]


def _make_lags_for_quarter(multiple, num_cycles=3):
    return [_make_lags(k * 4 // multiple, 1) for k in range(1, num_cycles + 1)]


def _make_lags_for_semi_month(multiple, num_cycles=3):
    # We use previous ``num_cycles`` years to generate lags
    return [_make_lags(k * 24 // multiple, 1) for k in range(1, num_cycles + 1)]


def get_lags_for_frequency(
    freq: str,
    lag_ub: int = 1200,
    num_lags: Optional[int] = None,
    num_default_lags: int = 7,
) -> List[int]:
    """
    Generates a list of lags that that are appropriate for the given frequency
    string.

    By default all frequencies have the following lags: [1, 2, 3, 4, 5, 6, 7].
    Remaining lags correspond to the same `season` (+/- `delta`) in previous
    `k` cycles. Here `delta` and `k` are chosen according to the existing code.

    Parameters
    ----------
    freq_str
        Frequency string of the form [multiple][granularity] such as "12H",
        "5min", "1D" etc.
    lag_ub
        The maximum value for a lag.
    num_lags
        Maximum number of lags; by default all generated lags are returned.
    num_default_lags
        The number of default lags; by default it is 7.
    """

    offset = pd.tseries.frequencies.to_offset(freq)
    offset_name = norm_freq_str(offset)

    if offset_name == "A":
        lags = []
    elif offset_name == "Q":
        lags = _make_lags_for_quarter(offset.n)
    elif offset_name == "M":
        lags = _make_lags_for_month(offset.n)
    elif offset_name == "SM":
        lags = _make_lags_for_semi_month(offset.n)
    elif offset_name == "W":
        lags = _make_lags_for_week(offset.n)
    elif offset_name == "D":
        lags = _make_lags_for_day(offset.n) + _make_lags_for_week(offset.n / 7.0)
    elif offset_name == "B":
        lags = _make_lags_for_day(offset.n, days_in_week=5, days_in_month=22) + _make_lags_for_week(offset.n / 5.0)
    elif offset_name == "H":
        lags = (
            _make_lags_for_hour(offset.n)
            + _make_lags_for_day(offset.n / 24)
            + _make_lags_for_week(offset.n / (24 * 7))
        )
    # business hour
    elif offset_name == "BH":
        lags = (
            _make_lags_for_business_hour(offset.n)
            + _make_lags_for_day(offset.n / 9)
            + _make_lags_for_week(offset.n / (9 * 7))
        )
    # minutes
    elif offset_name == "T":
        lags = (
            _make_lags_for_minute(offset.n)
            + _make_lags_for_hour(offset.n / 60)
            + _make_lags_for_day(offset.n / (60 * 24))
            + _make_lags_for_week(offset.n / (60 * 24 * 7))
        )
    # second
    elif offset_name == "S":
        lags = (
            _make_lags_for_second(offset.n)
            + _make_lags_for_minute(offset.n / 60)
            + _make_lags_for_hour(offset.n / (60 * 60))
        )
    elif offset_name == "L":
        lags = (
            _make_lags_for_second(offset.n / 1e3)
            + _make_lags_for_minute(offset.n / (60 * 1e3))
            + _make_lags_for_hour(offset.n / (60 * 60 * 1e3))
        )
    elif offset_name == "U":
        lags = (
            _make_lags_for_second(offset.n / 1e6)
            + _make_lags_for_minute(offset.n / (60 * 1e6))
            + _make_lags_for_hour(offset.n / (60 * 60 * 1e6))
        )
    elif offset_name == "N":
        lags = (
            _make_lags_for_second(offset.n / 1e9)
            + _make_lags_for_minute(offset.n / (60 * 1e9))
            + _make_lags_for_hour(offset.n / (60 * 60 * 1e9))
        )
    else:
        raise Exception(f"invalid frequency {freq}")

    # flatten lags list and filter
    lags = [int(lag) for sub_list in lags for lag in sub_list if 7 < lag <= lag_ub]
    lags = list(range(1, num_default_lags + 1)) + sorted(list(set(lags)))

    return sorted(set(lags))[:num_lags]
