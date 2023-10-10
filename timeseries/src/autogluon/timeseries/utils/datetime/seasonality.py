from typing import Union

import pandas as pd

from .base import norm_freq_str

DEFAULT_SEASONALITIES = {
    "A": 1,
    "Q": 4,
    "M": 12,
    "SM": 24,
    "W": 1,
    "D": 7,
    "B": 5,
    "BH": 9,
    "H": 24,
    "T": 60 * 24,
    "S": 1,
    "L": 1,
    "U": 1,
    "N": 1,
}


def get_seasonality(freq: Union[str, None]) -> int:
    """Return the seasonality of a given frequency. Adapted from ``gluonts.time_feature.seasonality``."""
    if freq is None:
        return 1

    offset = pd.tseries.frequencies.to_offset(freq)
    offset_name = norm_freq_str(offset)
    base_seasonality = DEFAULT_SEASONALITIES.get(offset_name, 1)

    seasonality, remainder = divmod(base_seasonality, offset.n)
    return seasonality if not remainder else 1
