from typing import Union

import pandas as pd


TO_MAJOR_FREQ = {
    "min": "T",
    "ms": "L",
    "us": "U",
    # business day
    "C": "B",
    # month
    "BM": "M",
    "CBM": "M",
    "MS": "M",
    "BMS": "M",
    "CBMS": "M",
    # semi-month
    "SMS": "SM",
    # quarter
    "BQ": "Q",
    "QS": "Q",
    # annual
    "Y": "A",
    "BA": "A",
    "BY": "A",
    "AS": "A",
    "YS": "A",
    "BAS": "A",
    "BYS": "A",
}


def get_freq_str(offset: pd.DateOffset) -> str:
    """Obtain frequency string from a pandas.DateOffset object.

    "Non-standard" frequencies are converted to their "standard" counterparts. For example, MS (month start) is mapped
    to M (month) since both correspond to the same seasonality, lags and time features.
    """
    base_freq = offset.name.split("-")[0]
    return TO_MAJOR_FREQ.get(base_freq, base_freq)


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
    norm_freq_str = get_freq_str(offset)
    base_seasonality = DEFAULT_SEASONALITIES.get(norm_freq_str, 1)

    seasonality, remainder = divmod(base_seasonality, offset.n)
    return seasonality if not remainder else 1


def get_lags_for_freq(freq: Union[str, None]):
    pass


def get_time_features_for_freq(freq: Union[str, None]):
    pass
