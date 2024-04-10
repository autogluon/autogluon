import pandas as pd

TO_MAJOR_FREQ = {
    # sub-daily
    "H": "h",
    "BH": "bh",
    "cbh": "bh",
    "CBH": "bh",
    "T": "min",
    "S": "s",
    "L": "ms",
    "U": "us",
    "N": "ns",
    # business day
    "C": "B",
    # month
    "M": "ME",
    "BM": "ME",
    "BME": "ME",
    "CBM": "ME",
    "CBME": "ME",
    "MS": "ME",
    "BMS": "ME",
    "CBMS": "ME",
    # semi-month
    "SM": "SME",
    "SMS": "SME",
    # quarter
    "Q": "QE",
    "BQ": "QE",
    "BQE": "QE",
    "QS": "QE",
    "BQS": "QE",
    # annual
    "A": "YE",
    "Y": "YE",
    "BA": "YE",
    "BY": "YE",
    "BYE": "YE",
    "AS": "YE",
    "YS": "YE",
    "BAS": "YE",
    "BYS": "YE",
}


def norm_freq_str(offset: pd.DateOffset) -> str:
    """Obtain frequency string from a pandas.DateOffset object.

    "Non-standard" frequencies are converted to their "standard" counterparts. For example, MS (month start) is mapped
    to ME (month end) since both correspond to the same seasonality, lags and time features.

    The frequencies are always mapped to the new non-deprecated aliases (pandas>=2.2), e.g., "H" is mapped to "h". The
    downstream functions like `get_seasonality` handle the new aliases even if older version of pandas is used.
    """
    base_freq = offset.name.split("-")[0]
    return TO_MAJOR_FREQ.get(base_freq, base_freq)
