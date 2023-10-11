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
    "BQS": "Q",
    # annual
    "Y": "A",
    "BA": "A",
    "BY": "A",
    "AS": "A",
    "YS": "A",
    "BAS": "A",
    "BYS": "A",
}


def norm_freq_str(offset: pd.DateOffset) -> str:
    """Obtain frequency string from a pandas.DateOffset object.

    "Non-standard" frequencies are converted to their "standard" counterparts. For example, MS (month start) is mapped
    to M (month) since both correspond to the same seasonality, lags and time features.
    """
    base_freq = offset.name.split("-")[0]
    return TO_MAJOR_FREQ.get(base_freq, base_freq)
