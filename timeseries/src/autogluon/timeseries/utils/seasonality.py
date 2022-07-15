import pandas as pd


DEFAULT_SEASONALITIES = {
    "T": 60 * 24,
    "H": 24,
    "D": 1,
    "W": 1,
    "M": 12,
    "B": 5,
    "Q": 4,
}


def get_seasonality(freq: str) -> int:
    """Return the seasonality of a given frequency. Adapted from
    ``gluonts.time_feature.seasonality``.
    """
    offset = pd.tseries.frequencies.to_offset(freq)
    norm_freq_str = offset.name.split("-")[0]
    base_seasonality = DEFAULT_SEASONALITIES.get(norm_freq_str, 1)

    seasonality, remainder = divmod(base_seasonality, offset.n)
    return seasonality if not remainder else 1
