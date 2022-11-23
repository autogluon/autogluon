import hashlib

import pandas as pd

from ..dataset.ts_dataframe import ITEMID, TIMESTAMP, TimeSeriesDataFrame


# TODO: Move to autogluon.timeseries.models.statsmodels.helper after we drop the sktime dependency
def hash_ts_dataframe_items(ts_dataframe: TimeSeriesDataFrame) -> pd.Series:
    """Hash each time series in the dataset to a 32-character hex string.

    Hash is computed based on the timestamps and values of the time series (item_id is ignored).

    This means that any model that doesn't use static features will make identical predictions for two time series
    with the same hash value (assuming no collisions).
    """
    df_with_timestamp = ts_dataframe.reset_index(level=TIMESTAMP)
    hash_per_timestep = pd.util.hash_pandas_object(df_with_timestamp, index=False)
    # groupby preserves the order of the timesteps
    return hash_per_timestep.groupby(level=ITEMID, sort=False).apply(lambda x: hashlib.md5(x.values).hexdigest())
