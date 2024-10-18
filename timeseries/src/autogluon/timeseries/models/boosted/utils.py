import numpy as np
import pandas as pd
from autogluon.timeseries.dataset.ts_dataframe import TimeSeriesDataFrame, ITEMID, TIMESTAMP

MEAN = "__mean"
SCALE = "__scale"


class StandardScaler:
    def __init__(self, target: str = "target", min_scale: float = 1e-2):
        self.target = target
        self.min_scale = min_scale
        self.stats_: pd.DataFrame = None

    def fit_transform(self, data: pd.DataFrame) -> TimeSeriesDataFrame:
        self.fit(data=data)
        return self.transform(data=data)

    def fit(self, data: TimeSeriesDataFrame) -> None:
        self.stats_ = (
            data.replace([np.inf, -np.inf], np.nan)
            .groupby(level=ITEMID, sort=False)[self.target]
            .agg(["mean", "std"])
            .rename(columns={"mean": MEAN, "std": SCALE})
        )
        self.stats_[SCALE] = self.stats_[SCALE].clip(lower=self.min_scale)

    def transform(self, data: TimeSeriesDataFrame) -> TimeSeriesDataFrame:
        data = data.reset_index().merge(self.stats_, on=ITEMID).set_index([ITEMID, TIMESTAMP])
        data[self.target] = (data[self.target] - data[MEAN]) / data[SCALE]
        data = data.drop(columns=[MEAN, SCALE])
        return data

    def inverse_transform(self, data: TimeSeriesDataFrame) -> TimeSeriesDataFrame:
        data = data.reset_index().merge(self.stats_, on=ITEMID).set_index([ITEMID, TIMESTAMP])
        for col in data.columns.drop([MEAN, SCALE]):
            data[col] = data[col] * data[SCALE] + data[MEAN]
        data = data.drop(columns=[MEAN, SCALE])
        return data
