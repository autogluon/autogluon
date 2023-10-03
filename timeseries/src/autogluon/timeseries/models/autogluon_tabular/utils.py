import numpy as np
import pandas as pd
from mlforecast.target_transforms import BaseTargetTransform


class StandardScaler(BaseTargetTransform):
    """Standardizes the series by subtracting mean and diving by standard deviation."""

    min_scale: float = 1e-2

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.stats_ = (
            df.replace([np.inf, -np.inf], np.nan)
            .groupby(self.id_col)[self.target_col]
            .agg(["mean", "std"])
            .rename(columns={"mean": "_mean", "std": "_scale"})
        )
        self.stats_["_scale"] = self.stats_["_scale"].clip(lower=self.min_scale)
        df = df.merge(self.stats_, on=self.id_col)
        df[self.target_col] = (df[self.target_col] - df["_mean"]) / df["_scale"]
        df = df.drop(columns=["_mean", "_scale"])
        return df

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.merge(self.stats_, on=self.id_col)
        for col in df.columns.drop([self.id_col, self.time_col, "_mean", "_scale"]):
            df[col] = df[col] * df["_scale"] + df["_mean"]
        df = df.drop(columns=["_mean", "_scale"])
        return df


class MeanAbsScaler(BaseTargetTransform):
    """Scales time series by diving by their mean absolute value."""

    min_scale: float = 1e-2

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        target = df[self.target_col].replace([np.inf, -np.inf], np.nan).abs()
        self.stats_ = target.groupby(df[self.id_col], sort=False).agg(["mean"]).rename(columns={"mean": "_scale"})
        self.stats_["_scale"] = self.stats_["_scale"].clip(lower=self.min_scale)
        df = df.merge(self.stats_, on=self.id_col)
        df[self.target_col] = df[self.target_col] / df["_scale"]
        df = df.drop(columns=["_scale"])
        return df

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.merge(self.stats_, on=self.id_col)
        for col in df.columns.drop([self.id_col, self.time_col, "_scale"]):
            df[col] = df[col] * df["_scale"]
        df = df.drop(columns=["_scale"])
        return df
