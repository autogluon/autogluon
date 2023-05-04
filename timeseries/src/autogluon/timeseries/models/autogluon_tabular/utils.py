import pandas as pd
from mlforecast.target_transforms import BaseTargetTransform


class StandardScaler(BaseTargetTransform):
    """Standardizes the series by dividing by their standard deviation."""

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.stats_ = df.groupby(self.id_col)[self.target_col].agg(["mean", "std"]).rename(columns={"mean": "_mean", "std": "_std"})
        df = df.merge(self.stats_, on=self.id_col)
        df[self.target_col] = (df[self.target_col] - df["_mean"]) / df["_std"].clip(lower=1e-2)
        df = df.drop(columns=["_mean", "_std"])
        return df

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.merge(self.stats_, on=self.id_col)
        for col in df.columns.drop([self.id_col, self.time_col, "_mean", "_std"]):
            df[col] = df[col] * df["_std"] + df["_mean"]
        df = df.drop(columns=["_mean", "_std"])
        return df
