import pandas as pd

from mlforecast.target_transforms import BaseTargetTransform


class StandardScaler(BaseTargetTransform):
    """Standardizes the series by subtracting their mean and dividing by their standard deviation."""

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.std_ = df.groupby(self.id_col)[self.target_col].std().rename("std").clip(lower=1e-2).to_frame()
        df = df.merge(self.std_, on=self.id_col)
        df[self.target_col] = df[self.target_col] / df["std"]
        df = df.drop(columns=["std"])
        return df

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.merge(self.std_, on=self.id_col)
        for col in df.columns.drop([self.id_col, self.time_col, "std"]):
            df[col] = df[col] * df["std"]
        df = df.drop(columns=["std"])
        return df
