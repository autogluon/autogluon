from typing import Literal, Union

import numpy as np
import pandas as pd
from mlforecast.target_transforms import (
    BaseTargetTransform,
    GroupedArray,
    _BaseGroupedArrayTargetTransform,
)

from autogluon.timeseries.dataset.ts_dataframe import (
    ITEMID,
    TIMESTAMP,
    TimeSeriesDataFrame,
)
from autogluon.timeseries.transforms.target_scaler import TargetScaler, get_target_scaler

from .utils import MLF_ITEMID, MLF_TIMESTAMP


class MLForecastScaler(BaseTargetTransform):
    def __init__(self, scaler_type: Literal["standard", "min_max", "mean_abs", "robust"]):
        # For backward compatibility
        self.scaler_type: Literal["standard", "min_max", "mean_abs", "robust"] = scaler_type
        self.ag_scaler: TargetScaler

    def _df_to_tsdf(self, df: pd.DataFrame) -> TimeSeriesDataFrame:
        return TimeSeriesDataFrame(
            df.rename(columns={self.id_col: ITEMID, self.time_col: TIMESTAMP}).set_index([ITEMID, TIMESTAMP])
        )

    def _tsdf_to_df(self, ts_df: TimeSeriesDataFrame) -> pd.DataFrame:
        return pd.DataFrame(ts_df).reset_index().rename(columns={ITEMID: self.id_col, TIMESTAMP: self.time_col})

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:  # type: ignore
        self.ag_scaler = get_target_scaler(name=self.scaler_type, target=self.target_col)
        transformed = self.ag_scaler.fit_transform(self._df_to_tsdf(df))
        return self._tsdf_to_df(transformed)

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:  # type: ignore
        assert self.ag_scaler is not None
        transformed = self.ag_scaler.inverse_transform(self._df_to_tsdf(df))
        return self._tsdf_to_df(transformed)


def apply_inverse_transform(
    df: pd.DataFrame,
    transform: Union[_BaseGroupedArrayTargetTransform, BaseTargetTransform],
) -> pd.DataFrame:
    """Apply inverse transformation to a dataframe, converting to GroupedArray if necessary"""
    if isinstance(transform, BaseTargetTransform):
        inverse_transformed = transform.inverse_transform(df=df)
        assert isinstance(inverse_transformed, pd.DataFrame)
        return inverse_transformed
    elif isinstance(transform, _BaseGroupedArrayTargetTransform):
        indptr = np.concatenate([[0], df[MLF_ITEMID].value_counts().cumsum()])
        assignment = {}
        for col in df.columns.drop([MLF_ITEMID, MLF_TIMESTAMP]):
            ga = GroupedArray(data=df[col].to_numpy(), indptr=indptr)
            assignment[col] = transform.inverse_transform(ga).data
        return df.assign(**assignment)
    else:
        raise ValueError(
            f"transform must be of type `_BaseGroupedArrayTargetTransform` or `BaseTargetTransform` (got {type(transform)})"
        )
