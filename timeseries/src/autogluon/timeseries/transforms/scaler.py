from typing import Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd

from autogluon.timeseries.dataset.ts_dataframe import ITEMID, TimeSeriesDataFrame


class LocalTargetScaler:
    """Applies an affine transformation (x - loc) / scale independently to each time series in the dataset."""

    def __init__(
        self,
        target: str = "target",
        min_scale: float = 1e-2,
    ):
        self.target = target
        self.min_scale = min_scale
        self.loc: Optional[pd.Series] = None
        self.scale: Optional[pd.Series] = None

    def _compute_loc_scale(self, target_series: pd.Series) -> Tuple[Optional[pd.Series], Optional[pd.Series]]:
        raise NotImplementedError

    def fit_transform(self, data: TimeSeriesDataFrame) -> TimeSeriesDataFrame:
        return self.fit(data=data).transform(data=data)

    def fit(self, data: TimeSeriesDataFrame) -> "LocalTargetScaler":
        target_series = data[self.target].replace([np.inf, -np.inf], np.nan)
        self.loc, self.scale = self._compute_loc_scale(target_series)
        if self.loc is not None:
            self.loc = self.loc.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if self.scale is not None:
            self.scale = self.scale.clip(lower=self.min_scale).replace([np.inf, -np.inf], np.nan).fillna(1.0)
        return self

    def _reindex_loc_scale(self, item_index: pd.Index) -> Tuple[Union[np.ndarray, float], Union[np.ndarray, float]]:
        """Reindex loc and scale parameters for the given item_ids and convert them to an array-like."""
        if self.loc is not None:
            loc = self.loc.reindex(item_index).to_numpy()
        else:
            loc = 0.0
        if self.scale is not None:
            scale = self.scale.reindex(item_index).to_numpy()
        else:
            scale = 1.0
        return loc, scale

    def transform(self, data: TimeSeriesDataFrame) -> TimeSeriesDataFrame:
        """Apply scaling to the target column in the dataframe."""
        loc, scale = self._reindex_loc_scale(item_index=data.index.get_level_values(ITEMID))
        return data.assign(**{self.target: (data[self.target] - loc) / scale})

    def inverse_transform(self, predictions: TimeSeriesDataFrame) -> TimeSeriesDataFrame:
        """Apply inverse scaling to all columns in the predictions dataframe."""
        loc, scale = self._reindex_loc_scale(item_index=predictions.index.get_level_values(ITEMID))
        return predictions.assign(**{col: predictions[col] * scale + loc for col in predictions.columns})


class LocalStandardScaler(LocalTargetScaler):
    """Applies standard scaling to each time series in the dataset.

    The resulting affine transformation is (x - loc) / scale, where scale = std(x), loc = mean(x).
    """

    def _compute_loc_scale(self, target_series: pd.Series) -> Tuple[pd.Series, pd.Series]:
        stats = target_series.groupby(level=ITEMID, sort=False).agg(["mean", "std"])
        return stats["mean"], stats["std"]


class LocalMeanAbsScaler(LocalTargetScaler):
    """Applies mean absolute scaling to each time series in the dataset."""

    def _compute_loc_scale(self, target_series: pd.Series) -> Tuple[pd.Series, pd.Series]:
        scale = target_series.abs().groupby(level=ITEMID, sort=False).agg("mean")
        return None, scale


class LocalMinMaxScaler(LocalTargetScaler):
    """Applies min/max scaling to each time series in the dataset.

    The resulting affine transformation is (x - loc) / scale, where scale = max(x) - min(x), loc = min(x) / scale.
    """

    def _compute_loc_scale(self, target_series: pd.Series) -> Tuple[pd.Series, pd.Series]:
        stats = target_series.abs().groupby(level=ITEMID, sort=False).agg(["min", "max"])
        scale = (stats["max"] - stats["min"]).clip(lower=self.min_scale)
        loc = stats["min"] / scale
        return loc, scale


class LocalRobustScaler(LocalTargetScaler):
    """Applies a robust scaler based on the interquartile range. Less sensitive to outliers compared to other scaler.

    The resulting affine transformation is (x - loc) / scale, where scale = quantile(x, 0.75) - quantile(x, 0.25), loc = median(x).
    """

    def __init__(
        self,
        target: str = "target",
        min_scale: float = 1e-2,
        **kwargs,
    ):
        super().__init__(target=target, min_scale=min_scale)
        self.q_min = 0.25
        self.q_max = 0.75
        assert 0 < self.q_min < self.q_max < 1

    def _compute_loc_scale(self, target_series: pd.Series) -> Tuple[pd.Series, pd.Series]:
        grouped = target_series.groupby(level=ITEMID, sort=False)
        loc = grouped.median()
        lower = grouped.quantile(self.q_min)
        upper = grouped.quantile(self.q_max)
        scale = upper - lower
        return loc, scale


AVAILABLE_SCALERS = {
    "standard": LocalStandardScaler,
    "mean_abs": LocalMeanAbsScaler,
    "min_max": LocalMinMaxScaler,
    "robust": LocalRobustScaler,
}


def get_target_scaler_from_name(
    name: Literal["standard", "mean_abs", "min_max", "robust"], **scaler_kwargs
) -> LocalTargetScaler:
    """Get LocalTargetScaler object from a string."""
    if name not in AVAILABLE_SCALERS:
        raise KeyError(f"Scaler type {name} not supported. Available scalers: {list(AVAILABLE_SCALERS)}")
    return AVAILABLE_SCALERS[name](**scaler_kwargs)
