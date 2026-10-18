from typing import Any, Iterator, Type

import numpy as np
import pandas as pd
from gluonts.dataset.common import Dataset as GluonTSDataset
from gluonts.dataset.field_names import FieldName

from autogluon.timeseries.dataset import TimeSeriesDataFrame
from autogluon.timeseries.utils.datetime import norm_freq_str


class SimpleGluonTSDataset(GluonTSDataset):
    """Wrapper for TimeSeriesDataFrame that is compatible with the GluonTS Dataset API."""

    def __init__(
        self,
        target_df: TimeSeriesDataFrame,
        freq: str,
        target_column: str = "target",
        feat_static_cat: np.ndarray | None = None,
        feat_static_real: np.ndarray | None = None,
        feat_dynamic_cat: np.ndarray | None = None,
        feat_dynamic_real: np.ndarray | None = None,
        past_feat_dynamic_cat: np.ndarray | None = None,
        past_feat_dynamic_real: np.ndarray | None = None,
        includes_future: bool = False,
        prediction_length: int | None = None,
    ):
        assert target_df is not None
        # Convert TimeSeriesDataFrame to pd.Series for faster processing
        self.target_array = target_df[target_column].to_numpy(np.float32)
        self.feat_static_cat = self._astype(feat_static_cat, dtype=np.int64)
        self.feat_static_real = self._astype(feat_static_real, dtype=np.float32)
        self.feat_dynamic_cat = self._astype(feat_dynamic_cat, dtype=np.int64)
        self.feat_dynamic_real = self._astype(feat_dynamic_real, dtype=np.float32)
        self.past_feat_dynamic_cat = self._astype(past_feat_dynamic_cat, dtype=np.int64)
        self.past_feat_dynamic_real = self._astype(past_feat_dynamic_real, dtype=np.float32)
        self.freq = self._get_freq_for_period(freq)

        # Necessary to compute indptr for known_covariates at prediction time
        self.includes_future = includes_future
        self.prediction_length = prediction_length

        # Replace inefficient groupby ITEMID with indptr that stores start:end of each time series
        self.item_ids = target_df.item_ids
        self.indptr = target_df.get_indptr()
        self.start_timestamps = target_df.index[self.indptr[:-1]].to_frame(index=False)[TimeSeriesDataFrame.TIMESTAMP]
        assert len(self.item_ids) == len(self.start_timestamps)

    @staticmethod
    def _astype(array: np.ndarray | None, dtype: Type[np.generic]) -> np.ndarray | None:
        if array is None:
            return None
        else:
            return array.astype(dtype)

    @staticmethod
    def _get_freq_for_period(freq: str) -> str:
        """Convert freq to format compatible with pd.Period.

        For example, ME freq must be converted to M when creating a pd.Period.
        """
        offset = pd.tseries.frequencies.to_offset(freq)
        assert offset is not None
        freq_name = norm_freq_str(offset)
        if freq_name == "SME":
            # Replace unsupported frequency "SME" with "2W"
            return "2W"
        elif freq_name == "bh":
            # Replace unsupported frequency "bh" with dummy value "Y"
            return "Y"
        else:
            freq_name_for_period = {"YE": "Y", "QE": "Q", "ME": "M"}.get(freq_name, freq_name)
            return f"{offset.n}{freq_name_for_period}"

    def __len__(self):
        return len(self.indptr) - 1  # noqa

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for j in range(len(self.indptr) - 1):
            start_idx = self.indptr[j]
            end_idx = self.indptr[j + 1]
            # GluonTS expects item_id to be a string
            ts = {
                FieldName.ITEM_ID: str(self.item_ids[j]),
                FieldName.START: pd.Period(self.start_timestamps.iloc[j], freq=self.freq),
                FieldName.TARGET: self.target_array[start_idx:end_idx],
            }
            if self.feat_static_cat is not None:
                ts[FieldName.FEAT_STATIC_CAT] = self.feat_static_cat[j]
            if self.feat_static_real is not None:
                ts[FieldName.FEAT_STATIC_REAL] = self.feat_static_real[j]
            if self.past_feat_dynamic_cat is not None:
                ts[FieldName.PAST_FEAT_DYNAMIC_CAT] = self.past_feat_dynamic_cat[start_idx:end_idx].T
            if self.past_feat_dynamic_real is not None:
                ts[FieldName.PAST_FEAT_DYNAMIC_REAL] = self.past_feat_dynamic_real[start_idx:end_idx].T

            # Dynamic features that may extend into the future
            if self.includes_future:
                assert self.prediction_length is not None, (
                    "Prediction length must be provided if includes_future is True"
                )
                start_idx = start_idx + j * self.prediction_length
                end_idx = end_idx + (j + 1) * self.prediction_length
            if self.feat_dynamic_cat is not None:
                ts[FieldName.FEAT_DYNAMIC_CAT] = self.feat_dynamic_cat[start_idx:end_idx].T
            if self.feat_dynamic_real is not None:
                ts[FieldName.FEAT_DYNAMIC_REAL] = self.feat_dynamic_real[start_idx:end_idx].T
            yield ts
