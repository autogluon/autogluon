import copy
import logging

import pandas as pd
from pandas import DataFrame

from autogluon.common.features.types import R_INT, R_FLOAT, S_BINNED

from .abstract import AbstractFeatureGenerator
from .. import binning
from ..utils import get_smallest_valid_dtype_int

logger = logging.getLogger(__name__)


# TODO: Add more parameters (possibly pass in binning function as an argument for full control)
class BinnedFeatureGenerator(AbstractFeatureGenerator):
    """BinnedFeatureGenerator bins incoming int and float features to num_bins unique int values, maintaining relative rank order."""
    def __init__(self, num_bins=10, **kwargs):
        super().__init__(**kwargs)
        self.num_bins = num_bins

    def _fit_transform(self, X: DataFrame, **kwargs) -> (DataFrame, dict):
        self._bin_map = self._get_bin_map(X=X)
        self._astype_map = {feature: get_smallest_valid_dtype_int(min_val=0, max_val=len(bin_index)) for feature, bin_index in self._bin_map.items()}
        X_out = self._transform(X)
        type_group_map_special = copy.deepcopy(self.feature_metadata_in.type_group_map_special)
        type_group_map_special[S_BINNED] += list(X_out.columns)
        return X_out, type_group_map_special

    def _transform(self, X: DataFrame) -> DataFrame:
        return self._transform_bin(X)

    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return dict(valid_raw_types=[R_INT, R_FLOAT])

    def _get_bin_map(self, X: DataFrame) -> dict:
        return binning.generate_bins(X, list(X.columns), ideal_bins=self.num_bins)

    def _transform_bin(self, X: DataFrame):
        X_out = dict()
        for column in self._bin_map:
            X_out[column] = binning.bin_column(series=X[column], bins=self._bin_map[column], dtype=self._astype_map[column])
        X_out = pd.DataFrame(X_out, index=X.index)
        return X_out

    def _remove_features_in(self, features: list):
        super()._remove_features_in(features)
        if self._bin_map:
            for feature in features:
                if feature in self._bin_map:
                    self._bin_map.pop(feature)
        if self._astype_map:
            for feature in features:
                if feature in self._astype_map:
                    self._astype_map.pop(feature)

    def _more_tags(self):
        return {'feature_interactions': False}
