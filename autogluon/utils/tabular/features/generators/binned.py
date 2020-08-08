import copy
import logging

from pandas import DataFrame

from .abstract import AbstractFeatureGenerator
from .. import binning

logger = logging.getLogger(__name__)


class BinnedFeatureGenerator(AbstractFeatureGenerator):
    def _fit_transform(self, X: DataFrame, **kwargs) -> (DataFrame, dict):
        self._bin_map = self._get_bin_map(X=X)
        X_out = self._transform(X)
        type_group_map_special = copy.deepcopy(self.feature_metadata_in.type_group_map_special)
        type_group_map_special['binned'] += list(X_out.columns)
        return X_out, type_group_map_special

    def _transform(self, X):
        return self._transform_bin(X)

    def _infer_features_in(self, X, y=None) -> list:
        features_to_bin = self.feature_metadata_in.type_group_map_special['text_special']
        return features_to_bin

    def _get_bin_map(self, X: DataFrame) -> dict:
        return binning.generate_bins(X, list(X.columns))

    # TODO: Compress further, uint16, etc.
    def _transform_bin(self, X: DataFrame):
        if self._bin_map:
            X = X.copy()
            for column in self._bin_map:
                X[column] = binning.bin_column(series=X[column], mapping=self._bin_map[column])
        return X
