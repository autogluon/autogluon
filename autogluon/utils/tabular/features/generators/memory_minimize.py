import logging

import numpy as np
from pandas import DataFrame

from . import AbstractFeatureGenerator
from ..utils import clip_and_astype

logger = logging.getLogger(__name__)


class CategoryMemoryMinimizeFeatureGenerator(AbstractFeatureGenerator):
    def __init__(self, inplace=False, **kwargs):
        super().__init__(**kwargs)
        self.inplace = inplace

    def _fit_transform(self, X: DataFrame, **kwargs) -> (DataFrame, dict):
        self._category_maps = self._get_category_map(X=X)

        X_out = self._transform(X)
        return X_out, self.feature_metadata_in.type_group_map_special

    def _transform(self, X):
        return self._minimize_categorical_memory_usage(X)

    def _infer_features_in(self, X, y=None) -> list:
        category_features = self.feature_metadata_in.type_group_map_raw['category']
        return category_features

    def _get_category_map(self, X: DataFrame) -> dict:
        category_maps = {}
        for column in X:
            old_categories = list(X[column].cat.categories.values)
            new_categories = list(range(len(old_categories)))
            category_maps[column] = {old_code: new_code for old_code, new_code in zip(old_categories, new_categories)}
        return category_maps

    # TODO: Compress further, uint16, etc.
    def _minimize_categorical_memory_usage(self, X: DataFrame):
        if self._category_maps:
            if not self.inplace:
                X = X.copy(deep=True)
            for column in self._category_maps:
                X[column].cat.rename_categories(self._category_maps[column], inplace=True)
        return X


# TODO: What about nulls / unknowns?
class NumericMemoryMinimizeFeatureGenerator(AbstractFeatureGenerator):
    def __init__(self, dtype_out=np.uint8, **kwargs):
        super().__init__(**kwargs)
        self.dtype_out, self._clip_min, self._clip_max = self._get_dtype_clip_args(dtype_out)

    def _fit_transform(self, X: DataFrame, **kwargs) -> (DataFrame, dict):
        X_out = self._transform(X)
        return X_out, self.feature_metadata_in.type_group_map_special

    def _transform(self, X):
        return self._minimize_numeric_memory_usage(X)

    @staticmethod
    def _get_dtype_clip_args(dtype) -> (np.dtype, int, int):
        try:
            dtype_info = np.iinfo(dtype)
        except ValueError:
            dtype_info = np.finfo(dtype)
        return dtype_info.dtype, dtype_info.min, dtype_info.max

    def _infer_features_in(self, X, y=None) -> list:
        numeric_features = [feature for feature in self.feature_metadata_in.get_features() if self.feature_metadata_in.get_feature_type_raw(feature) in ['int']]  # TODO: floats?
        return numeric_features

    def _minimize_numeric_memory_usage(self, X: DataFrame):
        return clip_and_astype(df=X, clip_min=self._clip_min, clip_max=self._clip_max, dtype=self.dtype_out)
