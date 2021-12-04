import logging

import numpy as np
from pandas import DataFrame, RangeIndex

from autogluon.common.features.types import R_CATEGORY, R_INT

from . import AbstractFeatureGenerator
from ..utils import clip_and_astype

logger = logging.getLogger(__name__)


class CategoryMemoryMinimizeFeatureGenerator(AbstractFeatureGenerator):
    """
    Minimizes memory usage of category features by converting the category values to monotonically increasing int values.
    This is important for category features with string values which can take up significant memory despite the string information not being used downstream.
    """
    def _fit_transform(self, X: DataFrame, **kwargs) -> (DataFrame, dict):
        self._category_maps = self._get_category_map(X=X)

        X_out = self._transform(X)
        return X_out, self.feature_metadata_in.type_group_map_special

    def _transform(self, X: DataFrame) -> DataFrame:
        return self._minimize_categorical_memory_usage(X)

    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return dict(valid_raw_types=[R_CATEGORY])

    def _get_category_map(self, X: DataFrame) -> dict:
        category_maps = {}
        for column in X:
            old_categories = list(X[column].cat.categories.values)
            new_categories = RangeIndex(len(old_categories))  # Memory optimal categories
            category_maps[column] = new_categories
        return category_maps

    def _minimize_categorical_memory_usage(self, X: DataFrame):
        if self._category_maps:
            X_renamed = dict()
            for column in self._category_maps:
                # rename_categories(inplace=True) is faster but it is deprecated as of pandas 1.3.0
                X_renamed[column] = X[column].cat.rename_categories(self._category_maps[column])
            X = DataFrame(X_renamed)
        return X

    def _remove_features_in(self, features: list):
        super()._remove_features_in(features)
        if self._category_maps:
            for feature in features:
                if feature in self._category_maps:
                    self._category_maps.pop(feature)

    def _more_tags(self):
        return {'feature_interactions': False}


# TODO: What about nulls / unknowns?
class NumericMemoryMinimizeFeatureGenerator(AbstractFeatureGenerator):
    """
    Clips and converts dtype of int features to minimize memory usage.

    dtype_out : np.dtype, default np.uint8
        dtype to clip and convert features to.
        Clipping will automatically use the correct min and max values for the dtype provided.
    **kwargs :
        Refer to :class:`AbstractFeatureGenerator` documentation for details on valid key word arguments.
    """
    def __init__(self, dtype_out=np.uint8, **kwargs):
        super().__init__(**kwargs)
        self.dtype_out, self._clip_min, self._clip_max = self._get_dtype_clip_args(dtype_out)

    def _fit_transform(self, X: DataFrame, **kwargs) -> (DataFrame, dict):
        X_out = self._transform(X)
        return X_out, self.feature_metadata_in.type_group_map_special

    def _transform(self, X):
        return self._minimize_numeric_memory_usage(X)

    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return dict(valid_raw_types=[R_INT])

    @staticmethod
    def _get_dtype_clip_args(dtype) -> (np.dtype, int, int):
        try:
            dtype_info = np.iinfo(dtype)
        except ValueError:
            dtype_info = np.finfo(dtype)
        return dtype_info.dtype, dtype_info.min, dtype_info.max

    def _minimize_numeric_memory_usage(self, X: DataFrame):
        return clip_and_astype(df=X, clip_min=self._clip_min, clip_max=self._clip_max, dtype=self.dtype_out)

    def _more_tags(self):
        return {'feature_interactions': False}
