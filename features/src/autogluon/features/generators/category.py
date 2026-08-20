import copy
import logging

import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas.api.types import CategoricalDtype

from autogluon.common.features.types import (
    R_BOOL,
    R_CATEGORY,
    R_OBJECT,
    S_DATETIME_AS_OBJECT,
    S_IMAGE_BYTEARRAY,
    S_IMAGE_PATH,
    S_TEXT,
    S_TEXT_AS_CATEGORY,
)

from .abstract import AbstractFeatureGenerator
from .memory_minimize import CategoryMemoryMinimizeFeatureGenerator

logger = logging.getLogger(__name__)


# TODO: Add hashing trick if minimize_memory=True to avoid storing full original mapping
# TODO: fill_nan add additional options: group_rares, possibly percentile based
class CategoryFeatureGenerator(AbstractFeatureGenerator):
    """
    CategoryFeatureGenerator is used to convert object types to category types, as well as remove rare categories and optimize memory usage.
    After fitting, previously unseen categories during transform are treated as missing values.

    Parameters
    ----------
    stateful_categories : bool, default True
        If True, categories from training are applied to transformed data, and any unknown categories from input data will be treated as missing values.
        It is recommended to keep this value as True to avoid strange downstream behavior.
    minimize_memory : bool, default True
        If True, minimizes category memory usage by converting all category values to sequential integers.
        This replaces any string data present in the categories but does not alter the behavior of models when using the category as a feature so long
        as the original string values are not required downstream.
        It is recommended to keep this value as True to dramatically reduce memory usage with no cost to accuracy.
    cat_order : str, default 'original'
        Determines the order in which categories are stored.
        This is important when minimize_memory is True, as the order will determine which categories are converted to which integer values.
        Valid values:
            'original' : Keep the original order. If the feature was originally an object, this is equivalent to 'alphanumeric'.
            'alphanumeric' : Sort the categories alphanumerically.
            'count' : Sort the categories by frequency (Least frequent in front with code of 0)
    minimum_cat_count : int, default 2
        The minimum number of occurrences a category must have in the training data to avoid being considered a rare category.
        Rare categories are removed and treated as missing values.
        If None, no minimum count is required. This includes categories that never occur in the data but are present in the category object
        as possible categories.
    maximum_num_cat : int, default None
        The maximum amount of categories that can be considered non-rare.
        Sorted by occurrence count, up to the N highest count categories will be kept if maximum_num_cat=N. All others will be considered rare categories.
    fillna : str, default None
        The method used to handle missing values. Only valid if stateful_categories=True.
        Missing values include the values that were originally NaN and values converted to NaN from other parameters such as minimum_cat_count.
        Valid values:
            None : Keep missing values as is. They will appear as NaN and have no category assigned to them.
            'mode' : Set missing values to the most frequent category in their feature.
            'rare' : Group missing values into a dedicated rare category.
                A new category is added to the feature (`max(categories) + 1` for integer categories, `'_NaN_'` otherwise)
                and all missing values are assigned to it. Because categories dropped by `minimum_cat_count` and
                `maximum_num_cat` have already become missing values at this point, they are grouped into the same category.
                If fewer than `minimum_cat_count` values are missing, no new category is added and missing values are
                instead assigned to the rarest surviving category, so as not to create a category below the minimum count.
                If no category satisfies `minimum_cat_count` and `maximum_num_cat`, missing values are left as NaN.
    **kwargs :
        Refer to :class:`AbstractFeatureGenerator` documentation for details on valid key word arguments.
    """

    def __init__(
        self,
        stateful_categories=True,
        minimize_memory=True,
        cat_order="original",
        minimum_cat_count: int = 2,
        maximum_num_cat: int = None,
        fillna: str = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._stateful_categories = stateful_categories
        if minimum_cat_count is not None and minimum_cat_count < 1:
            minimum_cat_count = None
        if cat_order not in ["original", "alphanumeric", "count"]:
            raise ValueError(f"cat_order must be one of {['original', 'alphanumeric', 'count']}, but was: {cat_order}")
        self.cat_order = cat_order
        self._minimum_cat_count = minimum_cat_count
        self._maximum_num_cat = maximum_num_cat
        self.category_map = None
        valid_fillna_values = (None, "mode", "rare")
        if fillna not in valid_fillna_values:
            raise ValueError(f"fillna={fillna} is not a valid value. Valid values: {list(valid_fillna_values)}")
        self._fillna = fillna
        self._fillna_flag = self._fillna is not None
        self._fillna_map = None

        if minimize_memory:
            self._post_generators = [CategoryMemoryMinimizeFeatureGenerator()] + self._post_generators

    def _fit_transform(self, X: DataFrame, **kwargs) -> (DataFrame, dict):
        if self._stateful_categories:
            X_out, self.category_map, self._fillna_map = self._generate_category_map(X=X)
            if self._fillna_map is not None:
                for column in self._fillna_map:
                    X_out[column] = X_out[column].fillna(self._fillna_map[column])
        else:
            X_out = self._transform(X)
        feature_metadata_out_type_group_map_special = copy.deepcopy(self.feature_metadata_in.type_group_map_special)
        if S_TEXT in feature_metadata_out_type_group_map_special:
            text_features = feature_metadata_out_type_group_map_special.pop(S_TEXT)
            feature_metadata_out_type_group_map_special[S_TEXT_AS_CATEGORY] += [
                feature
                for feature in text_features
                if feature not in feature_metadata_out_type_group_map_special[S_TEXT_AS_CATEGORY]
            ]
        return X_out, feature_metadata_out_type_group_map_special

    def _transform(self, X: DataFrame) -> DataFrame:
        return self._generate_features_category(X)

    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return dict(
            valid_raw_types=[R_OBJECT, R_CATEGORY, R_BOOL],
            invalid_special_types=[S_DATETIME_AS_OBJECT, S_IMAGE_PATH, S_IMAGE_BYTEARRAY],
        )

    def _generate_features_category(self, X: DataFrame) -> DataFrame:
        if self.features_in:
            X_category = dict()
            if self.category_map is not None:
                for column, column_map in self.category_map.items():
                    X_category[column] = pd.Categorical(X[column], categories=column_map)
                X_category = DataFrame(X_category, index=X.index)
                if self._fillna_map is not None:
                    for column, column_map in self._fillna_map.items():
                        X_category[column] = X_category[column].fillna(column_map)
        else:
            X_category = DataFrame(index=X.index)
        return X_category

    def _generate_category_map(self, X: DataFrame) -> (DataFrame, dict):
        if self.features_in:
            fill_nan_map = dict()
            category_map = dict()
            X_category = X.astype("category")
            for column in X_category:
                rank = X_category[column].value_counts().sort_index().sort_values(ascending=True, kind="stable")
                if self._minimum_cat_count is not None:
                    rank = rank[rank >= self._minimum_cat_count]
                if self._maximum_num_cat is not None:
                    rank = rank[-self._maximum_num_cat :]
                if (
                    self.cat_order == "count"
                    or self._minimum_cat_count is not None
                    or self._maximum_num_cat is not None
                ):
                    category_list = list(rank.index)  # category_list in 'count' order
                    if len(category_list) > 1:
                        if self.cat_order == "original":
                            original_cat_order = list(X_category[column].cat.categories)
                            set_category_list = set(category_list)
                            category_list = [cat for cat in original_cat_order if cat in set_category_list]
                        elif self.cat_order == "alphanumeric":
                            category_list.sort()
                    X_category[column] = X_category[column].astype(
                        CategoricalDtype(categories=category_list)
                    )  # TODO: Remove columns if all NaN after this?
                    X_category[column] = X_category[column].cat.reorder_categories(category_list)
                elif self.cat_order == "alphanumeric":
                    category_list = list(X_category[column].cat.categories)
                    category_list.sort()
                    X_category[column] = X_category[column].astype(CategoricalDtype(categories=category_list))
                    X_category[column] = X_category[column].cat.reorder_categories(category_list)
                if self._fillna_flag and len(rank) > 0:
                    if self._fillna == "mode":
                        fill_nan_map[column] = rank.index[-1]
                    elif self._fillna == "rare":
                        nan_count = X_category[column].isna().sum()
                        if self._minimum_cat_count is None or (nan_count >= self._minimum_cat_count):
                            # Add a dedicated category for the missing values. This also absorbs the
                            # values that `minimum_cat_count`/`maximum_num_cat` turned into NaN above.
                            category_list = list(X_category[column].cat.categories)
                            # `isinstance(c, int)` alone would miss numpy integers, which is what a
                            # category index is normally backed by, sending integer columns down the
                            # string branch and leaving them with mixed-type categories.
                            if all(isinstance(c, (int, np.integer)) for c in category_list):
                                fillna_category = max(rank.index) + 1
                            else:
                                fillna_category = "_NaN_"
                            category_list.append(fillna_category)
                            X_category[column] = X_category[column].astype(CategoricalDtype(categories=category_list))
                            X_category[column] = X_category[column].cat.reorder_categories(category_list)
                            fill_nan_map[column] = fillna_category
                        else:
                            # Too few missing values to justify their own category: fold them into the
                            # rarest surviving one rather than create a category below the minimum count.
                            fill_nan_map[column] = rank.index[0]
                category_map[column] = copy.deepcopy(X_category[column].cat.categories)
            if not self._fillna_flag:
                fill_nan_map = None
            return X_category, category_map, fill_nan_map
        else:
            return DataFrame(index=X.index), None, None

    def _remove_features_in(self, features: list):
        super()._remove_features_in(features)
        if self.category_map:
            for feature in features:
                if feature in self.category_map:
                    self.category_map.pop(feature)
        if self._fillna_map:
            for feature in features:
                if feature in self._fillna_map:
                    self._fillna_map.pop(feature)

    def _more_tags(self):
        return {"feature_interactions": False}
