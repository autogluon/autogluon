import copy
import logging
import math
from collections import defaultdict

import numpy as np
import pandas as pd
import psutil
from pandas import DataFrame, Series

from .feature_metadata import FeatureMetadata
from .generators.dummy import DummyFeatureGenerator
from .generators.bulk import BulkFeatureGenerator
from .types import get_type_map_raw, get_type_map_real, get_type_group_map_special
from ..utils.decorators import calculate_time

logger = logging.getLogger(__name__)


# TODO: Add feature of # of observation counts to high cardinality categorical features
# TODO: Use code from problem type detection for column types! Ints/Floats could be Categorical through this method! Maybe try both?
class AbstractPipelineFeatureGenerator(BulkFeatureGenerator):
    def __init__(self, pre_drop_useless=True, **kwargs):
        super().__init__(pre_drop_useless=pre_drop_useless, **kwargs)

        self._feature_metadata_in_real: FeatureMetadata = None  # FeatureMetadata object based on the original input features real dtypes (will contain dtypes such as 'int16' and 'float32' instead of 'int' and 'float').

        self._is_dummy = False  # If True, returns a single dummy feature as output. Occurs if fit with no useful features.

        self._useless_features_in: list = None

        self.pre_memory_usage = None
        self.pre_memory_usage_per_row = None
        self.post_memory_usage = None
        self.post_memory_usage_per_row = None

    def _preprocess(self, X: DataFrame):
        for column in X.columns:
            if self.feature_metadata_in.type_map_raw[column] == 'object':
                X[column].fillna('', inplace=True)
            else:
                X[column].fillna(np.nan, inplace=True)
        return X

    @calculate_time
    def fit_transform(self, X: DataFrame, y=None, feature_metadata_in: FeatureMetadata = None, **kwargs) -> DataFrame:
        X_index = copy.deepcopy(X.index)
        X = X.reset_index(drop=True)  # TODO: Theoretically inplace=True avoids data copy, but can lead to altering of original DataFrame outside of method context.
        if y is not None and isinstance(y, Series):
            y = y.reset_index(drop=True)  # TODO: this assumes y and X had matching indices prior
        X.columns = X.columns.astype(str)  # Ensure all column names are strings

        X_len = len(X)
        self.pre_memory_usage = self._get_approximate_df_mem_usage(X, sample_ratio=0.2).sum()
        self.pre_memory_usage_per_row = self.pre_memory_usage / X_len
        available_mem = psutil.virtual_memory().available
        pre_memory_usage_percent = self.pre_memory_usage / (available_mem + self.pre_memory_usage)
        logger.log(20, f'Available Memory:                    {(round((self.pre_memory_usage + available_mem) / 1e6, 2))} MB')
        logger.log(20, f'Train Data (Original)  Memory Usage: {round(self.pre_memory_usage / 1e6, 2)} MB ({round(pre_memory_usage_percent * 100, 1)}% of available memory)')
        if pre_memory_usage_percent > 0.05:
            logger.warning(f'Warning: Data size prior to feature transformation consumes {round(pre_memory_usage_percent * 100, 1)}% of available memory. Consider increasing memory or subsampling the data to avoid instability.')

        X_out = super().fit_transform(X=X, y=y, feature_metadata_in=feature_metadata_in, **kwargs)
        X_out.index = X_index

        self.post_memory_usage = self._get_approximate_df_mem_usage(X_out, sample_ratio=0.2).sum()
        self.post_memory_usage_per_row = self.post_memory_usage / X_len

        available_mem = psutil.virtual_memory().available
        post_memory_usage_percent = self.post_memory_usage / (available_mem + self.post_memory_usage + self.pre_memory_usage)
        logger.log(20, f'Train Data (Processed) Memory Usage: {round(self.post_memory_usage / 1e6, 2)} MB ({round(post_memory_usage_percent * 100, 1)}% of available memory)')
        if post_memory_usage_percent > 0.15:
            logger.warning(f'Warning: Data size post feature transformation consumes {round(post_memory_usage_percent*100, 1)}% of available memory. Consider increasing memory or subsampling the data to avoid instability.')

        logger.log(20, 'Feature Generator processed %s data points with %s features' % (X_len, len(self.features_out)))
        self.print_feature_metadata_info()

        return X_out

    # TODO: Save this to disk and remove from memory if large categoricals!
    def _fit_transform(self, X: DataFrame, y=None, **kwargs):
        X = self._preprocess(X)
        X_out, type_group_map_special = super()._fit_transform(X=X, y=y)
        X_out, type_group_map_special = self._fit_transform_custom(X_out=X_out, type_group_map_special=type_group_map_special, y=y)

        if len(list(X_out.columns)) == 0:
            self._is_dummy = True
            logger.warning(f'WARNING: No useful features were detected in the data! AutoGluon will train using 0 features, and will always predict the same value. Ensure that you are passing the correct data to AutoGluon!')
            dummy_generator = DummyFeatureGenerator()
            X_out = dummy_generator.fit_transform(X=X_out)
            type_group_map_special = copy.deepcopy(dummy_generator.feature_metadata.type_group_map_special)
            self.generators = [dummy_generator]
            self.post_generators = []

        return X_out, type_group_map_special

    def _fit_transform_custom(self, X_out: DataFrame, type_group_map_special: dict, y=None) -> (DataFrame, dict):
        feature_metadata = FeatureMetadata(type_map_raw=get_type_map_raw(X_out), type_group_map_special=type_group_map_special)

        features_to_remove_post = self._get_features_to_remove_post(X_out, feature_metadata)
        X_out.drop(features_to_remove_post, axis=1, inplace=True)
        feature_metadata.remove_features(features=features_to_remove_post, inplace=True)

        return X_out, feature_metadata.type_group_map_special

    def transform(self, X: DataFrame) -> DataFrame:
        X_index = copy.deepcopy(X.index)
        X = X.reset_index(drop=True)
        X.columns = X.columns.astype(str)  # Ensure all column names are strings
        X_out = super().transform(X=X)
        X_out.index = X_index
        return X_out

    def _transform(self, X: DataFrame) -> DataFrame:
        int_features = self.feature_metadata_in.type_group_map_raw['int']
        if int_features:
            null_count = X[int_features].isnull().sum()
            with_null = null_count[null_count != 0]
            # If int feature contains null during inference but not during fit.
            if len(with_null) > 0:
                # TODO: Consider imputing to mode? This is tricky because training data had no missing values.
                # TODO: Add unit test for this situation, to confirm it is handled properly.
                with_null_features = list(with_null.index)
                logger.warning(f'WARNING: Int features contain null values at inference time! Imputing nulls to 0. To avoid this, pass the features as floats during fit!')
                logger.warning(f'WARNING: Int features with nulls: {with_null_features}')
                X[with_null_features] = X[with_null_features].fillna(0)
        if self._feature_metadata_in_real.type_map_raw:
            # TODO: Confirm this works with sparse and other feature types!
            X = X.astype(self._feature_metadata_in_real.type_map_raw)

        X = self._preprocess(X)
        X_out = super()._transform(X)
        X_out = X_out[self._features_out_internal]

        return X_out

    def _infer_features_in_full(self, X: DataFrame, y: Series = None, feature_metadata_in: FeatureMetadata = None):
        super()._infer_features_in_full(X=X, y=y, feature_metadata_in=feature_metadata_in)
        type_map_real = get_type_map_real(X[self.feature_metadata_in.get_features()])
        self._feature_metadata_in_real = FeatureMetadata(type_map_raw=type_map_real, type_group_map_special=self.feature_metadata_in.type_group_map_raw)

    def _remove_features_in(self, features):
        super()._remove_features_in(features)
        self._feature_metadata_in_real = self._feature_metadata_in_real.remove_features(features=features)

    @staticmethod
    def _get_features_to_remove_post(X: DataFrame, feature_metadata: FeatureMetadata) -> list:
        features_to_remove_post = []
        X_len = len(X)
        # TODO: Consider making 0.99 a parameter to FeatureGenerator
        max_unique_value_count = X_len * 0.99
        for column in X:
            unique_value_count = len(X[column].unique())
            if unique_value_count == 1:
                features_to_remove_post.append(column)
            elif column in feature_metadata.type_group_map_raw['category'] and (unique_value_count > max_unique_value_count):
                features_to_remove_post.append(column)
        return features_to_remove_post

    @staticmethod
    def _get_type_group_map_special(X: DataFrame) -> defaultdict:
        return get_type_group_map_special(X)

    # TODO: Move this outside of here
    @staticmethod
    def _get_approximate_df_mem_usage(df, sample_ratio=0.2):
        if sample_ratio >= 1:
            return df.memory_usage(deep=True)
        else:
            num_rows = len(df)
            num_rows_sample = math.ceil(sample_ratio * num_rows)
            sample_ratio = num_rows_sample / num_rows
            exact_dtypes = ['int', 'float']
            category_dtypes = ['category']
            dtypes_raw = get_type_map_raw(df)
            columns_category = [column for column in df if dtypes_raw[column] in category_dtypes]
            columns_inexact = [column for column in df if dtypes_raw[column] not in exact_dtypes + category_dtypes]
            memory_usage = df.memory_usage()
            if columns_category:
                for column in columns_category:
                    num_categories = len(df[column].cat.categories)
                    num_categories_sample = math.ceil(sample_ratio * num_categories)
                    sample_ratio_cat = num_categories_sample / num_categories
                    memory_usage[column] = df[column].cat.codes.dtype.itemsize * num_rows + df[column].cat.categories[:num_categories_sample].memory_usage(deep=True) / sample_ratio_cat
            if columns_inexact:
                memory_usage_inexact = df[columns_inexact].head(num_rows_sample).memory_usage(deep=True)[columns_inexact] / sample_ratio
                memory_usage = memory_usage_inexact.combine_first(memory_usage)
            return memory_usage

    def print_feature_metadata_info(self):
        if self._useless_features_in:
            logger.log(20, f'Useless Original Features (Count: {len(self._useless_features_in)}): {list(self._useless_features_in)}')
            logger.log(20, f'\tThese features carry no predictive signal and should be manually investigated.')  # TODO: What about features with 1 unique value but also np.nan?
            logger.log(20, f'\tThese features do not need to be present at inference time for this FeatureGenerator.')
        if self._feature_metadata_in_unused.get_features():
            logger.log(20, f'Unused Original Features (Count: {len(self._feature_metadata_in_unused.get_features())}): {self._feature_metadata_in_unused.get_features()}')
            logger.log(20, f'\tThese features were not valid input to any of the feature generators. Add a feature generator compatible with these features to utilize them.')
            logger.log(20, f'\tThese features do not need to be present at inference time for this FeatureGenerator.')
            self._feature_metadata_in_unused.print_feature_metadata_full('\t')
        logger.log(20, 'Original Features (exact raw dtype, raw dtype):')
        self._feature_metadata_in_real.print_feature_metadata_full('\t', print_only_one_special=True)
        super().print_feature_metadata_info()
