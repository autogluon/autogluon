import copy
import logging
import math
from collections import defaultdict

import numpy as np
import pandas as pd
import psutil
from pandas import DataFrame, Series
from pandas.api.types import CategoricalDtype

from . import binning
from .feature_metadata import FeatureMetadata
from .generators.abstract import AbstractFeatureGenerator
from .generators.dummy import DummyFeatureGenerator
from .types import get_type_map_real, get_type_group_map_special
from .utils import check_if_useless_feature, clip_and_astype
from ..utils.decorators import calculate_time

logger = logging.getLogger(__name__)


# TODO: Add feature of # of observation counts to high cardinality categorical features
# TODO: Use code from problem type detection for column types! Ints/Floats could be Categorical through this method! Maybe try both?
class AbstractPipelineFeatureGenerator(AbstractFeatureGenerator):
    def __init__(self, generators):
        super().__init__()

        self.generators = generators

        self._feature_metadata_in_unused: FeatureMetadata = None  # FeatureMetadata object based on the original input features that were unused by any feature generator.
        self._feature_metadata_in_real: FeatureMetadata = None  # FeatureMetadata object based on the original input features real dtypes (will contain dtypes such as 'int16' and 'float32' instead of 'int' and 'float').

        self._is_dummy = False  # If True, returns a single dummy feature as output. Occurs if fit with no useful features.

        self._features_category_code_map = defaultdict()  # Categorical features original value -> category code mapping
        self._features_binned_map = defaultdict()  # Binned features original value ranges -> binned category code mapping
        self._minimize_categorical_memory_usage_flag = True

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
    def fit_transform(self, X: DataFrame, y=None, banned_features=None, fix_categoricals=False, drop_duplicates=False, **kwargs) -> DataFrame:
        X_index = copy.deepcopy(X.index)
        X = X.reset_index(drop=True)  # TODO: Theoretically inplace=True avoids data copy, but can lead to altering of original DataFrame outside of method context.
        X.columns = X.columns.astype(str)  # Ensure all column names are strings
        if banned_features:
            features_to_remove = [str(feature) for feature in banned_features]
        else:
            features_to_remove = []
        self._useless_features_in = [feature for feature in self._get_useless_features(X) if feature not in features_to_remove]
        features_to_remove += self._useless_features_in
        X.drop(features_to_remove, axis=1, errors='ignore', inplace=True)

        X_features = super().fit_transform(X=X, y=y, fix_categoricals=fix_categoricals, drop_duplicates=drop_duplicates, **kwargs)
        X_features.index = X_index

        logger.log(20, 'Feature Generator processed %s data points with %s features' % (len(X_features), len(self.features_out)))
        self.print_feature_metadata_info()

        return X_features

    # TODO: Save this to disk and remove from memory if large categoricals!
    @calculate_time
    def _fit_transform(self, X: DataFrame, y=None, fix_categoricals=False, drop_duplicates=False, **kwargs):
        X_len = len(X)
        self.pre_memory_usage = self._get_approximate_df_mem_usage(X, sample_ratio=0.2).sum()
        self.pre_memory_usage_per_row = self.pre_memory_usage / X_len
        available_mem = psutil.virtual_memory().available
        pre_memory_usage_percent = self.pre_memory_usage / (available_mem + self.pre_memory_usage)
        logger.log(20, f'Available Memory:                    {(round((self.pre_memory_usage + available_mem)/1e6, 2))} MB')
        logger.log(20, f'Train Data (Original)  Memory Usage: {round(self.pre_memory_usage/1e6, 2)} MB ({round(pre_memory_usage_percent*100, 1)}% of available memory)')
        if pre_memory_usage_percent > 0.05:
            logger.warning(f'Warning: Data size prior to feature transformation consumes {round(pre_memory_usage_percent*100, 1)}% of available memory. Consider increasing memory or subsampling the data to avoid instability.')

        X_features, type_group_map_special = self._fit_transform_custom(X=X, y=y, fix_categoricals=fix_categoricals, drop_duplicates=drop_duplicates)

        if len(list(X_features.columns)) == 0:
            self._is_dummy = True
            logger.warning(f'WARNING: No useful features were detected in the data! AutoGluon will train using 0 features, and will always predict the same value. Ensure that you are passing the correct data to AutoGluon!')
            dummy_generator = DummyFeatureGenerator()
            X_features = dummy_generator.fit_transform(X=X_features)
            type_group_map_special = copy.deepcopy(dummy_generator.feature_metadata.type_group_map_special)  # TODO: Remove
            self.generators = [dummy_generator]

        self.post_memory_usage = self._get_approximate_df_mem_usage(X_features, sample_ratio=0.2).sum()
        self.post_memory_usage_per_row = self.post_memory_usage / X_len

        available_mem = psutil.virtual_memory().available
        post_memory_usage_percent = self.post_memory_usage / (available_mem + self.post_memory_usage + self.pre_memory_usage)
        logger.log(20, f'Train Data (Processed) Memory Usage: {round(self.post_memory_usage / 1e6, 2)} MB ({round(post_memory_usage_percent * 100, 1)}% of available memory)')
        if post_memory_usage_percent > 0.15:
            logger.warning(f'Warning: Data size post feature transformation consumes {round(post_memory_usage_percent*100, 1)}% of available memory. Consider increasing memory or subsampling the data to avoid instability.')

        return X_features, type_group_map_special

    def _fit_transform_custom(self, X: DataFrame, y=None, fix_categoricals=False, drop_duplicates=False) -> (DataFrame, dict):
        # TODO: Add ability for user to specify type_group_map_special as input to fit_transform
        X_features = self._generate_features(X)

        if self.generators:
            feature_metadata_in_generators = FeatureMetadata.join_metadatas([generator.feature_metadata_in for generator in self.generators], allow_shared_raw_features=True)
        else:
            feature_metadata_in_generators = FeatureMetadata(type_map_raw=dict())

        used_features_in = feature_metadata_in_generators.get_features()
        unused_features_in = [feature for feature in self.feature_metadata_in.get_features() if feature not in used_features_in]
        self._feature_metadata_in_unused = self.feature_metadata_in.keep_features(features=unused_features_in)
        self._remove_features_in(features=unused_features_in)

        if self.generators:
            self.feature_metadata = FeatureMetadata.join_metadatas([generator.feature_metadata for generator in self.generators], allow_shared_raw_features=False)
        else:
            self.feature_metadata = FeatureMetadata(type_map_raw=dict())

        # TODO: Remove the need for this
        if self.feature_metadata_in.type_group_map_special['text']:
            self.feature_metadata.type_group_map_special['text_as_category'] += [feature for feature in self.feature_metadata_in.type_group_map_special['text'] if feature in self.feature_metadata.type_group_map_raw['category']]

        features_to_remove_post = self._get_features_to_remove_post(X_features, self.feature_metadata)
        X_features.drop(features_to_remove_post, axis=1, inplace=True)
        self.feature_metadata.remove_features(features=features_to_remove_post, inplace=True)

        feature_names = self.feature_metadata.get_features()

        features_binned = self.feature_metadata.type_group_map_special['text_special']
        self._features_binned_map = binning.generate_bins(X_features, features_binned)
        for column in features_binned:
            X_features[column] = binning.bin_column(series=X_features[column], mapping=self._features_binned_map[column])
        self.feature_metadata.type_group_map_special['binned'] = copy.deepcopy(features_binned)  # TODO: Avoid doing this, instead have it occur in _generate_features naturally

        if fix_categoricals:  # if X_test is not used in fit_transform and the model used is from SKLearn
            X_features = self._fix_categoricals_for_sklearn(X_features=X_features)

        if drop_duplicates:
            X_features = self._drop_duplicate_features(X_features)

        feature_names_post = list(X_features.columns)
        if set(feature_names) != set(feature_names_post):
            features_to_remove_post = [feature for feature in feature_names if feature not in feature_names_post]
            self.feature_metadata.remove_features(features=features_to_remove_post, inplace=True)

        for column in self.feature_metadata.type_group_map_raw['category']:
            self._features_category_code_map[column] = copy.deepcopy(X_features[column].cat.categories)

        X_features = self._minimize_memory_usage(X_features=X_features)

        return X_features, self.feature_metadata.type_group_map_special

    def transform(self, X: DataFrame) -> DataFrame:
        if not self._is_fit:
            raise AssertionError('FeatureGenerator has not yet been fit.')
        X_index = copy.deepcopy(X.index)
        X = X.reset_index(drop=True)
        X_features = self._transform(X=X)
        X_features.index = X_index

        return X_features

    def _transform(self, X: DataFrame) -> DataFrame:
        X.columns = X.columns.astype(str)  # Ensure all column names are strings
        try:
            X = X[self.features_in]
        except KeyError:
            missing_cols = []
            for col in self.features_in:
                if col not in X.columns:
                    missing_cols.append(col)
            raise KeyError(f'{len(missing_cols)} required columns are missing from the provided dataset. Missing columns: {missing_cols}')

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

        X_features = self._generate_features(X)
        for column in self.feature_metadata.type_group_map_special['binned']:
            X_features[column] = binning.bin_column(series=X_features[column], mapping=self._features_binned_map[column])

        X_features = X_features[self.features_out]
        for column in self.feature_metadata.type_group_map_raw['category']:
            X_features[column].cat.set_categories(self._features_category_code_map[column], inplace=True)

        X_features = self._minimize_memory_usage(X_features=X_features)

        return X_features

    # TODO: Add _fit_transform_generators, _transform_generators instead of this
    def _generate_features(self, X: DataFrame) -> DataFrame:
        X = self._preprocess(X)

        feature_df_list = []
        for generator in self.generators:
            if not self._is_fit:
                X_out = generator.fit_transform(X, feature_metadata_in=self.feature_metadata_in)
            else:
                X_out = generator.transform(X)
            feature_df_list.append(X_out)

        if not self._is_fit:
            self.generators = [generator for i, generator in enumerate(self.generators) if feature_df_list[i] is not None and len(feature_df_list[i].columns) > 0]
            feature_df_list = [feature_df for feature_df in feature_df_list if feature_df is not None and len(feature_df.columns) > 0]

        if not feature_df_list:
            X_features = DataFrame(index=X.index)
        elif len(feature_df_list) == 1:
            X_features = feature_df_list[0]
        else:
            X_features = pd.concat(feature_df_list, axis=1, ignore_index=False, copy=False)

        return X_features

    def _infer_features_in_full(self, X: DataFrame, y: Series = None, feature_metadata_in: FeatureMetadata = None):
        super()._infer_features_in_full(X=X, y=y, feature_metadata_in=feature_metadata_in)
        type_map_real = get_type_map_real(X)
        self._feature_metadata_in_real = FeatureMetadata(type_map_raw=type_map_real, type_group_map_special=self.feature_metadata_in.type_group_map_raw)

    def _remove_features_in(self, features):
        self.feature_metadata_in = self.feature_metadata_in.remove_features(features=features)
        self._feature_metadata_in_real = self._feature_metadata_in_real.remove_features(features=features)
        self.features_in = self.feature_metadata_in.get_features()

    def _minimize_memory_usage(self, X_features: DataFrame) -> DataFrame:
        if self._minimize_categorical_memory_usage_flag:
            self._minimize_categorical_memory_usage(X_features=X_features)
        X_features = self._minimize_ngram_memory_usage(X_features=X_features)
        X_features = self._minimize_binned_memory_usage(X_features=X_features)
        return X_features

    def _minimize_ngram_memory_usage(self, X_features: DataFrame) -> DataFrame:
        return clip_and_astype(df=X_features, columns=self.feature_metadata.type_group_map_special['text_ngram'], clip_min=0, clip_max=255, dtype='uint8')

    def _minimize_binned_memory_usage(self, X_features: DataFrame) -> DataFrame:
        return clip_and_astype(df=X_features, columns=self.feature_metadata.type_group_map_special['binned'], clip_min=0, clip_max=255, dtype='uint8')

    # TODO: Compress further, uint16, etc.
    # Performs in-place updates
    def _minimize_categorical_memory_usage(self, X_features: DataFrame):
        for column in self.feature_metadata.type_group_map_raw['category']:
            new_categories = list(range(len(X_features[column].cat.categories.values)))
            X_features[column].cat.rename_categories(new_categories, inplace=True)

    def _fix_categoricals_for_sklearn(self, X_features: DataFrame) -> DataFrame:
        features_to_remove = []
        for column in self.feature_metadata.type_group_map_raw['category']:
            rank = X_features[column].value_counts().sort_values(ascending=True)
            rank = rank[rank >= 3]
            rank = rank.reset_index()
            val_list = list(rank['index'].values)
            if len(val_list) <= 1:
                features_to_remove.append(column)
            else:
                X_features[column] = X_features[column].astype(CategoricalDtype(categories=val_list))
        if features_to_remove:
            X_features.drop(features_to_remove, axis=1, inplace=True)
        return X_features

    def _get_features_to_remove_post(self, X_features: DataFrame, feature_metadata: FeatureMetadata) -> list:
        features_to_remove_post = []
        X_len = len(X_features)
        # TODO: Consider making 0.99 a parameter to FeatureGenerator
        max_unique_value_count = X_len * 0.99
        for column in X_features:
            unique_value_count = len(X_features[column].unique())
            if unique_value_count == 1:
                features_to_remove_post.append(column)
            elif column in feature_metadata.type_group_map_raw['category'] and (unique_value_count > max_unique_value_count):
                features_to_remove_post.append(column)
        return features_to_remove_post

    @staticmethod
    def _get_useless_features(X: DataFrame) -> list:
        useless_features = []
        for column in X:
            if check_if_useless_feature(X[column]):
                useless_features.append(column)
        return useless_features

    @staticmethod
    def _get_type_group_map_special(X: DataFrame) -> defaultdict:
        return get_type_group_map_special(X)

    # TODO: optimize by not considering columns with unique sums/means
    # TODO: Multithread?
    @staticmethod
    def _drop_duplicate_features(X: DataFrame) -> DataFrame:
        X_without_dups = X.T.drop_duplicates().T
        logger.debug(f"X_without_dups.shape: {X_without_dups.shape}")

        columns_orig = X.columns.values
        columns_new = X_without_dups.columns.values
        columns_removed = [column for column in columns_orig if column not in columns_new]

        del X_without_dups

        logger.log(15, 'Warning: duplicate columns removed ')
        logger.log(15, columns_removed)
        logger.log(15, f'Removed {len(columns_removed)} duplicate columns before training models')

        return X[columns_new]

    # TODO: Move this outside of here
    # TODO: Not accurate for categoricals, will count categorical mapping dict as taking more memory than it actually does.
    @staticmethod
    def _get_approximate_df_mem_usage(df, sample_ratio=0.2):
        if sample_ratio >= 1:
            return df.memory_usage(deep=True)
        else:
            num_rows = len(df)
            num_rows_sample = math.ceil(sample_ratio * num_rows)
            return df.head(num_rows_sample).memory_usage(deep=True) / sample_ratio

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
        self._feature_metadata_in_real.print_feature_metadata_full('\t')
        super().print_feature_metadata_info()
