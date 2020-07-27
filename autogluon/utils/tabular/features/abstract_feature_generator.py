import copy
import logging
import math
from collections import defaultdict

import numpy as np
import psutil
from pandas import DataFrame
from pandas.api.types import CategoricalDtype

from . import binning
from .feature_types_metadata import FeatureTypesMetadata
from .generators.dummy import DummyFeatureGenerator
from .types import get_type_group_map_real, get_type_map_raw, get_type_map_real, get_type_map_special, get_type_group_map_special
from .utils import check_if_useless_feature, clip_and_astype
from ..utils.decorators import calculate_time
from ..utils.savers import save_pkl

logger = logging.getLogger(__name__)


# TODO: Add feature of # of observation counts to high cardinality categorical features
# TODO: Use code from problem type detection for column types! Ints/Floats could be Categorical through this method! Maybe try both?
class AbstractFeatureGenerator:  # TODO: RENAME
    def __init__(self, generators):
        self.features_init_to_keep = []  # Original features to use as input to feature generation
        self.features_init_types = dict()  # Initial feature types prior to transformation
        self.features_categorical_final_mapping = defaultdict()  # Categorical features original value -> category code mapping
        self.features_binned = []  # Features to be binned
        self.features_binned_mapping = defaultdict()  # Binned features original value ranges -> binned category code mapping
        self.features = []  # Final list of features after transformation

        self._is_fit = False  # Whether the feature generation has been fit

        self.minimize_categorical_memory_usage_flag = True

        self.pre_memory_usage = None
        self.pre_memory_usage_per_row = None
        self.post_memory_usage = None
        self.post_memory_usage_per_row = None
        self._is_dummy = False  # If True, returns a single dummy feature as output. Occurs if fit with no useful features.

        self.generators = generators

        self._feature_metadata_in: FeatureTypesMetadata = None  # FeatureTypesMetadata object based on the original input features.
        self._feature_metadata_in_real: FeatureTypesMetadata = None  # FeatureTypesMetadata object based on the original input features real dtypes (will contain dtypes such as 'int16' and 'float32' instead of 'int' and 'float').
        self.feature_types_metadata: FeatureTypesMetadata = None  # FeatureTypesMetadata object based on the processed features. Passed to models to enable advanced functionality.

    def _preprocess(self, X: DataFrame):
        for column in X.columns:
            if self.features_init_types[column].name == 'object':
                X[column].fillna('', inplace=True)
            else:
                X[column].fillna(np.nan, inplace=True)
        return X

    # TODO: Save this to disk and remove from memory if large categoricals!
    @calculate_time
    def fit_transform(self, X: DataFrame, y=None, banned_features=None, fix_categoricals=False, drop_duplicates=True):
        if self._is_fit:
            raise AssertionError('fit_transform cannot be called on an already fit feature generator.')
        X_len = len(X)
        self.pre_memory_usage = self._get_approximate_df_mem_usage(X, sample_ratio=0.2).sum()
        self.pre_memory_usage_per_row = self.pre_memory_usage / X_len
        available_mem = psutil.virtual_memory().available
        pre_memory_usage_percent = self.pre_memory_usage / (available_mem + self.pre_memory_usage)
        if pre_memory_usage_percent > 0.05:
            logger.warning(f'Warning: Data size prior to feature transformation consumes {round(pre_memory_usage_percent*100, 1)}% of available memory. Consider increasing memory or subsampling the data to avoid instability.')

        X_index = copy.deepcopy(X.index)
        X = X.reset_index(drop=True)  # TODO: Theoretically inplace=True avoids data copy, but can lead to altering of original DataFrame outside of method context.

        X_features = self._fit_transform(X=X, y=y, banned_features=banned_features, fix_categoricals=fix_categoricals, drop_duplicates=drop_duplicates)

        if len(list(X_features.columns)) == 0:
            self._is_dummy = True
            logger.warning(f'WARNING: No useful features were detected in the data! AutoGluon will train using 0 features, and will always predict the same value. Ensure that you are passing the correct data to AutoGluon!')
            dummy_generator = DummyFeatureGenerator()
            X_features = dummy_generator.fit_transform(X=X_features)
            self.feature_types_metadata = copy.deepcopy(dummy_generator.feature_types_metadata)
            self.generators = [(dummy_generator, None)]

        X_features.index = X_index
        self.features = list(X_features.columns)

        self.post_memory_usage = self._get_approximate_df_mem_usage(X_features, sample_ratio=0.2).sum()
        self.post_memory_usage_per_row = self.post_memory_usage / X_len

        available_mem = psutil.virtual_memory().available
        post_memory_usage_percent = self.post_memory_usage / (available_mem + self.post_memory_usage + self.pre_memory_usage)

        if post_memory_usage_percent > 0.15:
            logger.warning(f'Warning: Data size post feature transformation consumes {round(post_memory_usage_percent*100, 1)}% of available memory. Consider increasing memory or subsampling the data to avoid instability.')

        self._is_fit = True
        logger.log(20, 'Feature Generator processed %s data points with %s features' % (X_len, len(self.features)))
        self.print_feature_type_info()

        return X_features

    def _fit_transform(self, X: DataFrame, y=None, banned_features=None, fix_categoricals=False, drop_duplicates=True) -> DataFrame:
        X.columns = X.columns.astype(str)  # Ensure all column names are strings

        if banned_features:
            features_to_remove = [str(feature) for feature in banned_features]
        else:
            features_to_remove = []
        features_to_remove += [feature for feature in self._get_useless_features(X) if feature not in features_to_remove]
        X.drop(features_to_remove, axis=1, errors='ignore', inplace=True)

        self.features_init_to_keep = list(X.columns)
        self.features_init_types = X.dtypes.to_dict()

        type_map_real = get_type_map_real(X)
        type_map_raw = get_type_map_raw(X)
        type_group_map_special = self._get_type_group_map_special(X)
        # TODO: Add ability for user to specify type_group_map_special as input to fit_transform
        self._feature_metadata_in_real = FeatureTypesMetadata(type_map_raw=type_map_real)
        self._feature_metadata_in = FeatureTypesMetadata(type_map_raw=type_map_raw, type_group_map_special=type_group_map_special)

        X_features = self._generate_features(X)

        features_to_remove_post = self._get_features_to_remove_post(X_features)
        X_features.drop(features_to_remove_post, axis=1, inplace=True)
        self._remove_features(features=features_to_remove_post)

        feature_names = self.feature_types_metadata.get_features()

        self.features_binned_mapping = binning.generate_bins(X_features, self.features_binned)
        for column in self.features_binned:
            X_features[column] = binning.bin_column(series=X_features[column], mapping=self.features_binned_mapping[column])

        if fix_categoricals:  # if X_test is not used in fit_transform and the model used is from SKLearn
            X_features = self._fix_categoricals_for_sklearn(X_features=X_features)

        if drop_duplicates:
            X_features = self._drop_duplicate_features(X_features)

        feature_names_post = list(X_features.columns)
        if set(feature_names) != set(feature_names_post):
            features_to_remove_post = [feature for feature in feature_names if feature not in feature_names_post]
            self._remove_features(features=features_to_remove_post)

        for column in self.feature_types_metadata.type_group_map_raw['category']:
            self.features_categorical_final_mapping[column] = copy.deepcopy(X_features[column].cat.categories)  # dict(enumerate(X_features[column].cat.categories))

        X_features = self._minimize_memory_usage(X_features=X_features)

        self.feature_types_metadata = FeatureTypesMetadata(type_map_raw=get_type_map_raw(X_features), type_group_map_special=self.feature_types_metadata.type_group_map_special)
        self.features_binned = [feature for feature in self.features_binned if feature in self.feature_types_metadata.type_map_raw]

        return X_features

    @calculate_time
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
            X = X[self.features_init_to_keep]
        except KeyError:
            missing_cols = []
            for col in self.features_init_to_keep:
                if col not in X.columns:
                    missing_cols.append(col)
            raise KeyError(f'{len(missing_cols)} required columns are missing from the provided dataset. Missing columns: {missing_cols}')

        if self.features_init_types:
            X = X.astype(self.features_init_types)
        X_features = self._generate_features(X)
        for column in self.features_binned:
            X_features[column] = binning.bin_column(series=X_features[column], mapping=self.features_binned_mapping[column])

        X_features = X_features[self.features]
        for column in self.feature_types_metadata.type_group_map_raw['category']:
            X_features[column].cat.set_categories(self.features_categorical_final_mapping[column], inplace=True)

        X_features = self._minimize_memory_usage(X_features=X_features)

        return X_features

    def _generate_features(self, X: DataFrame) -> DataFrame:
        raise NotImplementedError()

    # TODO: Remove self.features_binned from here
    def _remove_features(self, features):
        self.feature_types_metadata.remove_features(features=features, inplace=True)
        self.features_binned = [feature for feature in self.features_binned if feature in self.feature_types_metadata.type_map_raw]

    def _minimize_memory_usage(self, X_features: DataFrame) -> DataFrame:
        if self.minimize_categorical_memory_usage_flag:
            self._minimize_categorical_memory_usage(X_features=X_features)
        X_features = self._minimize_ngram_memory_usage(X_features=X_features)
        X_features = self._minimize_binned_memory_usage(X_features=X_features)
        return X_features

    def _minimize_ngram_memory_usage(self, X_features: DataFrame) -> DataFrame:
        return clip_and_astype(df=X_features, columns=self.feature_types_metadata.type_group_map_special['text_ngram'], clip_min=0, clip_max=255, dtype='uint8')

    def _minimize_binned_memory_usage(self, X_features: DataFrame) -> DataFrame:
        return clip_and_astype(df=X_features, columns=self.features_binned, clip_min=0, clip_max=255, dtype='uint8')

    # TODO: Compress further, uint16, etc.
    # Performs in-place updates
    def _minimize_categorical_memory_usage(self, X_features: DataFrame):
        for column in self.feature_types_metadata.type_group_map_raw['category']:
            new_categories = list(range(len(X_features[column].cat.categories.values)))
            X_features[column].cat.rename_categories(new_categories, inplace=True)

    def _fix_categoricals_for_sklearn(self, X_features: DataFrame) -> DataFrame:
        features_to_remove = []
        for column in self.feature_types_metadata.type_group_map_raw['category']:
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

    def _get_features_to_remove_post(self, X_features: DataFrame) -> list:
        features_to_remove_post = []
        X_len = len(X_features)
        for column in X_features:
            unique_value_count = len(X_features[column].unique())
            if unique_value_count == 1:
                features_to_remove_post.append(column)
            # TODO: Consider making 0.99 a parameter to FeatureGenerator
            elif column in self._feature_metadata_in.type_group_map_raw['object'] and (unique_value_count / X_len > 0.99):
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

    def get_feature_types_metadata_full(self):
        feature_types_metadata_full = copy.deepcopy(self.feature_types_metadata.type_group_map_special)

        for key_raw in self.feature_types_metadata.type_group_map_raw:
            values = self.feature_types_metadata.type_group_map_raw[key_raw]
            for key_special in self.feature_types_metadata.type_group_map_special:
                values = [value for value in values if value not in self.feature_types_metadata.type_group_map_special[key_special]]
            if values:
                feature_types_metadata_full[key_raw] += values

        return feature_types_metadata_full

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

    def print_feature_type_info(self):
        logger.log(20, 'Original Features (raw dtypes):')
        for key, val in self._feature_metadata_in_real.type_group_map_raw.items():
            if val: logger.log(20, '\t%s features: %s' % (key, len(val)))
        logger.log(20, 'Original Features (inferred dtypes):')
        for key, val in self._feature_metadata_in.type_group_map_special.items():
            if val: logger.log(20, '\t%s features: %s' % (key, len(val)))
        logger.log(20, 'Generated Features (special dtypes):')
        for key, val in self.feature_types_metadata.type_group_map_special.items():
            if val: logger.log(20, '\t%s features: %s' % (key, len(val)))
        logger.log(20, 'Processed Features (raw dtypes):')
        for key, val in self.feature_types_metadata.type_group_map_raw.items():
            if val: logger.log(20, '\t%s features: %s' % (key, len(val)))
        logger.log(20, 'Processed Features:')
        for key, val in self.get_feature_types_metadata_full().items():
            if val: logger.log(20, '\t%s features: %s' % (key, len(val)))

    def save(self, path):
        save_pkl.save(path=path, object=self)
