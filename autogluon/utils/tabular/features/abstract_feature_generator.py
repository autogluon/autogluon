import copy
import logging
import math
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import psutil
from pandas import DataFrame, Series
from pandas.api.types import CategoricalDtype

from .feature_types_metadata import FeatureTypesMetadata
from .utils import get_type_family, get_type_groups_df, get_type_family_groups_df
from ..utils.decorators import calculate_time
from ..utils.savers import save_pkl

logger = logging.getLogger(__name__)


# TODO: Add feature of # of observation counts to high cardinality categorical features
# TODO: Use code from problem type detection for column types! Ints/Floats could be Categorical through this method! Maybe try both?
class AbstractFeatureGenerator:
    def __init__(self):
        self.features_init_to_keep = []  # Original features to use as input to feature generation
        self.features_to_remove = []  # Original features to remove prior to feature generation
        self.features_to_remove_post = []  # Final features to remove prior to sending to models
        self.features_init_types = dict()  # Initial feature types prior to transformation
        self.feature_type_family_init_raw = defaultdict(list)  # Feature types of original features, without inferring.
        self.feature_type_family = defaultdict(list)  # Feature types of original features, after inferring.
        self.feature_type_family_generated = defaultdict(list)  # Feature types (special) of generated features.
        self.feature_transformations = defaultdict(list)  # Dictionary of transformation pipelines (keys) and the list of original features transformed through them (values)
        self.features_categorical_final = []  # Categorical features
        self.features_categorical_final_mapping = defaultdict()  # Categorical features original value -> category code mapping
        self.features_binned = []  # Features to be binned
        self.features_binned_mapping = defaultdict()  # Binned features original value ranges -> binned category code mapping
        self.features = []  # Final list of features after transformation

        self.fit = False  # Whether the feature generation has been fit

        self.minimize_categorical_memory_usage_flag = True

        self.pre_memory_usage = None
        self.pre_memory_usage_per_row = None
        self.post_memory_usage = None
        self.post_memory_usage_per_row = None
        self.is_dummy = False  # If True, returns a single dummy feature as output. Occurs if fit with no useful features.

        self.feature_types_metadata: FeatureTypesMetadata = None  # FeatureTypesMetadata object based on the final features. Passed to models to enable advanced functionality.

    def preprocess(self, X: DataFrame):
        return X

    # TODO: Save this to disk and remove from memory if large categoricals!
    @calculate_time
    def fit_transform(self, X: DataFrame, y=None, banned_features=None, fix_categoricals=False, drop_duplicates=True):
        self.fit = False
        X_len = len(X)
        self.pre_memory_usage = self.get_approximate_df_mem_usage(X, sample_ratio=0.2).sum()
        self.pre_memory_usage_per_row = self.pre_memory_usage / X_len
        available_mem = psutil.virtual_memory().available
        pre_memory_usage_percent = self.pre_memory_usage / (available_mem + self.pre_memory_usage)
        if pre_memory_usage_percent > 0.05:
            logger.warning(f'Warning: Data size prior to feature transformation consumes {round(pre_memory_usage_percent*100, 1)}% of available memory. Consider increasing memory or subsampling the data to avoid instability.')

        X_index = copy.deepcopy(X.index)
        X = X.reset_index(drop=True)  # TODO: Theoretically inplace=True avoids data copy, but can lead to altering of original DataFrame outside of method context.

        X_features = self._fit_transform(X=X, y=y, banned_features=banned_features, fix_categoricals=fix_categoricals, drop_duplicates=drop_duplicates)

        X_features.index = X_index

        for feature_type in self.feature_type_family_generated:
            self.feature_type_family_generated[feature_type] = list(set(self.feature_type_family_generated[feature_type]) - set(self.features_to_remove_post))

        self.post_memory_usage = self.get_approximate_df_mem_usage(X_features, sample_ratio=0.2).sum()
        self.post_memory_usage_per_row = self.post_memory_usage / X_len

        available_mem = psutil.virtual_memory().available
        post_memory_usage_percent = self.post_memory_usage / (available_mem + self.post_memory_usage + self.pre_memory_usage)

        if post_memory_usage_percent > 0.15:
            logger.warning(f'Warning: Data size post feature transformation consumes {round(post_memory_usage_percent*100, 1)}% of available memory. Consider increasing memory or subsampling the data to avoid instability.')

        if len(list(X_features.columns)) == 0:
            self.is_dummy = True
            logger.warning(f'WARNING: No useful features were detected in the data! AutoGluon will train using 0 features, and will always predict the same value. Ensure that you are passing the correct data to AutoGluon!')
            X_features['__dummy__'] = 0
            self.feature_type_family_generated['int'] = ['__dummy__']

        feature_types_raw = get_type_family_groups_df(X_features)
        self.feature_types_metadata = FeatureTypesMetadata(feature_types_raw=feature_types_raw, feature_types_special=self.feature_type_family_generated)

        self.features = list(X_features.columns)
        self.fit = True
        logger.log(20, 'Feature Generator processed %s data points with %s features' % (X_len, len(self.features)))
        self.print_feature_type_info()

        return X_features

    def _fit_transform(self, X: DataFrame, y=None, banned_features=None, fix_categoricals=False, drop_duplicates=True):
        if banned_features:
            self.features_to_remove += [str(feature) for feature in banned_features]

        X.columns = X.columns.astype(str)  # Ensure all column names are strings
        self.get_feature_types(X)
        X = X.drop(self.features_to_remove, axis=1, errors='ignore')
        self.features_init_to_keep = copy.deepcopy(list(X.columns))
        self.features_init_types = X.dtypes.to_dict()
        self.feature_type_family_init_raw = get_type_groups_df(X)

        X_features = self.generate_features(X)
        X_len = len(X)
        for column in X_features:
            unique_value_count = len(X_features[column].unique())
            if unique_value_count == 1:
                self.features_to_remove_post.append(column)
            # TODO: Consider making 0.99 a parameter to FeatureGenerator
            elif 'object' in self.feature_type_family and column in self.feature_type_family['object'] and (unique_value_count / X_len > 0.99):
                self.features_to_remove_post.append(column)

        self.features_binned = list(set(self.features_binned) - set(self.features_to_remove_post))
        self.features_binned_mapping = self.generate_bins(X_features, self.features_binned)
        for column in self.features_binned:  # TODO: Should binned columns be continuous or categorical if they were initially continuous? (Currently categorical)
            X_features[column] = self.bin_column(series=X_features[column], mapping=self.features_binned_mapping[column])
        X_features = X_features.drop(self.features_to_remove_post, axis=1)
        if drop_duplicates:
            X_features = self.drop_duplicate_features(X_features)
        self.features_categorical_final = list(X_features.select_dtypes(include='category').columns.values)
        if fix_categoricals:  # if X_test is not used in fit_transform and the model used is from SKLearn
            X_features = self.fix_categoricals_for_sklearn(X_features=X_features)
        for column in self.features_categorical_final:
            self.features_categorical_final_mapping[column] = copy.deepcopy(X_features[column].cat.categories)  # dict(enumerate(X_features[column].cat.categories))

        X_features = self.minimize_memory_usage(X_features=X_features)

        return X_features

    @calculate_time
    def transform(self, X: DataFrame):
        if not self.fit:
            raise AssertionError('FeatureGenerator has not yet been fit.')
        if self.features is None:
            raise AssertionError('FeatureGenerator.features is None, have you called fit() yet?')
        X_index = copy.deepcopy(X.index)
        X = X.reset_index(drop=True)
        X.columns = X.columns.astype(str)  # Ensure all column names are strings
        X = X.drop(self.features_to_remove, axis=1, errors='ignore')
        X_columns = X.columns.tolist()
        # Create any columns present in the training dataset that are now missing from this dataframe:
        missing_cols = []
        for col in self.features_init_to_keep:
            if col not in X_columns:
                missing_cols.append(col)
        if len(missing_cols) > 0:
            raise ValueError(f'Required columns are missing from the provided dataset. Missing columns: {missing_cols}')

        X = X.astype(self.features_init_types)
        X_features = self.generate_features(X)
        for column in self.features_binned:
            X_features[column] = self.bin_column(series=X_features[column], mapping=self.features_binned_mapping[column])
        if self.is_dummy:
            X_features.index = X_index
            X_features['__dummy__'] = 0
            return X_features
        X_features = X_features[self.features]
        for column in self.features_categorical_final:
            X_features[column].cat.set_categories(self.features_categorical_final_mapping[column], inplace=True)
        X_features = self.minimize_memory_usage(X_features=X_features)
        X_features.index = X_index

        return X_features

    def get_feature_types(self, X: DataFrame):
        features_init = list(X.columns)
        features_init = [feature for feature in features_init if feature not in self.features_to_remove]
        for column in features_init:
            mark_for_removal = False
            col_val = X[column]
            dtype = col_val.dtype
            num_unique = len(col_val.unique())

            type_family = self.get_type_family(dtype)
            if self.check_if_datetime_feature(col_val):
                type_family = 'datetime'
            elif self.check_if_nlp_feature(col_val):
                type_family = 'text'

            if num_unique == 1:
                mark_for_removal = True
            if mark_for_removal:
                self.features_to_remove.append(column)
            else:
                self.feature_type_family[type_family].append(column)

    def generate_features(self, X: DataFrame):
        raise NotImplementedError()

    # TODO: Expand to int64 -> date features (milli from epoch etc)
    def check_if_datetime_feature(self, X: Series):
        type_family = self.get_type_family(X.dtype)
        # TODO: Check if low numeric numbers, could be categorical encoding!
        # TODO: If low numeric, potentially it is just numeric instead of date
        if X.isnull().all():
            return False
        if type_family == 'datetime':
            return True
        if type_family != 'object':  # TODO: seconds from epoch support
            return False
        try:
            X.apply(pd.to_datetime)
            return True
        except:
            return False

    def check_if_nlp_feature(self, X: Series):
        type_family = self.get_type_family(X.dtype)
        if type_family != 'object':
            return False
        X_unique = X.unique()
        num_unique = len(X_unique)
        num_rows = len(X)
        unique_ratio = num_unique / num_rows
        # print(X_unique)
        if unique_ratio <= 0.01:
            return False
        avg_words = np.mean([len(re.sub(' +', ' ', value).split(' ')) if isinstance(value, str) else 0 for value in X_unique])
        # print(avg_words)
        if avg_words < 3:
            return False

        return True

    def minimize_memory_usage(self, X_features):
        if self.minimize_categorical_memory_usage_flag:
            X_features = self.minimize_categorical_memory_usage(X_features=X_features)
        X_features = self.minimize_ngram_memory_usage(X_features=X_features)
        X_features = self.minimize_binned_memory_usage(X_features=X_features)
        return X_features

    def minimize_ngram_memory_usage(self, X_features):
        if 'text_ngram' in self.feature_type_family_generated and self.feature_type_family_generated['text_ngram']:
            X_features[self.feature_type_family_generated['text_ngram']] = np.clip(X_features[self.feature_type_family_generated['text_ngram']], 0, 255).astype('uint8')
        return X_features

    def minimize_binned_memory_usage(self, X_features):
        if self.features_binned:
            X_features[self.features_binned] = np.clip(X_features[self.features_binned], 0, 255).astype('uint8')
        return X_features

    # TODO: Compress further, uint16, etc.
    # Warning: Performs in-place updates
    def minimize_categorical_memory_usage(self, X_features):
        cat_columns = X_features.select_dtypes('category').columns
        for column in cat_columns:
            new_categories = list(range(len(X_features[column].cat.categories.values)))
            X_features[column].cat.rename_categories(new_categories, inplace=True)
        return X_features

    def fix_categoricals_for_sklearn(self, X_features):
        for column in self.features_categorical_final:
            rank = X_features[column].value_counts().sort_values(ascending=True)
            rank = rank[rank >= 3]
            rank = rank.reset_index()
            val_list = list(rank['index'].values)
            if len(val_list) <= 1:
                self.features_to_remove_post.append(column)
                self.features_categorical_final = [feature for feature in self.features_categorical_final if feature != column]
                logger.debug(f'Dropping {column}')
            else:
                X_features[column] = X_features[column].astype(CategoricalDtype(categories=val_list))
        return X_features

    # TODO: add option for user to specify dtypes on load
    @staticmethod
    def get_type_family(dtype):
        return get_type_family(dtype=dtype)

    # TODO: optimize by not considering columns with unique sums/means
    # TODO: Multithread?
    @staticmethod
    def drop_duplicate_features(X):
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
        feature_types_metadata_full = copy.deepcopy(self.feature_types_metadata.feature_types_special)

        for key_raw in self.feature_types_metadata.feature_types_raw:
            values = self.feature_types_metadata.feature_types_raw[key_raw]
            for key_special in self.feature_types_metadata.feature_types_special:
                values = [value for value in values if value not in self.feature_types_metadata.feature_types_special[key_special]]
            if values:
                feature_types_metadata_full[key_raw] += values

        return feature_types_metadata_full

    @staticmethod
    def bin_column(series, mapping):
        mapping_dict = {k: v for v, k in enumerate(list(mapping))}
        series_out = pd.cut(series, mapping)
        # series_out.cat.categories = [str(g) for g in series_out.cat.categories]  # LightGBM crashes at end of training without this
        series_out_int = [mapping_dict[val] for val in series_out]
        return series_out_int

    # TODO: Rewrite with normalized value counts as binning technique, will be more performant and optimal
    @staticmethod
    def generate_bins(X_features: DataFrame, features_to_bin):
        X_len = len(X_features)
        ideal_cats = 10
        starting_cats = 1000
        bin_index_starting = [np.floor(X_len * (num + 1) / starting_cats) for num in range(starting_cats - 1)]
        bin_epsilon = 0.000000001
        bin_mapping = defaultdict()
        max_iterations = 20
        for column in features_to_bin:
            num_cats_initial = starting_cats
            bins_value_counts = X_features[column].value_counts(ascending=False, normalize=True)
            max_bins = len(bins_value_counts)

            if max_bins <= ideal_cats:
                bins = pd.Series(data=sorted(X_features[column].unique()))
                num_cats_initial = max_bins
                cur_len = max_bins
                bin_index = range(num_cats_initial)
                interval_index = AbstractFeatureGenerator.get_bins(bins=bins, bin_index=bin_index, bin_epsilon=bin_epsilon)
            else:
                cur_len = X_len
                bins = X_features[column].sort_values(ascending=True)
                interval_index = AbstractFeatureGenerator.get_bins(bins=bins, bin_index=bin_index_starting, bin_epsilon=bin_epsilon)

            max_desired_bins = min(ideal_cats, max_bins)
            min_desired_bins = min(ideal_cats, max_bins)

            if (len(interval_index) >= min_desired_bins) and (len(interval_index) <= max_desired_bins):
                is_satisfied = True
            else:
                is_satisfied = False

            num_cats_current = num_cats_initial
            # print(column, min_desired_bins, max_desired_bins)
            cur_iteration = 0
            while not is_satisfied:
                if len(interval_index) > max_desired_bins:
                    pass
                elif len(interval_index) < min_desired_bins:
                    pass
                ratio_reduction = max_desired_bins / len(interval_index)
                num_cats_current = int(np.floor(num_cats_current * ratio_reduction))
                bin_index = [np.floor(cur_len * (num + 1) / num_cats_current) for num in range(num_cats_current - 1)]
                interval_index = AbstractFeatureGenerator.get_bins(bins=bins, bin_index=bin_index, bin_epsilon=bin_epsilon)

                if (len(interval_index) >= min_desired_bins) and (len(interval_index) <= max_desired_bins):
                    is_satisfied = True
                    # print('satisfied', column, len(interval_index))
                cur_iteration += 1
                if cur_iteration >= max_iterations:
                    is_satisfied = True
                    # print('max_iterations met, stopping prior to satisfaction!', column, len(interval_index))

            bin_mapping[column] = interval_index
        return bin_mapping

    # TODO: Move this outside of here
    # TODO: Not accurate for categoricals, will count categorical mapping dict as taking more memory than it actually does.
    @staticmethod
    def get_approximate_df_mem_usage(df, sample_ratio=0.2):
        if sample_ratio >= 1:
            return df.memory_usage(deep=True)
        else:
            num_rows = len(df)
            num_rows_sample = math.ceil(sample_ratio * num_rows)
            return df.head(num_rows_sample).memory_usage(deep=True) / sample_ratio

    # TODO: Clean code
    # bins is a sorted int/float series, ascending=True
    @staticmethod
    def get_bins(bins: Series, bin_index, bin_epsilon):
        max_val = bins.max()
        bins_2 = bins.iloc[bin_index]
        bins_3 = list(bins_2.values)
        bins_unique = sorted(list(set(bins_3)))
        bins_with_epsilon_max = set([i for i in bins_unique] + [i - bin_epsilon for i in bins_unique if i == max_val])
        removal_bins = set([bins_unique[index - 1] for index, i in enumerate(bins_unique[1:], start=1) if i == max_val])
        bins_4 = sorted(list(bins_with_epsilon_max - removal_bins))
        bins_5 = [np.inf if (x == max_val) else x for x in bins_4]
        bins_6 = sorted(list(set([-np.inf] + bins_5 + [np.inf])))
        bins_7 = [(bins_6[i], bins_6[i + 1]) for i in range(len(bins_6) - 1)]
        interval_index = pd.IntervalIndex.from_tuples(bins_7)
        return interval_index

    def print_feature_type_info(self):
        logger.log(20, 'Original Features (raw dtypes):')
        for key, val in self.feature_type_family_init_raw.items():
            if val: logger.log(20, '\t%s features: %s' % (key, len(val)))
        logger.log(20, 'Original Features (inferred dtypes):')
        for key, val in self.feature_type_family.items():
            if val: logger.log(20, '\t%s features: %s' % (key, len(val)))
        logger.log(20, 'Generated Features (special dtypes):')
        for key, val in self.feature_types_metadata.feature_types_special.items():
            if val: logger.log(20, '\t%s features: %s' % (key, len(val)))
        logger.log(20, 'Processed Features (raw dtypes):')
        for key, val in self.feature_types_metadata.feature_types_raw.items():
            if val: logger.log(20, '\t%s features: %s' % (key, len(val)))
        logger.log(20, 'Processed Features:')
        for key, val in self.get_feature_types_metadata_full().items():
            if val: logger.log(20, '\t%s features: %s' % (key, len(val)))

    def save_self(self, path):
        save_pkl.save(path=path, object=self)
