import copy
import logging
import re
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from pandas.api.types import CategoricalDtype

from ..utils.decorators import calculate_time
from ..utils.savers import save_pkl

logger = logging.getLogger(__name__)


# TODO: Add optimization to make Vectorizer smaller in size by deleting key dictionary
# TODO: Add feature of # of observation counts to high cardinality categorical features
# TODO: Use code from problem type detection for column types! Ints/Floats could be Categorical through this method! Maybe try both?
class AbstractFeatureGenerator:
    def __init__(self):
        self.features_init = []
        self.features_init_to_keep = []
        self.features_to_remove = []
        self.features_to_remove_post = []
        self.features_to_keep_raw = []
        self.features_object = []
        self.features_init_types = dict()
        self.feature_types = defaultdict(list)
        self.feature_type_family = defaultdict(list)
        self.feature_type_family_generated = defaultdict(list)
        self.features_bool = []
        self.features_nlp = []
        self.features_nlp_ratio = []
        self.features_datetime = []
        self.features_categorical = []
        self.features_categorical_final = []
        self.features_categorical_final_mapping = defaultdict()
        self.features_binned = []
        self.features_binned_mapping = defaultdict()
        self.features_vectorizers = []
        self.features = []
        self.banned_features = []
        self.fit = False

    @property
    def feature_types_metadata(self):
        feature_types_metadata = copy.deepcopy(
            {
                'nlp': self.features_nlp,
                'vectorizers': self.features_vectorizers,
                **self.feature_type_family
            }
        )
        for key, val in self.feature_type_family_generated.items():
            if key in feature_types_metadata:
                feature_types_metadata[key] += val
            else:
                feature_types_metadata[key] = val
        return feature_types_metadata

    @property
    def feature_types_metadata_generated(self):
        feature_types_metadata_generated = copy.deepcopy(
            {**self.feature_type_family_generated}
        )
        if 'int' in feature_types_metadata_generated:  # TODO: Clean this, feature_vectorizers should already be handled
            feature_types_metadata_generated['int'] += self.features_vectorizers
        elif len(self.features_vectorizers) > 0:
            feature_types_metadata_generated['int'] = self.features_vectorizers
        return feature_types_metadata_generated

    @property
    def feature_types_metadata_full(self):
        feature_types_metadata_full = copy.deepcopy(
            {**self.feature_type_family}
        )
        for key, val in self.feature_type_family_generated.items():
            if key in feature_types_metadata_full:
                feature_types_metadata_full[key] += val
            else:
                feature_types_metadata_full[key] = val
        if 'int' in feature_types_metadata_full:  # TODO: Clean this, feature_vectorizers should already be handled
            feature_types_metadata_full['int'] += self.features_vectorizers
        elif len(self.features_vectorizers) > 0:
            feature_types_metadata_full['int'] = self.features_vectorizers
        return feature_types_metadata_full

    @staticmethod
    def train_vectorizer(text_list, vectorizer):
        logger.log(15, 'Fitting vectorizer...')
        transform_matrix = vectorizer.fit_transform(text_list)  # TODO: Consider upgrading to pandas 0.25.0 to benefit from sparse attribute improvements / bug fixes! https://pandas.pydata.org/pandas-docs/stable/whatsnew/v0.25.0.html
        vectorizer.stop_words_ = None  # Reduces object size by 100x+ on large datasets, no effect on usability
        logger.log(15, f'Vectorizer fit with vocabulary size = {len(vectorizer.vocabulary_)}')
        return vectorizer, transform_matrix

    def preprocess(self, X: DataFrame):
        return X

    @calculate_time
    def fit_transform(self, X: DataFrame, y=None, banned_features=None, fix_categoricals=False, drop_duplicates=True):
        self.fit = False
        X_len = len(X)
        if banned_features:
            self.banned_features = banned_features
            self.features_to_remove += self.banned_features
        X_index = copy.deepcopy(X.index)
        self.get_feature_types(X)
        X = X.drop(self.features_to_remove, axis=1, errors='ignore')
        self.features_init_to_keep = copy.deepcopy(list(X.columns))
        self.features_init_types = X.dtypes.to_dict()
        X.reset_index(drop=True, inplace=True)
        X_features = self.generate_features(X)
        for column in X_features:
            unique_value_count = len(X_features[column].unique())
            if unique_value_count == 1:
                self.features_to_remove_post.append(column)
            elif column in self.feature_type_family['object'] and (unique_value_count / X_len > 0.99):
                self.features_to_remove_post.append(column)

        self.features_binned = set(self.features_binned) - set(self.features_to_remove_post)
        self.features_binned_mapping = self.generate_bins(X_features, self.features_binned)
        for column in self.features_binned:  # TODO: Should binned columns be continuous or categorical if they were initially continuous? (Currently categorical)
            X_features[column] = self.bin_column(series=X_features[column], mapping=self.features_binned_mapping[column])
            # print(X_features[column].value_counts().sort_index())
        X_features = X_features.drop(self.features_to_remove_post, axis=1)
        if drop_duplicates:
            X_features = self.drop_duplicate_features(X_features)
        self.features_categorical_final = list(X_features.select_dtypes(include='category').columns.values)
        if fix_categoricals:  # if X_test is not used in fit_transform and the model used is from SKLearn
            X_features = self.fix_categoricals_for_sklearn(X_features=X_features)
        for column in self.features_categorical_final:
            self.features_categorical_final_mapping[column] = X_features[column].cat.categories  # dict(enumerate(X_features[column].cat.categories))
        X_features.index = X_index
        self.features = list(X_features.columns)
        self.feature_type_family_generated['int'] += self.features_binned
        self.fit = True

        logger.log(20, 'Feature Generator processed %s data points with %s features' % (X_len, len(self.features)))
        logger.log(20, 'Original Features:')
        for key, val in self.feature_type_family.items():
            logger.log(20, '\t%s features: %s' % (key, len(val)))
        logger.log(20, 'Generated Features:')
        for key, val in self.feature_types_metadata_generated.items():
            logger.log(20, '\t%s features: %s' % (key, len(val)))
        logger.log(20, 'All Features:')
        for key, val in self.feature_types_metadata_full.items():
            logger.log(20, '\t%s features: %s' % (key, len(val)))

        return X_features

    @calculate_time
    def transform(self, X: DataFrame):
        if not self.fit:
            raise AssertionError('FeatureGenerator has not yet been fit.')
        if self.features is None:
            raise AssertionError('FeatureGenerator.features is None, have you called fit() yet?')
        X_index = copy.deepcopy(X.index)
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
        X.reset_index(drop=True, inplace=True)
        X_features = self.generate_features(X)
        for column in self.features_binned:
            X_features[column] = self.bin_column(series=X_features[column], mapping=self.features_binned_mapping[column])
        X_features = X_features[self.features]
        for column in self.features_categorical_final:
            X_features[column].cat.set_categories(self.features_categorical_final_mapping[column], inplace=True)
        X_features.index = X_index
        return X_features

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

    def get_feature_types(self, X: DataFrame):
        self.features_init = list(X.columns)
        self.features_init = [feature for feature in self.features_init if feature not in self.features_to_remove]
        for column in self.features_init:
            mark_for_removal = False
            col_val = X[column]
            dtype = col_val.dtype
            num_unique = len(col_val.unique())
            unique_counts = col_val.value_counts()

            # num_unique_duplicates = len(unique_counts[unique_counts > 100])
            # num_rows = len(col_val)
            # unique_ratio = num_unique / float(num_rows)
            # print(column)
            # print(num_unique)
            # # print(num_rows)
            # # print(unique_ratio)
            # print(dtype)

            type_family = self.get_type_family(dtype)
            # print(num_unique, '\t', num_unique_duplicates, '\t', unique_ratio, '\t', type_family, '\t', column,)

            if num_unique == 1:
                mark_for_removal = True

            # if num_unique == num_rows:
            #     print('fully unique!')
            # if unique_ratio > 0.5:
            #     print('fairly unique!')
            # print(col_val.value_counts())

            if self.check_if_datetime_feature(col_val):
                type_family = 'datetime'  # TODO: Verify
                dtype = 'datetime'
                self.features_datetime.append(column)
                logger.debug(f'date: {column}')
                logger.debug(unique_counts.head(5))
            elif self.check_if_nlp_feature(col_val):
                self.features_nlp.append(column)
                self.features_nlp_ratio.append(column)
                logger.debug(f'nlp: {column}')
                logger.debug(unique_counts.head(5))
            # print(is_nlp, '\t', column)

            if mark_for_removal:
                self.features_to_remove.append(column)
            else:
                self.feature_type_family[type_family].append(column)
                if type_family == 'object':
                    self.features_categorical.append(column)
                elif type_family != 'datetime':
                    self.features_to_keep_raw.append(column)
                self.feature_types[dtype].append(column)

        pass

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

    def generate_text_features(self, X: Series, feature: str) -> DataFrame:
        X: DataFrame = X.to_frame(name=feature)
        X[feature + '.char_count'] = [self.char_count(value) for value in X[feature]]
        X[feature + '.word_count'] = [self.word_count(value) for value in X[feature]]
        X[feature + '.capital_ratio'] = [self.capital_ratio(value) for value in X[feature]]
        X[feature + '.lower_ratio'] = [self.lower_ratio(value) for value in X[feature]]
        X[feature + '.digit_ratio'] = [self.digit_ratio(value) for value in X[feature]]
        X[feature + '.special_ratio'] = [self.special_ratio(value) for value in X[feature]]

        symbols = ['!', '?', '@', '%', '$', '*', '&', '#', '^', '.', ':', ' ', '/', ';', '-', '=']
        for symbol in symbols:
            X[feature + '.symbol_count.' + symbol] = [self.symbol_in_string_count(value, symbol) for value in X[feature]]
            X[feature + '.symbol_ratio.' + symbol] = X[feature + '.symbol_count.' + symbol] / X[feature + '.char_count']
            X[feature + '.symbol_ratio.' + symbol].fillna(0, inplace=True)

        X = X.drop(feature, axis=1)

        return X

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
    def get_type_family(type):
        try:
            if 'datetime' in type.name:
                return 'datetime'
            elif np.issubdtype(type, np.integer):
                return 'int'
            elif np.issubdtype(type, np.floating):
                return 'float'
        except Exception as err:
            logger.exception('Warning: dtype %s is not recognized as a valid dtype by numpy! AutoGluon may incorrectly handle this feature...')
            logger.exception(err)

        if type.name in ['bool', 'bool_']:
            return 'bool'
        elif type.name in ['str', 'string', 'object']:
            return 'object'
        else:
            return type.name

    @staticmethod
    def word_count(string):
        return len(string.split())

    @staticmethod
    def char_count(string):
        return len(string)

    @staticmethod
    def special_ratio(string):
        string = string.replace(' ', '')
        if not string:
            return 0
        new_str = re.sub(r'[\w]+', '', string)
        return len(new_str) / len(string)

    @staticmethod
    def digit_ratio(string):
        string = string.replace(' ', '')
        if not string:
            return 0
        return sum(c.isdigit() for c in string) / len(string)

    @staticmethod
    def lower_ratio(string):
        string = string.replace(' ', '')
        if not string:
            return 0
        return sum(c.islower() for c in string) / len(string)

    @staticmethod
    def capital_ratio(string):
        string = string.replace(' ', '')
        if not string:
            return 0
        return sum(1 for c in string if c.isupper()) / len(string)

    @staticmethod
    def symbol_in_string_count(string, character):
        if not string:
            return 0
        return sum(1 for c in string if c == character)

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

    def save_self(self, path):
        save_pkl.save(path=path, object=self)
