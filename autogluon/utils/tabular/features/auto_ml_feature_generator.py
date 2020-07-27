import copy
import logging
import re
import traceback

import numpy as np
import pandas as pd
import psutil
from pandas import DataFrame, Series

from .abstract_feature_generator import AbstractFeatureGenerator
from .vectorizers import get_ngram_freq, downscale_vectorizer
from .vectorizers import vectorizer_auto_ml_default

logger = logging.getLogger(__name__)


# TODO: Add verbose descriptions of each special dtype this generator can create.
class AutoMLFeatureGenerator(AbstractFeatureGenerator):
    def __init__(self, enable_text_ngram_features=True, enable_text_special_features=True,
                 enable_categorical_features=True, enable_raw_features=True, enable_datetime_features=True,
                 vectorizer=None):
        super().__init__()
        self.enable_nlp_features = enable_text_ngram_features
        self.enable_text_special_features = enable_text_special_features
        self.enable_categorical_features = enable_categorical_features
        self.enable_raw_features = enable_raw_features
        self.enable_datetime_features = enable_datetime_features
        if vectorizer is None:
            self.vectorizer_default_raw = vectorizer_auto_ml_default()
        else:
            self.vectorizer_default_raw = vectorizer
        self.vectorizers = []

    def _compute_feature_transformations(self):
        """Determines which features undergo which feature transformations."""
        if self.enable_categorical_features:
            if 'object' in self.feature_type_family:
                self.feature_transformations['category'] += self.feature_type_family['object']
            if 'text' in self.feature_type_family:
                self.feature_transformations['category'] += self.feature_type_family['text']

        if 'text' in self.feature_type_family:
            text_features = self.feature_type_family['text']
            if self.enable_text_special_features:
                self.feature_transformations['text_special'] += text_features
            if self.enable_nlp_features:
                self.feature_transformations['text_ngram'] += text_features

        if 'datetime' in self.feature_type_family:
            datetime_features = self.feature_type_family['datetime']
            if self.enable_datetime_features:
                self.feature_transformations['datetime'] += datetime_features

        if self.enable_raw_features:
            for type_family in self.feature_type_family:
                if type_family not in ['object', 'text', 'datetime']:
                    self.feature_transformations['raw'] += self.feature_type_family[type_family]

    # TODO: Parallelize with decorator!
    def generate_features(self, X: DataFrame):
        if not self.fit:
            self._compute_feature_transformations()
        X_features = pd.DataFrame(index=X.index)
        for column in X.columns:
            if X[column].dtype.name == 'object':
                X[column].fillna('', inplace=True)
            else:
                X[column].fillna(np.nan, inplace=True)

        X = self.preprocess(X)

        if self.feature_transformations['raw']:
            X_features = X_features.join(X[self.feature_transformations['raw']])

        if self.feature_transformations['category']:
            X_categoricals = X[self.feature_transformations['category']]
            # TODO: Add stateful categorical generator, merge rare cases to an unknown value
            # TODO: What happens when training set has no unknown/rare values but test set does? What models can handle this?
            if not self.fit:
                if 'text' in self.feature_type_family:
                    self.feature_type_family_generated['text_as_category'] += self.feature_type_family['text']
            X_categoricals = X_categoricals.astype('category')
            X_features = X_features.join(X_categoricals)

        if self.feature_transformations['text_special']:
            X_text_special_combined = []
            for nlp_feature in self.feature_transformations['text_special']:
                X_text_special = self.generate_text_special(X[nlp_feature], nlp_feature)
                X_text_special_combined.append(X_text_special)
            X_text_special_combined = pd.concat(X_text_special_combined, axis=1)
            if not self.fit:
                self.features_binned += list(X_text_special_combined.columns)
                self.feature_type_family_generated['text_special'] += list(X_text_special_combined.columns)
            X_features = X_features.join(X_text_special_combined)

        if self.feature_transformations['datetime']:
            for datetime_feature in self.feature_transformations['datetime']:
                X_features[datetime_feature] = pd.to_datetime(X[datetime_feature])
                X_features[datetime_feature] = pd.to_numeric(X_features[datetime_feature])  # TODO: Use actual date info
                if not self.fit:
                    self.feature_type_family_generated['datetime'].append(datetime_feature)
                # TODO: Add fastai date features

        if self.feature_transformations['text_ngram']:
            # Combine Text Fields
            features_nlp_current = ['__nlp__']

            if not self.fit:
                features_nlp_to_remove = []
                logger.log(15, 'Fitting vectorizer for text features: ' + str(self.feature_transformations['text_ngram']))
                for nlp_feature in features_nlp_current:
                    # TODO: Preprocess text?
                    if nlp_feature == '__nlp__':
                        text_list = list(set(['. '.join(row) for row in X[self.feature_transformations['text_ngram']].values]))
                    else:
                        text_list = list(X[nlp_feature].drop_duplicates().values)
                    vectorizer_raw = copy.deepcopy(self.vectorizer_default_raw)
                    try:
                        vectorizer_fit, _ = self.train_vectorizer(text_list, vectorizer_raw)
                        self.vectorizers.append(vectorizer_fit)
                    except ValueError:
                        logger.debug("Removing 'text_ngram' features due to error")
                        features_nlp_to_remove = self.feature_transformations['text_ngram']

                self.feature_transformations['text_ngram'] = [feature for feature in self.feature_transformations['text_ngram'] if feature not in features_nlp_to_remove]

            X_features_cols_prior_to_nlp = list(X_features.columns)
            downsample_ratio = None
            nlp_failure_count = 0
            keep_trying_nlp = True
            while keep_trying_nlp:
                try:
                    X_nlp_features_combined = self.generate_text_ngrams(X=X, features_nlp_current=features_nlp_current, downsample_ratio=downsample_ratio)

                    if self.feature_transformations['text_ngram']:
                        X_features = X_features.join(X_nlp_features_combined)

                    if not self.fit:
                        self.feature_type_family_generated['text_ngram'] += list(X_nlp_features_combined.columns)
                    keep_trying_nlp = False
                except Exception as err:
                    nlp_failure_count += 1
                    if self.fit:
                        logger.exception('Error: OOM error during NLP feature transform, unrecoverable. Increase memory allocation or reduce data size to avoid this error.')
                        raise
                    traceback.print_tb(err.__traceback__)

                    X_features = X_features[X_features_cols_prior_to_nlp]
                    skip_nlp = False
                    for vectorizer in self.vectorizers:
                        vocab_size = len(vectorizer.vocabulary_)
                        if vocab_size <= 50:
                            skip_nlp = True
                            break
                    else:
                        if nlp_failure_count >= 3:
                            skip_nlp = True

                    if skip_nlp:
                        logger.log(15, 'Warning: ngrams generation resulted in OOM error, removing ngrams features. If you want to use ngrams for this problem, increase memory allocation for AutoGluon.')
                        logger.debug(str(err))
                        self.vectorizers = []
                        if 'text_ngram' in self.feature_transformations:
                            self.feature_transformations.pop('text_ngram')
                        if 'text_ngram' in self.feature_type_family_generated:
                            self.feature_type_family_generated.pop('text_ngram')
                        self.enable_nlp_features = False
                        keep_trying_nlp = False
                    else:
                        logger.log(15, 'Warning: ngrams generation resulted in OOM error, attempting to reduce ngram feature count. If you want to optimally use ngrams for this problem, increase memory allocation for AutoGluon.')
                        logger.debug(str(err))
                        downsample_ratio = 0.25

        return X_features

    def generate_text_ngrams(self, X, features_nlp_current, downsample_ratio: int = None):
        X_nlp_features_combined = []
        for i, nlp_feature in enumerate(features_nlp_current):
            vectorizer_fit = self.vectorizers[i]

            if nlp_feature == '__nlp__':
                text_data = ['. '.join(row) for row in X[self.feature_transformations['text_ngram']].values]
            else:
                text_data = X[nlp_feature].values
            transform_matrix = vectorizer_fit.transform(text_data)

            if not self.fit:
                predicted_ngrams_memory_usage_bytes = len(X) * 8 * (transform_matrix.shape[1] + 1) + 80
                mem_avail = psutil.virtual_memory().available
                mem_rss = psutil.Process().memory_info().rss
                # TODO: 0.25 causes OOM error with 72 GB ram on nyc-wendykan-lending-club-loan-data, fails on NN or Catboost, distributed.worker spams logs with memory warnings
                # TODO: 0.20 causes OOM error with 64 GB ram on NN with several datasets. LightGBM and CatBoost succeed
                max_memory_percentage = 0.15  # TODO: Finetune this, or find a better metric
                predicted_rss = mem_rss + predicted_ngrams_memory_usage_bytes
                predicted_percentage = predicted_rss / mem_avail
                if downsample_ratio is None:
                    if predicted_percentage > max_memory_percentage:
                        downsample_ratio = max_memory_percentage / predicted_percentage
                        logger.warning('Warning: Due to memory constraints, ngram feature count is being reduced. Allocate more memory to maximize model quality.')

                if downsample_ratio is not None:
                    if (downsample_ratio >= 1) or (downsample_ratio <= 0):
                        raise ValueError(f'downsample_ratio must be >0 and <1, but downsample_ratio is {downsample_ratio}')
                    vocab_size = len(vectorizer_fit.vocabulary_)
                    downsampled_vocab_size = int(np.floor(vocab_size * downsample_ratio))
                    logger.log(20, f'Reducing Vectorizer vocab size from {vocab_size} to {downsampled_vocab_size} to avoid OOM error')
                    ngram_freq = get_ngram_freq(vectorizer=vectorizer_fit, transform_matrix=transform_matrix)
                    downscale_vectorizer(vectorizer=vectorizer_fit, ngram_freq=ngram_freq, vocab_size=downsampled_vocab_size)
                    # TODO: This doesn't have to be done twice, can update transform matrix based on new vocab instead of calling .transform
                    #  If we have this functionality, simply update transform_matrix each time OOM occurs instead of re-calling .transform
                    transform_matrix = vectorizer_fit.transform(text_data)

            nlp_features_names = vectorizer_fit.get_feature_names()

            X_nlp_features = pd.DataFrame(transform_matrix.toarray())  # FIXME
            X_nlp_features.columns = [f'{nlp_feature}.{x}' for x in nlp_features_names]
            X_nlp_features[nlp_feature + '._total_'] = X_nlp_features.gt(0).sum(axis=1)

            X_nlp_features_combined.append(X_nlp_features)

        if self.feature_transformations['text_ngram']:
            X_nlp_features_combined = pd.concat(X_nlp_features_combined, axis=1)

        return X_nlp_features_combined

    def generate_text_special(self, X: Series, feature: str) -> DataFrame:
        X_text_special: DataFrame = DataFrame(index=X.index)
        X_text_special[feature + '.char_count'] = [self.char_count(value) for value in X]
        X_text_special[feature + '.word_count'] = [self.word_count(value) for value in X]
        X_text_special[feature + '.capital_ratio'] = [self.capital_ratio(value) for value in X]
        X_text_special[feature + '.lower_ratio'] = [self.lower_ratio(value) for value in X]
        X_text_special[feature + '.digit_ratio'] = [self.digit_ratio(value) for value in X]
        X_text_special[feature + '.special_ratio'] = [self.special_ratio(value) for value in X]

        symbols = ['!', '?', '@', '%', '$', '*', '&', '#', '^', '.', ':', ' ', '/', ';', '-', '=']
        for symbol in symbols:
            X_text_special[feature + '.symbol_count.' + symbol] = [self.symbol_in_string_count(value, symbol) for value in X]
            X_text_special[feature + '.symbol_ratio.' + symbol] = X_text_special[feature + '.symbol_count.' + symbol] / X_text_special[feature + '.char_count']
            X_text_special[feature + '.symbol_ratio.' + symbol].fillna(0, inplace=True)

        return X_text_special

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

    @staticmethod
    def train_vectorizer(text_list, vectorizer):
        logger.log(15, 'Fitting vectorizer...')
        transform_matrix = vectorizer.fit_transform(text_list)  # TODO: Consider upgrading to pandas 0.25.0 to benefit from sparse attribute improvements / bug fixes! https://pandas.pydata.org/pandas-docs/stable/whatsnew/v0.25.0.html
        vectorizer.stop_words_ = None  # Reduces object size by 100x+ on large datasets, no effect on usability
        logger.log(15, f'Vectorizer fit with vocabulary size = {len(vectorizer.vocabulary_)}')
        return vectorizer, transform_matrix
