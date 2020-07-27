import copy
import logging
import traceback

import numpy as np
import pandas as pd
import psutil
from pandas import DataFrame

from .abstract import AbstractFeatureGenerator
from ..vectorizers import get_ngram_freq, downscale_vectorizer, vectorizer_auto_ml_default

logger = logging.getLogger(__name__)


# TODO: Add verbose descriptions of each special dtype this generator can create.
class TextNgramFeatureGenerator(AbstractFeatureGenerator):
    def __init__(self, vectorizer=None, **kwargs):
        super().__init__(**kwargs)

        self.vectorizers = []

        if vectorizer is None:
            self.vectorizer_default_raw = vectorizer_auto_ml_default()
        else:
            self.vectorizer_default_raw = vectorizer

    def fit(self, X, y=None):
        self.fit_transform(X, y=y)

    def _fit_transform(self, X, y=None):
        X_out = self._transform(X)
        type_family_groups_special = dict(
            text_ngram=list(X_out.columns)
        )
        return X_out, type_family_groups_special

    def _transform(self, X):
        return self._generate_features_text_ngram(X)

    def _generate_features_text_ngram(self, X: DataFrame):
        X_text_ngram = None
        if self.features_in:
            # Combine Text Fields
            features_nlp_current = ['__nlp__']

            if not self._is_fit:
                features_nlp_to_remove = []
                logger.log(15, 'Fitting vectorizer for text features: ' + str(self.features_in))
                for nlp_feature in features_nlp_current:
                    # TODO: Preprocess text?
                    if nlp_feature == '__nlp__':
                        text_list = list(set(['. '.join(row) for row in X[self.features_in].values]))
                    else:
                        text_list = list(X[nlp_feature].drop_duplicates().values)
                    vectorizer_raw = copy.deepcopy(self.vectorizer_default_raw)
                    try:
                        vectorizer_fit, _ = self._train_vectorizer(text_list, vectorizer_raw)
                        self.vectorizers.append(vectorizer_fit)
                    except ValueError:
                        logger.debug("Removing 'text_ngram' features due to error")
                        features_nlp_to_remove = self.features_in

                self.features_in = [feature for feature in self.features_in if feature not in features_nlp_to_remove]

            downsample_ratio = None
            nlp_failure_count = 0
            keep_trying_nlp = True
            while keep_trying_nlp:
                try:
                    X_text_ngram = self._generate_text_ngrams(X=X, features_nlp_current=features_nlp_current, downsample_ratio=downsample_ratio)
                    keep_trying_nlp = False
                except Exception as err:
                    nlp_failure_count += 1
                    if self._is_fit:
                        logger.exception('Error: OOM error during NLP feature transform, unrecoverable. Increase memory allocation or reduce data size to avoid this error.')
                        raise
                    traceback.print_tb(err.__traceback__)

                    X_text_ngram = None
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
                        self.features_in = []
                        keep_trying_nlp = False
                    else:
                        logger.log(15, 'Warning: ngrams generation resulted in OOM error, attempting to reduce ngram feature count. If you want to optimally use ngrams for this problem, increase memory allocation for AutoGluon.')
                        logger.debug(str(err))
                        downsample_ratio = 0.25
        if X_text_ngram is None:
            X_text_ngram = DataFrame(index=X)
        return X_text_ngram

    def _generate_text_ngrams(self, X, features_nlp_current, downsample_ratio: int = None):
        X_nlp_features_combined = []
        for i, nlp_feature in enumerate(features_nlp_current):
            vectorizer_fit = self.vectorizers[i]

            if nlp_feature == '__nlp__':
                text_data = ['. '.join(row) for row in X[self.features_in].values]
            else:
                text_data = X[nlp_feature].values
            transform_matrix = vectorizer_fit.transform(text_data)

            if not self._is_fit:
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

        if X_nlp_features_combined:
            X_nlp_features_combined = pd.concat(X_nlp_features_combined, axis=1)

        return X_nlp_features_combined

    @staticmethod
    def _train_vectorizer(text_list, vectorizer):
        logger.log(15, 'Fitting vectorizer...')
        transform_matrix = vectorizer.fit_transform(text_list)  # TODO: Consider upgrading to pandas 0.25.0 to benefit from sparse attribute improvements / bug fixes! https://pandas.pydata.org/pandas-docs/stable/whatsnew/v0.25.0.html
        vectorizer.stop_words_ = None  # Reduces object size by 100x+ on large datasets, no effect on usability
        logger.log(15, f'Vectorizer fit with vocabulary size = {len(vectorizer.vocabulary_)}')
        return vectorizer, transform_matrix
