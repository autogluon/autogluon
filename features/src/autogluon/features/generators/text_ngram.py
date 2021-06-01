import copy
import logging
import traceback

import numpy as np
import pandas as pd
import psutil
from pandas import DataFrame, Series
from sklearn.feature_selection import SelectKBest, f_classif, f_regression

from autogluon.core.features.types import S_TEXT, S_TEXT_NGRAM

from .abstract import AbstractFeatureGenerator
from ..vectorizers import get_ngram_freq, downscale_vectorizer, vectorizer_auto_ml_default

logger = logging.getLogger(__name__)


# TODO: Add argument to define the text preprocessing logic
# TODO: Add argument to output ngrams as a sparse matrix
# TODO: Add HashingVectorizer support
# TODO: Add TFIDF support
# TODO: Documentation
class TextNgramFeatureGenerator(AbstractFeatureGenerator):
    """
    Generates ngram features from text features.

    Parameters
    ----------
    vectorizer : :class:`sklearn.feature_extraction.text.CountVectorizer`, default CountVectorizer(min_df=30, ngram_range=(1, 3), max_features=10000, dtype=np.uint8)
        sklearn CountVectorizer which is used to generate the ngrams given the text data.
    vectorizer_strategy : str, default 'combined'
        If 'combined', all text features are concatenated together to fit the vectorizer. Features generated in this way have their names prepended with '__nlp__.'.
        If 'separate', all text features are fit separately with their own copy of the vectorizer. Their ngram features are then concatenated together to form the output.
        If 'both', the outputs of 'combined' and 'separate' are concatenated together to form the output.
        It is generally recommended to keep vectorizer_strategy as 'combined' unless the text features are not associated with each-other, as fitting separate vectorizers could increase memory usage and model training time.
        Valid values: ['combined', 'separate', 'both']
    max_memory_ratio : float, default 0.15
        Safety measure to avoid out-of-memory errors downstream in model training.
        The number of ngrams generated will be capped to take at most max_memory_ratio proportion of total available memory, treating the ngrams as float32 values.
        ngram features will be removed in least frequent to most frequent order.
        Note: For vectorizer_strategy values other than 'combined', the resulting ngrams may use more than this value.
        It is recommended to only increase this value above 0.15 if confident that higher values will not result in out-of-memory errors.
    **kwargs :
        Refer to :class:`AbstractFeatureGenerator` documentation for details on valid key word arguments.
    """
    def __init__(self, vectorizer=None, vectorizer_strategy='combined', max_memory_ratio=0.15, prefilter_tokens=False, prefilter_token_count=100, **kwargs):
        super().__init__(**kwargs)
        self.vectorizers = []
        # TODO: 0.20 causes OOM error with 64 GB ram on NN with several datasets. LightGBM and CatBoost succeed
        # TODO: Finetune this, or find a better way to ensure stability
        # TODO: adjust max_memory_ratio correspondingly if prefilter_tokens==True
        self.max_memory_ratio = max_memory_ratio  # Ratio of maximum memory the output ngram features are allowed to use in dense int32 form.
        
        if vectorizer is None:
            self.vectorizer_default_raw = vectorizer_auto_ml_default()
        else:
            self.vectorizer_default_raw = vectorizer

        if vectorizer_strategy not in ['combined', 'separate', 'both']:
            raise ValueError(f"vectorizer_strategy must be one of {['combined', 'separate', 'both']}, but value is: {vectorizer_strategy}")
        self.vectorizer_strategy = vectorizer_strategy
        self.vectorizer_features = None
        self.prefilter_tokens = prefilter_tokens 
        self.prefilter_token_count = prefilter_token_count 
        self.token_mask = None
        self._feature_names_dict = dict()

    def _fit_transform(self, X: DataFrame, y: Series = None, problem_type: str = None, **kwargs) -> (DataFrame, dict):
        
        X_out = self._fit_transform_ngrams(X)
        
        if (self.prefilter_tokens and self.prefilter_token_count>=X_out.shape[1]):
            logger.warning('`prefilter_tokens` was enabled but `prefilter_token_count` larger than the vocabulary. Disabling `prefilter_tokens`.')
            self.prefilter_tokens=False

        if self.prefilter_tokens and problem_type not in ['binary','regression']:
           logger.warning('`prefilter_tokens` was enabled but invalid `problem_type`. Disabling `prefilter_tokens`.')
           self.prefilter_tokens = False

        if self.prefilter_tokens and y is None:
           logger.warning('`prefilter_tokens` was enabled but `y` values were not provided to fit_transform. Disabling `prefilter_tokens`.')
           self.prefilter_tokens = False

        if self.prefilter_tokens:
            scoring_function = f_classif if problem_type=='binary' else f_regression
            selector = SelectKBest(scoring_function, k=self.prefilter_token_count)
            selector.fit(X_out, y)
            self.token_mask = selector.get_support()
            X_out = X_out[ X_out.columns[self.token_mask] ] # select the columns that are most correlated with y

        type_family_groups_special = {
            S_TEXT_NGRAM: list(X_out.columns)
        }
        return X_out, type_family_groups_special

    def _transform(self, X: DataFrame) -> DataFrame:
        # TODO: Optimize for inference
        if not self.features_in:
            return DataFrame(index=X.index)
        try:
            X_out = self._generate_ngrams(X=X)
            if self.prefilter_tokens:
                X_out = X_out[ X_out.columns[self.token_mask] ] # select the columns identified during training
        except Exception:
            self._log(40, '\tError: OOM error during NLP feature transform, unrecoverable. Increase memory allocation or reduce data size to avoid this error.')
            raise
        return X_out

    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return dict(required_special_types=[S_TEXT])

    def _fit_transform_ngrams(self, X):
        if not self.features_in:
            return DataFrame(index=X.index)
        features_nlp_to_remove = []
        if self.vectorizer_strategy == 'combined':
            self.vectorizer_features = ['__nlp__']
        elif self.vectorizer_strategy == 'separate':
            self.vectorizer_features = copy.deepcopy(self.features_in)
        elif self.vectorizer_strategy == 'both':
            self.vectorizer_features = ['__nlp__'] + copy.deepcopy(self.features_in)
        else:
            raise ValueError(f"vectorizer_strategy must be one of {['combined', 'separate', 'both']}, but value is: {self.vectorizer_features}")
        self._log(20, f'Fitting {self.vectorizer_default_raw.__class__.__name__} for text features: ' + str(self.features_in), self.log_prefix + '\t')
        self._log(15, f'{self.vectorizer_default_raw}', self.log_prefix + '\t\t')
        for nlp_feature in self.vectorizer_features:
            # TODO: Preprocess text?
            if nlp_feature == '__nlp__':  # Combine Text Fields
                text_list = list(set(['. '.join(row) for row in X[self.features_in].values]))
            else:
                text_list = list(X[nlp_feature].drop_duplicates().values)
            vectorizer_raw = copy.deepcopy(self.vectorizer_default_raw)
            try:
                vectorizer_fit, _ = self._train_vectorizer(text_list, vectorizer_raw)  # Don't use transform_matrix output because it may contain fewer rows due to drop_duplicates call.
                self._log(20, f'{vectorizer_fit.__class__.__name__} fit with vocabulary size = {len(vectorizer_fit.vocabulary_)}', self.log_prefix + '\t')
            except ValueError:
                self._log(30, f"Removing text_ngram feature due to error: '{nlp_feature}'", self.log_prefix + '\t')
                if nlp_feature == '__nlp__':
                    self.vectorizer_features = []
                    features_nlp_to_remove = self.features_in
                    break
                else:
                    features_nlp_to_remove.append(nlp_feature)
            else:
                self.vectorizers.append(vectorizer_fit)
        self._remove_features_in(features_nlp_to_remove)

        downsample_ratio = None
        nlp_failure_count = 0
        X_text_ngram = None
        keep_trying_nlp = True
        while keep_trying_nlp:
            try:
                X_text_ngram = self._generate_ngrams(X=X, downsample_ratio=downsample_ratio)
                keep_trying_nlp = False
            except Exception as err:
                nlp_failure_count += 1
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
                    self._log(30, 'Warning: ngrams generation resulted in OOM error, removing ngrams features. If you want to use ngrams for this problem, increase memory allocation for AutoGluon.', self.log_prefix + '\t')
                    self._log(10, str(err))
                    self.vectorizers = []
                    self.features_in = []
                    keep_trying_nlp = False
                else:
                    self._log(20, 'Warning: ngrams generation resulted in OOM error, attempting to reduce ngram feature count. If you want to optimally use ngrams for this problem, increase memory allocation for AutoGluon.', self.log_prefix + '\t')
                    self._log(10, str(err))
                    downsample_ratio = 0.25
        if X_text_ngram is None:
            X_text_ngram = DataFrame(index=X.index)
        return X_text_ngram

    def _generate_ngrams(self, X, downsample_ratio: int = None):
        X_nlp_features_combined = []
        for nlp_feature, vectorizer_fit in zip(self.vectorizer_features, self.vectorizers):
            if nlp_feature == '__nlp__':
                text_data = ['. '.join(row) for row in X.values]
            else:
                text_data = X[nlp_feature].values
            transform_matrix = vectorizer_fit.transform(text_data)

            if not self._is_fit:
                transform_matrix = self._adjust_vectorizer_memory_usage(transform_matrix=transform_matrix, text_data=text_data, vectorizer_fit=vectorizer_fit, downsample_ratio=downsample_ratio)
                nlp_features_names = vectorizer_fit.get_feature_names()
                nlp_features_names_final = np.array([f'{nlp_feature}.{x}' for x in nlp_features_names]
                                                    + [f'{nlp_feature}._total_']
                                                    )
                self._feature_names_dict[nlp_feature] = nlp_features_names_final

            transform_array = transform_matrix.toarray()
            # This count could technically overflow in absurd situations. Consider making dtype a variable that is computed.
            nonzero_count = np.count_nonzero(transform_array, axis=1).astype(np.uint16)
            transform_array = np.append(transform_array, np.expand_dims(nonzero_count, axis=1), axis=1)
            X_nlp_features = pd.DataFrame(transform_array, columns=self._feature_names_dict[nlp_feature], index=X.index)  # TODO: Consider keeping sparse
            X_nlp_features_combined.append(X_nlp_features)

        if X_nlp_features_combined:
            if len(X_nlp_features_combined) == 1:
                X_nlp_features_combined = X_nlp_features_combined[0]
            else:
                X_nlp_features_combined = pd.concat(X_nlp_features_combined, axis=1)
        else:
            X_nlp_features_combined = DataFrame(index=X.index)

        return X_nlp_features_combined

    # TODO: REMOVE NEED FOR text_data input!
    def _adjust_vectorizer_memory_usage(self, transform_matrix, text_data, vectorizer_fit, downsample_ratio: int = None):
        # This assumes that the ngrams eventually turn into int32/float32 downstream
        predicted_ngrams_memory_usage_bytes = len(text_data) * 4 * (transform_matrix.shape[1] + 1) + 80
        mem_avail = psutil.virtual_memory().available
        mem_rss = psutil.Process().memory_info().rss
        predicted_rss = mem_rss + predicted_ngrams_memory_usage_bytes
        predicted_percentage = predicted_rss / mem_avail
        if downsample_ratio is None:
            if self.max_memory_ratio is not None and predicted_percentage > self.max_memory_ratio:
                downsample_ratio = self.max_memory_ratio / predicted_percentage
                self._log(30, 'Warning: Due to memory constraints, ngram feature count is being reduced. Allocate more memory to maximize model quality.')

        if downsample_ratio is not None:
            if (downsample_ratio >= 1) or (downsample_ratio <= 0):
                raise ValueError(f'downsample_ratio must be >0 and <1, but downsample_ratio is {downsample_ratio}')
            vocab_size = len(vectorizer_fit.vocabulary_)
            downsampled_vocab_size = int(np.floor(vocab_size * downsample_ratio))
            self._log(20, f'Reducing Vectorizer vocab size from {vocab_size} to {downsampled_vocab_size} to avoid OOM error')
            ngram_freq = get_ngram_freq(vectorizer=vectorizer_fit, transform_matrix=transform_matrix)
            downscale_vectorizer(vectorizer=vectorizer_fit, ngram_freq=ngram_freq, vocab_size=downsampled_vocab_size)
            # TODO: This doesn't have to be done twice, can update transform matrix based on new vocab instead of calling .transform
            #  If we have this functionality, simply update transform_matrix each time OOM occurs instead of re-calling .transform
            transform_matrix = vectorizer_fit.transform(text_data)

        return transform_matrix

    @staticmethod
    def _train_vectorizer(text_data: list, vectorizer):
        transform_matrix = vectorizer.fit_transform(text_data)  # TODO: Consider upgrading to pandas 0.25.0 to benefit from sparse attribute improvements / bug fixes! https://pandas.pydata.org/pandas-docs/stable/whatsnew/v0.25.0.html
        vectorizer.stop_words_ = None  # Reduces object size by 100x+ on large datasets, no effect on usability
        return vectorizer, transform_matrix

    def _remove_features_in(self, features):
        super()._remove_features_in(features)
        if features:
            self.vectorizer_features = [feature for feature in self.vectorizer_features if feature not in features]
