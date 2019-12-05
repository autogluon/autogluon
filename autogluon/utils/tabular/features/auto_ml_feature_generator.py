import pandas as pd
from pandas import DataFrame
import numpy as np
import copy
import traceback
import psutil
# from fastai.tabular.transform import add_datepart

from .abstract_feature_generator import AbstractFeatureGenerator
from .vectorizers import vectorizer_auto_ml_default
from .vectorizers import get_ngram_freq, downscale_vectorizer


class AutoMLFeatureGenerator(AbstractFeatureGenerator):
    def __init__(self, enable_nlp_vectorizer_features=True, enable_nlp_ratio_features=True, enable_categorical_features=True, enable_raw_features=True, enable_datetime_features=True,
                 vectorizer=None):
        super().__init__()
        self.enable_nlp_features = enable_nlp_vectorizer_features
        self.enable_nlp_ratio_features = enable_nlp_ratio_features
        self.enable_categorical_features = enable_categorical_features
        self.enable_raw_features = enable_raw_features
        self.enable_datetime_features = enable_datetime_features
        if vectorizer is None:
            self.vectorizer_default_raw = vectorizer_auto_ml_default()
        else:
            self.vectorizer_default_raw = vectorizer
        self.vectorizers = []

    # TODO: Parallelize with decorator!
    def generate_features(self, X: DataFrame):
        X_features = pd.DataFrame(index=X.index)
        for column in X.columns:
            if X[column].dtype.name == 'object':
                X[column].fillna('', inplace=True)
            else:
                X[column].fillna(np.nan, inplace=True)

        X_text_features_combined = []
        if self.enable_nlp_ratio_features and self.features_nlp_ratio:
            for nlp_feature in self.features_nlp_ratio:
                X_text_features = self.generate_text_features(X[nlp_feature], nlp_feature)
                if not self.fit:
                    self.features_binned += list(X_text_features.columns)
                X_text_features_combined.append(X_text_features)
            X_text_features_combined = pd.concat(X_text_features_combined, axis=1)

        X = self.preprocess(X)

        if self.enable_raw_features and self.features_to_keep_raw:
            X_features = X_features.join(X[self.features_to_keep_raw])

        if self.enable_categorical_features and self.features_categorical:
            X_categoricals = X[self.features_categorical]
            # TODO: Add stateful categorical generator, merge rare cases to an unknown value
            # TODO: What happens when training set has no unknown/rare values but test set does? What models can handle this?
            X_categoricals = X_categoricals.astype('category')
            X_features = X_features.join(X_categoricals)

        if self.enable_nlp_ratio_features and self.features_nlp_ratio:
            X_features = X_features.join(X_text_features_combined)

        if self.enable_datetime_features and self.features_datetime:
            for datetime_feature in self.features_datetime:
                X_features[datetime_feature] = pd.to_datetime(X[datetime_feature])
                X_features[datetime_feature] = pd.to_numeric(X_features[datetime_feature])  # TODO: Use actual date info
                # TODO: Add fastai date features

        if self.enable_nlp_features and self.features_nlp:
            # Combine Text Fields
            txt = ['. '.join(row) for row in X[self.features_nlp].values]
            # print(txt[:10])
            # txt_cleaned = LemmatizerPreprocessor().process(txt)
            # print(txt_cleaned[:10])
            # txt_cleaned_2 = StopWordsRemover().process(txt_cleaned)
            # print(txt_cleaned_2[:10])

            X['__nlp__'] = txt  # Could potentially find faster methods if this ends up being slow

            features_nlp_current = ['__nlp__']

            if not self.fit:
                features_nlp_to_remove = []
                print('fitting vectorizer for nlp features:', self.features_nlp)

                for nlp_feature in features_nlp_current:
                    # TODO: Preprocess text?
                    # print('fitting vectorizer for', nlp_feature, '...')
                    text_list = (list(X[nlp_feature].drop_duplicates().values))
                    vectorizer_raw = copy.deepcopy(self.vectorizer_default_raw)
                    try:
                        vectorizer_fit, _ = self.train_vectorizer(text_list, vectorizer_raw)
                        self.vectorizers.append(vectorizer_fit)
                    except ValueError:
                        print('removing nlp feature')
                        features_nlp_to_remove = self.features_nlp
                    except:
                        raise
                self.features_nlp = [feature for feature in self.features_nlp if feature not in features_nlp_to_remove]
            X_features_cols_prior_to_nlp = list(X_features.columns)
            keep_trying_nlp = True

            downsample_ratio = None
            nlp_failure_count = 0
            while keep_trying_nlp:
                try:
                    X_nlp_features_combined = self.generate_nlp_ngrams(X=X, features_nlp_current=features_nlp_current, downsample_ratio=downsample_ratio)

                    if self.features_nlp:
                        X_features = X_features.join(X_nlp_features_combined)

                    if not self.fit:
                        self.features_vectorizers = self.features_vectorizers + list(X_nlp_features_combined.columns)
                    keep_trying_nlp = False
                except Exception as err:
                    nlp_failure_count += 1
                    if self.fit:
                        print('Error: OOM error during NLP feature transform, unrecoverable. Increase memory allocation or reduce data size to avoid this error.')
                        raise
                    traceback.print_tb(err.__traceback__)

                    X_nlp_features_combined = None
                    X_features = X_features[X_features_cols_prior_to_nlp]
                    skip_nlp = False
                    for vectorizer in self.vectorizers:
                        vocab_size = len(vectorizer.vocabulary_)
                        if vocab_size <= 50:
                            skip_nlp = True
                    if nlp_failure_count >= 3:
                        skip_nlp = True
                    if skip_nlp:
                        print('Warning: ngrams generation resulted in OOM error, removing ngrams features. If you want to use ngrams for this problem, increase memory allocation for AutoGluon.')
                        print(err)
                        self.vectorizers = []
                        self.features_nlp = []
                        self.features_vectorizers = []
                        self.enable_nlp_features = False
                        keep_trying_nlp = False
                    else:
                        print('Warning: ngrams generation resulted in OOM error, attempting to reduce ngram feature count. If you want to optimally use ngrams for this problem, increase memory allocation for AutoGluon.')
                        print(err)
                        downsample_ratio = 0.25

        return X_features

    def generate_nlp_ngrams(self, X, features_nlp_current, downsample_ratio: int = None):
        X_nlp_features_combined = []
        for i, nlp_feature in enumerate(features_nlp_current):
            vectorizer_fit = self.vectorizers[i]

            transform_matrix = vectorizer_fit.transform(X[nlp_feature].values)

            predicted_ngrams_memory_usage_bytes = len(X) * 8 * (transform_matrix.shape[1] + 1) + 80
            mem_avail = psutil.virtual_memory().available
            mem_rss = psutil.Process().memory_info().rss
            # TODO: 0.25 causes OOM error with 72 GB ram on nyc-wendykan-lending-club-loan-data, fails on NN or Catboost, distributed.worker spams logs with memory warnings
            max_memory_percentage = 0.2  # TODO: Finetune this, or find a better metric, this mostly hinges on LightGBM, as it expands all features to float32 upon training start, which causes a huge memory spike. 0.20 is safe, higher might be unsafe
            predicted_rss = mem_rss + predicted_ngrams_memory_usage_bytes
            predicted_percentage = predicted_rss / mem_avail
            if not self.fit:
                if downsample_ratio is None:
                    if predicted_percentage > max_memory_percentage:
                        downsample_ratio = max_memory_percentage / predicted_percentage
                        print('Warning: Due to memory constraints, ngram feature count is being reduced. Allocate more memory to maximize model quality.')

                if downsample_ratio is not None:
                    if (downsample_ratio >= 1) or (downsample_ratio <= 0):
                        raise ValueError('downsample_ratio must be <1 and >0, but downsample_ratio is ' + str(downsample_ratio))
                    vocab_size = len(vectorizer_fit.vocabulary_)
                    downsampled_vocab_size = int(np.floor(vocab_size * downsample_ratio))
                    print('Reducing Vectorizer vocab size from', vocab_size, 'to', downsampled_vocab_size, 'to avoid OOM error.')
                    ngram_freq = get_ngram_freq(vectorizer=vectorizer_fit, transform_matrix=transform_matrix)
                    downscale_vectorizer(vectorizer=vectorizer_fit, ngram_freq=ngram_freq, vocab_size=downsampled_vocab_size)
                    # TODO: This doesn't have to be done twice, can update transform matrix based on new vocab instead of calling .transform
                    #  If we have this functionality, simply update transform_matrix each time OOM occurs instead of re-calling .transform
                    transform_matrix = vectorizer_fit.transform(X[nlp_feature].values)

            nlp_features_names = vectorizer_fit.get_feature_names()

            X_nlp_features = pd.DataFrame(transform_matrix.toarray())  # FIXME
            X_nlp_features.columns = [nlp_feature + '.' + str(x) for x in nlp_features_names]
            X_nlp_features[nlp_feature + '._total_'] = X_nlp_features.gt(0).sum(axis=1)

            X_nlp_features_combined.append(X_nlp_features)

        if self.features_nlp:
            X_nlp_features_combined = pd.concat(X_nlp_features_combined, axis=1)

        return X_nlp_features_combined
