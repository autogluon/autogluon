
import pandas as pd
from pandas import DataFrame, Series
import numpy as np
import copy

from tabular.feature_generators.abstract_feature_generator import AbstractFeatureGenerator
# from fastai.tabular.transform import add_datepart
from tabular.ml.vectorizers import vectorizer_auto_ml_default


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
            X_nlp_features_combined = []
            for i, nlp_feature in enumerate(features_nlp_current):
                vectorizer_fit = self.vectorizers[i]
                nlp_features_names = vectorizer_fit.get_feature_names()

                X_nlp_features = pd.DataFrame(vectorizer_fit.transform(X[nlp_feature].values).toarray())  # FIXME
                X_nlp_features.columns = [nlp_feature + '.' + str(x) for x in nlp_features_names]
                X_nlp_features[nlp_feature + '._total_'] = X_nlp_features.gt(0).sum(axis=1)

                self.features_vectorizers = self.features_vectorizers + list(X_nlp_features.columns)

                X_nlp_features_combined.append(X_nlp_features)

            if self.features_nlp:
                X_nlp_features_combined = pd.concat(X_nlp_features_combined, axis=1)
                print(X_nlp_features_combined)
                X_features = X_features.join(X_nlp_features_combined)

        if self.enable_categorical_features and self.features_categorical:
            X_categoricals = X[self.features_categorical]
            X_categoricals = X_categoricals.astype('category')
            X_features = X_features.join(X_categoricals)

        if self.enable_raw_features and self.features_to_keep_raw:
            X_features = X_features.join(X[self.features_to_keep_raw])

        if self.enable_nlp_ratio_features and self.features_nlp_ratio:
            X_features = X_features.join(X_text_features_combined)

        print('transformed...')

        return X_features
