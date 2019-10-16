
import pandas as pd
from pandas import DataFrame, Series
import numpy as np

from tabular.feature_generators.abstract_feature_generator import AbstractFeatureGenerator
# from fastai.tabular.transform import add_datepart
from tabular.sandbox.techops.vectorizers import vectorizer_auto_ml_default
import copy


class AutoMLFeatureGenerator(AbstractFeatureGenerator):
    def __init__(self, enable_nlp_vectorizer_features=True, enable_nlp_ratio_features=True, enable_categorical_features=True, enable_raw_features=True, enable_datetime_features=True):
        super().__init__()
        self.enable_nlp_features = enable_nlp_vectorizer_features
        self.enable_nlp_ratio_features = enable_nlp_ratio_features
        self.enable_categorical_features = enable_categorical_features
        self.enable_raw_features = enable_raw_features
        self.enable_datetime_features = enable_datetime_features
        self.vectorizer_default_raw = vectorizer_auto_ml_default()
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
        if self.enable_nlp_ratio_features and self.features_nlp:
            for nlp_feature in self.features_nlp:
                X_text_features = self.generate_text_features(X[nlp_feature], nlp_feature)
                if not self.fit:
                    self.features_binned += list(X_text_features.columns)
                X_text_features_combined.append(X_text_features)
            X_text_features_combined = pd.concat(X_text_features_combined, axis=1)

        X = self.preprocess(X)

        if self.enable_datetime_features and self.features_datetime:
            for datetime_feature in self.features_datetime:
                X_features[datetime_feature] = pd.to_datetime(X[datetime_feature], infer_datetime_format=True)
                X_features[datetime_feature] = pd.to_numeric(X_features[datetime_feature])  # TODO: Use actual date info, call self.generate_datetime_features()
        
        # TODO: Experiment with concatenating nlp fields to a single feature for a single vectorizer fit
        if self.enable_nlp_features and self.features_nlp:
            if not self.fit:
                for nlp_feature in self.features_nlp:
                    # TODO: Preprocess text?
                    print('fitting vectorizer for', nlp_feature, '...')
                    text_list = (list(X[nlp_feature].drop_duplicates().values))
                    vectorizer_raw = copy.deepcopy(self.vectorizer_default_raw)
                    vectorizer_fit, _ = self.train_vectorizer(text_list, vectorizer_raw)
                    self.vectorizers.append(vectorizer_fit)
            X_nlp_features_combined = []
            for i, nlp_feature in enumerate(self.features_nlp):
                vectorizer_fit = self.vectorizers[i]
                nlp_features_names = vectorizer_fit.get_feature_names()

                X_nlp_features = pd.DataFrame(vectorizer_fit.transform(X[nlp_feature].values).toarray())  # FIXME
                X_nlp_features.columns = [nlp_feature + '.' + str(x) for x in nlp_features_names]
                X_nlp_features[nlp_feature + '._total_'] = X_nlp_features.gt(0).sum(axis=1)

                X_nlp_features_combined.append(X_nlp_features)
            X_nlp_features_combined = pd.concat(X_nlp_features_combined, axis=1)
            print(X_nlp_features_combined)
            X_features = X_features.join(X_nlp_features_combined)

        if self.enable_categorical_features and self.features_categorical:
            X_categoricals = X[self.features_categorical]
            X_categoricals = X_categoricals.astype('category')
            X_features = X_features.join(X_categoricals)

        if self.enable_raw_features and self.features_to_keep_raw:
            X_features = X_features.join(X[self.features_to_keep_raw])

        if self.enable_nlp_ratio_features and self.features_nlp:
            X_features = X_features.join(X_text_features_combined)

        print('transformed...')

        return X_features
