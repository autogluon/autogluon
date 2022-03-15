from sklearn.base import BaseEstimator, TransformerMixin

from autogluon.features.generators import OneHotEncoderFeatureGenerator


class OheFeaturesGenerator(BaseEstimator, TransformerMixin):
    def __init__(self):
        self._feature_names = []
        self._encoder = None

    def fit(self, X, y=None):
        self._encoder = OneHotEncoderFeatureGenerator(max_levels=10000, verbosity=0)
        self._encoder.fit(X)
        self._feature_names = self._encoder.features_out
        return self

    def transform(self, X, y=None):
        return self._encoder.transform_ohe(X)

    def get_feature_names(self):
        return self._feature_names


class NlpDataPreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self, nlp_cols):
        self.nlp_cols = nlp_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X[self.nlp_cols].copy()
        for c in self.nlp_cols:
            X[c] = X[c].astype(str).fillna(' ')
        X = X.apply(' '.join, axis=1).str.replace('[ ]+', ' ', regex=True)
        return X.values.tolist()
