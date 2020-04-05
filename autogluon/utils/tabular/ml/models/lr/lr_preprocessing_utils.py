from scipy.sparse import hstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder


class OheFeaturesGenerator(BaseEstimator, TransformerMixin):
    missing_category_str = '!missing!'

    def __init__(self, cats_cols):
        self._feature_names = []
        self.cats = cats_cols
        self.ohe_encs = None
        self.labels = None

    def fit(self, X, y=None):
        self.ohe_encs = {f: OneHotEncoder(handle_unknown='ignore') for f in self.cats}
        self.labels = {}

        for c in self.cats:
            self.ohe_encs[c].fit(self._normalize(X[c]))
            self.labels[c] = self.ohe_encs[c].categories_
        return self

    def transform(self, X, y=None):
        Xs = [self.ohe_encs[c].transform(self._normalize(X[c])) for c in self.cats]

        # Update feature names
        self._feature_names = []
        for k, v in self.labels.items():
            for f in k + '_' + v[0]:
                self._feature_names.append(f)

        return hstack(Xs)

    def _normalize(self, col):
        return col.astype(str).fillna(self.missing_category_str).values.reshape(-1, 1)

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


class NumericDataPreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self, cont_cols):
        self.cont_cols = cont_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X[self.cont_cols].copy()
        return X.values.tolist()
