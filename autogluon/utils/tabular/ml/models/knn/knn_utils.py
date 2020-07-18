import numpy as np
from pandas import DataFrame
from scipy.stats import mode
from sklearn.utils.extmath import weighted_mode
from .....try_import import try_import_faiss

import logging


logger = logging.getLogger(__name__)


# Rather than try to import non-public sklearn internals, we implement our own weighting functions here
# These support the same operations as the sklearn functions - at least as far as possible with FAISS
def _check_weights(weights):
    """Check to make sure weights are valid"""
    if weights in (None, 'uniform', 'distance'):
        return weights
    elif callable(weights):
        return weights
    else:
        raise ValueError("weights not recognized: should be 'uniform', 'distance', or a callable function")


def _get_weights(dist, weights):
    """Get the weights from an array of distances and a parameter weights"""
    if weights in (None, 'uniform'):
        return None
    elif weights == 'distance':
        # if user attempts to classify a point that was zero distance from one
        # or more training points, those training points are weighted as 1.0
        # and the other points as 0.0
        with np.errstate(divide='ignore'):
            dist = 1. / dist
        inf_mask = np.isinf(dist)
        inf_row = np.any(inf_mask, axis=1)
        dist[inf_row] = inf_mask[inf_row]
        return dist
    elif callable(weights):
        return weights(dist)
    else:
        raise ValueError("weights not recognized: should be 'uniform', 'distance', or a callable function")


class FAISSNeighborsRegressor:
    def __init__(self, n_neighbors=5, weights='uniform', n_jobs=-1, index_factory_string="Flat"):
        """
        Creates a KNN regressor model based on FAISS. FAISS allows you to compose different
        near-neighbor search algorithms from several different preprocessing / search algorithms
        This composition is specified by the string that is passed to the FAISS index_factory. 
        Here are good guidelines for choosing the index string: 
        https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index

        The model itself is a clone of the sklearn one
        """
        try_import_faiss()
        import faiss
        self.faiss = faiss
        self.index_factory_string = index_factory_string
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.n_jobs = n_jobs
        if n_jobs > 0:
            # global config, affects all faiss indexes
            faiss.omp_set_num_threads(n_jobs)

    def fit(self, X_train, y_train):
        if isinstance(X_train, DataFrame):
            X_train = X_train.to_numpy(dtype=np.float32)
        else:
            X_train = X_train.astype(np.float32)
        if not X_train.flags['C_CONTIGUOUS']:
            X_train = np.ascontiguousarray(X_train)
        d = X_train.shape[1]
        self.index = self.faiss.index_factory(d, self.index_factory_string)
        self.y = np.array(y_train)
        self.index.train(X_train)
        self.index.add(X_train)
        return self

    def predict(self, X):
        X = X.astype(np.float32)
        X = np.ascontiguousarray(X)
        if X.ndim == 1:
            X = X[np.newaxis]
        D, I = self.index.search(X, self.n_neighbors)
        outputs = np.squeeze(self.y[I])

        weights = _get_weights(D, self.weights)

        if weights is None:
            y_pred = np.mean(outputs, axis=1)
        else:
            denom = np.sum(weights, axis=1)
            if outputs.ndim == 1:
                y_pred = np.sum(weights * outputs, axis=1)
                y_pred /= denom
            else:
                y_pred = np.sum(weights * outputs, axis=1)
                y_pred /= denom

        return y_pred

    def __getstate__(self):
        state = {}
        for k, v in self.__dict__.items():
            if (v is not self.index) and (v is not self.faiss):
                state[k] = v
            else:
                state[k] = self.faiss.serialize_index(self.index)
        return state

    def __setstate__(self, state):
        try_import_faiss()
        import faiss
        self.__dict__.update(state)
        self.faiss = faiss
        self.index = self.faiss.deserialize_index(self.index)


class FAISSNeighborsClassifier:
    def __init__(self, n_neighbors=5, weights='uniform', n_jobs=-1, index_factory_string="Flat"):
        """
        Creates a KNN classifier model based on FAISS. FAISS allows you to compose different
        near-neighbor search algorithms from several different preprocessing / search algorithms
        This composition is specified by the string that is passed to the FAISS index_factory. 
        Here are good guidelines for choosing the index string: 
        https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index

        The model itself is a clone of the sklearn one
        """
        try_import_faiss()
        import faiss
        self.faiss = faiss
        self.index_factory_string = index_factory_string
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.classes = []
        self.n_jobs = n_jobs
        if n_jobs > 0:
            # global config, affects all faiss indexes
            faiss.omp_set_num_threads(n_jobs)

    def fit(self, X_train, y_train):
        if isinstance(X_train, DataFrame):
            X_train = X_train.to_numpy(dtype=np.float32)
        else:
            X_train = X_train.astype(np.float32)
        if not X_train.flags['C_CONTIGUOUS']:
            X_train = np.ascontiguousarray(X_train)
        d = X_train.shape[1]
        self.index = self.faiss.index_factory(d, self.index_factory_string)
        self.labels = np.array(y_train)
        self.index.train(X_train)
        self.index.add(X_train)
        self.classes = np.unique(y_train)
        return self

    def predict(self, X):
        X = X.astype(np.float32)
        X = np.ascontiguousarray(X)
        if X.ndim == 1:
            X = X[np.newaxis]
        D, I = self.index.search(X, self.n_neighbors)
        outputs = np.squeeze(self.labels[I])
        weights = _get_weights(D, self.weights)
        if weights is None:
            y_pred, _ = mode(outputs, axis=1)
        else:
            y_pred, _ = weighted_mode(outputs, weights, axis=1)
        return y_pred

    def predict_proba(self, X):
        X = X.astype(np.float32)
        X = np.ascontiguousarray(X)
        if X.ndim == 1:
            X = X[np.newaxis]
        D, I = self.index.search(X, self.n_neighbors)
        outputs = np.squeeze(self.labels[I])
        weights = _get_weights(D, self.weights)
        if weights is None:
            weights = np.ones_like(I)

        probabilities = np.empty((X.shape[0], len(self.classes)), dtype=np.float64)
        for k, class_k in enumerate(self.classes):
            proba_k = np.sum(np.multiply(outputs == class_k, weights), axis=1)
            probabilities[:, k] = proba_k

        normalizer = np.sum(probabilities, axis=1)
        normalizer[normalizer == 0.0] = 1.0
        probabilities /= normalizer[:, np.newaxis]
        return probabilities

    def __getstate__(self):
        state = {}
        for k, v in self.__dict__.items():
            if (v is not self.index) and (v is not self.faiss):
                state[k] = v
            else:
                state[k] = self.faiss.serialize_index(self.index)
        return state

    def __setstate__(self, state):
        try_import_faiss()
        import faiss
        self.__dict__.update(state)
        self.faiss = faiss
        self.index = self.faiss.deserialize_index(self.index)
