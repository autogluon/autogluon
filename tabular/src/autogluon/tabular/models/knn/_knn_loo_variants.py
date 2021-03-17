import numpy as np
from scipy import stats
from sklearn.neighbors import KNeighborsClassifier as _KNeighborsClassifier, KNeighborsRegressor as _KNeighborsRegressor
from sklearn.neighbors._base import _get_weights
from sklearn.utils.extmath import weighted_mode

# These are variants of sklearn KNN models which have the ability to calculate unbiased predictions on the training data via leave-one-out (LOO) calculation.
# TODO: Consider contributing to sklearn officially
# TODO: This uses private methods in sklearn, could potentially break without warning in future sklearn releases
# TODO: Code is largely identical to `predict` and `predict_proba` methods, but due to how those methods are coded, we can't call them directly.
#  This means if code within those methods changes, the LOO equivalents may start to become outdated.

__all__ = ['KNeighborsClassifier', 'KNeighborsRegressor']


class KNeighborsClassifier(_KNeighborsClassifier):
    def predict_loo(self):
        """Predict the class labels for the training data via leave-one-out.

        Returns
        -------
        y : ndarray of shape (n_queries,) or (n_queries, n_outputs)
            Class labels for each training data sample.
        """

        neigh_dist, neigh_ind = self.kneighbors()
        classes_ = self.classes_
        _y = self._y
        if not self.outputs_2d_:
            _y = self._y.reshape((-1, 1))
            classes_ = [self.classes_]

        n_outputs = len(classes_)
        n_queries = len(neigh_dist)
        weights = _get_weights(neigh_dist, self.weights)

        y_pred = np.empty((n_queries, n_outputs), dtype=classes_[0].dtype)
        for k, classes_k in enumerate(classes_):
            if weights is None:
                mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
            else:
                mode, _ = weighted_mode(_y[neigh_ind, k], weights, axis=1)

            mode = np.asarray(mode.ravel(), dtype=np.intp)
            y_pred[:, k] = classes_k.take(mode)

        if not self.outputs_2d_:
            y_pred = y_pred.ravel()

        return y_pred

    def predict_proba_loo(self):
        """Return probability estimates for the training data via leave-one-out.

        Returns
        -------
        p : ndarray of shape (n_queries, n_classes), or a list of n_outputs
            of such arrays if n_outputs > 1.
            The class probabilities of the training data samples. Classes are ordered
            by lexicographic order.
        """

        neigh_dist, neigh_ind = self.kneighbors()

        classes_ = self.classes_
        _y = self._y
        if not self.outputs_2d_:
            _y = self._y.reshape((-1, 1))
            classes_ = [self.classes_]

        n_queries = len(neigh_dist)

        weights = _get_weights(neigh_dist, self.weights)
        if weights is None:
            weights = np.ones_like(neigh_ind)

        all_rows = np.arange(n_queries)
        probabilities = []
        for k, classes_k in enumerate(classes_):
            pred_labels = _y[:, k][neigh_ind]
            proba_k = np.zeros((n_queries, classes_k.size))

            # a simple ':' index doesn't work right
            for i, idx in enumerate(pred_labels.T):  # loop is O(n_neighbors)
                proba_k[all_rows, idx] += weights[:, i]

            # normalize 'votes' into real [0,1] probabilities
            normalizer = proba_k.sum(axis=1)[:, np.newaxis]
            normalizer[normalizer == 0.0] = 1.0
            proba_k /= normalizer

            probabilities.append(proba_k)

        if not self.outputs_2d_:
            probabilities = probabilities[0]

        return probabilities


class KNeighborsRegressor(_KNeighborsRegressor):
    def predict_loo(self):
        """Predict the target for the training data via leave-one-out.

        Returns
        -------
        y : ndarray of shape (n_queries,) or (n_queries, n_outputs), dtype=int
            Target values.
        """
        neigh_dist, neigh_ind = self.kneighbors()

        weights = _get_weights(neigh_dist, self.weights)

        _y = self._y
        if _y.ndim == 1:
            _y = _y.reshape((-1, 1))

        if weights is None:
            y_pred = np.mean(_y[neigh_ind], axis=1)
        else:
            y_pred = np.empty((len(neigh_dist), _y.shape[1]), dtype=np.float64)
            denom = np.sum(weights, axis=1)

            for j in range(_y.shape[1]):
                num = np.sum(_y[neigh_ind, j] * weights, axis=1)
                y_pred[:, j] = num / denom

        if self._y.ndim == 1:
            y_pred = y_pred.ravel()

        return y_pred
