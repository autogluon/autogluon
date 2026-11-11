from unittest.mock import Mock

import numpy as np
import pytest
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from autogluon.tabular.models.knn.knn_utils import FAISSNeighborsClassifier, FAISSNeighborsRegressor


@pytest.mark.parametrize("n_queries", [1, 3])
@pytest.mark.parametrize("n_neighbors", [1, 3])
@pytest.mark.parametrize("weights", ["uniform", "distance"])
@pytest.mark.parametrize("classification", [False, True])
def test_faiss_neighbors_preserve_query_and_neighbor_axes(n_queries, n_neighbors, weights, classification):
    X = np.array([[0.0], [1.0], [3.0], [5.0], [9.0]], dtype=np.float32)
    X_query = np.array([[0.2], [4.2], [8.0]], dtype=np.float32)[:n_queries]
    y = np.array([0, 0, 1, 1, 1]) if classification else np.array([0.0, 1.0, 4.0, 10.0, 20.0])
    sklearn_type = KNeighborsClassifier if classification else KNeighborsRegressor
    reference = sklearn_type(n_neighbors=n_neighbors, weights=weights).fit(X, y)
    distances, indices = reference.kneighbors(X_query)

    # Exercise the prediction wrappers without requiring an optional FAISS installation.
    model_type = FAISSNeighborsClassifier if classification else FAISSNeighborsRegressor
    model = model_type.__new__(model_type)
    model.index = Mock()
    model.index.search.return_value = (distances, indices)
    model.n_neighbors = n_neighbors
    model.weights = weights
    if classification:
        model.labels = y
        model.classes = np.unique(y)
    else:
        model.y = y

    prediction = model.predict(X_query)
    np.testing.assert_allclose(prediction.ravel(), reference.predict(X_query))
    if classification:
        np.testing.assert_allclose(model.predict_proba(X_query), reference.predict_proba(X_query))
    else:
        assert prediction.shape == (n_queries,)
