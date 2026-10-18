import numpy as np
import pytest
import sklearn

from autogluon.core.metrics import balanced_accuracy


def _sklearn_balanced_accuracy(y_true, y_pred) -> float:
    with pytest.warns(UserWarning, match="y_pred contains classes not in y_true"):
        return sklearn.metrics.balanced_accuracy_score(y_true, y_pred)


def test_balanced_accuracy_ignores_predicted_classes_absent_from_y_true():
    """A class that is only predicted has no recall and must not count as perfectly recalled.

    Predicting a class absent from `y_true` scored better than predicting a present but wrong class.
    """
    y_true = np.array([0, 0, 1, 1])
    error_to_present_class = balanced_accuracy(y_true, np.array([0, 1, 1, 1]))
    error_to_absent_class = balanced_accuracy(y_true, np.array([0, 2, 1, 1]))
    assert error_to_present_class == pytest.approx(0.75)
    assert error_to_absent_class == pytest.approx(0.75)


@pytest.mark.parametrize("seed", range(5))
def test_balanced_accuracy_multiclass_matches_sklearn_with_absent_classes(seed):
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 3, size=300)
    y_pred = y_true.copy()
    errors = rng.random(300) < 0.3
    y_pred[errors] = rng.integers(0, 5, size=errors.sum())  # classes 3 and 4 never occur in y_true
    assert balanced_accuracy(y_true, y_pred) == pytest.approx(_sklearn_balanced_accuracy(y_true, y_pred))


@pytest.mark.parametrize(
    "y_true,y_pred",
    [
        (np.array([0, 0, 0, 0]), np.array([0, 0, 1, 1])),
        (np.array([1, 1, 1, 1]), np.array([1, 0, 1, 0])),
    ],
)
def test_balanced_accuracy_binary_single_class_y_true_matches_sklearn(y_true, y_pred):
    """With one class in `y_true`, the score is that class's recall, not the average with a recall of 1."""
    assert balanced_accuracy(y_true, y_pred) == pytest.approx(_sklearn_balanced_accuracy(y_true, y_pred))


@pytest.mark.parametrize("seed", range(5))
def test_balanced_accuracy_matches_sklearn_when_all_classes_are_present(seed):
    rng = np.random.default_rng(seed)
    for num_classes in [2, 4]:
        y_true = rng.integers(0, num_classes, size=200)
        y_pred = np.where(rng.random(200) < 0.7, y_true, rng.integers(0, num_classes, size=200))
        expected = sklearn.metrics.balanced_accuracy_score(y_true, y_pred)
        assert balanced_accuracy(y_true, y_pred) == pytest.approx(expected)
