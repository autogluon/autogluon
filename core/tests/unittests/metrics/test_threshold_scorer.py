import numpy as np
import pytest
import sklearn

from autogluon.core.metrics import average_precision, roc_auc

_rng = np.random.default_rng(0)
_Y_TRUE = _rng.integers(0, 2, size=200)
_Y_PRED_PROBA = np.clip(_Y_TRUE * 0.3 + _rng.random(200) * 0.7, 0, 1)  # positive-class probabilities


@pytest.mark.parametrize(
    "scorer,sklearn_metric",
    [(roc_auc, sklearn.metrics.roc_auc_score), (average_precision, sklearn.metrics.average_precision_score)],
    ids=["roc_auc", "average_precision"],
)
def test_threshold_metrics_accept_2d_binary_proba(scorer, sklearn_metric):
    """2-column probabilities and a single column score like the 1-D positive-class probabilities.

    Passed through as 2-D, `roc_auc` sorted each row instead of the samples and returned ~0.5 or raised, depending on
    the data, and `average_precision` raised; a single column made `roc_auc` raise.
    """
    expected = sklearn_metric(_Y_TRUE, _Y_PRED_PROBA)
    assert scorer(_Y_TRUE, _Y_PRED_PROBA) == pytest.approx(expected)
    y_pred_proba_2d = np.stack([1 - _Y_PRED_PROBA, _Y_PRED_PROBA], axis=1)
    assert scorer(_Y_TRUE, y_pred_proba_2d) == pytest.approx(expected)
    assert scorer(_Y_TRUE, _Y_PRED_PROBA.reshape(-1, 1)) == pytest.approx(expected)
