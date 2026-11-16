import numpy as np
import pytest

from autogluon.core.metrics import pac


def test_pac_binary_2d_proba_scores_like_1d():
    """2-column binary probabilities score like the 1-D positive-class probabilities.

    The positive-class column was taken as 1-D while the labels were (n, 1), so the log loss broadcast to (n, n) and
    the score was wrong (negative instead of 0.70 here).
    """
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=200)
    y_pred_proba = np.clip(y_true * 0.7 + rng.random(200) * 0.3, 0, 1)
    expected = pac(y_true, y_pred_proba)
    assert pac(y_true, np.stack([1 - y_pred_proba, y_pred_proba], axis=1)) == pytest.approx(expected)
    assert pac(y_true, y_pred_proba.reshape(-1, 1)) == pytest.approx(expected)
