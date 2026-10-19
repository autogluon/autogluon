import numpy as np
import pytest

from autogluon.core.metrics import log_loss
from autogluon.core.metrics.softclass_metrics import soft_log_loss


def test_soft_log_loss_error_is_the_loss():
    """Like `log_loss`, the optimum is a loss of 0, so the error is the loss itself.

    Without `optimum=0` the scorer took the default optimum of 1, making the error `loss - 1`: negative for any loss
    below 1, as in error-format leaderboards and learning curves.
    """
    assert soft_log_loss.optimum == 0
    y_true = np.array([0, 1, 1])
    y_pred_proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.4, 0.6]])
    true_probs = np.eye(2)[y_true]  # one-hot soft labels, for which soft_log_loss equals log_loss
    error = soft_log_loss.error(true_probs, y_pred_proba)
    assert error == pytest.approx(-soft_log_loss(true_probs, y_pred_proba))
    assert error == pytest.approx(log_loss.error(y_true, y_pred_proba))
    assert error > 0
