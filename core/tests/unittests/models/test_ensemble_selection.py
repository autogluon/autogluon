import numpy as np
import pytest

from autogluon.core.metrics import log_loss, make_scorer, root_mean_squared_error
from autogluon.core.models.greedy_ensemble.ensemble_selection import EnsembleSelection


def _labels_and_preds(n_models=2):
    labels = np.array([0.0, 1.0, 2.0, 3.0])
    predictions = [np.full(labels.shape, float(i)) for i in range(n_models)]
    return labels, predictions


def _ensemble_selection(metric, **kwargs):
    kwargs.setdefault("ensemble_size", 5)
    kwargs.setdefault("problem_type", "regression")
    return EnsembleSelection(metric=metric, **kwargs)


def _always_nan_metric():
    return make_scorer("always_nan", lambda y_true, y_pred: np.nan, optimum=0, greater_is_better=False)


def test_select_best_by_score_all_nan_raises():
    with pytest.raises(ValueError, match="all ensemble scores non-finite"):
        EnsembleSelection._select_best_by_score(np.array([np.nan, np.nan, np.nan]))


def test_select_best_by_score_nonfinite_lose_to_finite():
    scores, all_best = EnsembleSelection._select_best_by_score(np.array([np.nan, 0.4, np.inf, -np.inf, 0.2]))
    assert np.array_equal(all_best, np.array([4]))
    assert np.isposinf(scores[0])
    assert np.isposinf(scores[2])
    assert np.isposinf(scores[3])
    assert scores[4] == 0.2


def test_ensemble_selection_all_nan_scores_raises():
    """All-NaN regrets used to raise ValueError from RandomState.choice([])."""
    labels, predictions = _labels_and_preds()
    ensemble = _ensemble_selection(_always_nan_metric())
    with pytest.raises(ValueError, match="all ensemble scores non-finite"):
        ensemble.fit(predictions=predictions, labels=labels)


def test_ensemble_selection_all_nan_predictions_raises():
    """All-NaN predictions produce all-NaN RMSE regrets and crashed on master."""
    labels = np.array([0.0, 1.0, 2.0, 3.0])
    predictions = [np.full_like(labels, np.nan), np.full_like(labels, np.nan)]
    ensemble = _ensemble_selection(root_mean_squared_error)
    with pytest.raises(ValueError, match="all ensemble scores non-finite"):
        ensemble.fit(predictions=predictions, labels=labels)


@pytest.mark.parametrize("dead_score", [np.nan, np.inf, -np.inf])
def test_ensemble_selection_all_nonfinite_scores_raises(dead_score):
    labels, predictions = _labels_and_preds(n_models=3)
    ensemble = _ensemble_selection(root_mean_squared_error)
    ensemble._calculate_regret = lambda *args, **kwargs: dead_score
    with pytest.raises(ValueError, match="all ensemble scores non-finite"):
        ensemble.fit(predictions=predictions, labels=labels)


def test_ensemble_selection_nonfinite_scores_lose_to_finite():
    labels, predictions = _labels_and_preds(n_models=3)
    ensemble = _ensemble_selection(root_mean_squared_error)
    call = {"i": 0}
    per_model = [np.nan, 0.25, np.inf]

    def _regret(*args, **kwargs):
        j = call["i"] % len(per_model)
        call["i"] += 1
        return per_model[j]

    ensemble._calculate_regret = _regret
    ensemble.fit(predictions=predictions, labels=labels)
    assert ensemble.weights_[1] == 1.0
    assert np.count_nonzero(ensemble.weights_) == 1
    # best_score written into trajectory is finite, so index(min) cannot miss NaN
    assert np.isfinite(ensemble.train_score_)


def test_ensemble_selection_second_metric_all_nan_tiebreak_raises():
    labels = np.array([0, 1, 0, 1])
    predictions = [
        np.array([0.2, 0.8, 0.3, 0.7]),
        np.array([0.1, 0.9, 0.4, 0.6]),
    ]
    ensemble = EnsembleSelection(
        ensemble_size=5,
        problem_type="binary",
        metric=make_scorer("tied", lambda y_true, y_pred: 1.0, optimum=0, greater_is_better=False),
        tie_breaker="second_metric",
    )

    def _regret(y_true, y_pred_proba, metric, sample_weight=None):
        if metric is log_loss or getattr(metric, "name", None) == "log_loss":
            return np.nan
        return 1.0

    ensemble._calculate_regret = _regret
    with pytest.raises(ValueError, match="all ensemble scores non-finite"):
        ensemble.fit(predictions=predictions, labels=labels)


def test_ensemble_selection_finite_scores_still_pick_better_model():
    labels = np.array([0.0, 1.0, 2.0, 3.0])
    predictions = [np.zeros_like(labels), labels.copy()]
    ensemble = _ensemble_selection(root_mean_squared_error)
    ensemble.fit(predictions=predictions, labels=labels)
    assert ensemble.weights_[1] > ensemble.weights_[0]
    assert np.isclose(np.sum(ensemble.weights_), 1.0)
    assert np.isfinite(ensemble.train_score_)


@pytest.mark.parametrize("shape", [(50,), (50, 3)])
def test_weight_pred_probas_is_the_row_by_row_weighted_sum(shape):
    """The weighted sum skips zero weights, leaves the inputs untouched and does not depend on the batch.

    `np.sum` over the stacked predictions summed a single row in a different (pairwise) order than a batch of
    rows once 8 or more models had a weight, so a row predicted alone could differ from the same row in a batch.
    """
    rng = np.random.default_rng(0)
    pred_probas = [rng.random(shape, dtype=np.float32) for _ in range(12)]
    pred_probas_before = [pred.copy() for pred in pred_probas]
    weights = rng.random(12)
    weights[[3, 7]] = 0
    weights /= weights.sum()

    out = EnsembleSelection.weight_pred_probas(pred_probas, weights)

    expected = np.zeros(shape)
    for pred, weight in zip(pred_probas, weights):
        if weight != 0:
            expected += pred * weight
    assert np.array_equal(out, expected)
    for pred, pred_before in zip(pred_probas, pred_probas_before):
        assert np.array_equal(pred, pred_before)
    for i in range(shape[0]):
        out_row = EnsembleSelection.weight_pred_probas([pred[i : i + 1] for pred in pred_probas], weights)
        assert np.array_equal(out_row, out[i : i + 1])
