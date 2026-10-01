"""A NaN prediction never gets a finite score: every metric either raises a ValueError or returns NaN."""

import numpy as np
import pandas as pd
import pytest

from autogluon.core.constants import BINARY, MULTICLASS, QUANTILE, REGRESSION
from autogluon.core.metrics import METRICS

QUANTILE_LEVELS = [0.1, 0.5, 0.9]
SCORERS = [
    (problem_type, name)
    for problem_type, scorers in METRICS.items()
    for name, scorer in scorers.items()
    if name == scorer.name  # skip aliases
]


def _inputs_with_nan(problem_type: str, scorer, n: int = 40) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    scores = scorer.needs_proba or scorer.needs_threshold
    if problem_type == BINARY:
        y_true = np.arange(n) % 2
        y_pred = rng.uniform(0.05, 0.95, n) if scores else (rng.uniform(size=n) > 0.5).astype(float)
    elif problem_type == MULTICLASS:
        y_true = np.arange(n) % 3
        y_pred = rng.dirichlet([1, 1, 1], n) if scores else rng.integers(0, 3, n).astype(float)
    elif problem_type == REGRESSION:
        y_true = rng.normal(size=n) + 5
        y_pred = y_true + 0.3 * rng.normal(size=n)
    elif problem_type == QUANTILE:
        y_true = rng.normal(size=n)
        y_pred = np.sort(rng.normal(size=(n, len(QUANTILE_LEVELS))), axis=1)
    else:
        raise AssertionError(f"no inputs for problem_type={problem_type!r}")
    y_pred[3] = np.nan  # one prediction (the whole row of a 2-d prediction)
    return y_true, y_pred


@pytest.mark.parametrize("as_pandas", [False, True])
@pytest.mark.parametrize("problem_type, metric", SCORERS)
def test_nan_prediction_is_not_scored(problem_type, metric, as_pandas):
    scorer = METRICS[problem_type][metric]
    y_true, y_pred = _inputs_with_nan(problem_type, scorer)
    if as_pandas:
        y_true = pd.Series(y_true)
        if y_pred.ndim == 1:
            y_pred = pd.Series(y_pred)
    kwargs = {"quantile_levels": QUANTILE_LEVELS} if problem_type == QUANTILE else {}
    try:
        score = scorer(y_true, y_pred, **kwargs)
    except ValueError:
        return
    assert np.isnan(score), f"{metric} scored a NaN prediction as {score}"
