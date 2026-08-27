"""`validation_mode="none"`: train on every row, combine with predetermined weights.

The motivating case is an in-context learner such as TabPFN or TabICL, which needs no validation
set, paired with an ensemble combination that is known up front rather than learned from held-out
predictions. These tests use fast models so they exercise the plumbing, not the backends.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autogluon.tabular import TabularPredictor

HYPERPARAMETERS = {"GBM": {}, "XGB": {}}
WEIGHTS = {"LightGBM": 0.5, "XGBoost": 0.5}


def _data(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, 5)), columns=[f"f{i}" for i in range(5)])
    df["label"] = (df.f0 + rng.normal(scale=0.3, size=n) > 0).astype(int)
    return df


def _fit(tmp_path, train, **kwargs):
    return TabularPredictor(label="label", path=str(tmp_path), verbosity=0).fit(
        train, hyperparameters=HYPERPARAMETERS, **kwargs
    )


def test_validation_mode_none_trains_on_every_row(tmp_path):
    """No holdout is carved, and no model gets a validation score."""
    train = _data()
    predictor = _fit(tmp_path, train, validation_mode="none", ensemble_weights=WEIGHTS)

    assert predictor._trainer._num_rows_train == len(train)
    leaderboard = predictor.leaderboard(silent=True)
    assert leaderboard["score_val"].isna().all()
    assert predictor.model_best == "WeightedEnsemble_L2"


def test_validation_mode_none_applies_the_given_weights(tmp_path):
    """The ensemble is exactly the weighted combination of its base models."""
    train = _data()
    test = _data(n=100, seed=1).drop(columns=["label"])
    weights = {"LightGBM": 0.25, "XGBoost": 0.75}
    predictor = _fit(tmp_path, train, validation_mode="none", ensemble_weights=weights)

    ensemble = predictor.predict_proba(test)[1].to_numpy()
    expected = sum(w * predictor.predict_proba(test, model=name)[1].to_numpy() for name, w in weights.items())
    np.testing.assert_allclose(ensemble, expected, atol=1e-6)


def test_validation_mode_none_normalizes_weights(tmp_path):
    """Weights need not sum to 1; the ratio is what matters."""
    train = _data()
    test = _data(n=60, seed=2).drop(columns=["label"])
    scaled = _fit(tmp_path / "a", train, validation_mode="none", ensemble_weights={"LightGBM": 2, "XGBoost": 6})
    unit = _fit(tmp_path / "b", train, validation_mode="none", ensemble_weights={"LightGBM": 0.25, "XGBoost": 0.75})

    np.testing.assert_allclose(
        scaled.predict_proba(test)[1].to_numpy(), unit.predict_proba(test)[1].to_numpy(), atol=1e-6
    )


def test_validation_mode_none_without_an_ensemble(tmp_path):
    """`fit_weighted_ensemble=False` is the other way to have nothing to learn weights for."""
    train = _data()
    predictor = _fit(tmp_path, train, validation_mode="none", fit_weighted_ensemble=False)

    assert predictor._trainer._num_rows_train == len(train)
    assert predictor.model_best in set(predictor.model_names())


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"num_bag_folds": 3}, "cannot be combined with bagging"),
        ({"tuning_data": "TRAIN"}, "cannot be combined with `tuning_data`"),
        ({"validation_structure": {"group_on": "f0"}}, "cannot be combined with `validation_structure`"),
    ],
)
def test_validation_mode_none_rejects_what_needs_a_holdout(tmp_path, kwargs, match):
    """Bagging, stacking and explicit tuning data are all defined by holding rows out."""
    train = _data()
    if kwargs.get("tuning_data") == "TRAIN":
        kwargs["tuning_data"] = train
    with pytest.raises(ValueError, match=match):
        _fit(tmp_path, train, validation_mode="none", ensemble_weights=WEIGHTS, **kwargs)


def test_validation_mode_none_requires_weights_for_an_ensemble(tmp_path):
    """Without validation data there is nothing to learn weights from, so they must be supplied."""
    with pytest.raises(ValueError, match="leaves no data to learn ensemble weights from"):
        _fit(tmp_path, _data(), validation_mode="none")


def test_ensemble_weights_rejected_when_validation_exists(tmp_path):
    """With validation data AutoGluon learns the weights; fixed weights would silently override."""
    with pytest.raises(ValueError, match="only supported with validation_mode='none'"):
        _fit(tmp_path, _data(), ensemble_weights=WEIGHTS)


def test_validation_mode_rejects_unknown_values(tmp_path):
    with pytest.raises(ValueError, match="validation_mode must be 'auto' or 'none'"):
        _fit(tmp_path, _data(), validation_mode="sometimes")


@pytest.mark.parametrize(
    "weights, match",
    [
        ({"Nope": 1.0}, "are not fitted models"),
        ({"LightGBM": 1.0}, "missing a weight for"),
        ({"LightGBM": 0.0, "XGBoost": 0.0}, "must sum to a positive value"),
    ],
)
def test_ensemble_weights_are_validated(tmp_path, weights, match):
    """A silent mismatch would give a model someone else's weight."""
    with pytest.raises(ValueError, match=match):
        _fit(tmp_path, _data(), validation_mode="none", ensemble_weights=weights)
