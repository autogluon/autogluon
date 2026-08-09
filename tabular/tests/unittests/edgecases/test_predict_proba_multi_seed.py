"""Tests for `predict_proba_multi(model_pred_probas=...)`: seeding precomputed predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autogluon.tabular import TabularPredictor


def _fit_bagged_predictor(path, problem_type: str) -> tuple[TabularPredictor, pd.DataFrame]:
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n)})
    if problem_type == "binary":
        df["label"] = (df["x1"] + rng.normal(scale=0.5, size=n) > 0).astype(int)
    else:
        df["label"] = rng.choice(["u", "v", "w"], size=n, p=[0.5, 0.3, 0.2])
    train, test = df.iloc[:300], df.iloc[300:].drop(columns=["label"])
    predictor = TabularPredictor(label="label", problem_type=problem_type, path=str(path), verbosity=0).fit(
        train,
        hyperparameters={"GBM": {}, "RF": {}},
        num_bag_folds=2,
    )
    return predictor, test


@pytest.mark.parametrize("problem_type", ["binary", "multiclass"])
@pytest.mark.parametrize("as_pandas", [True, False])
@pytest.mark.parametrize("as_multiclass", [True, False])
def test_seeded_ensemble_prediction_matches_full(tmp_path, problem_type, as_pandas, as_multiclass):
    """Seeding base model predictions reproduces the dependent ensemble's predictions exactly."""
    predictor, test = _fit_bagged_predictor(tmp_path, problem_type)
    base_models = [m for m in predictor.model_names() if "WeightedEnsemble" not in m]
    ensemble = next(m for m in predictor.model_names() if "WeightedEnsemble" in m)

    full = predictor.predict_proba_multi(test, as_pandas=as_pandas, as_multiclass=as_multiclass)
    seeded = predictor.predict_proba_multi(
        test,
        models=[ensemble],
        as_pandas=as_pandas,
        as_multiclass=as_multiclass,
        model_pred_probas={m: full[m] for m in base_models},
    )
    assert np.allclose(np.asarray(full[ensemble]), np.asarray(seeded[ensemble]))


def test_seeds_are_used_not_recomputed(tmp_path):
    """A corrupted seed must change the dependent ensemble's output."""
    predictor, test = _fit_bagged_predictor(tmp_path, "multiclass")
    base_models = [m for m in predictor.model_names() if "WeightedEnsemble" not in m]
    ensemble = next(m for m in predictor.model_names() if "WeightedEnsemble" in m)

    full = predictor.predict_proba_multi(test)
    corrupted = {m: full[m] for m in base_models}
    flipped = full[base_models[0]].iloc[::-1].reset_index(drop=True)
    flipped.index = full[base_models[0]].index
    corrupted[base_models[0]] = flipped
    seeded = predictor.predict_proba_multi(test, models=[ensemble], model_pred_probas=corrupted)
    assert not np.allclose(full[ensemble].to_numpy(), seeded[ensemble].to_numpy())


def test_row_count_mismatch_raises(tmp_path):
    predictor, test = _fit_bagged_predictor(tmp_path, "multiclass")
    base_models = [m for m in predictor.model_names() if "WeightedEnsemble" not in m]
    ensemble = next(m for m in predictor.model_names() if "WeightedEnsemble" in m)

    full = predictor.predict_proba_multi(test)
    with pytest.raises(ValueError, match="rows"):
        predictor.predict_proba_multi(
            test, models=[ensemble], model_pred_probas={base_models[0]: full[base_models[0]].iloc[:10]}
        )


def test_seeding_without_data_raises(tmp_path):
    predictor, _ = _fit_bagged_predictor(tmp_path, "multiclass")
    model = predictor.model_names()[0]
    with pytest.raises(ValueError, match="requires"):
        predictor.predict_proba_multi(model_pred_probas={model: np.zeros((10, 3))})


@pytest.mark.parametrize("problem_type", ["binary", "multiclass"])
def test_predict_multi_seeded_matches_full(tmp_path, problem_type):
    """predict_multi seeded with predict_proba_multi output reproduces unseeded predictions."""
    predictor, test = _fit_bagged_predictor(tmp_path, problem_type)
    base_models = [m for m in predictor.model_names() if "WeightedEnsemble" not in m]

    pred_probas = predictor.predict_proba_multi(test)
    full = predictor.predict_multi(test)
    seeded = predictor.predict_multi(test, model_pred_probas={m: pred_probas[m] for m in base_models})
    for m in full:
        assert full[m].equals(seeded[m]), m


def test_predict_multi_seeded_respects_decision_threshold(tmp_path):
    """Seeded models' predictions are derived from the seeds with the requested threshold."""
    predictor, test = _fit_bagged_predictor(tmp_path, "binary")
    base_models = [m for m in predictor.model_names() if "WeightedEnsemble" not in m]

    pred_probas = predictor.predict_proba_multi(test)
    seeds = {m: pred_probas[m] for m in base_models}
    low = predictor.predict_multi(test, model_pred_probas=seeds, decision_threshold=0.01)
    high = predictor.predict_multi(test, model_pred_probas=seeds, decision_threshold=0.99)
    m = base_models[0]
    assert low[m].mean() > high[m].mean()


def test_predict_multi_seeded_regression(tmp_path):
    """For regression, predict_multi seeds are predictions and pass through unchanged."""
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n)})
    df["label"] = df["x1"] * 2 + rng.normal(scale=0.1, size=n)
    train, test = df.iloc[:300], df.iloc[300:].drop(columns=["label"])
    predictor = TabularPredictor(label="label", problem_type="regression", path=str(tmp_path), verbosity=0).fit(
        train,
        hyperparameters={"GBM": {}, "RF": {}},
        num_bag_folds=2,
    )
    base_models = [m for m in predictor.model_names() if "WeightedEnsemble" not in m]

    full = predictor.predict_multi(test)
    seeded = predictor.predict_multi(test, model_pred_probas={m: full[m] for m in base_models})
    for m in full:
        assert np.allclose(full[m].to_numpy(), seeded[m].to_numpy()), m
