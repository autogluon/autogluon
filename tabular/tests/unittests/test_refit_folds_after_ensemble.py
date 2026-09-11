"""`refit_folds="after_ensemble"`: bag for the out-of-fold predictions, refit only what the ensemble keeps."""

from __future__ import annotations

import numpy as np
import pandas as pd

from autogluon.tabular import TabularPredictor

# Two bagged LightGBM configs; RandomForest uses its out-of-bag predictions as a single child and
# never bags, so it has no folds to defer.
HYPERPARAMETERS = {"GBM": [{}, {"num_leaves": 8}], "RF": {}}


def _data(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, 5)), columns=[f"f{i}" for i in range(5)])
    df["label"] = (df.f0 + rng.normal(scale=0.3, size=n) > 0).astype(int)
    return df


def _fit(path, **kwargs) -> TabularPredictor:
    return TabularPredictor(label="label", path=str(path), verbosity=0).fit(
        _data(), hyperparameters=HYPERPARAMETERS, num_bag_folds=2, keep_only_best=False, **kwargs
    )


def test_refit_folds_after_ensemble_refits_only_the_ensemble_members(tmp_path):
    predictor = _fit(tmp_path, ag_args_ensemble={"refit_folds": "after_ensemble"})
    trainer = predictor._trainer
    bags = [m for m in predictor.model_names() if m.startswith("LightGBM") and m.endswith("_BAG_L1")]
    assert len(bags) == 2 and trainer.models_with_refit_pending() == bags
    for name in bags:
        assert not trainer.get_model_attribute(name, "can_infer")
    assert trainer.get_model_attribute("RandomForest_BAG_L1", "can_infer")

    # `get_model_best` scores the fitted models, so it still names the pre-refit best; the refit
    # covered exactly that model's set, and the predictor serves its refit copy.
    used = set(trainer.get_minimum_model_set(trainer.get_model_best()))
    refit_map = predictor.model_refit_map()
    assert set(refit_map) == used, (refit_map, used)
    for name in bags:
        assert (name in refit_map) == (name in used)
    assert predictor.model_best == refit_map[trainer.get_model_best()]
    assert len(predictor.predict(_data(50, seed=1).drop(columns="label"))) == 50


def test_refit_folds_after_ensemble_matches_an_immediate_refit(tmp_path):
    """Deferring the refit changes when it happens, not what is fit."""
    kwargs = dict(hyperparameters={"GBM": {}}, num_bag_folds=2, fit_weighted_ensemble=False, keep_only_best=False)
    deferred = TabularPredictor(label="label", path=str(tmp_path / "deferred"), verbosity=0).fit(
        _data(), ag_args_ensemble={"refit_folds": "after_ensemble"}, **kwargs
    )
    immediate = TabularPredictor(label="label", path=str(tmp_path / "immediate"), verbosity=0).fit(
        _data(), ag_args_ensemble={"refit_folds": True}, **kwargs
    )
    X = _data(100, seed=2).drop(columns="label")
    assert deferred.model_best == "LightGBM_BAG_L1_FULL"
    np.testing.assert_array_equal(
        deferred.predict_proba(X).to_numpy(), immediate.predict_proba(X, model="LightGBM_BAG_L1").to_numpy()
    )


def test_refit_folds_after_ensemble_with_an_explicit_refit_full(tmp_path):
    """A caller asking for `refit_full="all"` keeps that: every bag is refit, pending or not."""
    predictor = _fit(tmp_path, ag_args_ensemble={"refit_folds": "after_ensemble"}, refit_full="all")
    bags = [m for m in predictor.model_names() if m.endswith("_BAG_L1")]
    assert all(name in predictor.model_refit_map() for name in bags)
