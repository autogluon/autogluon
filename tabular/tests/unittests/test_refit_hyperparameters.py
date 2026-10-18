"""`ag.refit_hyperparameters`: fit the folds one way, refit on all the data another.

A refit trains a single model on every row with no validation data, so it can afford settings the
fitted model could not. The motivating case is an in-context model whose ensemble size is a direct
cost multiplier -- fit cheaply, spend the budget once on the model that is served.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autogluon.tabular import TabularPredictor


def _data(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, 5)), columns=[f"f{i}" for i in range(5)])
    df["label"] = (df.f0 + rng.normal(scale=0.3, size=n) > 0).astype(int)
    return df


def _child_param(predictor, model_name: str, param: str):
    """The value a bagged model's child actually holds, or the model's own if not bagged."""
    model = predictor._trainer.load_model(model_name)
    if getattr(model, "models", None):
        return model.load_child(model.models[0]).params.get(param)
    return model.params.get(param)


def test_refit_hyperparameters_apply_to_a_bagged_refit(tmp_path):
    """`refit_full` on a bagged model refits the child with the override."""
    predictor = TabularPredictor(label="label", path=str(tmp_path), verbosity=0).fit(
        _data(),
        hyperparameters={"GBM": {"num_leaves": 5, "ag.refit_hyperparameters": {"num_leaves": 40}}},
        num_bag_folds=3,
        fit_weighted_ensemble=False,
    )
    predictor.refit_full()

    assert _child_param(predictor, "LightGBM_BAG_L1", "num_leaves") == 5
    assert _child_param(predictor, "LightGBM_BAG_L1_FULL", "num_leaves") == 40


def test_refit_hyperparameters_apply_to_a_holdout_refit(tmp_path):
    """The same override works without bagging, where the refit follows a holdout fit."""
    predictor = TabularPredictor(label="label", path=str(tmp_path), verbosity=0).fit(
        _data(),
        hyperparameters={"GBM": {"num_leaves": 5, "ag.refit_hyperparameters": {"num_leaves": 40}}},
        num_bag_folds=0,
        fit_weighted_ensemble=False,
    )
    predictor.refit_full()

    assert _child_param(predictor, "LightGBM", "num_leaves") == 5
    assert _child_param(predictor, "LightGBM_FULL", "num_leaves") == 40


def test_refit_hyperparameters_apply_when_refit_folds_refits_during_fit(tmp_path):
    """`refit_folds=True` refits inside `fit`, discarding the folds -- as TabICLv2 does by default.

    Only the refit survives, so the override has to have reached it during the original fit.
    """
    predictor = TabularPredictor(label="label", path=str(tmp_path), verbosity=0).fit(
        _data(),
        hyperparameters={
            "GBM": {
                "num_leaves": 5,
                "ag.refit_hyperparameters": {"num_leaves": 40},
                "ag.ens.refit_folds": True,
            }
        },
        num_bag_folds=3,
        fit_weighted_ensemble=False,
    )

    assert _child_param(predictor, "LightGBM_BAG_L1", "num_leaves") == 40


def test_refit_override_beats_a_value_the_fit_concluded_with(tmp_path):
    """LightGBM records its early-stopped iteration count in `params_trained`.

    That value carries into the refit by default, which is what makes refits cheap. An explicit
    override of the same key has to win, or the option could not change anything the fit settled.
    """
    data = _data(n=500, seed=1)
    common = dict(num_bag_folds=0, fit_weighted_ensemble=False)

    default = TabularPredictor(label="label", path=str(tmp_path / "a"), verbosity=0).fit(
        data, hyperparameters={"GBM": {"num_boost_round": 200}}, **common
    )
    default.refit_full()
    learned = default._trainer.load_model("LightGBM").params_trained["num_boost_round"]
    assert _child_param(default, "LightGBM_FULL", "num_boost_round") == learned

    overridden = TabularPredictor(label="label", path=str(tmp_path / "b"), verbosity=0).fit(
        data,
        hyperparameters={"GBM": {"num_boost_round": 200, "ag.refit_hyperparameters": {"num_boost_round": 7}}},
        **common,
    )
    overridden.refit_full()
    assert _child_param(overridden, "LightGBM_FULL", "num_boost_round") == 7


def test_refit_hyperparameters_is_consumed_by_the_refit(tmp_path):
    """The override is applied once, not carried into the refit's own auxiliary parameters."""
    predictor = TabularPredictor(label="label", path=str(tmp_path), verbosity=0).fit(
        _data(),
        hyperparameters={"GBM": {"num_leaves": 5, "ag.refit_hyperparameters": {"num_leaves": 40}}},
        num_bag_folds=0,
        fit_weighted_ensemble=False,
    )
    predictor.refit_full()

    refit = predictor._trainer.load_model("LightGBM_FULL")
    assert refit.aux_params.refit_hyperparameters is None


def test_no_refit_hyperparameters_leaves_the_refit_unchanged(tmp_path):
    """Without the option the refit inherits the fitted hyperparameters, as before."""
    predictor = TabularPredictor(label="label", path=str(tmp_path), verbosity=0).fit(
        _data(),
        hyperparameters={"GBM": {"num_leaves": 5}},
        num_bag_folds=0,
        fit_weighted_ensemble=False,
    )
    predictor.refit_full()

    assert _child_param(predictor, "LightGBM_FULL", "num_leaves") == 5


def test_refit_hyperparameters_can_set_an_auxiliary_parameter(tmp_path):
    """An `ag.`-prefixed key inside the override routes to the refit's auxiliary parameters."""
    predictor = TabularPredictor(label="label", path=str(tmp_path), verbosity=0).fit(
        _data(),
        hyperparameters={"GBM": {"ag.refit_hyperparameters": {"ag.max_memory_usage_ratio": 3.0}}},
        num_bag_folds=0,
        fit_weighted_ensemble=False,
    )
    predictor.refit_full()

    assert predictor._trainer.load_model("LightGBM_FULL").aux_params.max_memory_usage_ratio == 3.0
