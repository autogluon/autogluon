"""`validation_folds`: the weighted ensemble is chosen on some of the bag folds, then the bags it uses are completed."""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import pytest

from autogluon.core.models import DummyModel
from autogluon.core.utils.exceptions import TimeLimitExceeded
from autogluon.tabular import TabularPredictor
from autogluon.tabular.configs.pipeline_presets import resolve_validation_folds_time_reserve
from autogluon.tabular.trainer.abstract_trainer import (
    COMPLETION_FOLD_TIME_FLOOR,
    COMPLETION_TIME_MARGIN,
    AbstractTabularTrainer,
)

HYPERPARAMETERS = {"GBM": {"num_boost_round": 30}, "LR": {}, "DUMMY": {}}
"""Bagged with real folds (no child out-of-fold predictions); the dummy model is one the ensemble leaves out."""
NUM_FOLDS = 4


def _frame(problem_type: str, rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(rows, 5))
    signal = X[:, 0] + 0.5 * X[:, 1] - 0.3 * X[:, 2] ** 2 + rng.normal(scale=0.5, size=rows)
    if problem_type == "regression":
        y = signal
    elif problem_type == "binary":
        y = (signal > 0).astype(int)
    else:
        y = np.digitize(signal, np.quantile(signal, [1 / 3, 2 / 3]))
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    df["label"] = y
    return df


def _fit(train: pd.DataFrame, tmp_path, **kwargs) -> TabularPredictor:
    kwargs.setdefault("num_bag_folds", NUM_FOLDS)
    kwargs.setdefault("num_bag_sets", 1)
    kwargs.setdefault("num_stack_levels", 0)
    kwargs.setdefault("hyperparameters", HYPERPARAMETERS)
    return TabularPredictor(label="label", path=str(tmp_path / "predictor"), verbosity=0).fit(
        train, num_gpus=0, dynamic_stacking=False, **kwargs
    )


def _bags(predictor: TabularPredictor) -> list[str]:
    return [m for m in predictor.model_names() if not m.startswith("WeightedEnsemble")]


def _proba(predictor: TabularPredictor, data: pd.DataFrame, model: str | None = None) -> np.ndarray:
    if predictor.problem_type == "regression":
        return predictor.predict(data, model=model).to_numpy()[:, None]
    return predictor.predict_proba(data, model=model).to_numpy()


def _uncovered(oof: np.ndarray) -> np.ndarray:
    return np.isnan(oof.reshape(len(oof), -1)).any(axis=1)


@pytest.mark.parametrize("problem_type", ["binary", "multiclass", "regression"], ids=["binary", "multiclass", "rmse"])
def test_the_ensembles_bags_are_completed_and_the_others_keep_their_first_folds(problem_type, tmp_path):
    train = _frame(problem_type, rows=400, seed=0)
    test = _frame(problem_type, rows=100, seed=1)
    predictor = _fit(train, tmp_path, validation_folds=1)
    trainer = predictor._trainer
    best = predictor.model_best
    ensemble = trainer.load_model(best)
    weights = ensemble._get_model_weights()
    used = set(trainer.get_minimum_model_set(best)) - {best}
    assert used == set(weights)
    assert set(_bags(predictor)) - used, "the test needs a model the ensemble leaves out"

    covered = ~_uncovered(trainer.get_model_oof(best))
    assert covered.sum() == pytest.approx(len(train) / NUM_FOLDS, abs=2)
    for name in _bags(predictor):
        bag = trainer.load_model(name)
        oof = trainer.get_model_oof(name)
        if name in used:
            assert bag.n_children == NUM_FOLDS
            assert not trainer.get_model_attribute(name, "folds_pending")
            assert not _uncovered(oof).any()
        else:
            assert bag.n_children == 1
            assert trainer.get_model_attribute(name, "folds_pending")
            np.testing.assert_array_equal(~_uncovered(oof), covered)
        # every validation score is on the first fold's rows
        y_covered = trainer.load_y().to_numpy()[covered]
        oof_covered = oof[covered]
        assert trainer.get_model_attribute(name, "val_score") == pytest.approx(
            bag.score_with_y_pred_proba(y=y_covered, y_pred_proba=oof_covered), rel=1e-5
        )

    # the ensemble kept the weights it chose on those rows: its out-of-fold predictions are the weighted members'
    oof_ensemble = trainer.get_model_oof(best).reshape(len(train), -1)[covered]
    oof_members = sum(w * trainer.get_model_oof(m).reshape(len(train), -1)[covered] for m, w in weights.items())
    np.testing.assert_allclose(oof_ensemble, oof_members, atol=1e-5)
    # and predicts with the completed bags
    pred = _proba(predictor, test)
    expected = sum(w * _proba(predictor, test, model=m) for m, w in weights.items())
    np.testing.assert_allclose(pred, expected, atol=1e-5)

    loaded = TabularPredictor.load(predictor.path)
    np.testing.assert_allclose(_proba(loaded, test), pred, atol=1e-6)
    assert loaded._trainer.models_with_folds_pending() == trainer.models_with_folds_pending()
    lb = predictor.leaderboard(silent=True)
    assert lb.score_val.notna().all()


def test_validation_folds_cost_less_than_full_bags(tmp_path):
    train = _frame("binary", rows=400, seed=0)
    full = _fit(train, tmp_path / "full")
    partial = _fit(train, tmp_path / "partial", validation_folds=1)
    n_children = lambda p: sum(p._trainer.load_model(m).n_children for m in _bags(p))  # noqa: E731
    assert n_children(partial) < n_children(full)


def test_a_refit_folds_bag_is_refit_and_not_completed(tmp_path):
    train = _frame("binary", rows=400, seed=0)
    hyperparameters = {"GBM": {"num_boost_round": 30, "ag_args_ensemble": {"refit_folds": True}}, "LR": {}}
    predictor = _fit(train, tmp_path, hyperparameters=hyperparameters, validation_folds=1)
    trainer = predictor._trainer
    bag = trainer.load_model("LightGBM_BAG_L1")
    assert bag.n_children == 1  # the all-rows refit
    assert not trainer.get_model_attribute("LightGBM_BAG_L1", "folds_pending")
    # its out-of-fold predictions come from the first fold, which the refit discarded
    assert _uncovered(trainer.get_model_oof("LightGBM_BAG_L1")).mean() == pytest.approx(1 - 1 / NUM_FOLDS, abs=0.01)
    assert len(bag.get_oof_fold_val_idx(*_train_xy(trainer))) == 1
    test = _frame("binary", rows=50, seed=1)
    assert np.isfinite(_proba(predictor, test)).all()


def _train_xy(trainer):
    return trainer.load_X(), trainer.load_y()


def test_refit_folds_after_ensemble_keeps_its_post_fit_refit(tmp_path):
    train = _frame("binary", rows=400, seed=0)
    hyperparameters = {"GBM": {"num_boost_round": 30, "ag_args_ensemble": {"refit_folds": "after_ensemble"}}, "LR": {}}
    predictor = _fit(train, tmp_path, hyperparameters=hyperparameters, validation_folds=1)
    assert predictor.model_best.endswith("_FULL")
    assert np.isfinite(_proba(predictor, _frame("binary", rows=50, seed=1))).all()


def test_a_size_curve_sets_validation_folds(tmp_path):
    curves = {"validation_folds": [[500, None], 1]}
    small = _fit(_frame("binary", rows=400, seed=0), tmp_path / "small", validation_size_curves=curves)
    large = _fit(_frame("binary", rows=800, seed=0), tmp_path / "large", validation_size_curves=curves)
    assert not small._trainer.models_with_folds_pending()
    assert all(small._trainer.load_model(m).n_children in (1, NUM_FOLDS) for m in _bags(small))
    assert large._trainer.models_with_folds_pending()


def test_refit_full_after_completion(tmp_path):
    train = _frame("multiclass", rows=400, seed=0)
    predictor = _fit(train, tmp_path, validation_folds=1, refit_full=True, set_best_to_refit_full=True)
    assert predictor.model_best.endswith("_FULL")
    pred = predictor.predict_proba(_frame("multiclass", rows=50, seed=1))
    np.testing.assert_allclose(pred.sum(axis=1), 1, atol=1e-5)


def test_temperature_scaling_on_the_validation_rows(tmp_path):
    train = _frame("multiclass", rows=400, seed=0)
    predictor = _fit(train, tmp_path, validation_folds=1, calibrate=True)
    pred = predictor.predict_proba(_frame("multiclass", rows=50, seed=1))
    assert np.isfinite(pred.to_numpy()).all()


def test_validation_folds_with_group_views(tmp_path):
    rng = np.random.default_rng(0)
    n_groups, rows = 80, 5
    group = np.repeat(np.arange(n_groups), rows)
    latent = rng.normal(size=(n_groups, 3))
    train = pd.DataFrame(latent[group] + rng.normal(scale=1.0, size=(len(group), 3)), columns=["f0", "f1", "f2"])
    train["patient"] = [f"p{g}" for g in group]
    train["label"] = (latent[:, 0] + 0.5 * latent[:, 1] > 0).astype(int)[group]
    predictor = _fit(
        train,
        tmp_path,
        hyperparameters={"GBM": {"num_boost_round": 20}, "LR": {}},
        validation_structure={"group_on": "patient"},
        group_views=True,
        validation_folds=1,
    )
    trainer = predictor._trainer
    used = set(trainer.get_minimum_model_set(predictor.model_best)) - {predictor.model_best}
    assert any("GroupCollapsed" in m for m in _bags(predictor))
    for name in _bags(predictor):
        assert trainer.load_model(name).n_children == (NUM_FOLDS if name in used else 1)
    test = train.sample(frac=1, random_state=0).head(100)
    pred = predictor.predict_proba(test)
    assert pred.groupby(test["patient"].to_numpy()).nunique().max().max() == 1


def test_a_value_of_at_least_num_bag_folds_fits_every_fold(tmp_path):
    predictor = _fit(_frame("binary", rows=400, seed=0), tmp_path, validation_folds=NUM_FOLDS)
    assert not predictor._trainer.models_with_folds_pending()


def test_errors(tmp_path):
    train = _frame("binary", rows=200, seed=0)
    for value in (0, True, 1.5, "one"):
        with pytest.raises(ValueError, match="positive int"):
            _fit(train, tmp_path / f"v{value}", validation_folds=value)
    cases = {
        "bag": dict(num_bag_folds=0),
        "sets": dict(num_bag_sets=2),
        "stack": dict(num_stack_levels=1),
        "weighted": dict(fit_weighted_ensemble=False),
    }
    for name, kwargs in cases.items():
        with pytest.raises(ValueError, match="validation_folds=1"):
            _fit(train, tmp_path / name, validation_folds=1, **kwargs)
        assert not (tmp_path / name / "predictor" / "models").exists()
    with pytest.raises(ValueError, match="validation_mode='none'"):
        _fit(
            train,
            tmp_path / "none",
            validation_folds=1,
            validation_mode="none",
            num_bag_folds=0,
            ensemble_weights={"LightGBM": 1.0},
            hyperparameters={"GBM": {"num_boost_round": 10}},
        )
    # a combination the curves resolve to only at some sizes names them
    with pytest.raises(ValueError, match="size curves"):
        _fit(
            train,
            tmp_path / "curves",
            num_bag_folds=None,
            validation_size_curves={"num_bag_folds": [[100, 4], 0], "validation_folds": [[100, None], 1]},
        )


class TimedDummyModel(DummyModel):
    """A dummy model whose fit takes `seconds` and records the time limit it gets (`time_limits`: rows, limit). With
    `refit_rows`, a time-limited fit on that many rows or more (a `refit_folds` refit on every row) runs out of time."""

    time_limits: list[tuple[int, float | None]] = []

    def _fit(self, X, y, time_limit=None, **kwargs):
        params = self._get_model_params()
        TimedDummyModel.time_limits.append((len(X), time_limit))
        if params.get("refit_rows") and len(X) >= params["refit_rows"] and time_limit is not None:
            raise TimeLimitExceeded
        time.sleep(params.get("seconds", 0.0))
        return super()._fit(X, y, **kwargs)


def _timed(**params) -> dict:
    return {TimedDummyModel: [params]}


def test_the_reserve_setting():
    assert resolve_validation_folds_time_reserve("auto", validation_folds=1) == "auto"
    assert resolve_validation_folds_time_reserve("auto", validation_folds=None) == 0
    assert resolve_validation_folds_time_reserve(0.3, validation_folds=1) == 0.3
    assert resolve_validation_folds_time_reserve(None, validation_folds=1) == 0
    for bad in ("half", 1.0, -0.1, True):
        with pytest.raises(ValueError, match="validation_folds_time_reserve"):
            resolve_validation_folds_time_reserve(bad, validation_folds=1)


def test_the_first_folds_are_fit_within_the_rest_of_the_time_limit(tmp_path, monkeypatch):
    limits = {}
    train_multi_levels = AbstractTabularTrainer.train_multi_levels
    train_multi_and_ensemble = AbstractTabularTrainer._train_multi_and_ensemble

    def spy_levels(self, *args, time_limit=None, **kwargs):
        limits["first"] = time_limit
        return train_multi_levels(self, *args, time_limit=time_limit, **kwargs)

    def spy_fit(self, *args, time_limit=None, **kwargs):
        limits["total"] = time_limit
        return train_multi_and_ensemble(self, *args, time_limit=time_limit, **kwargs)

    monkeypatch.setattr(AbstractTabularTrainer, "train_multi_levels", spy_levels)
    monkeypatch.setattr(AbstractTabularTrainer, "_train_multi_and_ensemble", spy_fit)
    train = _frame("binary", rows=400, seed=0)
    _fit(train, tmp_path / "reserve", validation_folds=1, time_limit=120, validation_folds_time_reserve=0.6)
    assert limits["first"] == pytest.approx(0.4 * limits["total"], rel=1e-6)
    _fit(train, tmp_path / "none", validation_folds=1, time_limit=120, validation_folds_time_reserve=None)
    assert limits["first"] == limits["total"]
    _fit(train, tmp_path / "every_fold", time_limit=120, validation_folds_time_reserve=0.6)
    assert limits["first"] == limits["total"]
    # "auto" keeps time back model by model, not up front
    _fit(train, tmp_path / "auto", validation_folds=1, time_limit=120)
    assert limits["first"] == limits["total"]


def test_a_bag_is_completed_only_if_it_fits_and_its_folds_get_the_first_folds_time(tmp_path, monkeypatch):
    train = _frame("binary", rows=400, seed=0)
    hyperparameters = {
        TimedDummyModel: [
            {"seconds": 0.3, "ag_args": {"name_suffix": "A"}},
            {"seconds": 0.3, "ag_args": {"name_suffix": "B"}},
        ]
    }
    # fit the first folds only, and complete by hand
    monkeypatch.setattr(AbstractTabularTrainer, "complete_bags", lambda self, *args, **kwargs: [])
    predictor = _fit(train, tmp_path, hyperparameters=hyperparameters, validation_folds=1)
    monkeypatch.undo()
    trainer = predictor._trainer
    a, b = sorted(_bags(predictor))
    assert trainer.load_model(a).n_children == trainer.load_model(b).n_children == 1
    fold_time = trainer.load_model(a).fit_time
    # A has the most weight per second; the time limit fits its completion but not B's after it
    monkeypatch.setattr(AbstractTabularTrainer, "_model_weights", lambda self, model: {a: 0.9, b: 0.1})
    TimedDummyModel.time_limits.clear()
    completed = trainer.complete_bags(trainer.load_X(), trainer.load_y(), models=[b, a], time_limit=5 * fold_time)
    assert completed == [a]
    assert trainer.load_model(a).n_children == NUM_FOLDS
    assert trainer.load_model(b).n_children == 1 and trainer.get_model_attribute(b, "folds_pending")
    # the completed folds each got at least the floor times the first fold's fit time (less the moments between the
    # limit and the fold's fit)
    assert len(TimedDummyModel.time_limits) == NUM_FOLDS - 1
    assert all(limit >= COMPLETION_FOLD_TIME_FLOOR * fold_time * 0.98 for _, limit in TimedDummyModel.time_limits)
    assert np.isfinite(_proba(predictor, _frame("binary", rows=50, seed=1), model=b)).all()


def test_a_refit_that_runs_out_of_time_leaves_the_bag_on_its_first_folds(tmp_path):
    train = _frame("binary", rows=400, seed=0)
    refit = _timed(refit_rows=len(train), ag_args_ensemble={"refit_folds": True})
    predictor = _fit(train, tmp_path / "limited", hyperparameters=refit, validation_folds=1, time_limit=120)
    trainer = predictor._trainer
    (bag,) = _bags(predictor)
    model = trainer.load_model(bag)
    assert model._refit_folds_fallback and model.n_children == 1
    assert trainer.get_model_attribute(bag, "folds_pending")  # predicts from its first fold, and is not completed
    assert np.isfinite(_proba(predictor, _frame("binary", rows=50, seed=1))).all()
    # without a time limit the refit fits as before
    predictor = _fit(train, tmp_path / "unlimited", hyperparameters=refit, validation_folds=1)
    model = predictor._trainer.load_model(_bags(predictor)[0])
    assert not model._refit_folds_fallback and model._refit_oof


def test_a_time_limited_refit_drops_the_folds_kept_for_the_fallback(tmp_path):
    train = _frame("binary", rows=400, seed=0)
    refit = _timed(ag_args_ensemble={"refit_folds": True})
    predictor = _fit(train, tmp_path, hyperparameters=refit, validation_folds=2, time_limit=120)
    (bag,) = _bags(predictor)
    model = predictor._trainer.load_model(bag)
    assert not model._refit_folds_fallback and model.n_children == 1
    assert not os.path.exists(os.path.join(model.path, "S1F2"))


def test_the_auto_reserve_is_what_completing_the_bags_greedy_uses_takes(tmp_path, monkeypatch):
    train = _frame("binary", rows=400, seed=0)
    monkeypatch.setattr(AbstractTabularTrainer, "complete_bags", lambda self, *args, **kwargs: [])
    hyperparameters = {"GBM": {"num_boost_round": 30}, "DUMMY": {}}
    trainer = _fit(train, tmp_path / "bag", hyperparameters=hyperparameters, validation_folds=1)._trainer
    gbm = trainer.load_model("LightGBM_BAG_L1")
    # greedy uses the LightGBM bag (not the constant), which has three of its four folds to fit
    needed = COMPLETION_TIME_MARGIN * gbm.fit_time * (NUM_FOLDS - 1)
    assert trainer._completion_reserve(trainer.load_y(), time_left=10 * needed) == pytest.approx(needed, rel=1e-6)
    # a completion that cannot fit in the time left is kept no time for: the bag keeps its first folds either way
    assert trainer._completion_reserve(trainer.load_y(), time_left=0.5 * needed) == 0
    # a bag refit on all rows has nothing to complete
    refit = {"GBM": {"num_boost_round": 30, "ag_args_ensemble": {"refit_folds": True}}, "DUMMY": {}}
    trainer = _fit(train, tmp_path / "refit", hyperparameters=refit, validation_folds=1)._trainer
    assert trainer._completion_reserve(trainer.load_y(), time_left=1e6) == 0


def test_a_model_is_fit_only_in_the_time_the_reserve_leaves(tmp_path, monkeypatch):
    # once a model is fit, the reserve takes every second left: the next model is not fit
    monkeypatch.setattr(
        AbstractTabularTrainer,
        "_completion_reserve",
        lambda self, y, time_left: time_left if self.get_model_names(level=1) else 0.0,
    )
    hyperparameters = {"GBM": {"num_boost_round": 30}, "LR": {}}
    predictor = _fit(
        _frame("binary", rows=400, seed=0),
        tmp_path,
        hyperparameters=hyperparameters,
        validation_folds=1,
        time_limit=120,
    )
    assert len(_bags(predictor)) == 1
