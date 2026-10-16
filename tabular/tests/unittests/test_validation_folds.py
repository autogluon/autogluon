"""`validation_folds`: the weighted ensemble is chosen on some of the bag folds, then the bags it uses are completed."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autogluon.tabular import TabularPredictor

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
