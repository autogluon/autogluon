from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autogluon.core.models import BaggedEnsembleModel
from autogluon.core.models.dummy.dummy_model import DummyModel
from autogluon.core.utils.exceptions import TimeLimitExceeded
from autogluon.tabular.models import TabPFN3Model, TabPFN35Model
from autogluon.tabular.models.tabpfnv2.tabpfn3_model import estimate_fit_time_v3


class TimedDummyModel(DummyModel):
    """DummyModel that declares a fit-time estimate of one second per training row and counts its fits."""

    fits: list[int] = []

    @classmethod
    def _estimate_fit_time_static(cls, *, X: pd.DataFrame, X_val: pd.DataFrame | None = None, **kwargs) -> float:
        return float(len(X))

    def _fit(self, X: pd.DataFrame, **kwargs):
        type(self).fits.append(len(X))
        return super()._fit(X=X, **kwargs)


class TimedRefitDummyModel(TimedDummyModel):
    """Like an in-context model: a fit costs 1 s, predicting the validation rows one second per training row, and
    it can refit on all rows."""

    @classmethod
    def _estimate_fit_time_static(cls, *, X: pd.DataFrame, X_val: pd.DataFrame | None = None, **kwargs) -> float:
        return 1.0 + (float(len(X)) if X_val is not None else 0.0)

    def _more_tags(self) -> dict:
        return {"can_refit_full": True}


@pytest.fixture(autouse=True)
def _reset_fits():
    TimedDummyModel.fits = []
    TimedRefitDummyModel.fits = []


def _data(n: int = 10):
    X = pd.DataFrame({"a": np.arange(float(n)), "b": np.arange(float(n))})
    y = pd.Series([0, 1] * (n // 2))
    return X, y


def _init_model(model_cls, hyperparameters: dict | None = None):
    X, y = _data()
    model = model_cls(
        path="",
        name=model_cls.__name__,
        problem_type="binary",
        eval_metric="accuracy",
        hyperparameters=hyperparameters,
    )
    model.initialize(X=X, y=y)
    return model, X, y


def test__estimate_fit_time__default_is_none():
    assert DummyModel.can_estimate_fit_time_static() is False
    model, X, y = _init_model(DummyModel)
    assert model.estimate_fit_time(X=X) is None
    # no estimate -> no check, however short the time limit
    assert model._validate_fit_time(time_limit=1e-3, X=X) is None


def test__estimate_fit_time_static__not_implemented_by_default():
    X, y = _data()
    with pytest.raises(NotImplementedError):
        DummyModel.estimate_fit_time_static(X=X, y=y)


def test__validate_fit_time__raises_when_estimate_exceeds_time_limit():
    model, X, y = _init_model(TimedDummyModel)
    assert TimedDummyModel.can_estimate_fit_time_static() is True
    with pytest.raises(TimeLimitExceeded):
        model._validate_fit_time(time_limit=5, X=X)
    assert model._validate_fit_time(time_limit=20, X=X) == 10


def test__validate_fit_time__skipped_without_time_limit_or_ratio():
    model, X, y = _init_model(TimedDummyModel)
    assert model._validate_fit_time(time_limit=None, X=X) is None
    model, X, y = _init_model(TimedDummyModel, hyperparameters={"ag.max_estimated_time_ratio": None})
    assert model._validate_fit_time(time_limit=1e-3, X=X) is None


def test__validate_fit_time__ratio_scales_the_time_limit():
    model, X, y = _init_model(TimedDummyModel, hyperparameters={"ag.max_estimated_time_ratio": 2.0})
    assert model._validate_fit_time(time_limit=6, X=X) == 10


def test__fit__skips_a_model_estimated_to_exceed_its_time_limit():
    """The model is skipped before `_fit` starts, not after it has spent its time."""
    X, y = _data()
    model = TimedDummyModel(path="", name="TimedDummyModel", problem_type="binary", eval_metric="accuracy")
    with pytest.raises(TimeLimitExceeded):
        model.fit(X=X, y=y, time_limit=5)
    assert TimedDummyModel.fits == []
    model = TimedDummyModel(path="", name="TimedDummyModel", problem_type="binary", eval_metric="accuracy")
    model.fit(X=X, y=y, time_limit=100)
    assert TimedDummyModel.fits == [10]


def test__bagged_fit__skips_before_the_first_fold(tmp_path):
    """A fold child checks its estimate against its share of the bag's time, so a bag that cannot
    fit its folds is skipped before any fold trains."""
    X, y = _data(80)
    base = TimedDummyModel(path=str(tmp_path), name="TimedDummy", problem_type="binary", eval_metric="accuracy")
    bag_hyperparameters = {"fold_fitting_strategy": "sequential_local"}
    bag = BaggedEnsembleModel(
        model_base=base, path=str(tmp_path), name="TimedDummy_BAG", hyperparameters=bag_hyperparameters
    )
    # 8 folds of 70 rows, each estimated at 70 s; 100 s for the bag leaves 12.5 s per fold
    with pytest.raises(TimeLimitExceeded):
        bag.fit(X=X, y=y, k_fold=8, time_limit=100)
    assert TimedDummyModel.fits == []
    bag = BaggedEnsembleModel(
        model_base=base, path=str(tmp_path), name="TimedDummy_BAG", hyperparameters=bag_hyperparameters
    )
    bag.fit(X=X, y=y, k_fold=8, time_limit=10_000)
    assert TimedDummyModel.fits == [70] * 8


def _timed_bag(tmp_path, model_cls, **bag_hyperparameters) -> BaggedEnsembleModel:
    base = model_cls(path=str(tmp_path), name="TimedDummy", problem_type="binary", eval_metric="accuracy")
    return BaggedEnsembleModel(
        model_base=base,
        path=str(tmp_path),
        name="TimedDummy_BAG",
        hyperparameters={"fold_fitting_strategy": "sequential_local", **bag_hyperparameters},
    )


def test__bagged_fit__a_fold_with_an_estimate_gets_its_whole_share(tmp_path):
    """8 folds of 70 rows estimated at 70 s each fit in 570 s: the fold's whole share (71 s), not the
    80 % a model without an estimate fits in (its out-of-fold predictions take the rest)."""
    X, y = _data(80)
    _timed_bag(tmp_path, TimedDummyModel).fit(X=X, y=y, k_fold=8, time_limit=570)
    assert TimedDummyModel.fits == [70] * 8


def test__bagged_refit__reserves_the_refit_estimate(tmp_path):
    """With `refit_folds` the bag keeps the refit's estimate (1 s) back rather than 1.2 folds' share: 8 folds
    estimated at 71 s each fit in 600 s, (600 - 1) / 8 = 74.9 s per fold, where 1.2 shares would leave 65 s."""
    X, y = _data(80)
    _timed_bag(tmp_path, TimedRefitDummyModel, refit_folds=True).fit(X=X, y=y, k_fold=8, time_limit=600)
    assert TimedRefitDummyModel.fits == [70] * 8 + [80]


def test__estimate_fit_time_v3__matches_measured_large_folds():
    """BeyondArena 1M-row splits, one fold of an 8-fold bag (875k rows fit, 125k predicted), measured on
    one RTX PRO 6000 with 8 estimators: maps_router_eta_1m (997 features) 3300 s, cooking_time_1m (205)
    2296 s, consumer_complaints_1m (114) 2149 s."""
    for n_features, measured in [(997, 3300), (205, 2296), (114, 2149)]:
        estimate = estimate_fit_time_v3(
            n_train=875_000,
            n_val=125_000,
            n_features=n_features,
            n_estimators="auto",
            max_batch_size=975_000,
            num_gpus=1,
        )
        assert 0.8 < estimate / measured < 1.25


def test__estimate_fit_time_v3__scaling():
    kwargs = dict(n_train=100_000, n_val=10_000, n_features=100, max_batch_size=None, num_gpus=1)
    auto = estimate_fit_time_v3(n_estimators="auto", **kwargs)
    assert estimate_fit_time_v3(n_estimators=8, **kwargs) == auto
    assert estimate_fit_time_v3(n_estimators=1, **kwargs) < auto / 4
    # each chunk of predicted rows passes over the training rows again
    assert estimate_fit_time_v3(n_estimators=8, **{**kwargs, "max_batch_size": 2_500}) > 3 * auto

    # each estimator's network sees at most 768 features; fitting preprocesses every feature
    def predict_time(n_features):
        estimate = dict(kwargs, n_estimators=8, n_features=n_features)
        return estimate_fit_time_v3(**estimate) - estimate_fit_time_v3(**{**estimate, "n_val": 0})

    assert predict_time(10_000) == pytest.approx(predict_time(768))
    assert predict_time(500) < predict_time(768)
    fit = dict(kwargs, n_estimators=8, n_val=0)
    assert estimate_fit_time_v3(**{**fit, "n_features": 10_000}) > estimate_fit_time_v3(**{**fit, "n_features": 768})
    # estimators spread over the GPUs
    assert estimate_fit_time_v3(n_estimators=8, **{**kwargs, "num_gpus": 2}) < 0.6 * auto
    # no validation rows: the fit alone
    assert estimate_fit_time_v3(n_estimators=8, **{**kwargs, "n_val": 0}) < 0.2 * auto


def test__estimate_fit_time_v3__fit_mode():
    """ "low_memory" fits faster and predicts as fast as "fit_preprocessors"; "fit_with_cache" passes the
    training rows once, at fit."""
    kwargs = dict(n_train=500_000, n_features=200, n_estimators=8, max_batch_size=None, num_gpus=1)
    refit = {
        mode: estimate_fit_time_v3(n_val=0, fit_mode=mode, **kwargs) for mode in ("fit_preprocessors", "low_memory")
    }
    assert refit["low_memory"] == pytest.approx(0.07 + 0.6 * (refit["fit_preprocessors"] - 0.07))
    fold = {
        mode: estimate_fit_time_v3(n_val=50_000, fit_mode=mode, **kwargs)
        for mode in ("fit_preprocessors", "low_memory")
    }
    assert fold["low_memory"] - refit["low_memory"] == pytest.approx(
        fold["fit_preprocessors"] - refit["fit_preprocessors"]
    )
    chunked = dict(kwargs, max_batch_size=10_000)
    assert estimate_fit_time_v3(n_val=50_000, fit_mode="fit_with_cache", **chunked) < estimate_fit_time_v3(
        n_val=50_000, **chunked
    )
    assert estimate_fit_time_v3(n_val=0, fit_mode="fit_with_cache", **kwargs) > 10 * refit["fit_preprocessors"]


def test__estimate_fit_time_v3__none_on_cpu():
    assert (
        estimate_fit_time_v3(
            n_train=1000, n_val=100, n_features=10, n_estimators="auto", max_batch_size=None, num_gpus=0
        )
        is None
    )


@pytest.mark.parametrize("model_cls", [TabPFN3Model, TabPFN35Model])
def test__tabpfn_estimate_fit_time_static(model_cls):
    X = pd.DataFrame(np.zeros((100_000, 50)))
    X_val = X.iloc[:12_500]
    assert model_cls.can_estimate_fit_time_static() is True
    estimate = model_cls.estimate_fit_time_static(X=X, X_val=X_val, problem_type="binary", num_classes=2, num_gpus=1)
    assert estimate == estimate_fit_time_v3(
        n_train=100_000,
        n_val=12_500,
        n_features=50,
        n_estimators="auto",
        max_batch_size=model_cls._resolve_auto_max_batch_size(n_train=100_000),
        num_gpus=1,
    )
    fewer = model_cls.estimate_fit_time_static(
        X=X, X_val=X_val, problem_type="binary", num_classes=2, num_gpus=1, hyperparameters={"n_estimators": 2}
    )
    assert fewer < estimate
    assert (
        model_cls.estimate_fit_time_static(X=X, X_val=X_val, problem_type="binary", num_classes=2, num_gpus=0) is None
    )
