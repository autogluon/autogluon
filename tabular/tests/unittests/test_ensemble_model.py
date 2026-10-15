"""`ensemble_model`: a model of the caller's choice combines the base models in place of greedy ensemble selection.

The toy ensemble models below exercise the plumbing: which columns the model sees, where it is used,
and that the fitted predictor behaves like any other (save/load, refit_full, leaderboard, info).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autogluon.common.features.types import S_STACK
from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION
from autogluon.core.models import AbstractModel
from autogluon.core.models.greedy_ensemble.greedy_weighted_ensemble_model import GreedyWeightedEnsembleModel
from autogluon.tabular import TabularPredictor
from autogluon.tabular.configs.pipeline_presets import resolve_ensemble_model

HYPERPARAMETERS = {"GBM": {"num_boost_round": 20}, "LR": {}}
PROBLEM_TYPES = [BINARY, MULTICLASS, REGRESSION]
# Not "regression": the test suite skips tests with that keyword unless run with --runregression.
PROBLEM_TYPE_IDS = ["binary", "multiclass", "reg"]


class _StackColumnsMixin:
    """Splits the stack columns of `X` by base model, as `GreedyWeightedEnsembleModel` lays them out."""

    def _base_model_columns(self) -> dict[str, list[str]]:
        stack_columns = self.feature_metadata.get_features(required_special_types=[S_STACK])
        if self.problem_type == MULTICLASS:
            names = [c[: -len("_0")] for c in stack_columns if c.endswith("_0")]
            return {name: [f"{name}_{k}" for k in range(self.num_classes)] for name in names}
        return {name: [name] for name in stack_columns}


class BestBaseModelEnsemble(_StackColumnsMixin, AbstractModel):
    """Serves the predictions of the base model that scores best on the rows it is fit on."""

    ag_key = "TEST_BEST_BASE"
    ag_name = "BestBaseModel"

    def _fit(self, X, y, **kwargs):
        columns = self._base_model_columns()
        scores = {name: self.score_with_y_pred_proba(y, self._proba(X, cols)) for name, cols in columns.items()}
        best = max(scores, key=scores.get)
        self.best_ = best
        self.model = columns[best]
        # Only the chosen base model is read, so the ensemble can drop the others.
        self.features = list(columns[best])

    def _proba(self, X, cols):
        values = X[cols].to_numpy()
        return values if self.problem_type == MULTICLASS else values[:, 0]

    def _predict_proba(self, X, **kwargs):
        return self._proba(X, self.model)


class MeanWithTableStatsEnsemble(_StackColumnsMixin, AbstractModel):
    """Averages every base model uniformly, and records statistics of the original features it was fit with."""

    ag_key = "TEST_MEAN_TABLE"
    ag_name = "MeanWithTableStats"

    def _fit(self, X, y, **kwargs):
        stack = {c for cols in self._base_model_columns().values() for c in cols}
        original = [c for c in X.columns if c not in stack]
        self.table_stats_ = {"n_rows": len(X), "n_original_features": len(original)}
        self.model = self._base_model_columns()

    def _get_model_weights(self) -> dict:
        return {name: 1 / len(self.model) for name in self.model}

    def _predict_proba(self, X, **kwargs):
        preds = [X[cols].to_numpy() for cols in self.model.values()]
        mean = np.mean(preds, axis=0)
        return mean if self.problem_type == MULTICLASS else mean[:, 0]


def _data(problem_type: str, n: int = 240, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, 4)), columns=[f"f{i}" for i in range(4)])
    signal = df.f0 + rng.normal(scale=0.3, size=n)
    if problem_type == BINARY:
        df["label"] = (signal > 0).astype(int)
    elif problem_type == MULTICLASS:
        df["label"] = np.digitize(signal, [-0.5, 0.5])
    else:
        df["label"] = signal
    df.loc[df.index[:10], "f3"] = np.nan  # a missing original feature must not drop the row from the ensemble's fit
    return df


def _fit(path, train, problem_type, **kwargs):
    kwargs.setdefault("num_bag_folds", 2)
    return TabularPredictor(label="label", problem_type=problem_type, path=str(path), verbosity=0).fit(
        train, hyperparameters=HYPERPARAMETERS, **kwargs
    )


def _ensemble(predictor):
    return predictor._trainer.load_model("WeightedEnsemble_L2")


def _child(predictor, name="WeightedEnsemble_L2"):
    model = predictor._trainer.load_model(name)
    return model.load_child(model.models[0])


@pytest.mark.parametrize("problem_type", PROBLEM_TYPES, ids=PROBLEM_TYPE_IDS)
def test_ensemble_model_replaces_greedy(tmp_path, problem_type):
    """The ensemble serves the chosen base model's predictions, under the weighted ensemble's name."""
    train = _data(problem_type)
    test = _data(problem_type, n=60, seed=1).drop(columns=["label"])
    predictor = _fit(tmp_path, train, problem_type, ensemble_model={BestBaseModelEnsemble: {}})

    assert predictor.model_best == "WeightedEnsemble_L2"
    child = _child(predictor)
    assert isinstance(child, BestBaseModelEnsemble)
    best = child.best_
    # Only the chosen base model is needed at prediction time.
    assert _ensemble(predictor).base_model_names == [best]
    if problem_type == REGRESSION:
        np.testing.assert_allclose(predictor.predict(test), predictor.predict(test, model=best), atol=1e-6)
    else:
        np.testing.assert_allclose(
            predictor.predict_proba(test).to_numpy(), predictor.predict_proba(test, model=best).to_numpy(), atol=1e-6
        )
    # No weights to report: the leaderboard and info still work.
    assert _ensemble(predictor)._get_model_weights() == {}
    assert not _ensemble(predictor)._get_tags()["print_weights"]
    assert "WeightedEnsemble_L2" in set(predictor.leaderboard(silent=True)["model"])
    assert "WeightedEnsemble_L2" in predictor.info()["model_info"]


@pytest.mark.parametrize("problem_type", PROBLEM_TYPES, ids=PROBLEM_TYPE_IDS)
def test_original_features_reach_the_ensemble_model_only_on_request(tmp_path, problem_type):
    train = _data(problem_type)
    test = _data(problem_type, n=60, seed=2).drop(columns=["label"])
    with_orig = _fit(
        tmp_path / "orig",
        train,
        problem_type,
        ensemble_model={MeanWithTableStatsEnsemble: {"ag_args_ensemble": {"use_orig_features": True}}},
    )
    stats = _child(with_orig).table_stats_
    assert stats == {"n_rows": len(train), "n_original_features": 4}, "every row, NaN features included"
    # The mean of the base models, also at prediction time when the original features ride along.
    names = list(_child(with_orig).model)
    if problem_type == REGRESSION:
        expected = np.mean([with_orig.predict(test, model=n).to_numpy() for n in names], axis=0)
        np.testing.assert_allclose(with_orig.predict(test).to_numpy(), expected, atol=1e-6)
    else:
        expected = np.mean([with_orig.predict_proba(test, model=n).to_numpy() for n in names], axis=0)
        np.testing.assert_allclose(with_orig.predict_proba(test).to_numpy(), expected, atol=1e-6)
    # Weights from the child are reported as greedy's are.
    assert _ensemble(with_orig)._get_model_weights() == {n: 0.5 for n in names}

    without = _fit(tmp_path / "stack", train, problem_type, ensemble_model={MeanWithTableStatsEnsemble: {}})
    assert _child(without).table_stats_["n_original_features"] == 0


def test_predictions_do_not_depend_on_the_batch(tmp_path):
    train = _data(BINARY)
    test = _data(BINARY, n=50, seed=3).drop(columns=["label"])
    predictor = _fit(
        tmp_path,
        train,
        BINARY,
        ensemble_model={MeanWithTableStatsEnsemble: {"ag_args_ensemble": {"use_orig_features": True}}},
    )
    full = predictor.predict_proba(test)[1].to_numpy()
    rows = np.concatenate([predictor.predict_proba(test.iloc[[i]])[1].to_numpy() for i in range(len(test))])
    np.testing.assert_allclose(rows, full, atol=1e-7)


def test_ensemble_model_survives_save_load_and_refit_full(tmp_path):
    train = _data(MULTICLASS)
    test = _data(MULTICLASS, n=40, seed=4).drop(columns=["label"])
    predictor = _fit(
        tmp_path,
        train,
        MULTICLASS,
        ensemble_model={MeanWithTableStatsEnsemble: {"ag_args_ensemble": {"use_orig_features": True}}},
    )
    before = predictor.predict_proba(test)
    loaded = TabularPredictor.load(str(tmp_path), verbosity=0)
    pd.testing.assert_frame_equal(loaded.predict_proba(test), before)

    refit_map = loaded.refit_full()
    refit = refit_map["WeightedEnsemble_L2"]
    assert isinstance(_child(loaded, refit), MeanWithTableStatsEnsemble)
    assert loaded.predict_proba(test, model=refit).shape == before.shape


def test_holdout_validation_fits_the_ensemble_model_on_the_validation_rows(tmp_path):
    train = _data(BINARY)
    predictor = _fit(
        tmp_path,
        train,
        BINARY,
        num_bag_folds=0,
        holdout_frac=0.25,
        ensemble_model={MeanWithTableStatsEnsemble: {"ag_args_ensemble": {"use_orig_features": True}}},
    )
    assert abs(_child(predictor).table_stats_["n_rows"] - len(train) * 0.25) <= 1, "the validation rows only"


@pytest.mark.parametrize("threshold, expected", [(100, GreedyWeightedEnsembleModel), (1_000, BestBaseModelEnsemble)])
def test_size_curve_switches_the_ensemble_model(tmp_path, threshold, expected):
    """At or below the threshold the curve's model is used, above it greedy ensemble selection."""
    train = _data(BINARY)
    predictor = _fit(
        tmp_path,
        train,
        BINARY,
        validation_size_curves={"ensemble_model": [[threshold, {BestBaseModelEnsemble: {}}], None]},
    )
    assert type(_child(predictor)) is expected


def test_explicit_ensemble_model_wins_over_the_curve(tmp_path):
    train = _data(BINARY)
    predictor = _fit(
        tmp_path,
        train,
        BINARY,
        ensemble_model={MeanWithTableStatsEnsemble: {}},
        validation_size_curves={"ensemble_model": [[1_000, {BestBaseModelEnsemble: {}}], None]},
    )
    assert isinstance(_child(predictor), MeanWithTableStatsEnsemble)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (
            {"validation_mode": "none", "num_bag_folds": 0, "ensemble_weights": {"LightGBM": 0.5, "LinearModel": 0.5}},
            "cannot be combined",
        ),
        ({"validation_mode": "none", "num_bag_folds": 0}, "validation_mode='none'"),
        ({"fit_weighted_ensemble": False}, "fit_weighted_ensemble=False"),
    ],
)
def test_ensemble_model_rejects_what_leaves_it_nothing_to_learn(tmp_path, kwargs, match):
    with pytest.raises(ValueError, match=match):
        _fit(tmp_path, _data(BINARY), BINARY, ensemble_model={BestBaseModelEnsemble: {}}, **kwargs)
    assert not (tmp_path / "models").exists(), "rejected before any model is fit"


@pytest.mark.parametrize(
    "ensemble_model, match",
    [
        ({BestBaseModelEnsemble: {}, MeanWithTableStatsEnsemble: {}}, "exactly one entry"),
        ({}, "exactly one entry"),
        ({"NOT_A_MODEL": {}}, "not a registered model key"),
        ({int: {}}, "AbstractModel subclass"),
        ({BestBaseModelEnsemble: [{}]}, "hyperparameters as a dict"),
    ],
)
def test_malformed_ensemble_model_is_rejected(ensemble_model, match):
    with pytest.raises(ValueError, match=match):
        resolve_ensemble_model(ensemble_model, num_train_rows=100)


def test_resolve_ensemble_model_reads_the_curve_at_the_effective_size():
    curves = {"ensemble_model": [[100, {BestBaseModelEnsemble: {}}], None]}
    assert resolve_ensemble_model(None, num_train_rows=100, validation_size_curves=curves) == {
        BestBaseModelEnsemble: {}
    }
    assert resolve_ensemble_model(None, num_train_rows=101, validation_size_curves=curves) is None
    # Sized on groups when asked to.
    assert resolve_ensemble_model(
        None, num_train_rows=5_000, validation_size_curves=curves, num_group_instances=50, size_on_groups=True
    ) == {BestBaseModelEnsemble: {}}
    assert resolve_ensemble_model(None, num_train_rows=100) is None
