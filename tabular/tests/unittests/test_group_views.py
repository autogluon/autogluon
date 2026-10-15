"""`group_views`: pooled and collapsed views of every bagged model on data with one label per group."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autogluon.common.features.types import S_STACK
from autogluon.core.models import AbstractModel
from autogluon.tabular import TabularPredictor
from autogluon.tabular.trainer.group_views import (
    GROUP_COLLAPSED_SUFFIX,
    _group_mode,
    group_codes,
    label_constant_within_groups,
    pool_within_groups,
)

HYPERPARAMETERS = {"GBM": {"num_boost_round": 20}, "LR": {}}


def _label_per_group_frame(problem_type: str, n_groups: int, rows: int, seed: int) -> pd.DataFrame:
    """Groups of `rows` noisy observations of a group-level signal; the label is the group's."""
    rng = np.random.default_rng(seed)
    group = np.repeat(np.arange(n_groups), rows)
    latent = rng.normal(size=(n_groups, 3))
    signal = latent[:, 0] + 0.5 * latent[:, 1]
    if problem_type == "regression":
        y = signal
    elif problem_type == "binary":
        y = (signal > 0).astype(int)
    else:
        y = np.digitize(signal, np.quantile(signal, [1 / 3, 2 / 3]))
    df = pd.DataFrame(latent[group] + rng.normal(scale=2.0, size=(len(group), 3)), columns=["f0", "f1", "f2"])
    df["color"] = rng.choice(["red", "green", "blue"], size=len(group))
    df["patient"] = [f"s{seed}_{g}" for g in group]
    df["label"] = y[group]
    return df


def _fit(train: pd.DataFrame, tmp_path, **kwargs) -> TabularPredictor:
    kwargs.setdefault("num_bag_folds", 3)
    return TabularPredictor(label="label", path=str(tmp_path / "predictor"), verbosity=0).fit(
        train, hyperparameters=HYPERPARAMETERS, num_gpus=0, dynamic_stacking=False, **kwargs
    )


def _predict(predictor: TabularPredictor, data: pd.DataFrame) -> pd.DataFrame:
    if predictor.problem_type == "regression":
        return predictor.predict(data).to_frame()
    return predictor.predict_proba(data)


def test_pool_within_groups_takes_the_geometric_mean_of_probabilities():
    codes = np.array([0, 0, 1, 1, 1])
    proba = np.array([[0.2, 0.8], [0.5, 0.5], [0.9, 0.1], [0.6, 0.4], [0.3, 0.7]])
    expected = np.empty_like(proba)
    for code in np.unique(codes):
        rows = codes == code
        geometric = np.exp(np.log(proba[rows]).mean(axis=0))
        expected[rows] = geometric / geometric.sum()
    np.testing.assert_allclose(pool_within_groups(proba, codes, "multiclass"), expected, rtol=1e-6)
    # binary: the positive class's probability alone, pooled over both classes
    np.testing.assert_allclose(pool_within_groups(proba[:, 1], codes, "binary"), expected[:, 1], rtol=1e-6)
    np.testing.assert_allclose(
        pool_within_groups(np.array([1.0, 3.0, 2.0, 4.0, 6.0]), codes, "regression"), [2, 2, 4, 4, 4]
    )


def test_group_codes_number_the_groups_of_the_frame():
    df = pd.DataFrame({"a": ["x", "y", "x", "z"], "b": [1, 1, 2, 1]})
    np.testing.assert_array_equal(group_codes(df, ["a"]), [0, 1, 0, 2])
    np.testing.assert_array_equal(group_codes(df, ["a", "b"]), [0, 1, 2, 3])
    with pytest.raises(KeyError, match="group_on"):
        group_codes(df, ["missing"])
    with pytest.raises(ValueError, match="NaN"):
        group_codes(df.assign(a=[None, "y", "x", "z"]), ["a"])


def test_label_constant_within_groups():
    codes = np.array([0, 0, 1, 1])
    assert label_constant_within_groups(pd.Series([1, 1, 0, 0]), codes)
    assert not label_constant_within_groups(pd.Series([1, 0, 0, 0]), codes)
    # a missing label does not count as a second value
    assert label_constant_within_groups(pd.Series([1, None, 0, 0]), codes)


def test_group_mode_is_order_invariant_and_breaks_ties_by_the_smallest_value():
    column = pd.Series(["b", "a", "a", "c", "c", None], dtype="category")
    group = np.array([0, 0, 1, 1, 1, 2])
    mode = _group_mode(column, group, 3)
    assert mode.tolist()[:2] == ["a", "c"] and pd.isna(mode.iloc[2])
    order = np.array([5, 4, 3, 2, 1, 0])
    assert _group_mode(column.iloc[order].reset_index(drop=True), group[order], 3).tolist()[:2] == ["a", "c"]
    assert mode.dtype == column.dtype


# Ids without "regression": the test suite reserves that keyword for its opt-in regression-test marker.
@pytest.mark.parametrize("problem_type", ["binary", "multiclass", "regression"], ids=["binary", "multiclass", "rmse"])
def test_group_views_fit_two_views_and_predict_per_group(problem_type, tmp_path):
    train = _label_per_group_frame(problem_type, n_groups=40, rows=5, seed=0)
    test = _label_per_group_frame(problem_type, n_groups=15, rows=5, seed=1)
    predictor = _fit(train, tmp_path, validation_structure={"group_on": "patient"}, group_views=True)

    names = predictor.model_names()
    for base in ("LightGBM", "LinearModel"):
        assert f"{base}_BAG_L1" in names and f"{base}{GROUP_COLLAPSED_SUFFIX}_BAG_L1" in names

    pred = _predict(predictor, test)
    # Every view is constant within a group, so the ensemble is too.
    assert pred.groupby(test["patient"].to_numpy()).nunique().max().max() == 1

    # The rows of a group are a set: order and the other groups of the frame do not matter.
    shuffled = test.sample(frac=1, random_state=0)
    pd.testing.assert_frame_equal(_predict(predictor, shuffled).loc[test.index], pred)
    one_group = test[test["patient"] == test["patient"].iloc[0]]
    pd.testing.assert_frame_equal(_predict(predictor, one_group), pred.loc[one_group.index])

    loaded = TabularPredictor.load(predictor.path)
    pd.testing.assert_frame_equal(_predict(loaded, test), pred)


class _UniformMeanEnsemble(AbstractModel):
    """Averages the base models' columns uniformly: an ensemble model that does nothing about groups itself."""

    ag_key = "TEST_GROUP_UNIFORM_MEAN"
    ag_name = "UniformMean"

    def _fit(self, X, y, **kwargs):
        stack = self.feature_metadata.get_features(required_special_types=[S_STACK])
        k = self.num_classes if self.problem_type == "multiclass" else 1
        self.model = [stack[i : i + k] for i in range(0, len(stack), k)]

    def _predict_proba(self, X, **kwargs):
        mean = np.mean([X[cols].to_numpy() for cols in self.model], axis=0)
        return mean if self.problem_type == "multiclass" else mean[:, 0]


@pytest.mark.parametrize("problem_type", ["binary", "multiclass", "regression"], ids=["binary", "multiclass", "rmse"])
def test_an_ensemble_model_combines_the_pooled_views(problem_type, tmp_path):
    """With `ensemble_hyperparameters`, the group-pooled ensemble hands its model pooled inputs, so its output is
    constant within a group although the model knows nothing about groups."""
    train = _label_per_group_frame(problem_type, n_groups=40, rows=5, seed=0)
    test = _label_per_group_frame(problem_type, n_groups=15, rows=5, seed=1)
    predictor = _fit(
        train,
        tmp_path,
        validation_structure={"group_on": "patient"},
        group_views=True,
        ensemble_hyperparameters={_UniformMeanEnsemble: {}},
    )
    ensemble = predictor._trainer.load_model(predictor.model_best)
    assert isinstance(ensemble._get_model_base(), _UniformMeanEnsemble)
    assert any(name.endswith(GROUP_COLLAPSED_SUFFIX + "_BAG_L1") for name in ensemble.base_model_names)

    pred = _predict(predictor, test)
    assert pred.groupby(test["patient"].to_numpy()).nunique().max().max() == 1
    shuffled = test.sample(frac=1, random_state=0)
    pd.testing.assert_frame_equal(_predict(predictor, shuffled).loc[test.index], pred)


def test_group_views_help_on_label_per_group_data(tmp_path):
    """A sanity check on a clear case, not a benchmark: groups of noisy rows, one label per group."""
    train = _label_per_group_frame("binary", n_groups=60, rows=8, seed=0)
    test = _label_per_group_frame("binary", n_groups=40, rows=8, seed=1)
    rows = _fit(train, tmp_path / "rows", validation_structure={"group_on": "patient"})
    views = _fit(train, tmp_path / "views", validation_structure={"group_on": "patient"}, group_views=True)
    assert views.evaluate(test, silent=True)["accuracy"] > rows.evaluate(test, silent=True)["accuracy"]


def test_auto_turns_on_only_for_one_label_per_group(tmp_path):
    train = _label_per_group_frame("binary", n_groups=30, rows=4, seed=0)
    on = _fit(train, tmp_path / "on", validation_structure={"group_on": "patient"}, group_views="auto")
    assert any(GROUP_COLLAPSED_SUFFIX in name for name in on.model_names())

    varying = train.assign(label=np.random.default_rng(0).integers(0, 2, len(train)))
    off = _fit(varying, tmp_path / "off", validation_structure={"group_on": "patient"}, group_views="auto")
    assert not any(GROUP_COLLAPSED_SUFFIX in name for name in off.model_names())

    ungrouped = _fit(train, tmp_path / "ungrouped", group_views="auto")
    assert not any(GROUP_COLLAPSED_SUFFIX in name for name in ungrouped.model_names())


def test_a_size_curve_can_turn_group_views_on(tmp_path):
    train = _label_per_group_frame("binary", n_groups=30, rows=4, seed=0)
    predictor = _fit(
        train,
        tmp_path,
        validation_structure={"group_on": "patient"},
        validation_size_curves={"group_views": [[50, False], "auto"]},
    )
    assert any(GROUP_COLLAPSED_SUFFIX in name for name in predictor.model_names())


def test_refit_full_keeps_both_views(tmp_path):
    train = _label_per_group_frame("multiclass", n_groups=30, rows=4, seed=0)
    test = _label_per_group_frame("multiclass", n_groups=10, rows=4, seed=1)
    predictor = _fit(
        train,
        tmp_path,
        validation_structure={"group_on": "patient"},
        group_views=True,
        refit_full=True,
        set_best_to_refit_full=True,
    )
    assert predictor.model_best.endswith("_FULL")
    assert any(name.endswith(f"{GROUP_COLLAPSED_SUFFIX}_BAG_L1_FULL") for name in predictor.model_names())
    pred = predictor.predict_proba(test)
    assert pred.groupby(test["patient"].to_numpy()).nunique().max().max() == 1


def test_errors(tmp_path):
    train = _label_per_group_frame("binary", n_groups=20, rows=4, seed=0)
    with pytest.raises(ValueError, match="no `group_on`"):
        _fit(train, tmp_path / "a", group_views=True)
    with pytest.raises(ValueError, match="not bagged"):
        _fit(train, tmp_path / "b", num_bag_folds=0, validation_structure={"group_on": "patient"}, group_views=True)
    with pytest.raises(ValueError, match="must be True, False or 'auto'"):
        _fit(train, tmp_path / "c", validation_structure={"group_on": "patient"}, group_views="yes")

    predictor = _fit(train, tmp_path / "d", validation_structure={"group_on": "patient"}, group_views=True)
    with pytest.raises(KeyError, match="group_on"):
        predictor.predict(train.drop(columns=["patient", "label"]))
