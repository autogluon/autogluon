"""`group_views`: pooled and collapsed views of every bagged model on data with one label per group."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autogluon.common.features.feature_metadata import FeatureMetadata
from autogluon.common.features.types import S_GROUP_KEY, S_STACK
from autogluon.core.models import AbstractModel
from autogluon.tabular import TabularPredictor
from autogluon.tabular.models import LGBModel
from autogluon.tabular.trainer.group_views import (
    GROUP_COLLAPSED_SUFFIX,
    GROUP_KEY_COLUMN,
    GROUP_LOG_ROWS_COLUMN,
    GROUP_ORDER_COLUMN,
    GROUP_POOLED_SEPARATOR,
    GroupCollapse,
    GroupCollapsedModel,
    GroupViews,
    _group_mode,
    group_codes,
    group_order,
    group_pooling_name,
    group_pooling_strategies,
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
    kwargs.setdefault("hyperparameters", HYPERPARAMETERS)
    return TabularPredictor(label="label", path=str(tmp_path / "predictor"), verbosity=0).fit(
        train, num_gpus=0, dynamic_stacking=False, **kwargs
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


def _max_pool(pred: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """A custom pooling strategy: each group's largest prediction, per column."""
    return pd.DataFrame(pred).groupby(groups).transform("max").to_numpy()


def test_pool_within_groups_by_mean_median_or_a_custom_strategy():
    codes = np.array([0, 0, 0, 1, 1])
    proba = np.array([[0.2, 0.8], [0.5, 0.5], [0.4, 0.6], [0.9, 0.1], [0.6, 0.4]])
    mean = np.repeat([[11 / 30, 19 / 30], [0.75, 0.25]], [3, 2], axis=0)
    np.testing.assert_allclose(pool_within_groups(proba, codes, "multiclass", "mean"), mean, rtol=1e-6)
    median = np.repeat([[0.4, 0.6], [0.75, 0.25]], [3, 2], axis=0)
    np.testing.assert_allclose(pool_within_groups(proba, codes, "multiclass", "median"), median, rtol=1e-6)
    np.testing.assert_allclose(pool_within_groups(proba[:, 1], codes, "binary", "median"), median[:, 1], rtol=1e-6)
    # a custom strategy's class probabilities are renormalized
    custom = np.repeat([[0.5 / 1.3, 0.8 / 1.3], [0.9 / 1.3, 0.4 / 1.3]], [3, 2], axis=0)
    np.testing.assert_allclose(pool_within_groups(proba, codes, "multiclass", _max_pool), custom, rtol=1e-6)

    y = np.array([1.0, 2.0, 6.0, 4.0, 8.0])
    np.testing.assert_allclose(pool_within_groups(y, codes, "regression"), [3, 3, 3, 6, 6])
    np.testing.assert_allclose(pool_within_groups(y, codes, "regression", "median"), [2, 2, 2, 6, 6])
    np.testing.assert_allclose(pool_within_groups(y, codes, "regression", _max_pool), [6, 6, 6, 8, 8])

    with pytest.raises(ValueError, match="does not apply to regression"):
        pool_within_groups(y, codes, "regression", "geometric_mean")
    with pytest.raises(ValueError, match="must be one of"):
        pool_within_groups(y, codes, "regression", "mode")
    with pytest.raises(ValueError, match="shape of its input"):
        pool_within_groups(proba, codes, "multiclass", lambda pred, groups: pred[:, :1])


def test_group_views_input_forms():
    assert GroupViews.from_input(True) == GroupViews()
    assert GroupViews.from_input("auto") == GroupViews(enabled="auto")
    assert GroupViews.from_input({"enabled": "auto", "pooling": "median"}) == GroupViews("auto", "median")
    assert GroupViews.from_input(GroupViews(pooling=_max_pool)).pooling is _max_pool
    for off in (None, False, {"enabled": False}):
        assert GroupViews.from_input(off) is None
    assert GroupViews().resolve_pooling("binary") == "geometric_mean"
    assert GroupViews().resolve_pooling("regression") == "mean"
    with pytest.raises(ValueError, match="Invalid `group_views` keys"):
        GroupViews.from_input({"pool": "mean"})
    with pytest.raises(ValueError, match="must be True, False or 'auto'"):
        GroupViews.from_input("yes")
    with pytest.raises(ValueError, match="must be one of"):
        GroupViews(pooling="mode")
    with pytest.raises(ValueError, match="must be picklable"):
        GroupViews(pooling=lambda pred, groups: pred).resolve_pooling("binary")
    # an in-memory predictor saves nothing, so it takes any callable
    assert callable(GroupViews(pooling=lambda pred, groups: pred).resolve_pooling("binary", check_picklable=False))

    listed = GroupViews.from_input({"pooling": ["mean", "median", _max_pool]})
    assert listed.pooling == ("mean", "median", _max_pool)
    assert hash(listed) == hash(GroupViews(pooling=("mean", "median", _max_pool)))
    assert listed.resolve_pooling("regression") == ("mean", "median", _max_pool)
    assert GroupViews(pooling=["median"]).resolve_pooling("binary") == "median"
    with pytest.raises(ValueError, match="at least one strategy"):
        GroupViews(pooling=[])
    with pytest.raises(ValueError, match="must be distinct"):
        GroupViews(pooling=["mean", "mean"])
    with pytest.raises(ValueError, match="must be one of"):
        GroupViews(pooling=["mean", "mode"])
    with pytest.raises(ValueError, match="does not apply to regression"):
        GroupViews(pooling=["mean", "geometric_mean"]).resolve_pooling("regression")
    with pytest.raises(ValueError, match="must be picklable"):
        GroupViews(pooling=["mean", lambda pred, groups: pred]).resolve_pooling("binary")

    assert GroupViews().views == ("pooled", "collapsed")
    assert GroupViews.from_input({"views": "collapsed"}).views == ("collapsed",)
    assert GroupViews(views=["collapsed", "pooled"]) == GroupViews()
    for views in ([], ["rows"], ["pooled", "pooled"]):
        with pytest.raises(ValueError, match="non-empty list of distinct views"):
            GroupViews(views=views)
    with pytest.raises(ValueError, match="fits only the collapsed ones"):
        GroupViews(pooling="median", views="collapsed")
    with pytest.raises(ValueError, match="fits only the pooled ones"):
        GroupViews(views="pooled", collapse={"other": "first"})
    assert GroupViews(views="collapsed", collapse={"other": "first"}).collapse == GroupCollapse(other="first")


def test_group_collapse_input_forms():
    assert GroupViews().collapse is None
    collapse = {"numeric": ["mean", "last"], "other": "last", "order_on": "visit", "exclude_special_types": ["agg"]}
    views = GroupViews.from_input({"pooling": "median", "collapse": collapse})
    assert views.collapse == GroupCollapse(
        numeric=("mean", "last"), other="last", order_on="visit", exclude_special_types=("agg",)
    )
    assert hash(views) == hash(GroupViews(pooling="median", collapse=GroupCollapse(**collapse)))
    assert views.collapse.numeric_summaries == ("mean", "last") and views.collapse.order_columns == ["visit"]
    assert GroupCollapse(numeric="median").numeric_summaries == ("median",)
    assert GroupCollapse(other="first", order_on=["a", "b"]).order_columns == ["a", "b"]
    assert not GroupCollapse().ordered and GroupCollapse(numeric=["mean", "first"]).ordered
    with pytest.raises(ValueError, match="Invalid `group_views` collapse keys"):
        GroupViews(collapse={"summaries": "mean"})
    with pytest.raises(ValueError, match="must be a dict or a GroupCollapse"):
        GroupViews(collapse="mean")
    with pytest.raises(ValueError, match="`numeric` must be one of"):
        GroupCollapse(numeric="mode")
    with pytest.raises(ValueError, match="at least one summary"):
        GroupCollapse(numeric=[])
    with pytest.raises(ValueError, match="must be distinct"):
        GroupCollapse(numeric=["mean", "mean"])
    with pytest.raises(ValueError, match="`other` must be one of"):
        GroupCollapse(other="mean")
    with pytest.raises(ValueError, match="neither `numeric` nor `other` uses"):
        GroupCollapse(order_on="visit")


def test_group_order_is_the_position_in_the_sorted_frame():
    df = pd.DataFrame({"t": [3.0, 1.0, None, 1.0], "u": [0, 1, 0, 0]})
    # ties keep the frame's order, missing values come first
    np.testing.assert_array_equal(group_order(df, ["t"]), [3, 1, 0, 2])
    np.testing.assert_array_equal(group_order(df, ["t", "u"]), [3, 2, 0, 1])
    with pytest.raises(KeyError, match="order_on"):
        group_order(df, ["missing"])


def _collapse_frame(
    collapse: GroupCollapse | None, X: pd.DataFrame, feature_metadata: FeatureMetadata
) -> pd.DataFrame:
    model = GroupCollapsedModel(
        model_base=LGBModel(), collapse=collapse, problem_type="binary", eval_metric="accuracy", name="collapsed"
    )
    model._features_internal = list(X.columns)
    model._feature_metadata = feature_metadata
    model._set_collapse_features()
    return model._collapse(X)[0]


def test_the_collapse_summarises_each_feature_as_set():
    X = pd.DataFrame(
        {
            "f": [1.0, np.nan, 3.0, 4.0, 5.0, 6.0],
            "c": pd.Categorical(["a", "b", None, "b", "a", "a"]),
            "agg": [9.0] * 6,
            GROUP_KEY_COLUMN: [0, 0, 0, 1, 1, 2],
            # group 0's rows in reverse order, group 1's in frame order
            GROUP_ORDER_COLUMN: group_order(pd.DataFrame({"t": [3, 2, 1, 1, 2, 5]}), ["t"]),
        }
    )
    feature_metadata = FeatureMetadata(
        type_map_raw={
            "f": "float",
            "c": "category",
            "agg": "float",
            GROUP_KEY_COLUMN: "int",
            GROUP_ORDER_COLUMN: "int",
        },
        type_group_map_special={S_GROUP_KEY: [GROUP_KEY_COLUMN, GROUP_ORDER_COLUMN], "agg_type": ["agg"]},
    )
    default = _collapse_frame(None, X, feature_metadata)
    assert list(default.columns) == ["f", "agg", "c", GROUP_LOG_ROWS_COLUMN]
    np.testing.assert_allclose(default["f"], [2.0, 4.5, 6.0])
    assert default["c"].tolist() == ["a", "a", "a"]  # group 0 ties "a" and "b": the smallest
    np.testing.assert_allclose(default[GROUP_LOG_ROWS_COLUMN], np.log([3, 2, 1]))

    collapse = GroupCollapse(numeric=["mean", "last", "std"], other="last", exclude_special_types=["agg_type"])
    table = _collapse_frame(collapse, X, feature_metadata)
    assert list(table.columns) == ["f.mean", "f.last", "f.std", "c", GROUP_LOG_ROWS_COLUMN]
    # the last non-missing value in the rows' order: group 0's earliest row is the frame's last
    np.testing.assert_allclose(table["f.last"], [1.0, 5.0, 6.0])
    assert table["c"].tolist() == ["a", "a", "a"]
    assert table["c"].dtype == X["c"].dtype
    first = _collapse_frame(GroupCollapse(numeric="first", other="first"), X, feature_metadata)
    np.testing.assert_allclose(first["f"], [3.0, 4.0, 6.0])
    assert first["c"].tolist() == ["b", "b", "a"]


def test_group_codes_number_the_groups_of_the_frame():
    df = pd.DataFrame({"a": ["x", "y", "x", "z"], "b": [1, 1, 2, 1]})
    np.testing.assert_array_equal(group_codes(df, ["a"]), [0, 1, 0, 2])
    np.testing.assert_array_equal(group_codes(df, ["a", "b"]), [0, 1, 2, 3])
    with pytest.raises(KeyError, match="group_on"):
        group_codes(df, ["missing"])
    with pytest.raises(ValueError, match="NaN"):
        group_codes(df.assign(a=[None, "y", "x", "z"]), ["a"])


@pytest.mark.parametrize(
    "a,b",
    [
        (["a||b", "a", "a||b"], ["c", "b||c", "c"]),
        ([1, "1", 1], ["x", "x", "x"]),
    ],
)
def test_group_codes_keep_distinct_composite_keys_separate(a, b):
    df = pd.DataFrame({"a": a, "b": b})
    codes = group_codes(df, ["a", "b"])
    np.testing.assert_array_equal(codes, [0, 1, 0])
    np.testing.assert_allclose(pool_within_groups(np.array([0.0, 10.0, 2.0]), codes, "regression"), [1, 10, 1])


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


@pytest.mark.parametrize("dtype", ["object", "string", "boolean", "category"])
def test_group_mode_handles_an_entirely_missing_column(dtype):
    column = pd.Series([None, None, None], dtype=dtype, name="missing")
    mode = _group_mode(column, np.array([0, 1, 0]), 2)
    assert len(mode) == 2
    assert mode.isna().all()
    assert mode.dtype == column.dtype
    assert mode.name == column.name


# Ids without "regression": the test suite reserves that keyword for its opt-in regression-test marker.
@pytest.mark.parametrize("problem_type", ["binary", "multiclass", "regression"], ids=["binary", "multiclass", "rmse"])
def test_group_views_fit_two_views_and_predict_per_group(problem_type, tmp_path):
    train = _label_per_group_frame(problem_type, n_groups=40, rows=5, seed=0)
    test = _label_per_group_frame(problem_type, n_groups=15, rows=5, seed=1)
    predictor = _fit(train, tmp_path, validation_structure={"group_on": "patient"}, group_views=True)

    names = predictor.model_names()
    for base in ("LightGBM", "LinearModel"):
        assert f"{base}_BAG_L1" in names and f"{base}{GROUP_COLLAPSED_SUFFIX}_BAG_L1" in names

    ensemble = predictor._trainer.load_model(predictor.model_best)
    assert ensemble.params["group_pooling"] == ("mean" if problem_type == "regression" else "geometric_mean")

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


def _pooled_greedy_prediction(predictor: TabularPredictor, data: pd.DataFrame, pooling) -> pd.DataFrame:
    """The weighted ensemble's prediction, recomputed: its weights over its inputs, each a base model's
    predictions pooled by one of the strategies of `pooling` (a collapsed model's by the first)."""
    ensemble = predictor._trainer.load_model(predictor.model_best)
    strategies = {group_pooling_name(strategy): strategy for strategy in group_pooling_strategies(pooling)}
    codes = group_codes(data, ["patient"])
    out = 0
    for name, weight in ensemble.load_child(ensemble.models[0])._get_model_weights().items():
        model, _, strategy = name.partition(GROUP_POOLED_SEPARATOR)
        strategy = strategies[strategy] if strategy else next(iter(strategies.values()))
        pred = _predict_model(predictor, data, model)
        values = pred.to_numpy()[:, 1] if predictor.problem_type == "binary" else pred.to_numpy()
        pooled = pool_within_groups(values, codes, predictor.problem_type, strategy)
        if predictor.problem_type == "binary":
            pooled = np.stack([1 - pooled, pooled], axis=1)
        out = out + weight * pooled.reshape(pred.shape)
    return pd.DataFrame(out, index=data.index, columns=pred.columns)


def _predict_model(predictor: TabularPredictor, data: pd.DataFrame, model: str) -> pd.DataFrame:
    if predictor.problem_type == "regression":
        return predictor.predict(data, model=model).to_frame()
    return predictor.predict_proba(data, model=model)


@pytest.mark.parametrize(
    "problem_type, pooling",
    [("multiclass", "mean"), ("binary", "median"), ("regression", "median"), ("multiclass", _max_pool)],
    ids=["multiclass-mean", "binary-median", "rmse-median", "multiclass-custom"],
)
def test_the_ensemble_pools_by_the_chosen_strategy(problem_type, pooling, tmp_path):
    train = _label_per_group_frame(problem_type, n_groups=40, rows=5, seed=0)
    test = _label_per_group_frame(problem_type, n_groups=15, rows=5, seed=1)
    predictor = _fit(train, tmp_path, validation_structure={"group_on": "patient"}, group_views={"pooling": pooling})
    ensemble = predictor._trainer.load_model(predictor.model_best)
    assert ensemble.params["group_pooling"] == pooling

    pred = _predict(predictor, test)
    pd.testing.assert_frame_equal(
        pred, _pooled_greedy_prediction(predictor, test, pooling), atol=1e-5, check_dtype=False
    )
    loaded = TabularPredictor.load(predictor.path)
    pd.testing.assert_frame_equal(_predict(loaded, test), pred)


@pytest.mark.parametrize("problem_type", ["multiclass", "regression"], ids=["multiclass", "rmse"])
def test_a_list_of_strategies_gives_every_model_a_pooled_view_per_strategy(problem_type, tmp_path):
    train = _label_per_group_frame(problem_type, n_groups=40, rows=5, seed=0)
    test = _label_per_group_frame(problem_type, n_groups=15, rows=5, seed=1)
    pooling = ["mean", "median"]
    predictor = _fit(train, tmp_path, validation_structure={"group_on": "patient"}, group_views={"pooling": pooling})
    ensemble = predictor._trainer.load_model(predictor.model_best)
    assert ensemble.params["group_pooling"] == ("mean", "median")
    input_weights = ensemble.load_child(ensemble.models[0])._get_model_weights()
    for name in input_weights:
        model, _, strategy = name.partition(GROUP_POOLED_SEPARATOR)
        assert strategy == ("" if model.endswith(GROUP_COLLAPSED_SUFFIX + "_BAG_L1") else strategy)
        assert strategy in ("", "mean", "median")
    # A model's weight is the sum of its inputs'.
    model_weights = ensemble._get_model_weights()
    assert set(model_weights) == {name.partition(GROUP_POOLED_SEPARATOR)[0] for name in input_weights}
    assert sum(model_weights.values()) == pytest.approx(sum(input_weights.values()))

    pred = _predict(predictor, test)
    pd.testing.assert_frame_equal(
        pred, _pooled_greedy_prediction(predictor, test, pooling), atol=1e-5, check_dtype=False
    )
    loaded = TabularPredictor.load(predictor.path)
    pd.testing.assert_frame_equal(_predict(loaded, test), pred)

    refit_map = predictor.refit_full()
    full = predictor._trainer.load_model(refit_map[ensemble.name])
    assert sorted(set(full.stack_column_prefix_lst)) == sorted(ensemble.stack_column_prefix_lst)
    assert len(full.base_model_names) == len(set(full.base_model_names))
    pred_full = _predict_model(predictor, test, full.name)
    assert pred_full.groupby(test["patient"].to_numpy()).nunique().max().max() == 1


def test_every_input_reaches_the_ensemble_model(tmp_path):
    """Below 1000 rows the ensemble keeps one base model per model type (`max_base_models_per_type="auto"`): a
    collapsed model counts under the type it collapses, as a type of its own, so every model's collapsed view is
    kept, not only the best one. With a list of strategies, each pooled view is an input of its own."""
    train = _label_per_group_frame("binary", n_groups=40, rows=5, seed=0)
    test = _label_per_group_frame("binary", n_groups=15, rows=5, seed=1)
    predictor = _fit(
        train,
        tmp_path,
        hyperparameters={"GBM": {"num_boost_round": 20}, "LR": {}, "KNN": {}},
        validation_structure={"group_on": "patient"},
        group_views={"pooling": ["geometric_mean", _max_pool]},
        ensemble_hyperparameters={_UniformMeanEnsemble: {}},
    )
    ensemble = predictor._trainer.load_model(predictor.model_best)
    base_models = ["KNeighbors", "LightGBM", "LinearModel"]
    assert sorted(ensemble.base_model_names) == sorted(
        f"{base}{suffix}_BAG_L1" for base in base_models for suffix in ("", GROUP_COLLAPSED_SUFFIX)
    )
    assert sorted(ensemble.stack_column_prefix_lst) == sorted(
        [f"{base}_BAG_L1@{strategy}" for base in base_models for strategy in ("geometric_mean", "_max_pool")]
        + [f"{base}{GROUP_COLLAPSED_SUFFIX}_BAG_L1" for base in base_models]
    )
    pred = _predict(predictor, test)
    assert pred.groupby(test["patient"].to_numpy()).nunique().max().max() == 1


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
        validation_size_curves={"group_views": [[50, False], {"enabled": "auto", "pooling": "median"}]},
    )
    assert any(GROUP_COLLAPSED_SUFFIX in name for name in predictor.model_names())
    assert predictor._trainer.load_model(predictor.model_best).params["group_pooling"] == "median"


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


def test_a_bag_that_refits_its_folds_gets_a_collapsed_view(tmp_path):
    """A bag fit with `refit_folds` keeps no fold models: the collapsed view's fold children are saved without
    their inner model, and the bag reloads them for their trained parameters before refitting."""
    train = _label_per_group_frame("binary", n_groups=30, rows=4, seed=0)
    test = _label_per_group_frame("binary", n_groups=10, rows=4, seed=1)
    hyperparameters = {"GBM": {"num_boost_round": 20, "ag_args_ensemble": {"refit_folds": True}}}
    predictor = _fit(
        train,
        tmp_path,
        hyperparameters=hyperparameters,
        validation_structure={"group_on": "patient"},
        group_views=True,
        raise_on_model_failure=True,
    )
    assert f"LightGBM{GROUP_COLLAPSED_SUFFIX}_BAG_L1" in predictor.model_names()
    pred = predictor.predict_proba(test)
    assert pred.groupby(test["patient"].to_numpy()).nunique().max().max() == 1
    loaded = TabularPredictor.load(predictor.path)
    pd.testing.assert_frame_equal(loaded.predict_proba(test), pred)


@pytest.mark.parametrize("problem_type", ["binary", "regression"], ids=["binary", "rmse"])
def test_the_collapsed_views_alone_skip_the_fits_on_the_rows(problem_type, tmp_path):
    train = _label_per_group_frame(problem_type, n_groups=40, rows=5, seed=0)
    test = _label_per_group_frame(problem_type, n_groups=15, rows=5, seed=1)
    predictor = _fit(
        train,
        tmp_path,
        validation_structure={"group_on": "patient"},
        group_views={"views": ["collapsed"]},
        refit_full=True,
    )
    bags = [name for name in predictor.model_names() if not name.startswith("WeightedEnsemble")]
    assert sorted(bags) == sorted(
        f"{base}{GROUP_COLLAPSED_SUFFIX}_BAG_L1{suffix}"
        for base in ("LightGBM", "LinearModel")
        for suffix in ("", "_FULL")
    )
    pred = _predict(predictor, test)
    assert pred.groupby(test["patient"].to_numpy()).nunique().max().max() == 1
    # the same collapsed models as with both views: same configuration, same folds
    both = _fit(train, tmp_path / "both", validation_structure={"group_on": "patient"}, group_views=True)

    def collapsed_lightgbm(p: TabularPredictor) -> pd.DataFrame:
        model = f"LightGBM{GROUP_COLLAPSED_SUFFIX}_BAG_L1"
        if problem_type == "regression":
            return p.predict(test, model=model).to_frame()
        return p.predict_proba(test, model=model)

    pd.testing.assert_frame_equal(collapsed_lightgbm(predictor), collapsed_lightgbm(both))
    loaded = TabularPredictor.load(predictor.path)
    pd.testing.assert_frame_equal(_predict(loaded, test), pred)


def test_the_pooled_views_alone_fit_no_collapsed_models(tmp_path):
    train = _label_per_group_frame("multiclass", n_groups=40, rows=5, seed=0)
    test = _label_per_group_frame("multiclass", n_groups=15, rows=5, seed=1)
    predictor = _fit(train, tmp_path, validation_structure={"group_on": "patient"}, group_views={"views": "pooled"})
    assert not any(GROUP_COLLAPSED_SUFFIX in name for name in predictor.model_names())
    pred = predictor.predict_proba(test)
    assert pred.groupby(test["patient"].to_numpy()).nunique().max().max() == 1


def test_a_collapse_by_last_value_follows_order_on(tmp_path):
    """The collapsed views summarise as `collapse` says, read each group's rows in the order of `order_on` (so the
    rows' order in the data does not matter), and leave out the features of `exclude_special_types`."""
    rng = np.random.default_rng(0)
    train = _label_per_group_frame("binary", n_groups=40, rows=5, seed=0)
    test = _label_per_group_frame("binary", n_groups=15, rows=5, seed=1)
    for data in (train, test):
        data["visit"] = rng.permutation(len(data))
        data["patient_mean_f0"] = data.groupby("patient")["f0"].transform("mean")
    feature_metadata = FeatureMetadata.from_df(train.drop(columns="label")).add_special_types(
        {"patient_mean_f0": ["per_group"]}
    )
    collapse = {
        "numeric": ["mean", "last"],
        "other": "last",
        "order_on": "visit",
        "exclude_special_types": ["per_group"],
    }
    predictor = _fit(
        train,
        tmp_path,
        validation_structure={"group_on": "patient"},
        group_views={"collapse": collapse},
        feature_metadata=feature_metadata,
        raise_on_model_failure=True,
    )
    bag = predictor._trainer.load_model(f"LightGBM{GROUP_COLLAPSED_SUFFIX}_BAG_L1")
    child = bag.load_child(bag.models[0])
    assert child.collapse == GroupCollapse(**collapse)
    numeric = ["f0", "f1", "f2", "visit"]
    assert child.model.features == [
        *(f"{f}.mean" for f in numeric),
        *(f"{f}.last" for f in numeric),
        "color",
        GROUP_LOG_ROWS_COLUMN,
    ]
    # The order travels as a group key feature that the other models leave out.
    assert GROUP_ORDER_COLUMN not in predictor._trainer.load_model("LightGBM_BAG_L1").features

    pred = _predict(predictor, test)
    assert pred.groupby(test["patient"].to_numpy()).nunique().max().max() == 1
    shuffled = test.sample(frac=1, random_state=0)
    pd.testing.assert_frame_equal(_predict(predictor, shuffled).loc[test.index], pred)
    loaded = TabularPredictor.load(predictor.path)
    pd.testing.assert_frame_equal(_predict(loaded, test), pred)
    with pytest.raises(KeyError, match="order_on"):
        predictor.predict(test.drop(columns=["visit", "label"]))


def test_errors(tmp_path):
    train = _label_per_group_frame("binary", n_groups=20, rows=4, seed=0)
    with pytest.raises(ValueError, match="no `group_on`"):
        _fit(train, tmp_path / "a", group_views=True)
    with pytest.raises(ValueError, match="not bagged"):
        _fit(train, tmp_path / "b", num_bag_folds=0, validation_structure={"group_on": "patient"}, group_views=True)
    with pytest.raises(ValueError, match="must be True, False or 'auto'"):
        _fit(train, tmp_path / "c", validation_structure={"group_on": "patient"}, group_views="yes")
    # A pooling strategy that cannot apply is rejected before any model is fit, "auto" or not.
    regression = _label_per_group_frame("regression", n_groups=20, rows=4, seed=0)
    for enabled in (True, "auto"):
        with pytest.raises(ValueError, match="does not apply to regression"):
            _fit(
                regression,
                tmp_path / f"e_{enabled}",
                validation_structure={"group_on": "patient"},
                group_views={"enabled": enabled, "pooling": "geometric_mean"},
            )
        assert not (tmp_path / f"e_{enabled}" / "predictor" / "models").exists()
    with pytest.raises(ValueError, match="must be picklable"):
        _fit(
            train,
            tmp_path / "f",
            validation_structure={"group_on": "patient"},
            group_views={"pooling": lambda pred, groups: pred},
        )

    predictor = _fit(train, tmp_path / "d", validation_structure={"group_on": "patient"}, group_views=True)
    with pytest.raises(KeyError, match="group_on"):
        predictor.predict(train.drop(columns=["patient", "label"]))
