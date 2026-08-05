from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autogluon.common.utils.validation_structure import ValidationStructure


def _toy_grouped(n_groups: int = 20, rows_per_group: int = 10, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    group = np.repeat(np.arange(n_groups), rows_per_group)
    X = pd.DataFrame(
        {
            "num": rng.normal(size=len(group)),
            "gid": [f"g{g}" for g in group],
        }
    )
    y = pd.Series((rng.random(len(group)) + (group % 2) > 0.75).astype(int))
    return X, y


def _toy_temporal(n: int = 200, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "num": rng.normal(size=n),
            # duplicated timestamps to exercise tie handling
            "ts": pd.to_datetime("2026-01-01") + pd.to_timedelta(np.sort(rng.integers(0, n // 4, size=n)), unit="D"),
        }
    )
    y = pd.Series(rng.integers(0, 2, size=n))
    return X, y


def test__from_input__dict_and_invalid_keys():
    vs = ValidationStructure.from_input({"group_on": "gid"})
    assert vs.group_on == "gid"
    assert ValidationStructure.from_input(None) is None
    with pytest.raises(ValueError, match="Invalid `validation_structure` keys"):
        ValidationStructure.from_input({"group": "gid"})
    with pytest.raises(NotImplementedError):
        ValidationStructure.from_input({"group_on": "gid", "time_on": "ts"})


def test__fold_ids__group_on__folds_are_group_disjoint():
    X, y = _toy_grouped()
    vs = ValidationStructure(group_on="gid")
    fold_ids = vs.fold_ids(X, y, n_splits=5)
    assert len(fold_ids) == len(X)
    assert fold_ids.nunique() == 5
    # every group lands in exactly one fold
    assert (X.groupby("gid", observed=True).apply(lambda g: fold_ids.loc[g.index].nunique()) == 1).all()


def test__fold_ids__group_on__clamps_to_group_count():
    X, y = _toy_grouped(n_groups=3)
    fold_ids = ValidationStructure(group_on="gid").fold_ids(X, y, n_splits=8)
    assert fold_ids.nunique() == 3


def test__fold_ids__time_on__blocks_are_contiguous_and_tie_safe():
    X, y = _toy_temporal()
    vs = ValidationStructure(time_on="ts")
    fold_ids = vs.fold_ids(X, y, n_splits=8)
    order = np.argsort(X["ts"].to_numpy(), kind="stable")
    # fold labels are non-decreasing in time order -> contiguous blocks
    assert (np.diff(fold_ids.to_numpy()[order]) >= 0).all()
    # identical timestamps never straddle two folds
    assert (X.groupby("ts").apply(lambda g: fold_ids.loc[g.index].nunique()) == 1).all()


def test__holdout__time_on__is_forward():
    X, y = _toy_temporal()
    train_idx, val_idx = ValidationStructure(time_on="ts").holdout_split_indices(X, y, holdout_frac=0.2)
    assert len(train_idx) + len(val_idx) == len(X)
    assert X["ts"].iloc[train_idx].max() <= X["ts"].iloc[val_idx].min()


def test__holdout__group_on__is_group_disjoint():
    X, y = _toy_grouped()
    train_idx, val_idx = ValidationStructure(group_on="gid").holdout_split_indices(X, y, holdout_frac=0.2)
    assert not set(X["gid"].iloc[train_idx]) & set(X["gid"].iloc[val_idx])


def test__holdout__stratify_only__defers_to_default():
    X, y = _toy_grouped()
    assert ValidationStructure(stratify_on="gid").holdout_split_indices(X, y, holdout_frac=0.2) is None


def test__predictor_fit__group_on__bagged_and_holdout():
    from autogluon.tabular import TabularPredictor

    X, y = _toy_grouped()
    train_data = X.copy()
    train_data["label"] = y

    # bagged path: explicit structure-aware splits, one child model per split
    predictor = TabularPredictor(label="label", verbosity=0).fit(
        train_data,
        hyperparameters={"DUMMY": {}},
        num_bag_folds=4,
        num_bag_sets=1,
        validation_structure={"group_on": "gid"},
        fit_weighted_ensemble=False,
    )
    assert predictor.model_best is not None
    children = predictor.info()["model_info"]["Dummy_BAG_L1"]["children_info"]
    assert len(children) == 4

    # holdout path: structure-aware split, no crash
    predictor = TabularPredictor(label="label", verbosity=0).fit(
        train_data,
        hyperparameters={"DUMMY": {}},
        validation_structure={"group_on": "gid"},
    )
    assert predictor.model_best is not None


def test__predictor_fit__groups_and_structure__raises():
    from autogluon.tabular import TabularPredictor

    X, y = _toy_grouped()
    train_data = X.copy()
    train_data["label"] = y
    with pytest.raises(ValueError, match="not both"):
        TabularPredictor(label="label", groups="gid", verbosity=0).fit(
            train_data,
            hyperparameters={"DUMMY": {}},
            num_bag_folds=4,
            validation_structure={"group_on": "gid"},
        )


def test__validation_structure__repeats_and_clamping():
    """Repeated grouped bagging, which the groups channel cannot express."""
    from autogluon.common.utils.validation_structure import ValidationStructure

    rng = np.random.default_rng(0)
    n = 120
    X = pd.DataFrame({"f1": rng.normal(size=n), "gid": np.repeat(np.arange(20), 6)})
    y = pd.Series(rng.integers(0, 2, n))

    vs = ValidationStructure(group_on="gid")
    splits, folds, repeats = vs.custom_splits(X, y, num_folds=5, num_repeats=5)
    assert (folds, repeats) == (5, 5)
    assert len(splits) == 25
    groups = X["gid"].to_numpy()
    for train_idx, val_idx in splits:
        assert len(train_idx) and len(val_idx)
        assert not (set(groups[train_idx]) & set(groups[val_idx]))  # group-disjoint

    # fewer groups than folds clamps folds and forces a single repeat
    X_small = pd.DataFrame({"f1": rng.normal(size=9), "gid": np.repeat(np.arange(3), 3)})
    y_small = pd.Series(rng.integers(0, 2, 9))
    _, folds_small, repeats_small = ValidationStructure(group_on="gid").custom_splits(
        X_small, y_small, num_folds=8, num_repeats=5
    )
    assert folds_small == 3 and repeats_small == 1


def test__validation_structure__group_time_on():
    """`group_time_on` splits are simultaneously group-disjoint and forward in time."""
    from autogluon.common.utils.validation_structure import ValidationStructure

    n_groups = 12
    X = pd.DataFrame({"f1": np.arange(n_groups * 5, dtype=float), "gid": np.repeat(np.arange(n_groups), 5)})
    y = pd.Series(np.tile([0, 1], n_groups * 5 // 2))

    vs = ValidationStructure(group_time_on="gid")
    splits, folds, repeats = vs.custom_splits(X, y, num_folds=4, num_repeats=3)
    assert repeats == 1  # temporal partition is deterministic
    groups = X["gid"].to_numpy()
    for train_idx, val_idx in splits:
        assert not (set(groups[train_idx]) & set(groups[val_idx]))
    # each fold is one contiguous block of groups in time; blocks tile the timeline
    ranges = sorted((groups[val_idx].min(), groups[val_idx].max()) for _, val_idx in splits)
    for (_, prev_max), (next_min, _) in zip(ranges, ranges[1:]):
        assert prev_max < next_min

    train_idx, val_idx = vs.holdout_split_indices(X, y, holdout_frac=0.25)
    assert groups[val_idx].min() > groups[train_idx].max()  # forward holdout


def test__validation_structure__group_and_time_guidance():
    """`group_on` + `time_on` points at `group_time_on` instead of inventing semantics."""
    from autogluon.common.utils.validation_structure import ValidationStructure

    with pytest.raises(NotImplementedError, match="group_time_on"):
        ValidationStructure(group_on="g", time_on="t")
    with pytest.raises(ValueError, match="mutually exclusive"):
        ValidationStructure(group_time_on="g", group_on="g2")


def test__validation_structure__time_blocks_are_contiguous_and_tie_safe():
    """Time blocks split observed values, keep ties together, and balance row counts."""
    from autogluon.common.utils.validation_structure import ValidationStructure

    # heavily tied timestamps: 4 distinct values, uneven row counts
    times = np.repeat([10, 20, 30, 40], [50, 10, 10, 30])
    X = pd.DataFrame({"f1": np.arange(len(times), dtype=float), "t": times})
    y = pd.Series(np.zeros(len(times)))

    splits, folds, repeats = ValidationStructure(time_on="t").custom_splits(X, y, num_folds=3, num_repeats=4)
    assert repeats == 1  # temporal partition is deterministic
    assert folds == 3

    t = X["t"].to_numpy()
    seen = set()
    for _, val_idx in splits:
        block_values = set(t[val_idx])
        # a block never splits rows sharing a timestamp
        for value in block_values:
            assert set(np.flatnonzero(t == value)) <= set(val_idx)
        assert not (block_values & seen)  # blocks are disjoint in time
        seen |= block_values
    assert seen == set(t)  # every timestamp is validated exactly once
    # blocks are contiguous in time (fold order itself is not meaningful)
    ranges = sorted((t[val_idx].min(), t[val_idx].max()) for _, val_idx in splits)
    for (_, prev_max), (next_min, _) in zip(ranges, ranges[1:]):
        assert prev_max < next_min

    # fewer distinct time values than requested blocks: fold count drops to what exists
    X_few = pd.DataFrame({"f1": [0.0, 1.0, 2.0, 3.0], "t": [1, 1, 2, 2]})
    y_few = pd.Series([0, 1, 0, 1])
    _, folds_few, _ = ValidationStructure(time_on="t").custom_splits(X_few, y_few, num_folds=5)
    assert folds_few == 2


def test__predictor_fit__structure_splits_reach_the_bagged_model():
    """The resolved splits must actually be consumed by the bagged model, not just computed.

    Child-model counts alone cannot show this: plain KFold produces the same count, so a
    dropped `custom_splits` looks identical from the outside. Assert on the splits the model
    was fit with, and on their group-disjointness.
    """
    from autogluon.tabular import TabularPredictor

    rng = np.random.default_rng(0)
    n = 120
    df = pd.DataFrame({"f1": rng.normal(size=n), "gid": np.repeat(np.arange(20), 6), "label": rng.integers(0, 2, n)})

    for num_bag_sets in (1, 2):
        predictor = TabularPredictor(label="label", verbosity=0).fit(
            df,
            hyperparameters={"DUMMY": {}},
            num_bag_folds=5,
            num_bag_sets=num_bag_sets,
            validation_structure={"group_on": "gid"},
            fit_weighted_ensemble=False,
        )
        bagged_model = predictor._trainer.load_model("Dummy_BAG_L1")
        splits = bagged_model.params.get("custom_splits")
        assert splits is not None, "validation_structure splits never reached the bagged model"
        assert len(splits) == 5 * num_bag_sets

        groups = df["gid"].to_numpy()
        for train_idx, val_idx in splits:
            assert not (set(groups[train_idx]) & set(groups[val_idx]))
        # each repeat validates every row exactly once
        first_repeat = np.concatenate([val_idx for _, val_idx in splits[:5]])
        assert sorted(first_repeat.tolist()) == list(range(n))


def test__predictor_fit__no_structure_leaves_default_splitting_untouched():
    """Without `validation_structure` nothing is injected: the default path is unchanged."""
    from autogluon.tabular import TabularPredictor

    rng = np.random.default_rng(0)
    n = 120
    df = pd.DataFrame({"f1": rng.normal(size=n), "label": rng.integers(0, 2, n)})

    predictor = TabularPredictor(label="label", verbosity=0).fit(
        df, hyperparameters={"DUMMY": {}}, num_bag_folds=5, num_bag_sets=2, fit_weighted_ensemble=False
    )
    bagged_model = predictor._trainer.load_model("Dummy_BAG_L1")
    assert bagged_model.params.get("custom_splits") is None
    assert predictor._trainer._groups is None
    assert len(predictor.info()["model_info"]["Dummy_BAG_L1"]["children_info"]) == 10


def _grouped_toy_data(n: int = 90, n_groups: int = 15, seed: int = 0) -> pd.DataFrame:
    """Small grouped classification frame: 15 groups of 6 rows, both classes per group."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "f1": rng.normal(size=n),
            "gid": np.repeat(np.arange(n_groups), n // n_groups),
            "label": np.tile([0, 1], n // 2),
        }
    )


def _captured_splits(monkeypatch) -> list[dict]:
    """Record every `custom_splits` resolution, so tests can assert on what was actually used."""
    from autogluon.common.utils.validation_structure import ValidationStructure

    captured: list[dict] = []
    original = ValidationStructure.custom_splits

    def spy(self, X, y, num_folds=None, num_repeats=None, random_state=0, **kwargs):
        splits, folds, repeats = original(
            self, X, y, num_folds=num_folds, num_repeats=num_repeats, random_state=random_state, **kwargs
        )
        captured.append({"n_rows": len(X), "splits": splits, "random_state": random_state})
        return splits, folds, repeats

    monkeypatch.setattr(ValidationStructure, "custom_splits", spy)
    return captured


def test__predictor_fit__use_bag_holdout__splits_index_the_bagged_rows(monkeypatch):
    """With `use_bag_holdout` the trainer holds rows back, so the splits must index what remains.

    Resolving splits over the full frame and holding rows back afterwards leaves indices
    pointing past the end of the reduced frame, which fails the fit outright.
    """
    from autogluon.tabular import TabularPredictor

    df = _grouped_toy_data()
    captured = _captured_splits(monkeypatch)

    predictor = TabularPredictor(label="label", verbosity=0).fit(
        df,
        hyperparameters={"DUMMY": {}},
        num_bag_folds=3,
        num_bag_sets=1,
        validation_structure={"group_on": "gid"},
        use_bag_holdout=True,
        fit_weighted_ensemble=False,
        raise_on_model_failure=True,
    )

    splits = predictor._trainer.load_model("Dummy_BAG_L1").params.get("custom_splits")
    assert splits is not None
    # the resolution that produced the bagged splits saw fewer rows than the input frame
    bagged = captured[-1]
    assert bagged["n_rows"] < len(df), "no rows were held back for the bag-holdout"
    max_index = max(max(train_idx.max(), val_idx.max()) for train_idx, val_idx in splits)
    assert max_index < bagged["n_rows"], "split indices address rows that are not in the bagged frame"


@pytest.mark.parametrize("validation_procedure", ["cv", "holdout"])
def test__predictor_fit__dynamic_stacking__honors_the_structure(validation_procedure, monkeypatch):
    """DyStack audits for stacked overfitting, so its own sub-fit splits must not leak either."""
    from autogluon.common.utils.validation_structure import ValidationStructure
    from autogluon.tabular import TabularPredictor

    df = _grouped_toy_data(n=60, n_groups=10)
    groups = df["gid"].to_numpy()

    holdouts: list[tuple[np.ndarray, np.ndarray]] = []
    original_holdout = ValidationStructure.holdout_split_indices

    def holdout_spy(self, X, y, holdout_frac, random_state=0, **kwargs):
        split = original_holdout(self, X, y, holdout_frac=holdout_frac, random_state=random_state, **kwargs)
        if split is not None:
            holdouts.append(split)
        return split

    monkeypatch.setattr(ValidationStructure, "holdout_split_indices", holdout_spy)
    captured = _captured_splits(monkeypatch)

    TabularPredictor(label="label", verbosity=0).fit(
        df,
        hyperparameters={"DUMMY": {}},
        num_bag_folds=2,
        num_bag_sets=1,
        num_stack_levels=1,  # DyStack is a no-op unless stacking would be used
        validation_structure={"group_on": "gid"},
        dynamic_stacking=True,
        ds_args={
            "validation_procedure": validation_procedure,
            "n_folds": 2,
            "n_repeats": 1,
            "enable_ray_logging": False,
            "memory_safe_fits": False,
            "clean_up_fits": True,
        },
        fit_weighted_ensemble=False,
    )

    # DyStack resolves its sub-fit split with its own seed (42); every split it uses,
    # and every split the sub-fits bag over, must be group-disjoint.
    if validation_procedure == "cv":
        ds_resolutions = [c for c in captured if c["random_state"] == 42]
        assert ds_resolutions, "DyStack CV did not use the declared structure"
        checked = [split for c in ds_resolutions for split in c["splits"]]
    else:
        assert holdouts, "DyStack holdout did not use the declared structure"
        checked = holdouts
    for train_idx, val_idx in checked:
        assert not (set(groups[train_idx]) & set(groups[val_idx]))


@pytest.mark.parametrize("structure_key", ["time_on", "group_time_on"])
def test__predictor_fit__temporal_structure_collapses_requested_repeats(structure_key):
    """A temporal partition is deterministic, so repeats must collapse to 1 for the whole fit.

    The count has to be corrected end to end, not just inside the resolver: the bagged model
    asserts ``len(custom_splits) == num_bag_folds * num_bag_sets``, so a request of 3x5 that
    yields 3 splits only works if the reduced repeat count propagates to the trainer.
    """
    import logging

    from autogluon.tabular import TabularPredictor

    rng = np.random.default_rng(0)
    n = 120
    df = pd.DataFrame(
        {
            "f1": rng.normal(size=n),
            "t": np.repeat(np.arange(30), 4),  # 30 distinct time values / groups-in-time
            "label": rng.integers(0, 2, n),
        }
    )

    # AutoGluon reconfigures logger propagation, so collect from the module logger directly
    # rather than through caplog's root handler.
    messages: list[str] = []

    class _Collect(logging.Handler):
        def emit(self, record):
            messages.append(record.getMessage())

    structure_logger = logging.getLogger("autogluon.common.utils.validation_structure")
    handler = _Collect()
    structure_logger.addHandler(handler)
    try:
        predictor = TabularPredictor(label="label", verbosity=2).fit(
            df,
            hyperparameters={"DUMMY": {}},
            num_bag_folds=3,
            num_bag_sets=5,  # would give 15 children on IID or grouped data
            validation_structure={structure_key: "t"},
            fit_weighted_ensemble=False,
            raise_on_model_failure=True,
        )
    finally:
        structure_logger.removeHandler(handler)

    splits = predictor._trainer.load_model("Dummy_BAG_L1").params.get("custom_splits")
    assert len(splits) == 3, "repeats were not collapsed for a temporal partition"
    assert len(predictor.info()["model_info"]["Dummy_BAG_L1"]["children_info"]) == 3
    # the reduction is reported rather than silently applied
    assert any("num_repeats reduced from 5 to 1" in message for message in messages)


def test__predictor_fit__grouped_structure_keeps_requested_repeats():
    """Contrast with the temporal case: grouped folds are not deterministic, so repeats stand."""
    from autogluon.tabular import TabularPredictor

    rng = np.random.default_rng(0)
    n = 120
    df = pd.DataFrame({"f1": rng.normal(size=n), "gid": np.repeat(np.arange(20), 6), "label": rng.integers(0, 2, n)})
    predictor = TabularPredictor(label="label", verbosity=0).fit(
        df,
        hyperparameters={"DUMMY": {}},
        num_bag_folds=3,
        num_bag_sets=5,
        validation_structure={"group_on": "gid"},
        fit_weighted_ensemble=False,
        raise_on_model_failure=True,
    )
    splits = predictor._trainer.load_model("Dummy_BAG_L1").params.get("custom_splits")
    assert len(splits) == 15
    assert len(predictor.info()["model_info"]["Dummy_BAG_L1"]["children_info"]) == 15


def test__predictor_fit__size_validation_on_groups_changes_the_chosen_method():
    """Sizing on groups instead of rows can select a different validation method.

    900 rows across 60 groups is large by rows and small by groups: on rows the preset gives
    8 folds and permits a stack layer, on groups 6 folds and none.
    """
    from autogluon.tabular import TabularPredictor

    rng = np.random.default_rng(0)
    n = 900
    df = pd.DataFrame({"f1": rng.normal(size=n), "gid": np.repeat(np.arange(60), 15), "label": rng.integers(0, 3, n)})

    def fit(structure: dict) -> tuple[int, list[str]]:
        predictor = TabularPredictor(label="label", verbosity=0).fit(
            df,
            hyperparameters={"DUMMY": {}},
            validation_structure=structure,
            auto_stack=True,
            dynamic_stacking=False,
            fit_weighted_ensemble=False,
        )
        model_info = predictor.info()["model_info"]
        bagged = next(name for name in model_info if "BAG_L1" in name)
        return len(model_info[bagged]["children_info"]), sorted(model_info)

    folds_on_rows, models_on_rows = fit({"group_on": "gid"})
    folds_on_groups, models_on_groups = fit({"group_on": "gid", "size_validation_on_groups": True})

    assert folds_on_rows == 8
    assert folds_on_groups == 6
    # the group count is below the stacking size threshold, so no second layer is added
    assert any("L2" in name for name in models_on_rows)
    assert not any("L2" in name for name in models_on_groups)


def test__validation_structure__group_instance_count_is_none_without_grouping():
    """The count is only meaningful for grouped structures."""
    from autogluon.common.utils.validation_structure import ValidationStructure

    X = pd.DataFrame({"f1": [0.0, 1.0, 2.0, 3.0], "gid": [0, 0, 1, 1], "t": [1, 2, 3, 4]})
    assert ValidationStructure(group_on="gid").num_group_instances(X) == 2
    assert ValidationStructure(group_time_on="gid").num_group_instances(X) == 2
    assert ValidationStructure(time_on="t").num_group_instances(X) is None
    assert ValidationStructure(stratify_on="gid").num_group_instances(X) is None


def test__predictor_fit__validation_curves_reach_the_fit():
    """`validation_size_curves` is a fit kwarg: a curve must change the bagging actually performed."""
    from autogluon.tabular import TabularPredictor

    rng = np.random.default_rng(0)
    n = 900
    df = pd.DataFrame({"f1": rng.normal(size=n), "gid": np.repeat(np.arange(60), 15), "label": rng.integers(0, 2, n)})

    def children(**fit_kwargs) -> int:
        predictor = TabularPredictor(label="label", verbosity=0).fit(
            df,
            hyperparameters={"DUMMY": {}},
            auto_stack=True,
            fit_weighted_ensemble=False,
            **fit_kwargs,
        )
        model_info = predictor.info()["model_info"]
        bagged = next(name for name in model_info if "BAG_L1" in name)
        return len(model_info[bagged]["children_info"])

    repeated = {"num_bag_folds": [[1_000, 5], 8], "num_bag_sets": [[1_000, 5], 5]}
    assert children() == 8  # default: 8 folds, 1 repeat
    assert children(validation_size_curves=repeated) == 25  # 5 folds x 5 repeats
    # an explicitly passed value still wins over the curve
    assert children(validation_size_curves={"num_bag_sets": [[1_000, 5], 1]}, num_bag_sets=1) == 8
    # curves are read at the group count when the structure asks for it (60 groups < 100 anchor)
    assert (
        children(
            validation_size_curves={"num_bag_folds": [[100, 5], 8], "num_bag_sets": [[100, 5], 5]},
            validation_structure={"group_on": "gid", "size_validation_on_groups": True},
        )
        == 25
    )


def test__custom_splits__temporal_forward_only__never_trains_on_the_future():
    """Forward-chaining: every fold's training rows precede its validation window."""
    X, y = _toy_temporal(n=200)
    vs = ValidationStructure(time_on="ts", temporal_forward_only=True)
    splits, num_folds, num_repeats = vs.custom_splits(X, y, num_folds=4, num_repeats=1)

    assert (num_folds, num_repeats) == (4, 1)
    assert len(splits) == 4
    for train_idx, val_idx in splits:
        assert X["ts"].iloc[train_idx].max() < X["ts"].iloc[val_idx].min()

    # Expanding window: each fold trains on everything validated so far, so train sets grow and
    # validation windows advance in time.
    train_sizes = [len(train_idx) for train_idx, _ in splits]
    assert train_sizes == sorted(train_sizes)
    assert train_sizes[0] < train_sizes[-1]
    val_starts = [X["ts"].iloc[val_idx].min() for _, val_idx in splits]
    assert val_starts == sorted(val_starts)


def test__custom_splits__temporal_forward_only__earliest_block_is_never_validated():
    """The cost of forward-chaining: the first block has no out-of-fold prediction."""
    X, y = _toy_temporal(n=200)
    vs = ValidationStructure(time_on="ts", temporal_forward_only=True)
    splits, _, _ = vs.custom_splits(X, y, num_folds=4, num_repeats=1)

    validated = np.concatenate([val_idx for _, val_idx in splits])
    # Each validated row appears exactly once, but not every row is validated.
    assert len(validated) == len(np.unique(validated))
    assert len(validated) < len(X)

    uncovered = vs.uncovered_rows(X, y, num_folds=4, num_repeats=1)
    assert len(uncovered) == len(X) - len(validated)
    # The gap is the earliest block: it precedes every validation window.
    assert X["ts"].iloc[uncovered].max() <= min(X["ts"].iloc[val_idx].min() for _, val_idx in splits)


def test__custom_splits__leave_one_block_out_covers_every_row():
    """The default temporal mode is the trade-off's other side: full coverage, some lookahead."""
    X, y = _toy_temporal(n=200)
    vs = ValidationStructure(time_on="ts")
    splits, _, _ = vs.custom_splits(X, y, num_folds=4, num_repeats=1)

    validated = np.concatenate([val_idx for _, val_idx in splits])
    assert sorted(validated) == list(range(len(X)))
    assert len(vs.uncovered_rows(X, y, num_folds=4, num_repeats=1)) == 0
    # At least one fold trains on rows later than its validation window (the lookahead).
    assert any(X["ts"].iloc[tr].max() > X["ts"].iloc[va].max() for tr, va in splits)


def test__temporal_forward_only__requires_a_time_column():
    with pytest.raises(ValueError, match="requires `time_on` or `group_time_on`"):
        ValidationStructure(group_on="gid", temporal_forward_only=True)


def test__temporal_forward_only__group_time_on_blocks_whole_groups():
    """`group_time_on` forward-chains over whole groups, staying group-disjoint."""
    n_groups, rows_per_group = 12, 5
    gids = np.repeat(np.arange(n_groups), rows_per_group)
    X = pd.DataFrame({"num": np.arange(len(gids), dtype=float), "sid": gids})
    y = pd.Series(np.tile([0, 1, 0, 1, 1], n_groups))

    vs = ValidationStructure(group_time_on="sid", temporal_forward_only=True)
    splits, _, _ = vs.custom_splits(X, y, num_folds=3, num_repeats=1)

    for train_idx, val_idx in splits:
        train_groups = set(X["sid"].iloc[train_idx])
        val_groups = set(X["sid"].iloc[val_idx])
        assert not (train_groups & val_groups)
        # Forward in group order too.
        assert max(train_groups) < min(val_groups)


def test__temporal_forward_only__too_few_blocks_raises():
    X = pd.DataFrame({"num": [0.0, 1.0, 2.0, 3.0], "ts": pd.to_datetime(["2026-01-01"] * 2 + ["2026-01-02"] * 2)})
    y = pd.Series([0, 1, 0, 1])
    vs = ValidationStructure(time_on="ts", temporal_forward_only=True)
    with pytest.raises(ValueError, match="at least 3 time blocks"):
        vs.custom_splits(X, y, num_folds=4, num_repeats=1)


def test__temporal_forward_only__weighted_ensemble_is_scored_on_covered_rows():
    """The weighted ensemble must be scored on the same rows as the models it is ranked against.

    Forward-chaining leaves the earliest time block unvalidated. Base models exclude those rows
    from their validation score (`score_with_oof` masks on fold coverage), so the ensemble must
    too -- otherwise its score covers a different row set and best-model selection is comparing
    unlike numbers. With a single base model at weight 1.0 the two scores must coincide exactly.
    """
    from autogluon.tabular import TabularPredictor

    rng = np.random.default_rng(0)
    n = 300
    data = pd.DataFrame(
        {
            "num": rng.normal(size=n),
            "ts": pd.to_datetime("2026-01-01") + pd.to_timedelta(np.sort(rng.integers(0, 60, size=n)), unit="D"),
        }
    )
    data["y"] = 3.0 * data["num"] + rng.normal(scale=0.5, size=n)

    predictor = TabularPredictor(label="y", problem_type="regression", verbosity=0).fit(
        data,
        hyperparameters={"GBM": [{"num_boost_round": 20}]},
        num_bag_folds=4,
        num_bag_sets=1,
        num_stack_levels=0,
        dynamic_stacking=False,
        fit_weighted_ensemble=True,
        validation_structure={"time_on": "ts", "temporal_forward_only": True},
    )

    leaderboard = predictor.leaderboard(silent=True).set_index("model")["score_val"]
    base = [m for m in leaderboard.index if "WeightedEnsemble" not in m]
    ensemble = [m for m in leaderboard.index if "WeightedEnsemble" in m]
    assert len(base) == 1 and len(ensemble) == 1
    assert leaderboard[ensemble[0]] == pytest.approx(leaderboard[base[0]])

    # The unvalidated rows are reported as missing rather than as a prediction of 0.
    oof = np.asarray(predictor._trainer.get_model_oof(base[0]), dtype=float)
    assert np.isnan(oof).any()
    assert not np.isnan(oof).all()


def test__temporal_forward_only__stacking_is_refused():
    """Stacking is not supported alongside forward-chaining, and says so."""
    from autogluon.tabular import TabularPredictor

    rng = np.random.default_rng(0)
    n = 200
    data = pd.DataFrame(
        {
            "num": rng.normal(size=n),
            "ts": pd.to_datetime("2026-01-01") + pd.to_timedelta(np.sort(rng.integers(0, 40, size=n)), unit="D"),
        }
    )
    data["y"] = rng.normal(size=n)

    with pytest.raises(ValueError, match="cannot be combined with num_stack_levels"):
        TabularPredictor(label="y", problem_type="regression", verbosity=0).fit(
            data,
            hyperparameters={"GBM": [{"num_boost_round": 5}]},
            num_bag_folds=3,
            num_bag_sets=1,
            num_stack_levels=1,
            dynamic_stacking=False,
            fit_weighted_ensemble=False,
            validation_structure={"time_on": "ts", "temporal_forward_only": True},
        )


def test__holdout__group_column_dropped_by_feature_generation():
    """The non-bagged holdout must be resolved before feature generation can drop its columns.

    A feature generator may legitimately consume a structure column and not emit it — TabArena's
    `tabarena_default` pipeline builds groupby-aggregate features from the group column and then
    drops it. The bagged path already resolves its splits on the raw frame; the holdout path used
    to leave the split to the trainer, which runs after feature generation, so the column was gone
    by then and the fit died with `KeyError: group_on column ... not found`.
    """
    from autogluon.features.generators import IdentityFeatureGenerator, PipelineFeatureGenerator
    from autogluon.tabular import TabularPredictor

    rng = np.random.default_rng(0)
    n_rows, n_groups = 90, 18
    data = pd.DataFrame(
        {
            "f1": rng.normal(size=n_rows),
            "f2": rng.normal(size=n_rows),
            "gid": np.repeat(np.arange(n_groups), n_rows // n_groups),
        }
    )
    data["y"] = (data.f1 + rng.normal(scale=0.3, size=n_rows) > 0).astype(int)

    # Emits only f1/f2, so `gid` does not survive into the feature set.
    feature_generator = PipelineFeatureGenerator(generators=[[IdentityFeatureGenerator(features_in=["f1", "f2"])]])
    predictor = TabularPredictor(label="y", verbosity=0).fit(
        data,
        hyperparameters={"GBM": [{"num_boost_round": 5}]},
        num_bag_folds=0,  # the holdout path
        fit_weighted_ensemble=False,
        feature_generator=feature_generator,
        validation_structure={"group_on": "gid"},
    )

    assert predictor.model_names(), "no model was trained"
    assert "gid" not in predictor.features()

    # The holdout it validated on is group-disjoint, which is the point of declaring the structure.
    structure = ValidationStructure(group_on="gid")
    train_idx, val_idx = structure.holdout_split_indices(
        data.drop(columns="y"), data["y"], holdout_frac=0.1, random_state=0, problem_type="binary"
    )
    groups = data["gid"]
    assert not (set(groups.iloc[train_idx]) & set(groups.iloc[val_idx]))


def test__holdout_coverage__weighted_ensemble_oof_stays_row_aligned():
    """A partially-covered validation scheme must not leave the ensemble's OOF a different length.

    Forward-chaining never validates the earliest time block, so the weighted ensemble cannot
    optimize weights on those rows and is fit without them. Its out-of-fold predictions must still
    be addressable by training-frame row, or anything pairing them with `y` breaks -- temperature
    scaling did exactly that, raising `Expected input batch_size (725) to match target batch_size
    (900)`. The uncovered rows are marked NaN, the same representation a bagged model uses for rows
    no fold validated.
    """
    from autogluon.tabular import TabularPredictor

    rng = np.random.default_rng(0)
    n_rows = 900
    data = pd.DataFrame(
        {
            "f1": rng.normal(size=n_rows),
            "f2": rng.normal(size=n_rows),
            "ts": pd.to_datetime("2026-01-01") + pd.to_timedelta(np.sort(rng.integers(0, 90, size=n_rows)), unit="D"),
        }
    )
    # multiclass + log_loss is what makes `calibrate=True` actually calibrate.
    data["y"] = rng.integers(0, 3, size=n_rows)

    predictor = TabularPredictor(label="y", problem_type="multiclass", eval_metric="log_loss", verbosity=0).fit(
        data,
        hyperparameters={"GBM": [{"num_boost_round": 10}]},
        num_bag_folds=4,
        num_bag_sets=1,
        num_stack_levels=0,
        dynamic_stacking=False,
        fit_weighted_ensemble=True,
        calibrate=True,
        validation_structure={"time_on": "ts", "temporal_forward_only": True},
    )

    ensemble = next(m for m in predictor.model_names() if "WeightedEnsemble" in m)
    n_train_rows = len(predictor._trainer.load_y())
    oof = np.asarray(predictor._trainer.get_model_oof(ensemble), dtype=float)

    # Row-aligned to the training frame, with the unvalidated rows marked rather than missing.
    assert oof.shape[0] == n_train_rows
    uncovered = np.isnan(oof).any(axis=1)
    assert uncovered.any(), "expected the earliest time block to be unvalidated"
    assert not uncovered.all()
    # The ensemble is still scored, on its covered rows.
    assert predictor.leaderboard(silent=True).set_index("model").loc[ensemble, "score_val"] is not None


def test__custom_splits__group_table__fewer_groups_per_class_than_folds():
    """Folds are built even when every class holds fewer *groups* than there are folds.

    The group table carries one row per group, so a class with plenty of rows can still have
    only a handful of groups. sklearn's ``StratifiedKFold`` refuses to split when every class
    is below the fold count, which is not a real constraint: two groups of a class landing in
    two different folds already leave every training split with a member of it. Routing
    through ``CVSplitter`` keeps such tasks fittable instead of aborting the fit.
    """
    n_classes, groups_per_class, rows_per_group, num_folds = 8, 6, 15, 8
    n_groups = n_classes * groups_per_class
    group_class = np.tile(np.arange(n_classes), groups_per_class)
    rng = np.random.default_rng(0)
    gid = np.repeat(np.arange(n_groups), rows_per_group)
    X = pd.DataFrame(
        {
            "f1": rng.normal(size=len(gid)),
            "gid": [f"g{g}" for g in gid],
            "cls": [f"c{group_class[g]}" for g in gid],
        }
    )
    y = pd.Series(X["cls"].to_numpy())

    # The rows are plentiful; only the groups behind them are scarce.
    assert X["cls"].value_counts().min() == groups_per_class * rows_per_group
    group_table = X.drop_duplicates("gid")
    assert group_table["cls"].value_counts().max() < num_folds

    vs = ValidationStructure(group_on="gid", stratify_on="cls")
    splits, folds, repeats = vs.custom_splits(X, y, num_folds=num_folds, num_repeats=1, problem_type="multiclass")
    assert (folds, repeats) == (num_folds, 1)
    assert len(splits) == num_folds

    groups, classes = X["gid"].to_numpy(), X["cls"].to_numpy()
    all_classes = set(pd.unique(classes))
    validated = np.concatenate([val_idx for _, val_idx in splits])
    assert sorted(validated) == list(range(len(X)))  # every row validated exactly once
    for train_idx, val_idx in splits:
        assert not (set(groups[train_idx]) & set(groups[val_idx]))  # group-disjoint
        # every class trainable in every fold, which is all the stratification has to deliver
        assert all_classes == set(pd.unique(classes[train_idx]))


def test__custom_splits__stratify_on__fold_count_not_capped_by_the_rarest_value():
    """A value occurring twice supports any number of folds, so the fold count stands.

    Capping ``n_splits`` at the rarest value's count silently shrinks the requested bagging:
    a task asking for 8 folds got 2 because one value happened to occur twice. Two members in
    two different folds is all k-fold needs -- every training split still retains one.
    """
    rng = np.random.default_rng(0)
    n_folds = 8
    # one value occurs twice, far below the fold count; the rest are plentiful
    cls = np.array(["rare"] * 2 + ["a"] * 40 + ["b"] * 40, dtype=object)
    rng.shuffle(cls)
    X = pd.DataFrame({"f1": rng.normal(size=len(cls)), "cls": cls})
    y = pd.Series(rng.integers(0, 2, len(cls)))

    vs = ValidationStructure(stratify_on="cls")
    splits, folds, repeats = vs.custom_splits(X, y, num_folds=n_folds, num_repeats=1)
    assert (folds, repeats) == (n_folds, 1)
    assert len(splits) == n_folds

    values = X["cls"].to_numpy()
    all_values = set(pd.unique(values))
    validated = np.concatenate([val_idx for _, val_idx in splits])
    assert sorted(validated) == list(range(len(X)))  # every row validated exactly once
    for train_idx, _ in splits:
        # the rare value is trainable in every fold, which is what two members buy
        assert all_values == set(pd.unique(values[train_idx]))

    # A single member is the case two members cannot rescue: the fold validating it trains on
    # none of it. AutoGluon duplicates such rows upstream (`augment_rare_classes`).
    X_single = X.copy()
    X_single.loc[X_single.index[X_single["cls"].to_numpy() == "rare"][0], "cls"] = "a"
    with pytest.raises(ValueError, match="fewer than 2 times"):
        ValidationStructure(stratify_on="cls").custom_splits(X_single, y, num_folds=n_folds, num_repeats=1)


def _rare_value_grouped(rare_groups: int, n_groups: int = 20, rows_per_group: int = 5, seed: int = 0):
    """Grouped frame whose 'rare' value occupies exactly ``rare_groups`` group(s)."""
    gid = np.repeat(np.arange(n_groups), rows_per_group)
    cls = np.where(gid % 2 == 0, "a", "b").astype(object)
    for group in range(rare_groups):
        cls[np.flatnonzero(gid == group)[:2]] = "rare"
    X = pd.DataFrame(
        {
            "f1": np.random.default_rng(seed).normal(size=len(gid)),
            "gid": [f"g{g}" for g in gid],
            "cls": cls,
        }
    )
    return X, pd.Series(cls)


def test__rare_stratification_values__counts_groups_not_rows():
    """Two rows inside one group are one independent unit, not two.

    The guarantee "at least two members" is only worth anything if the two land in different
    folds; two rows of one group never do. Row counts cannot express that, so rarity is
    measured in groups wherever grouping is declared.
    """
    vs = ValidationStructure(group_on="gid", stratify_on="cls")

    X, y = _rare_value_grouped(rare_groups=1)
    assert X["cls"].value_counts()["rare"] == 2  # ample by the row count `augment_rare_classes` uses
    assert vs.rare_stratification_values(X, y, min_units=2, problem_type="multiclass") == {"rare": 1}

    # spread over two groups, the same two rows now satisfy it
    X, y = _rare_value_grouped(rare_groups=2)
    assert vs.rare_stratification_values(X, y, min_units=2, problem_type="multiclass") == {}

    # ungrouped structures fall back to rows, where a count of 2 is the whole requirement
    ungrouped = ValidationStructure(stratify_on="cls")
    assert ungrouped.rare_stratification_values(X, y, min_units=2, problem_type="multiclass") == {}


def test__custom_splits__group_on__scarce_value_does_not_shrink_folds_or_repeats():
    """A value scarcer than the fold count is a scorability problem, not a partition problem."""
    X, y = _rare_value_grouped(rare_groups=1)
    splits, folds, repeats = ValidationStructure(group_on="gid", stratify_on="cls").custom_splits(
        X, y, num_folds=5, num_repeats=3, problem_type="multiclass"
    )
    assert (folds, repeats) == (5, 3)  # previously collapsed to (2, 1) off the row count
    assert len(splits) == 15
    groups = X["gid"].to_numpy()
    validated = np.concatenate([val_idx for _, val_idx in splits])
    assert sorted(validated) == sorted(list(range(len(X))) * 3)  # every row validated once per repeat
    for train_idx, val_idx in splits:
        assert not (set(groups[train_idx]) & set(groups[val_idx]))


def test__holdout__group_on__moves_whole_groups_to_keep_every_class_trainable():
    """The structured holdout enforces the per-class floor the unstructured split always did."""
    X, y = _rare_value_grouped(rare_groups=1)
    vs = ValidationStructure(group_on="gid", stratify_on="cls")
    values, groups = X["cls"].to_numpy(), X["gid"].to_numpy()

    train_idx, val_idx = vs.holdout_split_indices(
        X, y, holdout_frac=0.25, problem_type="multiclass", min_cls_count_train=2
    )
    assert (values[train_idx] == "rare").sum() >= 2
    # repaired by moving the whole group, so the split is still group-disjoint
    assert not (set(groups[train_idx]) & set(groups[val_idx]))
    assert len(val_idx) > 0


def test__holdout__group_on__keeps_the_split_when_repair_would_empty_the_holdout():
    """Coarse grouping can make the repair cost everything; an imperfect holdout beats none."""
    X, y = _rare_value_grouped(rare_groups=1, n_groups=4, rows_per_group=5)
    train_idx, val_idx = ValidationStructure(group_on="gid", stratify_on="cls").holdout_split_indices(
        X, y, holdout_frac=0.25, problem_type="multiclass", min_cls_count_train=2
    )
    assert len(val_idx) > 0 and len(train_idx) > 0
    groups = X["gid"].to_numpy()
    assert not (set(groups[train_idx]) & set(groups[val_idx]))


def test__custom_splits__binary__fold_count_respects_scorability():
    """Binary folds must hold both classes: AutoGluon scores every child on its own fold.

    ``roc_auc`` is undefined on a single-class fold, so a fold missing a class fails the fit
    rather than degrading it — unlike multiclass, where ``log_loss`` is still defined. The fold
    count therefore cannot exceed the number of groups carrying the rarer class.
    """
    n_groups, rows_per_group, minority_groups = 20, 5, 3
    gid = np.repeat(np.arange(n_groups), rows_per_group)
    group_class = np.array(["1"] * n_groups, dtype=object)
    group_class[:minority_groups] = "0"
    cls = group_class[gid]
    X = pd.DataFrame({"f1": np.random.default_rng(0).normal(size=len(gid)), "gid": [f"g{g}" for g in gid], "cls": cls})
    y = pd.Series(cls)

    vs = ValidationStructure(group_on="gid", stratify_on="cls")
    assert vs.max_scorable_folds(X, y, problem_type="binary") == minority_groups
    splits, folds, _ = vs.custom_splits(X, y, num_folds=8, num_repeats=1, problem_type="binary")
    assert folds == minority_groups
    values = np.asarray(y)
    for _, val_idx in splits:
        assert len(set(pd.unique(values[val_idx]))) == 2  # scorable

    # Multiclass is unbounded: the requested folds stand even with a class in 2 groups.
    group_class = np.array([str(i % 8) for i in range(n_groups)], dtype=object)
    group_class[:2] = "rare"
    X_multi = X.assign(cls=group_class[gid])
    y_multi = pd.Series(X_multi["cls"].to_numpy())
    assert (
        ValidationStructure(group_on="gid", stratify_on="cls").max_scorable_folds(
            X_multi, y_multi, problem_type="multiclass"
        )
        is None
    )
    _, folds_multi, _ = ValidationStructure(group_on="gid", stratify_on="cls").custom_splits(
        X_multi, y_multi, num_folds=8, num_repeats=1, problem_type="multiclass"
    )
    assert folds_multi == 8


def test__fold_ids__binary__ungrouped_fold_count_respects_scorability():
    """Same ceiling without grouping, where the independent unit is the row."""
    cls = np.array(["1"] * 100, dtype=object)
    cls[:4] = "0"
    X = pd.DataFrame({"f1": np.random.default_rng(0).normal(size=100), "cls": cls})
    y = pd.Series(cls)
    vs = ValidationStructure(stratify_on="cls")
    assert vs.max_scorable_folds(X, y, problem_type="binary") == 4
    fold_ids = vs.fold_ids(X, y, n_splits=8, problem_type="binary")
    assert fold_ids.nunique() == 4
    values = np.asarray(y)
    for fold in fold_ids.unique():
        assert len(set(pd.unique(values[fold_ids.to_numpy() == fold]))) == 2
