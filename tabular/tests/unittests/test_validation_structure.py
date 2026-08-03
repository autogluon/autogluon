from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autogluon.core.utils.validation_structure import ValidationStructure


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
    from autogluon.core.utils.validation_structure import ValidationStructure

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
    from autogluon.core.utils.validation_structure import ValidationStructure

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
    from autogluon.core.utils.validation_structure import ValidationStructure

    with pytest.raises(NotImplementedError, match="group_time_on"):
        ValidationStructure(group_on="g", time_on="t")
    with pytest.raises(ValueError, match="mutually exclusive"):
        ValidationStructure(group_time_on="g", group_on="g2")


def test__validation_structure__time_blocks_are_contiguous_and_tie_safe():
    """Time blocks split observed values, keep ties together, and balance row counts."""
    from autogluon.core.utils.validation_structure import ValidationStructure

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
    from autogluon.core.utils.validation_structure import ValidationStructure

    captured: list[dict] = []
    original = ValidationStructure.custom_splits

    def spy(self, X, y, num_folds=None, num_repeats=None, random_state=0):
        splits, folds, repeats = original(
            self, X, y, num_folds=num_folds, num_repeats=num_repeats, random_state=random_state
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
    from autogluon.core.utils.validation_structure import ValidationStructure
    from autogluon.tabular import TabularPredictor

    df = _grouped_toy_data(n=60, n_groups=10)
    groups = df["gid"].to_numpy()

    holdouts: list[tuple[np.ndarray, np.ndarray]] = []
    original_holdout = ValidationStructure.holdout_split_indices

    def holdout_spy(self, X, y, holdout_frac, random_state=0):
        split = original_holdout(self, X, y, holdout_frac=holdout_frac, random_state=random_state)
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
