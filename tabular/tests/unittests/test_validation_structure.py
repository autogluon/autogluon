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
    # blocks advance in time: each fold's groups are later than the previous fold's
    fold_max = [groups[val_idx].max() for _, val_idx in splits]
    assert fold_max == sorted(fold_max)

    train_idx, val_idx = vs.holdout_split_indices(X, y, holdout_frac=0.25)
    assert groups[val_idx].min() > groups[train_idx].max()  # forward holdout


def test__validation_structure__group_and_time_guidance():
    """`group_on` + `time_on` points at `group_time_on` instead of inventing semantics."""
    from autogluon.core.utils.validation_structure import ValidationStructure

    with pytest.raises(NotImplementedError, match="group_time_on"):
        ValidationStructure(group_on="g", time_on="t")
    with pytest.raises(ValueError, match="mutually exclusive"):
        ValidationStructure(group_time_on="g", group_on="g2")
