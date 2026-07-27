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

    # bagged path: fold-ids ride the groups channel
    predictor = TabularPredictor(label="label", verbosity=0).fit(
        train_data,
        hyperparameters={"DUMMY": {}},
        num_bag_folds=4,
        validation_structure={"group_on": "gid"},
    )
    assert predictor.model_best is not None
    trainer_groups = predictor._trainer._groups
    assert trainer_groups is not None
    assert trainer_groups.nunique() == 4
    # groups never straddle folds
    joined = pd.DataFrame({"gid": X["gid"].to_numpy(), "fold": trainer_groups.to_numpy()})
    assert (joined.groupby("gid", observed=True)["fold"].nunique() == 1).all()

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
