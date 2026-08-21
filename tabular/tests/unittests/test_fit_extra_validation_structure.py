"""`fit_extra` must validate new models on the same splits as the original `fit`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autogluon.common.utils import cv_splitter as cvs
from autogluon.tabular import TabularPredictor


def _grouped_frame(n_groups: int = 3, rows_per_group: int = 8, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_groups * rows_per_group
    df = pd.DataFrame(
        {
            "f1": rng.random(n),
            "f2": rng.random(n),
            "grp": np.repeat([f"g{i}" for i in range(n_groups)], rows_per_group),
        }
    )
    df["label"] = (df["f1"] + df["f2"] > 1.0).astype(int)
    return df


def test_fit_extra_reuses_the_validation_structure_from_fit(monkeypatch):
    """Models added by `fit_extra` must not be validated on leaky splits.

    A `validation_structure` is resolved once, in the learner, and reaches bagging as
    `ag_args_ensemble["custom_splits"]`. `fit_extra` builds its own `core_kwargs`, so it used to
    fall back to plain k-fold: the same predictor then held group-disjoint models from `fit` and
    group-leaking models from `fit_extra`, and compared their `score_val`s against each other.
    """
    df = _grouped_frame()
    groups = df["grp"].to_numpy()
    seen: list[tuple[str, bool, bool]] = []
    phase = ["fit"]
    original_split = cvs.CVSplitter.split

    def spy(self, X, y):
        splits = original_split(self, X, y)
        disjoint = all(
            set(groups[np.asarray(train_idx)]).isdisjoint(set(groups[np.asarray(val_idx)]))
            for train_idx, val_idx in splits
        )
        seen.append((phase[0], self.custom_splits is not None, disjoint))
        return splits

    monkeypatch.setattr(cvs.CVSplitter, "split", spy)

    predictor = TabularPredictor(label="label", verbosity=0).fit(
        df,
        hyperparameters={"DUMMY": {}},
        validation_structure={"group_on": "grp"},
        num_bag_folds=3,
        num_bag_sets=1,
        dynamic_stacking=False,
        fit_weighted_ensemble=False,
        num_gpus=0,
    )
    phase[0] = "fit_extra"
    predictor.fit_extra(
        hyperparameters={"DUMMY": {}}, num_bag_sets=1, fit_weighted_ensemble=False, name_suffix="_extra"
    )

    assert [p for p, _, _ in seen].count("fit_extra") >= 1, "no bagged model was fit by fit_extra"
    for phase_name, has_custom, disjoint in seen:
        assert has_custom, f"{phase_name}: structure-resolved splits were not used"
        assert disjoint, f"{phase_name}: a fold put the same group on both sides"


def test_fit_extra_rejects_an_explicit_validation_structure():
    """It was accepted and silently ignored, which is worse than refusing it.

    It cannot be honored: a structure is resolved off the raw frame before feature generation
    transforms or drops the columns it names, and `fit_extra` starts from transformed data.
    """
    df = _grouped_frame()
    predictor = TabularPredictor(label="label", verbosity=0).fit(
        df,
        hyperparameters={"DUMMY": {}},
        validation_structure={"group_on": "grp"},
        num_bag_folds=3,
        num_bag_sets=1,
        dynamic_stacking=False,
        fit_weighted_ensemble=False,
        num_gpus=0,
    )
    with pytest.raises(ValueError, match="cannot be specified in `fit_extra`"):
        predictor.fit_extra(
            hyperparameters={"DUMMY": {}},
            fit_weighted_ensemble=False,
            validation_structure={"group_on": "grp"},
        )


def test_remembered_splits_are_not_reused_on_a_different_row_count():
    """The splits are positional indices, so a changed row set must not reuse them."""
    from autogluon.tabular.trainer.abstract_trainer import AbstractTabularTrainer

    trainer = AbstractTabularTrainer.__new__(AbstractTabularTrainer)
    splits = [(np.array([0, 1]), np.array([2])), (np.array([1, 2]), np.array([0]))]
    trainer._custom_splits = splits
    trainer._custom_splits_num_rows = 3

    same_rows = trainer._resolve_custom_splits(ag_args_ensemble=None, X=pd.DataFrame({"a": [0, 1, 2]}))
    assert same_rows["custom_splits"] is splits

    more_rows = trainer._resolve_custom_splits(ag_args_ensemble=None, X=pd.DataFrame({"a": [0, 1, 2, 3]}))
    assert more_rows is None, "stale positional splits must not be reused"

    # An explicitly supplied set always wins, and replaces what was remembered.
    other = [(np.array([0]), np.array([1]))]
    out = trainer._resolve_custom_splits(ag_args_ensemble={"custom_splits": other}, X=pd.DataFrame({"a": [0, 1]}))
    assert out["custom_splits"] is other
    assert trainer._custom_splits is other
    assert trainer._custom_splits_num_rows == 2


def test_save_space_drops_the_fit_only_split_state():
    """The splits index the training data, so they are dead weight once that data is removed.

    `clone_for_deployment` goes through `save_space`, and neither the splits nor the group vector
    is read during inference -- roughly `2 * n_rows * num_repeats` int64s of pure overhead in a
    deployment artifact.
    """
    df = _grouped_frame()
    predictor = TabularPredictor(label="label", verbosity=0).fit(
        df,
        hyperparameters={"DUMMY": {}},
        validation_structure={"group_on": "grp"},
        num_bag_folds=3,
        num_bag_sets=1,
        dynamic_stacking=False,
        fit_weighted_ensemble=False,
        num_gpus=0,
    )
    assert predictor._trainer._custom_splits is not None, "the splits should be remembered after fit"

    predictor.save_space()

    trainer = predictor._trainer
    assert trainer._custom_splits is None
    assert trainer._custom_splits_num_rows is None
    # ...and it survives a round trip through disk, which is what a deployment clone loads.
    reloaded = TabularPredictor.load(predictor.path)
    assert reloaded._trainer._custom_splits is None
    assert len(reloaded.predict(df)) == len(df)
