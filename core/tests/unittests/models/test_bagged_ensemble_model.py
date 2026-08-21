import numpy as np
import pandas as pd

from autogluon.common.utils.cv_splitter import CVSplitter
from autogluon.core.models import BaggedEnsembleModel
from autogluon.core.models.dummy.dummy_model import DummyModel


def test_generate_fold_configs():
    y = pd.Series([0, 0, 0, 1, 1, 1, 1, 1])
    X = pd.DataFrame([[0], [0], [0], [0], [0], [0], [0], [0]])

    k_fold_start = 2
    k_fold_end = 1
    k_fold = 3
    n_repeat_start = 2
    n_repeats = 5

    cv_splitter = CVSplitter(n_splits=k_fold, n_repeats=n_repeats, stratify=True, random_state=0)

    fold_fit_args_list, n_repeats_started, n_repeats_finished = BaggedEnsembleModel._generate_fold_configs(
        X=X,
        y=y,
        cv_splitter=cv_splitter,
        k_fold_start=k_fold_start,
        k_fold_end=k_fold_end,
        n_repeat_start=n_repeat_start,
        n_repeat_end=n_repeats,
        vary_seed_across_folds=True,
        random_seed_offset=0,
    )

    assert fold_fit_args_list[0]["model_name_suffix"] == "S3F3"
    assert fold_fit_args_list[-1]["model_name_suffix"] == "S5F1"
    assert len(fold_fit_args_list) == 5
    assert n_repeats_started == 2
    assert n_repeats_finished == 2

    assert fold_fit_args_list[0]["is_last_fold"] is False
    assert fold_fit_args_list[1]["is_last_fold"] is False
    assert fold_fit_args_list[2]["is_last_fold"] is False
    assert fold_fit_args_list[3]["is_last_fold"] is False
    assert fold_fit_args_list[4]["is_last_fold"] is True

    assert fold_fit_args_list[0]["random_seed"] == 8
    assert fold_fit_args_list[1]["random_seed"] == 9
    assert fold_fit_args_list[2]["random_seed"] == 10
    assert fold_fit_args_list[3]["random_seed"] == 11
    assert fold_fit_args_list[4]["random_seed"] == 12

    k_fold_start = 0
    k_fold_end = 3
    k_fold = 3
    n_repeat_start = 0
    n_repeats = 5
    cv_splitter = CVSplitter(n_splits=k_fold, n_repeats=n_repeats, stratify=True, random_state=0)

    fold_fit_args_list, n_repeats_started, n_repeats_finished = BaggedEnsembleModel._generate_fold_configs(
        X=X,
        y=y,
        cv_splitter=cv_splitter,
        k_fold_start=k_fold_start,
        k_fold_end=k_fold_end,
        n_repeat_start=n_repeat_start,
        n_repeat_end=n_repeats,
        vary_seed_across_folds=False,
        random_seed_offset=0,
    )

    assert fold_fit_args_list[0]["random_seed"] == 0
    assert fold_fit_args_list[1]["random_seed"] == 0
    assert fold_fit_args_list[2]["random_seed"] == 0
    assert fold_fit_args_list[3]["random_seed"] == 0
    assert fold_fit_args_list[4]["random_seed"] == 0

    fold_fit_args_list, n_repeats_started, n_repeats_finished = BaggedEnsembleModel._generate_fold_configs(
        X=X,
        y=y,
        cv_splitter=cv_splitter,
        k_fold_start=k_fold_start,
        k_fold_end=k_fold_end,
        n_repeat_start=n_repeat_start,
        n_repeat_end=n_repeats,
        vary_seed_across_folds=True,
        random_seed_offset=42,
    )

    assert fold_fit_args_list[0]["random_seed"] == 42
    assert fold_fit_args_list[1]["random_seed"] == 43
    assert fold_fit_args_list[2]["random_seed"] == 44
    assert fold_fit_args_list[3]["random_seed"] == 45
    assert fold_fit_args_list[4]["random_seed"] == 46


def test_generate_fold_configs_with_offset_index():
    """Integration: BaggedEnsembleModel._generate_fold_configs uses positional indices
    from custom_splits regardless of the DataFrame's own index labels.
    """
    n = 8
    index = pd.RangeIndex(start=1000, stop=1000 + n)
    X = pd.DataFrame({"f": range(n)}, index=index)
    y = pd.Series([i % 2 for i in range(n)], index=index, name="label")

    mid = n // 2
    splits = [
        (np.arange(mid, n), np.arange(0, mid)),
        (np.arange(0, mid), np.arange(mid, n)),
    ]
    cv = CVSplitter(n_splits=2, n_repeats=1, custom_splits=splits)

    fold_fit_args_list, _, _ = BaggedEnsembleModel._generate_fold_configs(
        X=X,
        y=y,
        cv_splitter=cv,
        k_fold_start=0,
        k_fold_end=2,
        n_repeat_start=0,
        n_repeat_end=1,
        vary_seed_across_folds=False,
        random_seed_offset=0,
    )

    assert len(fold_fit_args_list) == 2
    _, test_idx_0 = fold_fit_args_list[0]["fold"]
    _, test_idx_1 = fold_fit_args_list[1]["fold"]
    # Positional indices: fold 0 tests rows 0..3, fold 1 tests rows 4..7
    np.testing.assert_array_equal(test_idx_0, np.arange(0, n // 2))
    np.testing.assert_array_equal(test_idx_1, np.arange(n // 2, n))
    # iloc access with those positional indices gives the correct labels
    assert X.iloc[test_idx_0].index[0] == 1000
    assert X.iloc[test_idx_1].index[0] == 1000 + n // 2


class OofCapableDummyModel(DummyModel):
    """A DummyModel that reports its own out-of-fold predictions, as forests / KNN do.

    Module scope rather than nested in a test, so a fitted bag can be pickled.
    """

    def predict_proba_oof(self, X, y=None, **kwargs):
        return self.predict_proba(X=X)

    def _more_tags(self):
        return {"valid_oof": True}


def test_use_child_oof_disabled_by_custom_splits():
    """`use_child_oof` must yield to explicit splits, which exist to prevent leakage.

    A child's internal out-of-bag estimate resamples rows independently, so on grouped or
    temporal data it scores across the very boundary the splits enforce. `custom_splits` is the
    channel grouped / temporal validation arrives through (directly via `ag_args_ensemble`, or
    resolved from a `validation_structure`), so it must force real cross-validation -- as
    `groups` already did.
    """
    n_rows = 12
    X = pd.DataFrame({"a": range(n_rows)})
    y = pd.Series([0, 1] * (n_rows // 2))
    splits = [
        (np.arange(4, n_rows), np.arange(0, 4)),
        (np.setdiff1d(np.arange(n_rows), np.arange(4, 8)), np.arange(4, 8)),
        (np.arange(0, 8), np.arange(8, n_rows)),
    ]

    bagged = BaggedEnsembleModel(
        model_base=DummyModel(),
        hyperparameters={"use_child_oof": True, "custom_splits": splits, "fold_fitting_strategy": "sequential_local"},
    )
    bagged.fit(X=X, y=y, k_fold=len(splits))

    # One child per supplied split (not the single child `use_child_oof` would have trained),
    # and every row has an out-of-fold prediction from the fold that held it out.
    assert bagged.n_children == len(splits)
    assert len(bagged._oof_pred_proba) == n_rows

    # The realized folds are the ones that were passed in.
    realized = [tuple(val_idx) for _, val_idx in bagged._cv_splitters[0].split(X=X, y=y)]
    assert realized == [tuple(val_idx) for _, val_idx in splits]


def test_use_child_oof_kept_without_custom_splits():
    """Without explicit splits there is no boundary to violate, so `use_child_oof` still applies.

    The counterpart of the test above: the guard must be specific to explicit splits rather than
    disabling `use_child_oof` generally.
    """
    X = pd.DataFrame({"a": range(12)})
    y = pd.Series([0, 1] * 6)

    bagged = BaggedEnsembleModel(
        model_base=OofCapableDummyModel(),
        hyperparameters={"use_child_oof": True, "fold_fitting_strategy": "sequential_local"},
    )
    bagged.fit(X=X, y=y, k_fold=3)

    # A single child, whose own estimate stands in for cross-validation.
    assert bagged.n_children == 1


def test_oof_is_nan_for_rows_no_fold_validated():
    """Rows outside every validation fold must read as missing, not as a prediction of 0.

    The out-of-fold accumulator starts at 0, so a row no fold validated would otherwise emerge as
    a confident 0 — silently wrong for anything consuming out-of-fold predictions as features or
    as a score. Partial coverage is reachable via explicit splits that skip rows, which is what
    forward-chaining temporal validation does with its earliest time block.
    """
    n_rows = 12
    X = pd.DataFrame({"a": range(n_rows)})
    y = pd.Series([0, 1] * (n_rows // 2))
    # Expanding-window splits: rows 0-3 are training data only and never validated.
    splits = [
        (np.arange(0, 4), np.arange(4, 8)),
        (np.arange(0, 8), np.arange(8, n_rows)),
    ]

    bagged = BaggedEnsembleModel(
        model_base=DummyModel(),
        hyperparameters={"custom_splits": splits, "fold_fitting_strategy": "sequential_local"},
    )
    bagged.fit(X=X, y=y, k_fold=len(splits))

    oof = bagged.predict_proba_oof()
    uncovered = np.arange(0, 4)
    covered = np.arange(4, n_rows)
    assert np.isnan(np.asarray(oof, dtype=float)[uncovered]).all()
    assert not np.isnan(np.asarray(oof, dtype=float)[covered]).any()

    # Scoring already excludes the uncovered rows, so a score is still produced.
    assert bagged.score_with_oof(y=y) is not None


def test_oof_has_no_nan_when_every_row_is_validated():
    """The ordinary case is untouched: full coverage means no NaN."""
    X = pd.DataFrame({"a": range(12)})
    y = pd.Series([0, 1] * 6)

    bagged = BaggedEnsembleModel(
        model_base=DummyModel(),
        hyperparameters={"fold_fitting_strategy": "sequential_local"},
    )
    bagged.fit(X=X, y=y, k_fold=3)

    assert not np.isnan(np.asarray(bagged.predict_proba_oof(), dtype=float)).any()
