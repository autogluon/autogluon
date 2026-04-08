import numpy as np
import pandas as pd

from autogluon.common.utils.cv_splitter import CVSplitter
from autogluon.core.models import BaggedEnsembleModel


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
