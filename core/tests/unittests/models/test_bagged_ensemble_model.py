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
