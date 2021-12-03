
import pandas as pd

from autogluon.core.models import BaggedEnsembleModel
from autogluon.core.utils.utils import CVSplitter


def test_generate_fold_configs():
    y = pd.Series([0, 0, 0, 1, 1, 1, 1, 1])
    X = pd.DataFrame([[0], [0], [0], [0], [0], [0], [0], [0]])

    k_fold_start = 2
    k_fold_end = 1
    k_fold = 3
    n_repeat_start = 2
    n_repeats = 5

    cv_splitter = CVSplitter(n_splits=k_fold, n_repeats=n_repeats, stratified=True, random_state=0)

    fold_fit_args_list, n_repeats_started, n_repeats_finished = BaggedEnsembleModel._generate_fold_configs(
        X=X,
        y=y,
        cv_splitter=cv_splitter,
        k_fold_start=k_fold_start,
        k_fold_end=k_fold_end,
        n_repeat_start=n_repeat_start,
        n_repeat_end=n_repeats,
    )

    assert fold_fit_args_list[0]['model_name_suffix'] == 'S3F3'
    assert fold_fit_args_list[-1]['model_name_suffix'] == 'S5F1'
    assert len(fold_fit_args_list) == 5
    assert n_repeats_started == 2
    assert n_repeats_finished == 2

    assert fold_fit_args_list[0]['is_last_fold'] is False
    assert fold_fit_args_list[1]['is_last_fold'] is False
    assert fold_fit_args_list[2]['is_last_fold'] is False
    assert fold_fit_args_list[3]['is_last_fold'] is False
    assert fold_fit_args_list[4]['is_last_fold'] is True
