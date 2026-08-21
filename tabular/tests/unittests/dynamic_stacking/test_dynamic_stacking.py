import shutil

import numpy as np
import pandas as pd
import pytest

from autogluon.core.constants import BINARY
from autogluon.core.metrics import METRICS
from autogluon.tabular.testing import FitHelper
from autogluon.tabular.testing.fit_helper import stacked_overfitting_assert

DS_ARGS_TEST_DEFAULTS = dict(
    validation_procedure="holdout",
    detection_time_frac=1 / 4,
    holdout_frac=1 / 9,
    n_folds=2,
    n_repeats=1,
    memory_safe_fits=True,
    clean_up_fits=True,
    holdout_data=None,
)


def test_spot_and_avoid_stacked_overfitting():
    """Tests that dynamic stacking works."""
    fit_args = dict(
        hyperparameters={"RF": {}, "GBM": {}},
        fit_weighted_ensemble=False,
        dynamic_stacking=True,
        num_stack_levels=1,
        num_bag_folds=2,
        num_bag_sets=1,
        time_limit=None,
        ds_args=DS_ARGS_TEST_DEFAULTS,
        ag_args_ensemble={"fold_fitting_strategy": "sequential_local"},
    )
    dataset_name = "adult"
    extra_metrics = list(METRICS[BINARY])

    FitHelper.fit_and_validate_dataset(
        dataset_name=dataset_name,
        fit_args=fit_args,
        extra_metrics=extra_metrics,
        expected_model_count=2,
        refit_full=False,
        allowed_dataset_features=["age"],
        expected_stacked_overfitting_at_test=False,
        expected_stacked_overfitting_at_val=True,
    )


def test_dynamic_stacking_hps():
    """Tests dynamic stacking arguments."""
    fit_args = dict(
        hyperparameters={"DUMMY": {}},
        fit_weighted_ensemble=False,
        dynamic_stacking=True,
        num_stack_levels=1,
        num_bag_folds=2,
        num_bag_sets=1,
        time_limit=None,
        ag_args_ensemble={"fold_fitting_strategy": "sequential_local"},
    )

    # Get custom val data (the test data)
    train_data, test_data, dataset_info = FitHelper.load_dataset(name="adult", directory_prefix="./datasets/")
    label = dataset_info["label"]
    allowed_cols = ["age", label]
    train_data = train_data[allowed_cols]
    test_data = test_data[allowed_cols]
    n_test_data = len(test_data)

    for ds_args_update in [
        dict(validation_procedure="holdout", holdout_frac=1 / 5),  # holdout
        dict(validation_procedure="cv"),  # 2-fold CV
        dict(validation_procedure="cv", n_repeats=2),  # 2-repeated 2-fold CV
        dict(memory_safe_fits=False, clean_up_fits=False),  # fit options False
        dict(holdout_data=test_data),
        dict(holdout_data=test_data, validation_procedure="cv", expect_raise=ValueError),
    ]:
        expect_raise = ds_args_update.pop("expect_raise", None)
        tmp_ds_args = DS_ARGS_TEST_DEFAULTS.copy()
        if ds_args_update is not None:
            tmp_ds_args.update(ds_args_update)
        tmp_fit_args = fit_args.copy()
        tmp_fit_args["ds_args"] = tmp_ds_args
        if expect_raise is None:
            predictor = FitHelper.fit_dataset(
                train_data=train_data, init_args=dict(label=label), fit_args=tmp_fit_args, sample_size=1000
            )
            if ("holdout_data" in ds_args_update) and (ds_args_update["holdout_data"] is not None):
                n_expected = 1000 + n_test_data
                assert len(predictor.predict_oof()) == n_expected, "Verify that holdout data was used for training"
            lb = predictor.leaderboard(test_data, extra_info=True)
            stacked_overfitting_assert(lb, predictor, False, False)
            shutil.rmtree(predictor.path)
        else:
            with pytest.raises(expect_raise):
                FitHelper.fit_dataset(
                    train_data=train_data, init_args=dict(label=label), fit_args=tmp_fit_args, sample_size=1000
                )


def test_no_dynamic_stacking():
    """Tests that dynamic stacking does not run if stacking is disabled."""
    fit_args = dict(
        hyperparameters={"DUMMY": {}},
        dynamic_stacking=True,
        fit_weighted_ensemble=False,
        num_stack_levels=0,
        ag_args_ensemble={"fold_fitting_strategy": "sequential_local"},
    )
    dataset_name = "adult"
    extra_metrics = list(METRICS[BINARY])

    predictor = FitHelper.fit_and_validate_dataset(
        dataset_name=dataset_name,
        fit_args=fit_args,
        extra_metrics=extra_metrics,
        expected_model_count=1,
        refit_full=False,
    )
    assert predictor._stacked_overfitting_occurred is None


def test_dynamic_stacking_fit_extra():
    """Tests that fit_extra works after dynamic stacking."""
    fit_args = dict(
        hyperparameters={"RF": {}},
        dynamic_stacking=True,
        fit_weighted_ensemble=False,
        num_bag_folds=2,
        num_bag_sets=1,
        num_stack_levels=1,
        ds_args=DS_ARGS_TEST_DEFAULTS,
        ag_args_ensemble={"fold_fitting_strategy": "sequential_local"},
    )
    dataset_name = "adult"
    extra_metrics = list(METRICS[BINARY])

    predictor = FitHelper.fit_and_validate_dataset(
        dataset_name=dataset_name,
        fit_args=fit_args,
        extra_metrics=extra_metrics,
        expected_model_count=1,
        refit_full=False,
        delete_directory=False,
        allowed_dataset_features=["age"],
        expected_stacked_overfitting_at_test=False,
        # This also check that we only consider something to be stacked overfitting if the dynamic stacking holdout score gets worse.
        expected_stacked_overfitting_at_val=True,
    )

    fit_extra_args = dict(
        hyperparameters={"GBM": {}},
        fit_weighted_ensemble=False,
    )

    predictor.fit_extra(**fit_extra_args)

    assert len(predictor.model_names()) == 2
    shutil.rmtree(predictor.path, ignore_errors=True)


def test_dynamic_stacking_with_time_limit():
    """Tests that dynamic stacking does not run if stacking is disabled."""
    ds_args = DS_ARGS_TEST_DEFAULTS.copy()
    ds_args["holdout_frac"] = 0.5
    fit_args = dict(
        hyperparameters={"DUMMY": {}},
        dynamic_stacking=True,
        fit_weighted_ensemble=False,
        num_bag_folds=2,
        num_bag_sets=1,
        num_stack_levels=1,
        time_limit=60,  # won't take 60s, but we need a number here instead of None.
        ds_args=ds_args,
        ag_args_ensemble={"fold_fitting_strategy": "sequential_local"},
    )
    dataset_name = "adult"
    extra_metrics = list(METRICS[BINARY])

    FitHelper.fit_and_validate_dataset(
        dataset_name=dataset_name,
        fit_args=fit_args,
        extra_metrics=extra_metrics,
        expected_model_count=2,
        refit_full=False,
        delete_directory=False,
        allowed_dataset_features=["age"],
        expected_stacked_overfitting_at_test=False,
        expected_stacked_overfitting_at_val=False,
    )


@pytest.mark.timeout(
    120
)  # if running AutoGluon twice fails due to a multiprocessing bug, we want to hang up and crash.
def test_dynamic_stacking_run_twice_parallel_fold_fitting_strategy():
    """Tests that dynamic stacking memory save fit works."""
    ds_args = DS_ARGS_TEST_DEFAULTS.copy()
    ds_args["memory_safe_fits"] = True  # guarantee for sanity
    fit_args = dict(
        hyperparameters={"DUMMY": {}},
        fit_weighted_ensemble=False,
        dynamic_stacking=True,
        num_stack_levels=1,
        num_bag_folds=2,
        num_bag_sets=1,
        time_limit=None,
        ds_args=ds_args,
    )

    # Get custom val data (the test data)
    train_data, test_data, dataset_info = FitHelper.load_dataset(name="adult", directory_prefix="./datasets/")
    label = dataset_info["label"]
    allowed_cols = ["age", label]
    train_data = train_data[allowed_cols]
    test_data = test_data[allowed_cols]

    for _ in range(2):
        predictor = FitHelper.fit_dataset(
            train_data=train_data, init_args=dict(label=label), fit_args=fit_args, sample_size=1000
        )
        lb = predictor.leaderboard(test_data, extra_info=True)
        stacked_overfitting_assert(lb, predictor, False, False)
        shutil.rmtree(predictor.path)


def _grouped_toy_for_dystack(n: int = 60, n_groups: int = 10, seed: int = 0) -> pd.DataFrame:
    """Grouped binary toy data: both classes in every group, 6 rows per group at n=60."""
    rng = np.random.default_rng(seed)
    rows_per = n // n_groups
    return pd.DataFrame(
        {
            "f1": rng.normal(size=n),
            "gid": np.repeat(np.arange(n_groups), rows_per),
            "label": np.tile([0, 1], n // 2),
        }
    )


def _assert_group_disjoint(splits: list, groups: np.ndarray) -> None:
    for train_idx, val_idx in splits:
        assert not (set(groups[train_idx]) & set(groups[val_idx]))


@pytest.mark.parametrize("validation_procedure", ["holdout", "cv"])
def test_dynamic_stacking_with_groups_is_group_disjoint(validation_procedure, monkeypatch):
    """DyStack + `groups` must not put the same group on both sides of a sub-fit split (#5533)."""
    from autogluon.common.utils.validation_structure import ValidationStructure
    from autogluon.tabular import TabularPredictor

    df = _grouped_toy_for_dystack()
    groups = df["gid"].to_numpy()

    holdouts: list[tuple[np.ndarray, np.ndarray]] = []
    cv_splits: list[tuple[np.ndarray, np.ndarray]] = []
    original_holdout = ValidationStructure.holdout_split_indices
    original_custom = ValidationStructure.custom_splits

    def holdout_spy(self, X, y, holdout_frac, random_state=0, **kwargs):
        split = original_holdout(self, X, y, holdout_frac=holdout_frac, random_state=random_state, **kwargs)
        if split is not None:
            holdouts.append(split)
        return split

    def custom_spy(self, X, y, **kwargs):
        splits, n_folds, n_repeats = original_custom(self, X, y, **kwargs)
        if kwargs.get("random_state") == 42:
            cv_splits.extend(splits)
        return splits, n_folds, n_repeats

    monkeypatch.setattr(ValidationStructure, "holdout_split_indices", holdout_spy)
    monkeypatch.setattr(ValidationStructure, "custom_splits", custom_spy)

    TabularPredictor(label="label", groups="gid", verbosity=0).fit(
        df,
        hyperparameters={"DUMMY": {}},
        num_bag_folds=2,
        num_bag_sets=1,
        num_stack_levels=1,
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
        raise_on_model_failure=True,
    )

    if validation_procedure == "holdout":
        assert holdouts, "DyStack holdout did not go through the group-aware splitter"
        _assert_group_disjoint(holdouts, groups)
    else:
        assert cv_splits, "DyStack CV did not go through the group-aware splitter"
        _assert_group_disjoint(cv_splits, groups)


def test_dynamic_stacking_with_groups_too_few_groups_raises():
    """Need 3+ groups: one held out, two left for LeaveOneGroupOut bagging."""
    from autogluon.tabular import TabularPredictor

    df = _grouped_toy_for_dystack(n=12, n_groups=2)
    with pytest.raises(ValueError, match="at least 3 unique groups"):
        TabularPredictor(label="label", groups="gid", verbosity=0).fit(
            df,
            hyperparameters={"DUMMY": {}},
            num_bag_folds=2,
            num_stack_levels=1,
            dynamic_stacking=True,
            ds_args={"enable_ray_logging": False, "memory_safe_fits": False},
            fit_weighted_ensemble=False,
        )


_DS_GROUPS = dict(enable_ray_logging=False, memory_safe_fits=False, clean_up_fits=True)


def _capture_dystack_splits(monkeypatch) -> list[dict]:
    """Record the train/val index splits `_dystack` actually receives."""
    import autogluon.tabular.predictor.predictor as pred_mod

    captured: list[dict] = []
    original = pred_mod._dystack

    def spy(predictor, train_data, time_limit, ds_fit_kwargs, ag_fit_kwargs, ag_post_fit_kwargs, holdout_data=None):
        rec = {
            "holdout_frac": ds_fit_kwargs.get("holdout_frac"),
            "train_indices": ds_fit_kwargs.get("train_indices"),
            "val_indices": ds_fit_kwargs.get("val_indices"),
        }
        captured.append(rec)
        return original(
            predictor, train_data, time_limit, ds_fit_kwargs, ag_fit_kwargs, ag_post_fit_kwargs, holdout_data
        )

    monkeypatch.setattr(pred_mod, "_dystack", spy)
    return captured


def _fit_dystack_groups(df, ds_args=None, groups="gid"):
    from autogluon.tabular import TabularPredictor

    args = dict(_DS_GROUPS)
    if ds_args:
        args.update(ds_args)
    init = dict(label="label", verbosity=0)
    if groups is not None:
        init["groups"] = groups
    return TabularPredictor(**init).fit(
        df,
        hyperparameters={"DUMMY": {}},
        num_bag_folds=2,
        num_bag_sets=1,
        num_stack_levels=1,
        dynamic_stacking=True,
        ds_args=args,
        fit_weighted_ensemble=False,
        raise_on_model_failure=True,
    )


def test_dynamic_stacking_with_groups_uses_indices_not_row_frac(monkeypatch):
    """Default DyStack + groups must take the index path, not holdout_frac (#5533)."""
    df = _grouped_toy_for_dystack()
    captured = _capture_dystack_splits(monkeypatch)
    predictor = _fit_dystack_groups(df)
    assert captured, "DyStack sub-fit did not run"
    rec = captured[0]
    assert rec["holdout_frac"] is None
    assert rec["train_indices"] is not None and rec["val_indices"] is not None
    groups = df["gid"].to_numpy()
    _assert_group_disjoint([(rec["train_indices"], rec["val_indices"])], groups)
    assert len(predictor.predict_oof()) == len(df), "full fit must train on all rows, not the sub-fit slice"
    shutil.rmtree(predictor.path)


def test_dynamic_stacking_without_groups_still_uses_holdout_frac(monkeypatch):
    """No groups= → DyStack keep the historical row-wise holdout_frac path."""
    df = _grouped_toy_for_dystack()
    captured = _capture_dystack_splits(monkeypatch)
    predictor = _fit_dystack_groups(df, groups=None)
    rec = captured[0]
    assert rec["holdout_frac"] is not None
    assert rec["train_indices"] is None
    shutil.rmtree(predictor.path)


def test_dynamic_stacking_with_groups_exactly_3_groups_is_feasible(monkeypatch):
    """3 groups is the minimum: 1 held out, 2 left for LeaveOneGroupOut."""
    df = _grouped_toy_for_dystack(n=18, n_groups=3)
    captured = _capture_dystack_splits(monkeypatch)
    predictor = _fit_dystack_groups(df)
    rec = captured[0]
    train_g = set(df["gid"].to_numpy()[rec["train_indices"]])
    val_g = set(df["gid"].to_numpy()[rec["val_indices"]])
    assert train_g.isdisjoint(val_g)
    assert len(train_g) >= 2
    assert len(val_g) >= 1
    shutil.rmtree(predictor.path)


def test_dynamic_stacking_with_groups_large_holdout_frac_keeps_logo_feasible(monkeypatch):
    """A 50% group holdout of 3 groups can leave 1 train group; repair must keep LOGO valid."""
    df = _grouped_toy_for_dystack(n=18, n_groups=3)
    captured = _capture_dystack_splits(monkeypatch)
    predictor = _fit_dystack_groups(df, ds_args={"holdout_frac": 0.5})
    rec = captured[0]
    train_g = set(df["gid"].to_numpy()[rec["train_indices"]])
    val_g = set(df["gid"].to_numpy()[rec["val_indices"]])
    assert train_g.isdisjoint(val_g)
    assert len(train_g) >= 2
    assert len(val_g) >= 1
    shutil.rmtree(predictor.path)


def test_dynamic_stacking_with_groups_cv_three_groups_keeps_logo_feasible(monkeypatch):
    """2-fold DyStack CV on 3 groups can produce a 1-group train fold; repair it."""
    df = _grouped_toy_for_dystack(n=18, n_groups=3)
    captured = _capture_dystack_splits(monkeypatch)
    predictor = _fit_dystack_groups(df, ds_args={"validation_procedure": "cv", "n_folds": 2, "n_repeats": 1})
    assert captured
    groups = df["gid"].to_numpy()
    for rec in captured:
        train_g = set(groups[rec["train_indices"]])
        val_g = set(groups[rec["val_indices"]])
        assert train_g.isdisjoint(val_g)
        assert len(train_g) >= 2
        assert len(val_g) >= 1
    shutil.rmtree(predictor.path)


def test_dynamic_stacking_false_allows_two_groups():
    """The 3-group floor is only for DyStack + groups, not for grouped bagging alone."""
    from autogluon.tabular import TabularPredictor

    df = _grouped_toy_for_dystack(n=12, n_groups=2)
    predictor = TabularPredictor(label="label", groups="gid", verbosity=0).fit(
        df,
        hyperparameters={"DUMMY": {}},
        num_bag_folds=2,
        num_stack_levels=0,
        dynamic_stacking=False,
        fit_weighted_ensemble=False,
        raise_on_model_failure=True,
    )
    assert len(predictor.predict_oof()) == len(df)
    shutil.rmtree(predictor.path)
