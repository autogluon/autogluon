import pytest

from autogluon.core.constants import BINARY
from autogluon.tabular.configs.pipeline_presets import get_validation_and_stacking_method


def test_default_get_validation_and_stacking_method():
    (
        num_bag_folds,
        num_bag_sets,
        num_stack_levels,
        dynamic_stacking,
        use_bag_holdout,
        holdout_frac,
        refit_full,
    ) = get_validation_and_stacking_method(
        auto_stack=False,
        # Default
        num_bag_folds=None,
        num_bag_sets=None,
        use_bag_holdout=None,
        holdout_frac=None,
        num_stack_levels=None,
        dynamic_stacking=None,
        refit_full=None,
        # Can change the default behavior
        num_train_rows=1000,
        hpo_enabled=False,
        # Is ignored due to `auto_stack=False`
        problem_type="N/A",
        n_samples_minority_class=None,
    )

    assert num_bag_folds == 0
    assert num_bag_sets == 1
    assert use_bag_holdout is False
    assert holdout_frac == 0.2
    assert refit_full is False
    assert dynamic_stacking is True


@pytest.mark.parametrize(
    "metadata_and_expected_result",
    [
        # Reaction to dataset size checks
        (dict(num_train_rows=40), dict(num_bag_folds=5)),
        (dict(num_train_rows=70), dict(num_bag_folds=7)),
        (dict(num_train_rows=10_000), dict(holdout_frac=0.1)),
        # (also checks that dynamic stacking is disabled for holdout validation)
        (
            dict(num_train_rows=1_000_000),
            dict(holdout_frac=0.01, dynamic_stacking=False, use_bag_holdout=True),
        ),
        # HPO On check
        (
            dict(num_train_rows=1_000_000, hpo_enabled=True),
            dict(holdout_frac=0.02, dynamic_stacking=False, use_bag_holdout=True),
        ),
        # No dynamic stacking, auto_stack check
        (
            dict(num_train_rows=749, dynamic_stacking=False),
            dict(dynamic_stacking=False, num_stack_levels=0),
        ),
        (
            dict(num_train_rows=750, dynamic_stacking=False),
            dict(dynamic_stacking=False, num_stack_levels=1),
        ),
        (
            dict(num_train_rows=750, dynamic_stacking=False, problem_type=BINARY),
            dict(dynamic_stacking=False, num_stack_levels=0),
        ),
        (
            dict(num_train_rows=750, dynamic_stacking=False, problem_type=BINARY, use_bag_holdout=True),
            dict(dynamic_stacking=False, num_stack_levels=1, use_bag_holdout=True),
        ),
        # Minority class handling when there are exactly enough samples
        (
            dict(num_train_rows=100, n_samples_minority_class=3),
            dict(num_bag_folds=2, num_stack_levels=1),
        ),
    ],
)
def test_auto_stack_get_validation_and_stacking_method(metadata_and_expected_result):
    metadata, expected_result = metadata_and_expected_result

    metadata["hpo_enabled"] = metadata.get("hpo_enabled", False)
    metadata["problem_type"] = metadata.get("problem_type", "N/A")
    metadata["dynamic_stacking"] = metadata.get("dynamic_stacking", None)
    metadata["use_bag_holdout"] = metadata.get("use_bag_holdout", None)
    metadata["n_samples_minority_class"] = metadata.get("n_samples_minority_class", None)

    num_bag_folds = expected_result.get("num_bag_folds", 8)
    num_bag_sets = expected_result.get("num_bag_sets", 1)
    num_stack_levels = expected_result.get("num_stack_levels", 1)
    dynamic_stacking = expected_result.get("dynamic_stacking", True)
    use_bag_holdout = expected_result.get("use_bag_holdout", False)
    holdout_frac = expected_result.get("holdout_frac", 0.2)

    assert get_validation_and_stacking_method(
        auto_stack=True,
        # Default
        num_bag_folds=None,
        num_bag_sets=None,
        holdout_frac=None,
        num_stack_levels=None,
        refit_full=None,
        **metadata,
    ) == (
        num_bag_folds,
        num_bag_sets,
        num_stack_levels,
        dynamic_stacking,
        use_bag_holdout,
        holdout_frac,
        False,
    )


def test_n_samples_minority_class_error_and_warning():
    # Case: too few minority samples should raise ValueError
    with pytest.raises(ValueError):
        get_validation_and_stacking_method(
            auto_stack=True,
            num_bag_folds=None,
            num_bag_sets=None,
            use_bag_holdout=None,
            holdout_frac=None,
            num_stack_levels=None,
            dynamic_stacking=None,
            refit_full=None,
            num_train_rows=100,
            hpo_enabled=False,
            problem_type="N/A",
            n_samples_minority_class=1,
        )

    # Case: exactly enough minority samples -> should return adjusted folds and emit a warning
    # dynamic_stacking defaults to True (since use_bag_holdout=False for 100 rows),
    # so extra_holdout_set=True, meaning supported_folds = 3 - 1 = 2
    with pytest.warns(UserWarning):
        result = get_validation_and_stacking_method(
            auto_stack=True,
            num_bag_folds=None,
            num_bag_sets=None,
            use_bag_holdout=None,
            holdout_frac=None,
            num_stack_levels=None,
            dynamic_stacking=None,
            refit_full=None,
            num_train_rows=100,
            hpo_enabled=False,
            problem_type="N/A",
            n_samples_minority_class=3,
        )

    # Ensure the folds were adjusted down (3 -> 2 due to extra holdout set required)
    assert result[0] == 2
    # Ensure other defaults hold as expected
    assert result[1] == 1
    assert result[2] == 1
    assert result[3] is True
    assert result[4] is False
    assert result[6] is False


def _call_get_validation(n_samples_minority_class, dynamic_stacking=None, use_bag_holdout=None, num_bag_folds=None):
    """Helper to call get_validation_and_stacking_method with common defaults."""
    return get_validation_and_stacking_method(
        auto_stack=True,
        num_bag_folds=num_bag_folds,
        num_bag_sets=None,
        use_bag_holdout=use_bag_holdout,
        holdout_frac=None,
        num_stack_levels=None,
        dynamic_stacking=dynamic_stacking,
        refit_full=None,
        num_train_rows=100,
        hpo_enabled=False,
        problem_type="N/A",
        n_samples_minority_class=n_samples_minority_class,
    )


def test_num_bag_folds_never_one():
    """num_bag_folds must be 0 or >= 2. The minority class adjustment must never produce 1."""
    for n_minority in range(2, 20):
        for ds in (True, False):
            for bh in (True, False):
                extra_holdout = ds or bh
                min_required = 3 if extra_holdout else 2
                if n_minority < min_required:
                    # Should raise ValueError when there aren't enough samples at all
                    with pytest.raises(ValueError):
                        _call_get_validation(
                            n_samples_minority_class=n_minority,
                            dynamic_stacking=ds,
                            use_bag_holdout=bh,
                        )
                else:
                    result = _call_get_validation(
                        n_samples_minority_class=n_minority,
                        dynamic_stacking=ds,
                        use_bag_holdout=bh,
                    )
                    num_bag_folds = result[0]
                    assert num_bag_folds == 0 or num_bag_folds >= 2, (
                        f"num_bag_folds={num_bag_folds} is invalid "
                        f"(n_minority={n_minority}, dynamic_stacking={ds}, use_bag_holdout={bh})"
                    )


def test_minority_class_none_has_no_effect():
    """When n_samples_minority_class is None, folds should not be adjusted."""
    result_none = _call_get_validation(n_samples_minority_class=None)
    # Default for 100 rows with auto_stack: 8 folds
    assert result_none[0] == 8


def test_minority_class_disables_bagging_when_too_few():
    """When minority samples can't support even 2 folds, bagging should be disabled (0 folds)."""
    # dynamic_stacking=False, use_bag_holdout=False -> extra_holdout_set=False
    # n_minority=2, supported=2, which is valid -> should get 2 folds
    result = _call_get_validation(n_samples_minority_class=2, dynamic_stacking=False, use_bag_holdout=False)
    assert result[0] == 2

    # dynamic_stacking=True, use_bag_holdout=False -> extra_holdout_set=True
    # n_minority=2, min_samples_per_class=3 > 2 -> ValueError (unrecoverable)
    with pytest.raises(ValueError):
        _call_get_validation(n_samples_minority_class=2, dynamic_stacking=True, use_bag_holdout=False)

    # dynamic_stacking=True, use_bag_holdout=False -> extra_holdout_set=True
    # n_minority=3, supported=3-1=2, clamped to 2 -> 2 folds
    with pytest.warns(UserWarning):
        result = _call_get_validation(n_samples_minority_class=3, dynamic_stacking=True, use_bag_holdout=False)
    assert result[0] == 2
    assert result[2] == 1  # stacking still enabled with 2 folds


def test_minority_class_caps_folds_at_n_samples():
    """Folds should not exceed minority class count (minus 1 if extra holdout)."""
    # No extra holdout: 5 minority samples -> supported=5, default folds=8 -> capped to 5
    with pytest.warns(UserWarning):
        result = _call_get_validation(n_samples_minority_class=5, dynamic_stacking=False, use_bag_holdout=False)
    assert result[0] == 5

    # With extra holdout: 5 minority samples -> supported=4, default folds=8 -> capped to 4
    with pytest.warns(UserWarning):
        result = _call_get_validation(n_samples_minority_class=5, dynamic_stacking=True, use_bag_holdout=False)
    assert result[0] == 4
