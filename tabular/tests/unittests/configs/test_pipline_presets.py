from autogluon.tabular.configs.pipeline_presets import get_validation_and_stacking_method
import pytest
from autogluon.core.constants import BINARY


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
    ],
)
def test_auto_stack_get_validation_and_stacking_method(metadata_and_expected_result):
    metadata, expected_result = metadata_and_expected_result

    metadata["hpo_enabled"] = metadata.get("hpo_enabled", False)
    metadata["problem_type"] = metadata.get("problem_type", "N/A")
    metadata["dynamic_stacking"] = metadata.get("dynamic_stacking", None)
    metadata["use_bag_holdout"] = metadata.get("use_bag_holdout", None)

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
