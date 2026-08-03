"""Size-curve resolution for the auto-selected validation method."""

from __future__ import annotations

import math

import pytest

from autogluon.tabular.configs.pipeline_presets import (
    USE_BAG_HOLDOUT_AUTO_THRESHOLD,
    _get_validation_preset,
    resolve_size_curve,
)


def test__resolve_size_curve__fixed_value_is_returned_as_is():
    assert resolve_size_curve(8, num_train_rows=10) == 8
    assert resolve_size_curve(False, num_train_rows=10**9) is False
    assert resolve_size_curve(None, num_train_rows=10) is None


def test__resolve_size_curve__anchor_is_inclusive_upper_bound():
    curve = [[100, "small"], [1000, "medium"], "large"]
    assert resolve_size_curve(curve, num_train_rows=1) == "small"
    assert resolve_size_curve(curve, num_train_rows=100) == "small"  # inclusive
    assert resolve_size_curve(curve, num_train_rows=101) == "medium"
    assert resolve_size_curve(curve, num_train_rows=1000) == "medium"
    assert resolve_size_curve(curve, num_train_rows=1001) == "large"


def test__resolve_size_curve__rejects_malformed_curves():
    with pytest.raises(ValueError, match="must not be empty"):
        resolve_size_curve([], num_train_rows=10)
    with pytest.raises(ValueError, match="ascend by rows"):
        resolve_size_curve([[1000, "a"], [100, "b"], "c"], num_train_rows=10)
    with pytest.raises(ValueError, match=r"\[rows, value\] pairs"):
        resolve_size_curve([[100, "a", "extra"], "b"], num_train_rows=10)


@pytest.mark.parametrize(
    "num_train_rows",
    [
        1,
        10,
        49,
        50,
        55,
        59,
        60,
        69,
        70,
        79,
        80,
        100,
        1000,
        USE_BAG_HOLDOUT_AUTO_THRESHOLD - 1,
        USE_BAG_HOLDOUT_AUTO_THRESHOLD,
        USE_BAG_HOLDOUT_AUTO_THRESHOLD + 1,
    ],
)
def test__validation_preset__matches_the_arithmetic_it_replaced(num_train_rows):
    """The curves are a refactor: same numbers as the expressions they replaced."""
    preset = _get_validation_preset(num_train_rows=num_train_rows, hpo_enabled=False)
    assert preset["num_bag_folds"] == min(8, max(5, math.floor(num_train_rows / 10)))
    assert preset["num_bag_sets"] == 1
    assert preset["use_bag_holdout"] == (num_train_rows >= USE_BAG_HOLDOUT_AUTO_THRESHOLD)


def test__validation_preset__curve_override_enables_repeats_on_small_data():
    """The point of the curves: retune one knob per size regime without editing the module."""
    curves = {"num_bag_sets": [[2_000, 5], 1]}
    small = _get_validation_preset(num_train_rows=500, hpo_enabled=False, validation_curves=curves)
    large = _get_validation_preset(num_train_rows=50_000, hpo_enabled=False, validation_curves=curves)
    assert (small["num_bag_folds"], small["num_bag_sets"]) == (8, 5)  # 8x5 CV on small data
    assert (large["num_bag_folds"], large["num_bag_sets"]) == (8, 1)  # unchanged above the anchor
    # overriding one knob leaves the others at their defaults
    assert small["use_bag_holdout"] is False


def test__validation_preset__curve_override_can_move_the_holdout_switch():
    curves = {"use_bag_holdout": [[10_000, False], True]}
    assert (
        _get_validation_preset(num_train_rows=10_000, hpo_enabled=False, validation_curves=curves)["use_bag_holdout"]
        is False
    )
    assert (
        _get_validation_preset(num_train_rows=10_001, hpo_enabled=False, validation_curves=curves)["use_bag_holdout"]
        is True
    )


def test__effective_sample_size__is_rows_unless_group_sizing_is_requested():
    from autogluon.tabular.configs.pipeline_presets import resolve_effective_sample_size

    # opt-in is required: a known group count alone does not change the size
    assert resolve_effective_sample_size(num_train_rows=4_672, num_group_instances=68) == 4_672
    assert resolve_effective_sample_size(num_train_rows=4_672, num_group_instances=68, size_on_groups=True) == 68
    # opting in without a group count falls back to rows rather than failing
    assert resolve_effective_sample_size(num_train_rows=4_672, size_on_groups=True) == 4_672


def test__validation_preset__group_sizing_is_off_by_default():
    """A known group count must not change the preset unless group sizing is requested."""
    curves = {"num_bag_folds": [[500, 5], 8], "num_bag_sets": [[500, 5], 1]}
    rows_only = _get_validation_preset(num_train_rows=4_672, hpo_enabled=False, validation_curves=curves)
    with_groups_known = _get_validation_preset(
        num_train_rows=4_672, hpo_enabled=False, validation_curves=curves, num_group_instances=68
    )
    assert rows_only == with_groups_known
    assert (rows_only["num_bag_folds"], rows_only["num_bag_sets"]) == (8, 1)


def test__validation_preset__group_sizing_flips_a_row_large_group_small_task():
    """The regime can differ by which sample size is used; grouped data is the case that matters.

    Mirrors a benchmark task with 4,672 rows across 68 groups: large by rows, small by groups.
    """
    curves = {"num_bag_folds": [[500, 5], 8], "num_bag_sets": [[500, 5], 1]}
    preset = _get_validation_preset(
        num_train_rows=4_672,
        hpo_enabled=False,
        validation_curves=curves,
        num_group_instances=68,
        size_on_groups=True,
    )
    assert (preset["num_bag_folds"], preset["num_bag_sets"]) == (5, 5)
    # holdout_frac is a fraction of the rows held out, so it stays sized on rows
    rows_only = _get_validation_preset(num_train_rows=4_672, hpo_enabled=False, validation_curves=curves)
    assert preset["holdout_frac"] == rows_only["holdout_frac"]
