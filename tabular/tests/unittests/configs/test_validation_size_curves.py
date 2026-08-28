"""Size-curve resolution for the auto-selected validation method."""

from __future__ import annotations

import math

import pytest

from autogluon.tabular.configs.pipeline_presets import (
    DEFAULT_VALIDATION_SIZE_CURVES,
    USE_BAG_HOLDOUT_AUTO_THRESHOLD,
    ValidationSizeCurves,
    _get_validation_preset,
    get_validation_and_stacking_method,
    resolve_hyperparameters_curve,
    resolve_size_curve,
    resolve_validation_mode,
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
    small = _get_validation_preset(num_train_rows=500, hpo_enabled=False, validation_size_curves=curves)
    large = _get_validation_preset(num_train_rows=50_000, hpo_enabled=False, validation_size_curves=curves)
    assert (small["num_bag_folds"], small["num_bag_sets"]) == (8, 5)  # 8x5 CV on small data
    assert (large["num_bag_folds"], large["num_bag_sets"]) == (8, 1)  # unchanged above the anchor
    # overriding one knob leaves the others at their defaults
    assert small["use_bag_holdout"] is False


def test__validation_preset__curve_override_can_move_the_holdout_switch():
    curves = {"use_bag_holdout": [[10_000, False], True]}
    assert (
        _get_validation_preset(num_train_rows=10_000, hpo_enabled=False, validation_size_curves=curves)[
            "use_bag_holdout"
        ]
        is False
    )
    assert (
        _get_validation_preset(num_train_rows=10_001, hpo_enabled=False, validation_size_curves=curves)[
            "use_bag_holdout"
        ]
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
    rows_only = _get_validation_preset(num_train_rows=4_672, hpo_enabled=False, validation_size_curves=curves)
    with_groups_known = _get_validation_preset(
        num_train_rows=4_672, hpo_enabled=False, validation_size_curves=curves, num_group_instances=68
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
        validation_size_curves=curves,
        num_group_instances=68,
        size_on_groups=True,
    )
    assert (preset["num_bag_folds"], preset["num_bag_sets"]) == (5, 5)
    # holdout_frac is a fraction of the rows held out, so it stays sized on rows
    rows_only = _get_validation_preset(num_train_rows=4_672, hpo_enabled=False, validation_size_curves=curves)
    assert preset["holdout_frac"] == rows_only["holdout_frac"]


@pytest.mark.parametrize("num_train_rows", [1, 749, 750, 751, 10_000])
@pytest.mark.parametrize("auto_stack", [True, False])
@pytest.mark.parametrize("dynamic_stacking", [True, False])
@pytest.mark.parametrize("problem_type", ["binary", "multiclass"])
def test__stack_levels__matches_the_condition_it_replaced(num_train_rows, auto_stack, dynamic_stacking, problem_type):
    """Stack depth now comes from a curve, gated by the same qualitative conditions as before."""
    from autogluon.tabular.configs.pipeline_presets import get_validation_and_stacking_method

    use_bag_holdout = False
    result = get_validation_and_stacking_method(
        num_bag_folds=None,
        num_bag_sets=None,
        use_bag_holdout=use_bag_holdout,
        holdout_frac=None,
        auto_stack=auto_stack,
        num_stack_levels=None,
        dynamic_stacking=dynamic_stacking,
        refit_full=None,
        num_train_rows=num_train_rows,
        problem_type=problem_type,
        hpo_enabled=False,
        n_samples_minority_class=None,
    )
    expected = (
        1
        if auto_stack
        and (dynamic_stacking or ((use_bag_holdout or problem_type != "binary") and num_train_rows >= 750))
        else 0
    )
    assert result[2] == expected


def test__stack_levels__curve_can_ask_for_a_deeper_layer_on_large_data():
    curves = {"num_stack_levels": [[749, 0], [100_000, 1], 2]}
    small = _get_validation_preset(num_train_rows=500, hpo_enabled=False, validation_size_curves=curves)
    medium = _get_validation_preset(num_train_rows=50_000, hpo_enabled=False, validation_size_curves=curves)
    large = _get_validation_preset(num_train_rows=500_000, hpo_enabled=False, validation_size_curves=curves)
    assert (small["num_stack_levels"], medium["num_stack_levels"], large["num_stack_levels"]) == (0, 1, 2)


def test__use_bag_holdout_and_stack_levels__can_both_be_sized_on_groups():
    """Both knobs read the same effective sample size, so group sizing applies to them too."""
    curves = {"use_bag_holdout": [[100, False], True], "num_stack_levels": [[100, 0], 1]}
    kwargs = dict(num_train_rows=4_672, hpo_enabled=False, validation_size_curves=curves, num_group_instances=68)
    on_rows = _get_validation_preset(**kwargs)
    on_groups = _get_validation_preset(**kwargs, size_on_groups=True)
    assert (on_rows["use_bag_holdout"], on_rows["num_stack_levels"]) == (True, 1)
    assert (on_groups["use_bag_holdout"], on_groups["num_stack_levels"]) == (False, 0)


def test__holdout_frac__has_no_default_curve_and_keeps_its_built_in_policy():
    """The built-in policy is a continuous function of rows, which a step curve cannot express."""
    from autogluon.core.utils.utils import default_holdout_frac

    for num_train_rows in (100, 900, 50_000):
        preset = _get_validation_preset(num_train_rows=num_train_rows, hpo_enabled=False)
        assert preset["holdout_frac"] == round(
            default_holdout_frac(num_train_rows=num_train_rows, hyperparameter_tune=False), 4
        )
    # a known group count must not change it either, absent an explicit curve
    assert (
        _get_validation_preset(num_train_rows=900, hpo_enabled=False, num_group_instances=60)["holdout_frac"]
        == _get_validation_preset(num_train_rows=900, hpo_enabled=False)["holdout_frac"]
    )


def test__holdout_frac__curve_replaces_the_built_in_policy():
    curves = {"holdout_frac": [[1_000, 0.2], 0.1]}
    assert (
        _get_validation_preset(num_train_rows=500, hpo_enabled=False, validation_size_curves=curves)["holdout_frac"]
        == 0.2
    )
    assert (
        _get_validation_preset(num_train_rows=5_000, hpo_enabled=False, validation_size_curves=curves)["holdout_frac"]
        == 0.1
    )


def test__holdout_frac__curve_is_read_at_the_effective_sample_size():
    """A supplied curve follows the same sizing as the other knobs, including group sizing."""
    curves = {"holdout_frac": [[100, 0.25], 0.1]}
    kwargs = dict(num_train_rows=9_000, hpo_enabled=False, validation_size_curves=curves, num_group_instances=60)
    assert _get_validation_preset(**kwargs)["holdout_frac"] == 0.1  # 9000 rows > 100
    assert _get_validation_preset(**kwargs, size_on_groups=True)["holdout_frac"] == 0.25  # 60 groups <= 100


def test__validation_size_curves__dataclass_and_dict_are_equivalent():
    """The dataclass is the explicit form; a dict is still accepted and normalized to it."""
    from autogluon.tabular.configs.pipeline_presets import ValidationSizeCurves

    as_dict = {"num_bag_folds": [[1_000, 5], 8], "num_bag_sets": [[1_000, 5], 5]}
    from_dict = _get_validation_preset(num_train_rows=500, hpo_enabled=False, validation_size_curves=as_dict)
    from_dataclass = _get_validation_preset(
        num_train_rows=500, hpo_enabled=False, validation_size_curves=ValidationSizeCurves(**as_dict)
    )
    assert from_dict == from_dataclass
    assert (from_dict["num_bag_folds"], from_dict["num_bag_sets"]) == (5, 5)

    assert ValidationSizeCurves.from_input(as_dict) == ValidationSizeCurves(**as_dict)
    assert ValidationSizeCurves.from_input(None) is None
    already = ValidationSizeCurves(num_bag_sets=5)
    assert ValidationSizeCurves.from_input(already) is already


def test__validation_size_curves__unset_knobs_keep_their_defaults():
    from autogluon.tabular.configs.pipeline_presets import DEFAULT_VALIDATION_SIZE_CURVES, ValidationSizeCurves

    curves = ValidationSizeCurves(num_bag_sets=5)
    assert curves.as_overrides() == {"num_bag_sets": 5}  # only what was set
    preset = _get_validation_preset(num_train_rows=900, hpo_enabled=False, validation_size_curves=curves)
    assert preset["num_bag_sets"] == 5
    # the untouched knobs still follow the defaults
    assert preset["num_bag_folds"] == resolve_size_curve(DEFAULT_VALIDATION_SIZE_CURVES["num_bag_folds"], 900)


def test__validation_size_curves__rejects_unknown_knobs():
    """A mistyped knob was previously accepted and silently ignored."""
    from autogluon.tabular.configs.pipeline_presets import ValidationSizeCurves

    with pytest.raises(ValueError, match="Invalid `validation_size_curves` keys"):
        ValidationSizeCurves.from_input({"num_bag_setz": 5})
    with pytest.raises(ValueError, match="must be a dict or ValidationSizeCurves"):
        ValidationSizeCurves.from_input(5)


def _resolve(*, auto_stack: bool, curves, num_train_rows: int = 400):
    """The (folds, sets, stack_levels) the validation method resolves to."""
    num_bag_folds, num_bag_sets, num_stack_levels, _, _, _, _ = get_validation_and_stacking_method(
        num_bag_folds=None,
        num_bag_sets=None,
        use_bag_holdout=None,
        holdout_frac=None,
        auto_stack=auto_stack,
        num_stack_levels=None,
        dynamic_stacking=None,
        refit_full=None,
        num_train_rows=num_train_rows,
        problem_type="binary",
        hpo_enabled=False,
        n_samples_minority_class=None,
        num_group_instances=None,
        size_on_groups=False,
        validation_size_curves=curves,
    )
    return num_bag_folds, num_bag_sets, num_stack_levels


def test__auto_stack_does_not_override_an_explicit_curve():
    """A curve for a knob is a request for size-driven selection of it, which `auto_stack` cannot undo.

    Without this, `auto_stack=False` (the default) replaced an explicit `num_bag_folds` curve with
    0 -- no bagging and so no out-of-fold predictions -- silently turning a benchmark run of bagged
    models into unbagged holdout fits.
    """
    curves = {"num_bag_folds": [[500, 5], 8], "num_bag_sets": [[500, 5], 1]}
    # 400 rows sits in the first anchor of both curves.
    assert _resolve(auto_stack=False, curves=curves) == (5, 5, 0)
    assert _resolve(auto_stack=True, curves=curves)[:2] == (5, 5)


def test__auto_stack_still_governs_knobs_without_a_curve():
    """Unchanged behavior where the caller said nothing: `auto_stack` decides."""
    assert _resolve(auto_stack=False, curves=None) == (0, 1, 0)
    assert _resolve(auto_stack=True, curves=None) == (8, 1, 1)


def test__explicit_curves_apply_per_knob():
    """Only the knobs given a curve escape `auto_stack`; the rest keep their gated defaults."""
    folds, sets, _ = _resolve(auto_stack=False, curves={"num_bag_sets": 3})
    assert sets == 3, "the specified knob is honored"
    assert folds == 0, "the unspecified knob still follows auto_stack=False"


def test__explicit_stack_curve_is_not_gated_by_auto_stack():
    """A `num_stack_levels` curve is the answer, not an input to the auto_stack conditions."""
    _, _, stack_levels = _resolve(auto_stack=False, curves={"num_stack_levels": 2})
    assert stack_levels == 2


def _resolve_dynamic_stacking(curves, *, dynamic_stacking=None, use_bag_holdout=None, num_train_rows: int = 400):
    """The (dynamic_stacking, use_bag_holdout) the validation method resolves to."""
    _, _, _, resolved_ds, resolved_ubh, _, _ = get_validation_and_stacking_method(
        num_bag_folds=None,
        num_bag_sets=None,
        use_bag_holdout=use_bag_holdout,
        holdout_frac=None,
        auto_stack=True,
        num_stack_levels=0,
        dynamic_stacking=dynamic_stacking,
        refit_full=None,
        num_train_rows=num_train_rows,
        problem_type="binary",
        hpo_enabled=False,
        n_samples_minority_class=None,
        num_group_instances=None,
        size_on_groups=False,
        validation_size_curves=curves,
    )
    return resolved_ds, resolved_ubh


def test__dynamic_stacking_curve_replaces_the_derived_default():
    """`dynamic_stacking` has no default curve: it is derived from `use_bag_holdout` unless given one."""
    # Derived: 400 rows -> use_bag_holdout False -> DyStack on.
    assert _resolve_dynamic_stacking(None) == (True, False)
    # A curve replaces that derivation in either direction.
    assert _resolve_dynamic_stacking({"dynamic_stacking": False})[0] is False
    assert _resolve_dynamic_stacking({"dynamic_stacking": True})[0] is True


def test__dynamic_stacking_curve_is_read_at_the_sample_size():
    """A step curve sizes DyStack independently of any other knob."""
    curve = {"dynamic_stacking": [[1_000, False], True]}
    assert _resolve_dynamic_stacking(curve, num_train_rows=400)[0] is False
    assert _resolve_dynamic_stacking(curve, num_train_rows=5_000)[0] is True


def test__explicit_dynamic_stacking_argument_beats_its_curve():
    """As for every other knob, an explicit argument wins over the curve."""
    assert _resolve_dynamic_stacking({"dynamic_stacking": True}, dynamic_stacking=False)[0] is False


def _resolve_method(num_train_rows: int, **kwargs):
    """Resolve the validation method, returning it as a dict of knob -> value."""
    kwargs.setdefault("num_bag_folds", None)
    kwargs.setdefault("holdout_frac", None)
    kwargs.setdefault("refit_full", None)
    kwargs.setdefault("validation_size_curves", None)
    resolved = get_validation_and_stacking_method(
        num_bag_sets=None,
        use_bag_holdout=None,
        auto_stack=False,
        num_stack_levels=None,
        dynamic_stacking=None,
        num_train_rows=num_train_rows,
        problem_type="binary",
        hpo_enabled=False,
        n_samples_minority_class=None,
        **kwargs,
    )
    keys = [
        "num_bag_folds",
        "num_bag_sets",
        "num_stack_levels",
        "dynamic_stacking",
        "use_bag_holdout",
        "holdout_frac",
        "refit_full",
    ]
    return dict(zip(keys, resolved))


def test__refit_full__has_no_default_curve():
    """Without a curve `refit_full` stays False, whatever the size."""
    assert _resolve_method(1_000)["refit_full"] is False
    assert _resolve_method(1_000_000)["refit_full"] is False


def test__refit_full__curve_sizes_it_like_any_other_knob():
    """Refit only in the regime where bagging has stopped, without a caller-side size check."""
    curves = {"num_bag_folds": [[50_000, 8], 0], "refit_full": [[50_000, False], True]}
    below = _resolve_method(10_000, validation_size_curves=curves)
    above = _resolve_method(60_000, validation_size_curves=curves)
    assert (below["num_bag_folds"], below["refit_full"]) == (8, False)
    assert (above["num_bag_folds"], above["refit_full"]) == (0, True)


def test__explicit_refit_full_argument_beats_its_curve():
    """As for every other knob, an explicit argument wins over the curve."""
    resolved = _resolve_method(60_000, validation_size_curves={"refit_full": True}, refit_full=False)
    assert resolved["refit_full"] is False


def test__holdout_frac__int_is_an_absolute_row_count():
    """An int passes through as rows; sklearn's `test_size` reads it that way."""
    assert _resolve_method(60_000, holdout_frac=10_000, num_bag_folds=0)["holdout_frac"] == 10_000


@pytest.mark.parametrize(
    "holdout_frac,message",
    [
        (1.5, "must be between 0 and 1"),
        (0.0, "must be between 0 and 1"),
        (0, "must be at least 1"),
        (True, "is a bool"),
        ("0.2", "must be an int .rows. or a float"),
    ],
)
def test__holdout_frac__rejects_values_that_are_not_a_size(holdout_frac, message):
    with pytest.raises(ValueError, match=message):
        _resolve_method(1_000, holdout_frac=holdout_frac, num_bag_folds=0)


@pytest.mark.parametrize("holdout_frac", [1_000, 2_000])
def test__holdout_frac__rejects_a_split_with_an_empty_side(holdout_frac):
    """Both sides need a row: the error says how the value was read and what it left."""
    with pytest.raises(ValueError, match="both sides of the split need at least 1 row"):
        _resolve_method(1_000, holdout_frac=holdout_frac, num_bag_folds=0)


def test__holdout_frac__is_not_validated_when_bagging_ignores_it():
    """Bagging ignores `holdout_frac`, so an unusable size stays as harmless as it was."""
    resolved = _resolve_method(5_000, holdout_frac=10_000, validation_size_curves={"num_bag_folds": 8})
    assert resolved["num_bag_folds"] == 8


def test__holdout_frac__built_in_policy_is_never_rejected():
    """The default is AutoGluon's own and must resolve at any size, however small."""
    assert _resolve_method(1, num_bag_folds=0)["holdout_frac"] > 0


def test__set_best_to_refit_full__is_allowed_alongside_a_refit_full_curve():
    """The curve decides refitting from the row count, after the kwargs are validated.

    `set_best_to_refit_full` needs no curve of its own: paired with a `refit_full` curve it
    says "serve the refit when there is one", and is inert in the sizes that do not refit.
    """
    from autogluon.tabular.predictor.predictor import _has_refit_full_curve

    assert _has_refit_full_curve({"validation_size_curves": {"refit_full": [[50_000, False], True]}})
    assert _has_refit_full_curve({"validation_size_curves": ValidationSizeCurves(refit_full=True)})
    # A curve for some other knob says nothing about refitting.
    assert not _has_refit_full_curve({"validation_size_curves": {"num_bag_folds": 8}})
    assert not _has_refit_full_curve({"validation_size_curves": None})
    assert not _has_refit_full_curve({})


@pytest.mark.parametrize("refit_full", [None, False])
def test__set_best_to_refit_full__is_disabled_with_a_warning_not_an_error(refit_full, caplog, monkeypatch):
    """Nothing to promote is not a fit-stopping problem; say so and carry on."""
    import logging

    import pandas as pd
    from sklearn.datasets import make_classification

    from autogluon.tabular import TabularPredictor

    X, y = make_classification(n_samples=200, n_features=4, random_state=0)
    train_data = pd.DataFrame(X).rename(columns=str)
    train_data["label"] = y

    # AutoGluon's logger does not propagate, so caplog's root handler never sees its records.
    monkeypatch.setattr(logging.getLogger("autogluon"), "propagate", True)

    predictor = TabularPredictor(label="label", eval_metric="roc_auc", verbosity=2)
    with caplog.at_level(logging.WARNING, logger="autogluon"):
        predictor.fit(
            train_data=train_data,
            hyperparameters={"GBM": [{}]},
            set_best_to_refit_full=True,
            refit_full=refit_full,
        )

    assert "`set_best_to_refit_full=True` is disabled" in caplog.text
    # The fit still produced a usable predictor, and nothing was promoted.
    assert not any(name.endswith("_FULL") for name in predictor.model_names())


def test_explicit_trailing_none_is_honoured_not_clamped():
    """A curve ending in `None` means None above the last anchor, not the last anchor's value.

    `None` is a legal curve value -- "no fixed weights", "use the built-in default" -- so
    `[[X, value], None]` is the natural way to spell "on below X, off above X". It previously
    returned `value` at every size, because the "no fallback supplied" state was itself `None`
    and the two could not be told apart.
    """
    below = {"A": 0.5}
    assert resolve_size_curve([[100, below], None], 40) == below
    assert resolve_size_curve([[100, below], None], 500) is None


def test_anchor_only_curve_still_clamps_to_the_last_anchor():
    """The clamp is right when no trailing value was given -- that is what it is for."""
    assert resolve_size_curve([[59, 5], [69, 6], [79, 7]], 500) == 7
    assert resolve_size_curve([[100, "x"]], 500) == "x"


def test_bare_none_curve_resolves_to_none():
    """A curve of only `None` has nothing to clamp to."""
    assert resolve_size_curve([None], 500) is None


@pytest.mark.parametrize("num_train_rows", [20, 59, 60, 79, 100, 749, 750, 5000, 100_000])
def test_default_curves_are_unchanged_by_the_fallback_fix(num_train_rows):
    """Every built-in curve ends in a non-None value, so none of them can be affected."""
    for key, curve in DEFAULT_VALIDATION_SIZE_CURVES.items():
        value = resolve_size_curve(curve, num_train_rows)
        assert value is not None, f"{key} resolved to None at {num_train_rows}"


def test_validation_mode_and_ensemble_weights_are_curve_keys():
    """Both resolve from curves like any other knob; a dict payload survives intact."""
    curves = {
        "validation_mode": [[100, "none"], "auto"],
        "ensemble_weights": [[100, {"TabPFN-3": 0.5, "TabICL": 0.5}], None],
    }
    assert resolve_size_curve(curves["validation_mode"], 60) == "none"
    assert resolve_size_curve(curves["validation_mode"], 300) == "auto"
    assert resolve_size_curve(curves["ensemble_weights"], 60) == {"TabPFN-3": 0.5, "TabICL": 0.5}
    assert resolve_size_curve(curves["ensemble_weights"], 300) is None


def _mode(num_train_rows, **kwargs):
    """Resolve the knobs, then the mode, exactly as `fit` does."""
    num_bag_folds, _, num_stack_levels, _, _, _, _ = get_validation_and_stacking_method(
        num_bag_folds=kwargs.get("num_bag_folds"),
        num_bag_sets=None,
        use_bag_holdout=None,
        holdout_frac=None,
        auto_stack=False,
        num_stack_levels=kwargs.get("num_stack_levels"),
        dynamic_stacking=None,
        refit_full=None,
        num_train_rows=num_train_rows,
        problem_type="binary",
        hpo_enabled=False,
        n_samples_minority_class=num_train_rows // 2,
        validation_size_curves=kwargs.get("validation_size_curves"),
    )
    return resolve_validation_mode(
        validation_mode=kwargs.get("validation_mode"),
        ensemble_weights=kwargs.get("ensemble_weights"),
        num_bag_folds=num_bag_folds,
        num_stack_levels=num_stack_levels,
        num_train_rows=num_train_rows,
        validation_size_curves=kwargs.get("validation_size_curves"),
    ) + (num_bag_folds, num_stack_levels)


def test_validation_mode_curve_switches_with_the_coupled_knobs():
    """The intended shape: every coupled knob flips at the same threshold."""
    curves = {
        "validation_mode": [[100, "none"], "auto"],
        "num_bag_folds": [[100, 0], 8],
        "num_stack_levels": [[100, 0], 1],
        "ensemble_weights": [[100, {"A": 0.5, "B": 0.5}], None],
    }
    mode, weights, folds, levels = _mode(60, validation_size_curves=curves)
    assert (mode, weights, folds, levels) == ("none", {"A": 0.5, "B": 0.5}, 0, 0)
    mode, weights, folds, _ = _mode(300, validation_size_curves=curves)
    assert (mode, weights, folds) == ("auto", None, 8)


def test_mismatched_curve_thresholds_are_reported_at_resolution():
    """Curves resolve one knob at a time, so a mistyped threshold makes a band of sizes disagree.

    Without this check the contradiction surfaces as a fit-time error at only some data sizes.
    """
    curves = {"validation_mode": [[100, "none"], "auto"], "num_bag_folds": [[50, 0], 8]}
    with pytest.raises(ValueError, match="every curve switches at the same threshold"):
        _mode(60, validation_size_curves=curves)
    # Below both thresholds the knobs agree, so it resolves.
    assert _mode(40, validation_size_curves=curves)[0] == "none"


def test_explicit_validation_mode_conflict_names_the_direct_fix():
    """The same check covers an explicitly passed combination, with a different remedy."""
    with pytest.raises(ValueError, match="Set them to 0, or drop"):
        _mode(60, validation_mode="none", num_bag_folds=8)


def test_explicit_validation_mode_beats_its_curve():
    """A caller-supplied value wins over a curve, as every other knob here does."""
    curves = {"validation_mode": [[100, "none"], "auto"]}
    assert _mode(60, validation_size_curves=curves)[0] == "none"
    assert _mode(60, validation_size_curves=curves, validation_mode="auto")[0] == "auto"


def test_hyperparameters_is_a_curve_key_read_at_the_row_count():
    """The portfolio switches with the rest of the bundle.

    Read at the row count, not the effective size: it resolves before `validation_structure` is
    known, so the group count is not available yet.
    """
    curve = [[100, {"TABPFN-3": {}, "TABICL": {}}], "default"]
    assert resolve_hyperparameters_curve(None, 60, {"hyperparameters": curve}) == {"TABPFN-3": {}, "TABICL": {}}
    assert resolve_hyperparameters_curve(None, 300, {"hyperparameters": curve}) == "default"


def test_explicit_hyperparameters_beats_its_curve():
    """A caller-supplied portfolio wins, as every other knob does."""
    curve = [[100, {"TABPFN-3": {}}], "default"]
    assert resolve_hyperparameters_curve({"GBM": {}}, 60, {"hyperparameters": curve}) == {"GBM": {}}


def test_no_hyperparameters_curve_leaves_the_value_alone():
    assert resolve_hyperparameters_curve(None, 60, {"num_bag_folds": [[100, 0], 8]}) is None
    assert resolve_hyperparameters_curve(None, 60, None) is None
