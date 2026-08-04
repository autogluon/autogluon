from __future__ import annotations

import warnings
from dataclasses import dataclass, fields

from autogluon.core.constants import BINARY, PROBLEM_TYPES
from autogluon.core.utils.utils import default_holdout_frac

USE_BAG_HOLDOUT_AUTO_THRESHOLD = 1_000_000

#: [EXPERIMENTAL] A validation knob as a function of training-set size: either a fixed value, or a
#: *size curve* -- ``[[rows, value], ..., fallback]`` -- read as "use ``value`` at or below
#: ``rows``", with the trailing entry applying above every anchor. Anchors must ascend by
#: ``rows``. This is the same declarative shape used for count-based fit callbacks, and it
#: replaces arithmetic like ``min(8, max(5, rows / 10))``: the thresholds become data that can
#: be inspected, overridden, and tested rather than an expression to re-derive.
#:
#: Examples::
#:
#:     8                            # always 8
#:     [[1_000, 5], 8]              # 5 folds up to 1k rows, 8 above
#:     [[2_000, 5], [10_000, 2], 1] # 5 repeats up to 2k rows, 2 up to 10k, 1 above
#:     [[1_000, 0.2], 0.1]          # hold out 20% up to 1k rows, 10% above
SizeCurve = "int | float | bool | None | list"

#: Defaults for the auto-selected validation method. Numerically identical to the arithmetic
#: they replaced, so behavior is unchanged until a curve is overridden.
#:
#: These curves are *policy*: what to ask for at a given data size. Feasibility is settled
#: separately and afterwards -- ``ValidationStructure.custom_splits`` reduces the fold and
#: repeat counts when the data cannot support what was asked (fewer groups than folds, a
#: stratification value rarer than the fold count, a deterministic temporal partition) and
#: returns what it used, which the caller adopts. So policy proposes and feasibility
#: reduces; neither needs to know the other's rules.
#:  - folds: max 8 because 8 cores per CPU is very common; down to 5 on small data so each
#:    validation set stays large enough to be representative.
#:  - repeats: 1. More repeats have not been observed to help, appearing to overfit the
#:    validation data; a curve is exposed so that can be revisited per size regime without
#:    editing this module.
#:  - holdout: an extra holdout only once the data is large enough that spending rows on it
#:    is cheap.
DEFAULT_VALIDATION_SIZE_CURVES: dict[str, SizeCurve] = {
    "num_bag_folds": [[59, 5], [69, 6], [79, 7], 8],
    "num_bag_sets": 1,
    "use_bag_holdout": [[USE_BAG_HOLDOUT_AUTO_THRESHOLD - 1, False], True],
    # Stack levels the data size permits. Whether stacking is used at all is a separate,
    # qualitative decision (see `get_validation_and_stacking_method`); this only says how deep
    # the size allows, so raising it to e.g. [[749, 0], [100_000, 1], 2] asks for a second
    # layer on large data without touching that decision.
    "num_stack_levels": [[749, 0], 1],
}


@dataclass(frozen=True)
class ValidationSizeCurves:
    """[EXPERIMENTAL] How the auto-selected validation method scales with data size.

    One :data:`SizeCurve` per knob; ``None`` leaves that knob at its default (see
    :data:`DEFAULT_VALIDATION_SIZE_CURVES`). Construct directly or from a dict via
    :meth:`from_input`, which rejects unknown knob names -- a typo would otherwise be
    silently ignored.

        ValidationSizeCurves(num_bag_sets=[[2_000, 5], 1])
        ValidationSizeCurves.from_input({"num_bag_sets": [[2_000, 5], 1]})  # equivalent

    ``holdout_frac`` and ``dynamic_stacking`` have no default curve, because neither built-in
    policy is a step function of size: ``holdout_frac`` is a continuous function of the row count,
    and ``dynamic_stacking`` is derived from another knob (it defaults to ``not use_bag_holdout``).
    A curve given here replaces that policy.
    """

    num_bag_folds: SizeCurve = None
    num_bag_sets: SizeCurve = None
    use_bag_holdout: SizeCurve = None
    num_stack_levels: SizeCurve = None
    holdout_frac: SizeCurve = None
    dynamic_stacking: SizeCurve = None

    @classmethod
    def from_input(cls, value: ValidationSizeCurves | dict | None) -> ValidationSizeCurves | None:
        if value is None or isinstance(value, ValidationSizeCurves):
            return value
        if isinstance(value, dict):
            valid = {f.name for f in fields(cls)}
            invalid = set(value) - valid
            if invalid:
                raise ValueError(
                    f"Invalid `validation_size_curves` keys: {sorted(invalid)}. Valid keys: {sorted(valid)}"
                )
            return cls(**value)
        raise ValueError(f"`validation_size_curves` must be a dict or ValidationSizeCurves, got: {type(value)}")

    def as_overrides(self) -> dict[str, SizeCurve]:
        """The knobs actually set, as a ``{knob: curve}`` dict; unset knobs are omitted."""
        return {f.name: getattr(self, f.name) for f in fields(self) if getattr(self, f.name) is not None}


def resolve_size_curve(curve: SizeCurve, num_train_rows: int) -> int | float | bool | None:
    """Read a :data:`SizeCurve` at ``num_train_rows``.

    A non-list ``curve`` is a fixed value. Otherwise the first ``[rows, value]`` anchor whose
    ``rows`` is >= ``num_train_rows`` wins, else the trailing fallback. A bare (non-pair)
    trailing entry is that fallback; a curve made only of anchors returns the last anchor's
    value above the final threshold.
    """
    if not isinstance(curve, list):
        return curve
    if not curve:
        raise ValueError("A size curve must not be empty.")

    fallback = None
    anchors: list[tuple[int, object]] = []
    for entry in curve:
        if isinstance(entry, (list, tuple)):
            if len(entry) != 2:
                raise ValueError(f"Size-curve anchors must be [rows, value] pairs, got: {entry!r}")
            anchors.append((entry[0], entry[1]))
        else:
            fallback = entry
    thresholds = [rows for rows, _ in anchors]
    if thresholds != sorted(thresholds):
        raise ValueError(f"Size-curve anchors must ascend by rows, got thresholds: {thresholds}")

    for rows, value in anchors:
        if num_train_rows <= rows:
            return value
    return fallback if fallback is not None or not anchors else anchors[-1][1]


def resolve_effective_sample_size(
    num_train_rows: int,
    num_group_instances: int | None = None,
    size_on_groups: bool = False,
) -> int:
    """The size the validation curves are read at.

    Rows by default. With ``size_on_groups`` and a known ``num_group_instances``, the group
    count is used instead: when rows within a group are not independent, the number of groups
    is the sample size that governs how many folds the data can support and how noisy each
    validation estimate is. The two can disagree sharply -- a benchmark task with 4,672 rows
    across 68 groups looks large by rows and small by groups -- so which is right is a
    judgement about the data, and this stays opt-in rather than inferred from the presence of
    a grouping.
    """
    if size_on_groups and num_group_instances is not None:
        return num_group_instances
    return num_train_rows


def _get_validation_preset(
    num_train_rows: int,
    hpo_enabled: bool,
    validation_size_curves: dict[str, SizeCurve] | None = None,
    num_group_instances: int | None = None,
    size_on_groups: bool = False,
) -> dict[str, int | float]:
    """Recommended validation preset, resolved from size curves at the effective sample size.

    ``validation_size_curves`` (EXPERIMENTAL: format subject to change) overrides individual entries
    of :data:`DEFAULT_VALIDATION_SIZE_CURVES`,
    so a caller can retune one knob (e.g. repeats on small data) without restating the rest.
    ``size_on_groups`` reads the curves at the group count instead of the row count -- see
    :func:`resolve_effective_sample_size`. ``holdout_frac`` is always sized on rows, since it
    is a fraction of the rows actually held out.
    """
    overrides = ValidationSizeCurves.from_input(validation_size_curves)
    curves = {**DEFAULT_VALIDATION_SIZE_CURVES, **(overrides.as_overrides() if overrides is not None else {})}
    effective_size = resolve_effective_sample_size(
        num_train_rows=num_train_rows,
        num_group_instances=num_group_instances,
        size_on_groups=size_on_groups,
    )
    resolved = {key: resolve_size_curve(curve, effective_size) for key, curve in curves.items()}
    if "holdout_frac" not in curves:
        # Not a default curve: the built-in policy is a continuous function of the row count
        # rather than a step curve, and a curve cannot reproduce it. A caller-supplied
        # `holdout_frac` curve replaces it, and is read at the effective size like the others.
        resolved["holdout_frac"] = round(
            default_holdout_frac(num_train_rows=num_train_rows, hyperparameter_tune=hpo_enabled), 4
        )
    return resolved


# TODO(refactor): use a data class for the config of the validation method.
# TODO(improvement): Implement a more sophisticated solution.
#   Could also use more metadata such as  num_features, num_models,
#   or time_limit for a heuristic.
#       num_features: The number of features in the dataset.
#       num_models: The number of models in the portfolio to fit.
#       time_limit: The time limit for fitting models.
#   Pointer for non-heuristic approach:
#       -> meta-learning like Auto-Sklearn 2.0, needs a lot of metadata
def get_validation_and_stacking_method(
    # Validation parameters
    num_bag_folds: int | None,
    num_bag_sets: int | None,
    use_bag_holdout: bool | None,
    holdout_frac: float | None,
    # Stacking/Pipeline parameters
    auto_stack: bool,
    num_stack_levels: int | None,
    dynamic_stacking: bool | None,
    refit_full: bool | None,
    # Metadata
    num_train_rows: int,
    problem_type: PROBLEM_TYPES,
    hpo_enabled: bool,
    n_samples_minority_class: int | None,
    num_group_instances: int | None = None,
    size_on_groups: bool = False,
    validation_size_curves: ValidationSizeCurves | dict[str, SizeCurve] | None = None,
) -> tuple[int, int, int, bool, bool, float, bool]:
    """Get the validation method for AutoGluon via a heuristic.

    Input variables are `None` if they were not specified by the user or have an explicit default.

    Parameters
    ----------
    num_bag_folds: int | None
        The number of folds for cross-validation.
    num_bag_sets: int | None
        The number of repeats for cross-validation.
    use_bag_holdout: bool | None
        Whether to use (additional) holdout validation.
    holdout_frac: float | None
        The fraction of data to holdout for validation.
    auto_stack: bool
        Whether to automatically determine the stacking method.
    num_stack_levels: int | None
        The number of stacking levels.
    dynamic_stacking: bool | None
        Whether to use dynamic stacking.
    refit_full: bool
        Whether to refit the full training dataset.
    num_train_rows: int
        The number of rows in the training dataset.
    problem_type: PROBLEM_TYPES
        The type of problem to solve.
    hpo_enabled: bool
        If True, HPO is enabled during the run of AutoGluon.
    n_samples_minority_class: int | None
        The number of samples in the minority class for classification problems.
        None for regression problems.
    num_group_instances: int | None
        The number of independent groups, when the data declares a grouping. None otherwise.
    size_on_groups: bool
        If True, size-dependent choices read `num_group_instances` instead of `num_train_rows`.
    validation_size_curves: ValidationSizeCurves | dict[str, SizeCurve] | None
        Per-knob overrides of `DEFAULT_VALIDATION_SIZE_CURVES`; only the knobs set are overridden.
        A dict is accepted and normalized via `ValidationSizeCurves.from_input`.

    Returns:
    --------
    Returns all variables needed to define the validation method.
    """
    cv_preset = _get_validation_preset(
        num_train_rows=num_train_rows,
        hpo_enabled=hpo_enabled,
        num_group_instances=num_group_instances,
        size_on_groups=size_on_groups,
        validation_size_curves=validation_size_curves,
    )

    # Which knobs the caller gave a curve for. Supplying one is itself a request for size-driven
    # selection of that knob, so it is honored wherever a knob would otherwise be decided by
    # `auto_stack` or derived from another knob.
    curve_overrides = ValidationSizeCurves.from_input(validation_size_curves)
    specified = set(curve_overrides.as_overrides()) if curve_overrides is not None else set()

    # Independent of `auto_stack`
    if use_bag_holdout is None:
        use_bag_holdout = cv_preset["use_bag_holdout"]
    if holdout_frac is None:
        holdout_frac = cv_preset["holdout_frac"]
    if dynamic_stacking is None:
        # Without a curve, DyStack follows from whether a bag-holdout is used; a curve replaces
        # that derivation, so it can be sized independently.
        dynamic_stacking = cv_preset["dynamic_stacking"] if "dynamic_stacking" in specified else not use_bag_holdout
    if refit_full is None:
        refit_full = False

    # Changed by `auto_stack` -- except where the caller gave a curve for the knob. Supplying a
    # curve is itself a request for size-driven selection of that knob, so `auto_stack` does not
    # get to overrule it: without this, `auto_stack=False` (the default) silently replaced an
    # explicit `num_bag_folds` curve with 0, i.e. no bagging and no out-of-fold predictions at all.
    # Knobs the caller left out still follow `auto_stack` exactly as before.
    if num_bag_folds is None:
        # `num_bag_folds == 0` -> only use holdout validation
        num_bag_folds = cv_preset["num_bag_folds"] if (auto_stack or "num_bag_folds" in specified) else 0
    if num_bag_sets is None:
        # `num_bag_sets == 1` -> no repeats
        num_bag_sets = cv_preset["num_bag_sets"] if (auto_stack or "num_bag_sets" in specified) else 1
    if num_stack_levels is None:
        # Disable multi-layer stacking by default
        num_stack_levels = 0
        # How deep the data size permits; the conditions below decide whether to stack at all.
        stack_levels_by_size = cv_preset["num_stack_levels"]

        if "num_stack_levels" in specified:
            # An explicit curve is the answer, not an input to the auto_stack conditions below.
            num_stack_levels = stack_levels_by_size
        elif auto_stack and dynamic_stacking:
            # Dynamic stacking detects stacked overfitting itself, so it is not size-gated.
            num_stack_levels = max(1, stack_levels_by_size)
        elif auto_stack and (use_bag_holdout or (problem_type != BINARY)):
            # Holdout validation or a non-binary problem: stack as deep as the size allows.
            num_stack_levels = stack_levels_by_size

    # Extra logic to handle cross-validation splits for classification
    #   - Avoid failure mode where we do not have enough samples to ensure the
    #    minority class is represented in each fold.
    #   - The failure mode only triggers if we use at least two folds.
    # FIXME:
    #   - This will still crash some models that need an extra holdout split (?)
    #   - Maybe it is better to just switch to no validation in some cases like this (?)
    if (n_samples_minority_class is not None) and (num_bag_folds >= 2):
        # 1 sample train, 1 sample test
        min_samples_per_class = 2

        # For dynamic stacking and use_bag_holdout, we need an extra sample
        # for validation outside of stacking.
        extra_holdout_set = dynamic_stacking or use_bag_holdout

        if extra_holdout_set:
            min_samples_per_class += 1

        # TODO: up-sample instead of raising an error?
        # Raise error in unrecoverable failure mode
        if n_samples_minority_class < min_samples_per_class:
            raise ValueError(
                "Number of samples per class must be >= minimum number of samples per class. "
                f"Got: {n_samples_minority_class} samples, need {min_samples_per_class}."
            )

        supported_num_bag_folds = n_samples_minority_class
        if extra_holdout_set:
            supported_num_bag_folds -= 1

        # num_bag_folds must be 0 or >= 2; clamp 1 down to 0
        supported_num_stack_levels = num_stack_levels
        if supported_num_bag_folds < 2:
            supported_num_bag_folds = 0
            supported_num_stack_levels = 0

        if supported_num_bag_folds < num_bag_folds:
            warnings.warn(
                f"Number of samples in minority class is {n_samples_minority_class}, "
                f"which is less than the requested number of folds {num_bag_folds}. "
                f"\n\tSetting num_bag_folds to {supported_num_bag_folds} to enable cross-validation."
                f"\n\tAccounting for an extra holdout set: {extra_holdout_set}."
                f"\n\tAdjusting stacking levels from {num_stack_levels} to {supported_num_stack_levels}.",
                UserWarning,
                stacklevel=2,
            )
            num_bag_folds = supported_num_bag_folds
            num_stack_levels = supported_num_stack_levels

    return (
        num_bag_folds,
        num_bag_sets,
        num_stack_levels,
        dynamic_stacking,
        use_bag_holdout,
        holdout_frac,
        refit_full,
    )
