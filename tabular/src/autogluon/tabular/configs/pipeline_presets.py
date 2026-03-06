from __future__ import annotations

import math
import warnings

from autogluon.core.constants import BINARY, PROBLEM_TYPES
from autogluon.core.utils.utils import default_holdout_frac

USE_BAG_HOLDOUT_AUTO_THRESHOLD = 1_000_000


def _get_validation_preset(num_train_rows: int, hpo_enabled: bool) -> dict[str, int | float]:
    """Recommended validation preset manually defined by the AutoGluon developers."""

    # -- Default recommendation
    #  max 8 due to 8 cores per CPU being very common.
    #  down to 5 folds for small datasets to have enough samples for a representative validation set.
    num_bag_folds = min(8, max(5, math.floor(num_train_rows / 10)))

    num_bag_sets = 1  # More repeats do not seem to help due to overfitting on val data.
    use_bag_holdout = num_train_rows >= USE_BAG_HOLDOUT_AUTO_THRESHOLD
    holdout_frac = round(default_holdout_frac(num_train_rows=num_train_rows, hyperparameter_tune=hpo_enabled), 4)

    return dict(
        num_bag_sets=num_bag_sets,
        num_bag_folds=num_bag_folds,
        use_bag_holdout=use_bag_holdout,
        holdout_frac=holdout_frac,
    )


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

    Returns:
    --------
    Returns all variables needed to define the validation method.
    """

    cv_preset = _get_validation_preset(num_train_rows=num_train_rows, hpo_enabled=hpo_enabled)

    # Independent of `auto_stack`
    if use_bag_holdout is None:
        use_bag_holdout = cv_preset["use_bag_holdout"]
    if holdout_frac is None:
        holdout_frac = cv_preset["holdout_frac"]
    if dynamic_stacking is None:
        dynamic_stacking = not use_bag_holdout
    if refit_full is None:
        refit_full = False

    # Changed by `auto_stack`
    if num_bag_folds is None:
        # `num_bag_folds == 0` -> only use holdout validation
        num_bag_folds = cv_preset["num_bag_folds"] if auto_stack else 0
    if num_bag_sets is None:
        # `num_bag_sets == 1` -> no repeats
        num_bag_sets = cv_preset["num_bag_sets"] if auto_stack else 1
    if num_stack_levels is None:
        # Disable multi-layer stacking by default
        num_stack_levels = 0

        # Activate multi-layer stacking for `auto_stack` if
        if auto_stack and (
            dynamic_stacking  # -> We use dynamic stacking
            or
            # -> We have holdout validation or a non-binary problem with more than 750 training rows
            ((use_bag_holdout or (problem_type != BINARY)) and (num_train_rows >= 750))
        ):
            num_stack_levels = 1


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
                "Number of samples per class must be >= minimum number of samples per class."
                f"Got: {n_samples_minority_class} samples, need {min_samples_per_class}."
            )

        supported_num_bag_folds = n_samples_minority_class
        if supported_num_bag_folds:
            supported_num_bag_folds -= 1

        if supported_num_bag_folds < num_bag_folds:
            warnings.warn(
                f"Number of samples in minority class is {n_samples_minority_class}, "
                f"which is less than the requested number of folds {num_bag_folds}. "
                f"\n\tSetting num_bag_folds to {supported_num_bag_folds} to enable cross-validation."
                f"\n\tAccounting for an extra holdout set: {extra_holdout_set}.",
                UserWarning,
            )
            num_bag_folds = supported_num_bag_folds

    return (
        num_bag_folds,
        num_bag_sets,
        num_stack_levels,
        dynamic_stacking,
        use_bag_holdout,
        holdout_frac,
        refit_full,
    )
