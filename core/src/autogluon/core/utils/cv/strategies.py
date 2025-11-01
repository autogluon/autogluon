"""Pre-built cross-validation strategies for time series and temporal data."""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def create_custom_cv_from_indices(
    train_indices_per_fold: Sequence[Sequence[int]],
    test_indices_per_fold: Sequence[Sequence[int]],
    n_samples: int,
) -> pd.DataFrame:
    """
    Create custom CV matrix from manually specified train/test indices.

    This function provides a convenient way to create custom cross-validation
    splits when you want full control over which samples go into which fold.

    Parameters
    ----------
    train_indices_per_fold : Sequence of Sequences of ints
        List of training indices for each fold.
        Each inner sequence contains the indices of samples to use for training in that fold.
        Example: [[0, 1, 2], [0, 1, 2, 3, 4], [5, 6, 7, 8]]
    test_indices_per_fold : Sequence of Sequences of ints
        List of test indices for each fold.
        Each inner sequence contains the indices of samples to use for testing in that fold.
        Example: [[3, 4], [5, 6], [9, 10]]
    n_samples : int
        Total number of samples in the dataset.
        Must be large enough to contain all specified indices.

    Returns
    -------
    pd.DataFrame
        CV matrix with shape (n_samples, n_folds) where n_folds = len(train_indices_per_fold)
        Values: 0=train, 1=test, 2=exclude

    Raises
    ------
    ValueError
        If train_indices_per_fold and test_indices_per_fold have different lengths
        If any index is >= n_samples or < 0
        If any index appears in both train and test for the same fold
        If any fold has no train or test samples

    Examples
    --------
    >>> from autogluon.core.utils.cv import create_custom_cv_from_indices
    >>>
    >>> # Create custom CV with non-contiguous splits
    >>> cv_matrix = create_custom_cv_from_indices(
    ...     train_indices_per_fold=[
    ...         [0, 1, 2, 3, 4, 5],              # fold 1 train
    ...         [0, 1, 2, 3, 4, 5, 6, 7, 8],     # fold 2 train (more data)
    ...         [10, 11, 12, 13, 14],            # fold 3 train (skip samples!)
    ...     ],
    ...     test_indices_per_fold=[
    ...         [6, 7, 8, 9, 10],                # fold 1 test
    ...         [20, 21, 22, 23],                # fold 2 test
    ...         [50, 51, 52],                    # fold 3 test
    ...     ],
    ...     n_samples=100
    ... )
    >>> print(cv_matrix.shape)
    (100, 3)

    >>> # Use with CustomCVSplitter
    >>> from autogluon.core.utils.cv import CustomCVSplitter
    >>> splitter = CustomCVSplitter(cv_matrix)
    >>> for train_idx, test_idx in splitter.split(X, y):
    ...     # Your training logic here
    ...     pass
    """
    # Validate inputs
    if len(train_indices_per_fold) != len(test_indices_per_fold):
        raise ValueError(
            f"train_indices_per_fold and test_indices_per_fold must have the same length. "
            f"Got {len(train_indices_per_fold)} and {len(test_indices_per_fold)}"
        )

    n_folds = len(train_indices_per_fold)

    if n_folds == 0:
        raise ValueError("Must specify at least one fold")

    if n_samples < 1:
        raise ValueError(f"n_samples must be at least 1, got {n_samples}")

    # Initialize matrix with all excluded (2)
    cv_matrix = np.full((n_samples, n_folds), 2, dtype=int)

    # Fill in train and test indices for each fold
    for fold_idx in range(n_folds):
        train_idx = list(train_indices_per_fold[fold_idx])
        test_idx = list(test_indices_per_fold[fold_idx])

        # Validate indices
        if len(train_idx) == 0:
            raise ValueError(
                f"Fold {fold_idx + 1} has no training samples. "
                f"Each fold must have at least one training sample."
            )

        if len(test_idx) == 0:
            raise ValueError(
                f"Fold {fold_idx + 1} has no test samples. "
                f"Each fold must have at least one test sample."
            )

        # Check for out-of-bounds indices
        all_indices = train_idx + test_idx
        if any(idx < 0 for idx in all_indices):
            raise ValueError(
                f"Fold {fold_idx + 1} contains negative indices. "
                f"All indices must be >= 0."
            )

        if any(idx >= n_samples for idx in all_indices):
            max_idx = max(all_indices)
            raise ValueError(
                f"Fold {fold_idx + 1} contains index {max_idx} which is >= n_samples ({n_samples}). "
                f"All indices must be < n_samples."
            )

        # Check for overlap between train and test
        train_set = set(train_idx)
        test_set = set(test_idx)
        overlap = train_set & test_set

        if overlap:
            raise ValueError(
                f"Fold {fold_idx + 1} has samples appearing in both train and test sets: {sorted(overlap)}. "
                f"A sample cannot be in both train and test for the same fold."
            )

        # Assign values
        cv_matrix[train_idx, fold_idx] = 0  # train
        cv_matrix[test_idx, fold_idx] = 1   # test

    # Convert to DataFrame with named columns
    cv_matrix_df = pd.DataFrame(
        cv_matrix,
        columns=[f"fold_{i+1}" for i in range(n_folds)]
    )

    return cv_matrix_df


def forward_chaining_cv(
    n_samples: int,
    n_folds: int,
    min_train_size: int | None = None,
    gap: int = 0,
) -> pd.DataFrame:
    """
    Generate forward chaining cross-validation matrix.

    Forward chaining (also called "walk-forward" or "rolling origin") ensures
    no data leakage in time series by always training on past data and testing
    on future data.

    Fold pattern:
        - Fold 1: train[0:t1], test[t1:t2]
        - Fold 2: train[0:t2], test[t2:t3]
        - Fold 3: train[0:t3], test[t3:t4]
        - ...

    Parameters
    ----------
    n_samples : int
        Total number of samples in the dataset
    n_folds : int
        Number of folds to create
    min_train_size : int, optional
        Minimum number of samples in the first training set.
        If None, uses approximately n_samples / (n_folds + 1)
    gap : int, default=0
        Number of samples to exclude between train and test sets.
        Useful to account for temporal lag or data collection delays.

    Returns
    -------
    pd.DataFrame
        CV matrix with shape (n_samples, n_folds)
        Values: 0=train, 1=test, 2=exclude

    Examples
    --------
    >>> from autogluon.core.utils.cv import forward_chaining_cv
    >>>
    >>> # Create forward chaining CV for 100 samples, 5 folds
    >>> cv_matrix = forward_chaining_cv(n_samples=100, n_folds=5)
    >>> print(cv_matrix.shape)
    (100, 5)
    >>>
    >>> # With gap to exclude samples between train/test
    >>> cv_matrix = forward_chaining_cv(n_samples=100, n_folds=5, gap=5)

    Raises
    ------
    ValueError
        If parameters result in invalid fold configuration
    """
    if n_samples < n_folds + 1:
        raise ValueError(
            f"n_samples ({n_samples}) must be greater than n_folds ({n_folds}) "
            f"to create meaningful train/test splits"
        )

    if n_folds < 1:
        raise ValueError(f"n_folds must be at least 1, got {n_folds}")

    if gap < 0:
        raise ValueError(f"gap must be non-negative, got {gap}")

    # Calculate test set size for each fold
    # Reserve enough samples so each fold has a test set
    if min_train_size is None:
        # Auto-calculate: distribute samples across folds
        # Leave enough for training in first fold and testing in all folds
        min_train_size = n_samples // (n_folds + 1)

    # Calculate test set size
    remaining_samples = n_samples - min_train_size
    test_size = remaining_samples // n_folds

    if test_size < 1:
        raise ValueError(
            f"Not enough samples to create {n_folds} folds. "
            f"With min_train_size={min_train_size}, cannot create valid test sets. "
            f"Try reducing n_folds or min_train_size."
        )

    # Verify gap is reasonable
    if gap >= test_size:
        logger.warning(
            f"gap ({gap}) is >= test_size ({test_size}), which may result in very small test sets"
        )

    # Initialize matrix with all excluded (value 2)
    cv_matrix = np.full((n_samples, n_folds), 2, dtype=int)

    # Fill in train/test splits for each fold
    for fold_idx in range(n_folds):
        # Calculate split points
        train_end = min_train_size + (fold_idx * test_size)
        gap_end = train_end + gap
        test_start = gap_end
        test_end = min(test_start + test_size, n_samples)

        # Assign train samples (0)
        cv_matrix[:train_end, fold_idx] = 0

        # Samples in gap are excluded (already 2)

        # Assign test samples (1)
        cv_matrix[test_start:test_end, fold_idx] = 1

    # Convert to DataFrame with named columns
    cv_matrix_df = pd.DataFrame(
        cv_matrix,
        columns=[f"fold_{i+1}" for i in range(n_folds)]
    )

    return cv_matrix_df


def sliding_window_cv(
    n_samples: int,
    n_folds: int,
    window_size: int | None = None,
    step_size: int | None = None,
    gap: int = 0,
) -> pd.DataFrame:
    """
    Generate sliding window cross-validation matrix.

    Sliding window maintains a fixed-size training window that slides forward
    over time. This is useful for time series where only recent history is
    relevant.

    Fold pattern:
        - Fold 1: train[0:w], test[w:w+s]
        - Fold 2: train[s:w+s], test[w+s:w+2s]
        - Fold 3: train[2s:w+2s], test[w+2s:w+3s]
        - ...

    Parameters
    ----------
    n_samples : int
        Total number of samples in the dataset
    n_folds : int
        Number of folds to create
    window_size : int, optional
        Size of the training window.
        If None, uses approximately 2 * (n_samples / n_folds)
    step_size : int, optional
        Step size for sliding the window forward.
        If None, uses approximately (n_samples - window_size) / n_folds
    gap : int, default=0
        Number of samples to exclude between train and test sets

    Returns
    -------
    pd.DataFrame
        CV matrix with shape (n_samples, n_folds)
        Values: 0=train, 1=test, 2=exclude

    Examples
    --------
    >>> from autogluon.core.utils.cv import sliding_window_cv
    >>>
    >>> # Create sliding window CV with auto-sized window
    >>> cv_matrix = sliding_window_cv(n_samples=100, n_folds=5)
    >>>
    >>> # With custom window and step sizes
    >>> cv_matrix = sliding_window_cv(
    ...     n_samples=100,
    ...     n_folds=5,
    ...     window_size=30,
    ...     step_size=15
    ... )

    Raises
    ------
    ValueError
        If parameters result in invalid fold configuration
    """
    if n_samples < n_folds + 1:
        raise ValueError(
            f"n_samples ({n_samples}) must be greater than n_folds ({n_folds})"
        )

    if n_folds < 1:
        raise ValueError(f"n_folds must be at least 1, got {n_folds}")

    if gap < 0:
        raise ValueError(f"gap must be non-negative, got {gap}")

    # Auto-calculate window_size if not provided
    if window_size is None:
        # Use roughly 2/3 of samples for training
        window_size = int(2 * n_samples / (n_folds + 1))

    if window_size < 1:
        raise ValueError(f"window_size must be at least 1, got {window_size}")

    if window_size >= n_samples:
        raise ValueError(
            f"window_size ({window_size}) must be less than n_samples ({n_samples})"
        )

    # Auto-calculate step_size if not provided
    if step_size is None:
        # Distribute remaining samples across folds
        remaining = n_samples - window_size
        step_size = max(1, remaining // n_folds)

    if step_size < 1:
        raise ValueError(f"step_size must be at least 1, got {step_size}")

    # Initialize matrix with all excluded (value 2)
    cv_matrix = np.full((n_samples, n_folds), 2, dtype=int)

    # Fill in train/test splits for each fold
    for fold_idx in range(n_folds):
        # Calculate window position
        window_start = fold_idx * step_size
        window_end = window_start + window_size

        if window_end > n_samples:
            logger.warning(
                f"Fold {fold_idx + 1}: Training window exceeds n_samples, truncating. "
                f"Consider reducing window_size or step_size."
            )
            window_end = n_samples

        # Calculate test set position
        gap_end = window_end + gap
        test_start = gap_end
        test_end = min(test_start + step_size, n_samples)

        if test_start >= n_samples:
            raise ValueError(
                f"Fold {fold_idx + 1}: No samples available for test set. "
                f"Consider reducing window_size, step_size, or gap."
            )

        # Assign train samples (0)
        cv_matrix[window_start:window_end, fold_idx] = 0

        # Samples in gap are excluded (already 2)

        # Assign test samples (1)
        if test_start < n_samples:
            cv_matrix[test_start:test_end, fold_idx] = 1

    # Validate that all folds have at least one test sample
    for fold_idx in range(n_folds):
        if np.sum(cv_matrix[:, fold_idx] == 1) == 0:
            raise ValueError(
                f"Fold {fold_idx + 1} has no test samples. "
                f"Adjust window_size, step_size, or gap parameters."
            )

    # Convert to DataFrame with named columns
    cv_matrix_df = pd.DataFrame(
        cv_matrix,
        columns=[f"fold_{i+1}" for i in range(n_folds)]
    )

    return cv_matrix_df


def time_series_cv(
    dates: pd.Series,
    n_folds: int,
    strategy: str = 'forward_chaining',
    gap: int | pd.Timedelta | None = None,
    **kwargs
) -> pd.DataFrame:
    """
    Generate time-series cross-validation matrix from datetime column.

    This is a convenience function that creates CV matrices aligned with
    a datetime series, automatically handling temporal ordering and gaps.

    Parameters
    ----------
    dates : pd.Series
        DateTime series indicating temporal order of samples.
        Will be sorted before creating CV matrix.
    n_folds : int
        Number of folds to create
    strategy : str, default='forward_chaining'
        CV strategy to use. Options:
            - 'forward_chaining': Expanding training window
            - 'sliding_window': Fixed-size training window
    gap : int, pd.Timedelta, or None, default=None
        Gap between train and test sets.
        If int, interpreted as number of samples.
        If pd.Timedelta, converted to number of samples based on dates.
        If None, no gap is used.
    **kwargs
        Additional arguments passed to the strategy function
        (e.g., min_train_size, window_size, step_size)

    Returns
    -------
    pd.DataFrame
        CV matrix aligned with dates.index
        Values: 0=train, 1=test, 2=exclude

    Examples
    --------
    >>> import pandas as pd
    >>> from autogluon.core.utils.cv import time_series_cv
    >>>
    >>> # Create datetime series
    >>> dates = pd.date_range('2020-01-01', periods=100, freq='D')
    >>>
    >>> # Generate forward chaining CV
    >>> cv_matrix = time_series_cv(dates, n_folds=5, strategy='forward_chaining')
    >>>
    >>> # With timedelta gap
    >>> cv_matrix = time_series_cv(
    ...     dates,
    ...     n_folds=5,
    ...     gap=pd.Timedelta(days=7)
    ... )

    Raises
    ------
    ValueError
        If strategy is not recognized or parameters are invalid
    """
    if not isinstance(dates, pd.Series):
        raise TypeError(f"dates must be pd.Series, got {type(dates)}")

    if strategy not in ['forward_chaining', 'sliding_window']:
        raise ValueError(
            f"Invalid strategy '{strategy}'. "
            f"Valid options: 'forward_chaining', 'sliding_window'"
        )

    # Sort dates to ensure temporal order
    dates_sorted = dates.sort_values()
    n_samples = len(dates_sorted)

    # Convert timedelta gap to number of samples if needed
    gap_samples = 0
    if gap is not None:
        if isinstance(gap, pd.Timedelta):
            # Calculate median time step
            time_diffs = dates_sorted.diff().dropna()
            if len(time_diffs) > 0:
                median_step = time_diffs.median()
                if median_step.total_seconds() > 0:
                    gap_samples = int(gap / median_step)
                else:
                    logger.warning("Cannot convert timedelta gap to samples (median step is 0), using gap=0")
                    gap_samples = 0
            else:
                logger.warning("Cannot convert timedelta gap to samples (only 1 sample), using gap=0")
                gap_samples = 0
        elif isinstance(gap, int):
            gap_samples = gap
        else:
            raise TypeError(
                f"gap must be int or pd.Timedelta, got {type(gap)}"
            )

    # Generate CV matrix using selected strategy
    if strategy == 'forward_chaining':
        cv_matrix = forward_chaining_cv(
            n_samples=n_samples,
            n_folds=n_folds,
            gap=gap_samples,
            **kwargs
        )
    else:  # sliding_window
        cv_matrix = sliding_window_cv(
            n_samples=n_samples,
            n_folds=n_folds,
            gap=gap_samples,
            **kwargs
        )

    # Align CV matrix index with original dates index (respecting sort order)
    cv_matrix.index = dates_sorted.index
    # Reorder to match original dates index
    cv_matrix = cv_matrix.loc[dates.index]

    return cv_matrix
