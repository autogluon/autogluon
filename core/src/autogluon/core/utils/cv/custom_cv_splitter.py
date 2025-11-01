"""Custom cross-validation splitter for AutoGluon."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator

logger = logging.getLogger(__name__)


class CustomCVSplitter(BaseCrossValidator):
    """
    Custom cross-validation splitter using user-defined fold matrix.

    This splitter allows users to define custom train/test splits via a matrix,
    enabling advanced CV strategies like forward chaining for time series,
    sliding windows, or any custom splitting logic.

    Parameters
    ----------
    cv_matrix : pd.DataFrame or np.ndarray
        Matrix where each column represents a fold and each row represents a sample.

        Values:
            - 0: Sample is in training set for this fold
            - 1: Sample is in test set for this fold
            - 2: Sample is excluded from this fold (neither train nor test)

        Example (forward chaining for 4 time periods, 3 folds):
            ```python
            cv_matrix = pd.DataFrame({
                'fold_1': [0, 1, 2, 2],  # train period 1, test period 2
                'fold_2': [0, 0, 1, 2],  # train periods 1-2, test period 3
                'fold_3': [0, 0, 0, 1],  # train periods 1-3, test period 4
            })
            ```

    Attributes
    ----------
    cv_matrix : pd.DataFrame
        The validated and processed CV matrix
    n_splits : int
        Number of folds (columns in cv_matrix)

    Examples
    --------
    >>> import pandas as pd
    >>> from autogluon.core.utils.cv import CustomCVSplitter
    >>>
    >>> # Create forward chaining CV matrix
    >>> cv_matrix = pd.DataFrame({
    ...     'fold_1': [0, 1, 2],
    ...     'fold_2': [0, 0, 1],
    ... })
    >>>
    >>> # Create splitter
    >>> splitter = CustomCVSplitter(cv_matrix)
    >>>
    >>> # Generate splits
    >>> X = pd.DataFrame({'a': [1, 2, 3]})
    >>> y = pd.Series([0, 1, 0])
    >>> for train_idx, test_idx in splitter.split(X, y):
    ...     print(f"Train: {train_idx}, Test: {test_idx}")
    Train: [0], Test: [1]
    Train: [0 1], Test: [2]
    """

    def __init__(self, cv_matrix: pd.DataFrame | np.ndarray):
        """
        Initialize CustomCVSplitter.

        Parameters
        ----------
        cv_matrix : pd.DataFrame or np.ndarray
            CV matrix defining train/test splits for each fold

        Raises
        ------
        ValueError
            If cv_matrix contains invalid values or structure
        """
        self.cv_matrix = self._validate_and_process_matrix(cv_matrix)
        self.n_splits = self.cv_matrix.shape[1]

    def _validate_and_process_matrix(self, cv_matrix: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        """
        Validate and convert cv_matrix to standard format.

        Parameters
        ----------
        cv_matrix : pd.DataFrame or np.ndarray
            Input CV matrix

        Returns
        -------
        pd.DataFrame
            Validated CV matrix as DataFrame

        Raises
        ------
        ValueError
            If matrix has invalid structure or values
        """
        # Convert to DataFrame if necessary
        if isinstance(cv_matrix, np.ndarray):
            n_folds = cv_matrix.shape[1] if cv_matrix.ndim == 2 else 1
            cv_matrix = pd.DataFrame(
                cv_matrix,
                columns=[f"fold_{i+1}" for i in range(n_folds)]
            )
        elif not isinstance(cv_matrix, pd.DataFrame):
            raise TypeError(
                f"cv_matrix must be pd.DataFrame or np.ndarray, got {type(cv_matrix)}"
            )

        # Check for empty matrix
        if cv_matrix.empty:
            raise ValueError("cv_matrix cannot be empty")

        # Check for at least one fold
        if cv_matrix.shape[1] == 0:
            raise ValueError("cv_matrix must have at least one fold (column)")

        # Validate values are 0, 1, or 2
        unique_values = set()
        for col in cv_matrix.columns:
            unique_values.update(cv_matrix[col].unique())

        invalid_values = unique_values - {0, 1, 2}
        if invalid_values:
            raise ValueError(
                f"cv_matrix contains invalid values: {invalid_values}. "
                f"Valid values are 0 (train), 1 (test), 2 (exclude)"
            )

        # Validate each fold has at least one train and one test sample
        for col_idx, col in enumerate(cv_matrix.columns):
            fold_values = cv_matrix[col].values

            n_train = np.sum(fold_values == 0)
            n_test = np.sum(fold_values == 1)

            if n_train == 0:
                raise ValueError(
                    f"Fold '{col}' (index {col_idx}) has no training samples. "
                    f"Each fold must have at least one sample with value 0."
                )

            if n_test == 0:
                raise ValueError(
                    f"Fold '{col}' (index {col_idx}) has no test samples. "
                    f"Each fold must have at least one sample with value 1."
                )

        return cv_matrix

    def split(self, X, y=None, groups=None):
        """
        Generate train/test indices for each fold.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data. Only used to check alignment with cv_matrix.
        y : array-like of shape (n_samples,), default=None
            Target variable. Ignored by this splitter.
        groups : array-like of shape (n_samples,), default=None
            Group labels. Ignored by this splitter.

        Yields
        ------
        train_index : np.ndarray
            Array of indices for training set
        test_index : np.ndarray
            Array of indices for test set

        Raises
        ------
        ValueError
            If X has different number of samples than cv_matrix
        """
        n_samples = len(X) if hasattr(X, '__len__') else X.shape[0]

        if n_samples != len(self.cv_matrix):
            raise ValueError(
                f"Number of samples in X ({n_samples}) does not match "
                f"number of rows in cv_matrix ({len(self.cv_matrix)}). "
                f"Ensure cv_matrix has one row per sample in your dataset."
            )

        # Generate splits for each fold
        for fold_idx in range(self.n_splits):
            fold_col = self.cv_matrix.iloc[:, fold_idx].values

            # Get train indices (value == 0)
            train_index = np.where(fold_col == 0)[0]

            # Get test indices (value == 1)
            test_index = np.where(fold_col == 1)[0]

            yield train_index, test_index

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """
        Returns the number of splitting iterations in the cross-validator.

        Parameters
        ----------
        X : object, default=None
            Always ignored, exists for compatibility.
        y : object, default=None
            Always ignored, exists for compatibility.
        groups : object, default=None
            Always ignored, exists for compatibility.

        Returns
        -------
        n_splits : int
            Number of folds
        """
        return self.n_splits

    def get_fold_stats(self) -> pd.DataFrame:
        """
        Get statistics about each fold.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
                - fold: Fold name/number
                - n_train: Number of training samples
                - n_test: Number of test samples
                - n_exclude: Number of excluded samples
                - train_pct: Percentage of samples used for training
                - test_pct: Percentage of samples used for testing
        """
        stats = []
        n_samples = len(self.cv_matrix)

        for col in self.cv_matrix.columns:
            fold_values = self.cv_matrix[col].values

            n_train = np.sum(fold_values == 0)
            n_test = np.sum(fold_values == 1)
            n_exclude = np.sum(fold_values == 2)

            stats.append({
                'fold': col,
                'n_train': n_train,
                'n_test': n_test,
                'n_exclude': n_exclude,
                'train_pct': 100.0 * n_train / n_samples,
                'test_pct': 100.0 * n_test / n_samples,
            })

        return pd.DataFrame(stats)

    def get_sample_coverage(self) -> dict:
        """
        Get statistics about sample coverage across folds.

        Returns
        -------
        dict
            Dictionary with keys:
                - tested_samples: Set of sample indices that appear in at least one test fold
                - never_tested: Set of sample indices that never appear in test folds
                - test_counts: pd.Series showing how many times each sample was tested
                - coverage_pct: Percentage of samples that were tested at least once
        """
        n_samples = len(self.cv_matrix)
        test_counts = np.zeros(n_samples, dtype=int)

        # Count how many times each sample appears in test sets
        for fold_idx in range(self.n_splits):
            fold_col = self.cv_matrix.iloc[:, fold_idx].values
            test_mask = (fold_col == 1)
            test_counts[test_mask] += 1

        tested_samples = set(np.where(test_counts > 0)[0])
        never_tested = set(np.where(test_counts == 0)[0])
        coverage_pct = 100.0 * len(tested_samples) / n_samples

        return {
            'tested_samples': tested_samples,
            'never_tested': never_tested,
            'test_counts': pd.Series(test_counts, index=self.cv_matrix.index),
            'coverage_pct': coverage_pct,
        }

    def __repr__(self) -> str:
        """String representation of the splitter."""
        return f"CustomCVSplitter(n_splits={self.n_splits}, n_samples={len(self.cv_matrix)})"
