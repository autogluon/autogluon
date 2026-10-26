import warnings

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .abstract import AbstractFeatureGenerator


class SpearmanFeatureSelector(AbstractFeatureGenerator):
    """Select features based on Spearman correlation with the target.

    Parameters
    ----------
    threshold : float, default=None
        Features with absolute Spearman correlation below this threshold will be removed.
        Ignored if None.
    max_features : int, default=2000
        Keep at most this many features, taking those with the largest absolute correlation.
    preserve_order : bool, default=True
        Emit the selected features in their input order rather than sorted by descending
        absolute correlation. The selected set is identical either way; only the column order
        differs. Sorting conveys no information to a downstream model that treats its columns
        symmetrically, while reordering is not free: models that subsample or window over
        columns see a different view, and the reordering hides from the caller that the frame
        came back permuted. Preserving the order also lets the selector skip its work entirely
        whenever the input cannot lose a column, which is the common case when the generator
        feeding it emits fewer features than ``max_features``. Pass ``False`` for the previous
        correlation-sorted output.
    """

    def __init__(
        self,
        threshold: float = None,
        max_features: int = 2000,
        preserve_order: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if threshold is None and max_features is None:
            raise ValueError("Either 'threshold' or 'max_features' must be provided.")
        self.threshold = threshold
        self.max_features = max_features
        self.preserve_order = preserve_order
        self.selected_features_ = []

    def _fit(self, X, y):
        # TODO: Add option for AUC for binary
        # TODO: Properly handle multiclass targets
        # Spearman correlation is undefined for constant (<= 1 distinct non-null value) columns:
        # scipy returns NaN and emits a ConstantInputWarning. Drop such columns up front so they
        # are excluded cleanly without the warning -- the `.dropna()` below would drop them anyway.
        X = X.loc[:, self._is_non_constant(X)]

        # Nothing can be dropped: with no threshold and at most `max_features` columns left, the
        # correlation cannot change the selection, only its order. Skipping it avoids ranking
        # every column against the target for no effect -- the dominant cost of this generator on
        # wide inputs. Only valid when the order is being preserved, since otherwise the output
        # order is itself derived from the correlation.
        if (
            self.preserve_order
            and self.threshold is None
            and self.max_features is not None
            and X.shape[1] <= self.max_features
        ):
            self.selected_features_ = list(X.columns)
            return

        abs_corr = self._abs_spearman(X, y).sort_values(ascending=False).dropna()

        if self.threshold is not None:
            selected = abs_corr[abs_corr >= self.threshold].index
        else:
            selected = abs_corr.head(self.max_features).index

        if self.preserve_order:
            selected = set(selected)
            self.selected_features_ = [c for c in X.columns if c in selected]
        else:
            self.selected_features_ = list(selected)

    @staticmethod
    def _is_non_constant(X: pd.DataFrame) -> np.ndarray:
        """Boolean mask of columns with more than one distinct non-null value.

        Equivalent to ``X.nunique(dropna=True) > 1`` for numeric input, but compares the min and
        max instead of hashing every value, which is an order of magnitude cheaper on wide frames.
        Non-numeric input falls back to ``nunique``.
        """
        numeric = X.select_dtypes(include=np.number)
        if numeric.shape[1] != X.shape[1]:
            return (X.nunique(dropna=True) > 1).to_numpy()
        non_constant = np.zeros(numeric.shape[1], dtype=bool)
        dtypes = numeric.dtypes
        for dtype in dtypes.unique():
            columns = (dtypes == dtype).to_numpy()
            block = numeric if columns.all() else numeric.loc[:, columns]
            if not isinstance(dtype, np.dtype):
                non_constant[columns] = (block.nunique(dropna=True) > 1).to_numpy()
                continue
            # Keep integer blocks in their original dtype: conversion to float64 can merge
            # distinct values above 2**53, incorrectly discarding an informative feature.
            values = block.to_numpy(copy=False)
            with warnings.catch_warnings():
                # All-null float columns produce NaN extrema and must be excluded explicitly.
                warnings.simplefilter("ignore", RuntimeWarning)
                low = np.nanmin(values, axis=0)
                high = np.nanmax(values, axis=0)
            non_constant[columns] = (low != high) & ~np.isnan(low)
        return non_constant

    @staticmethod
    def _abs_spearman(X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """``X.corrwith(y, method="spearman").abs()``, computed column-wise in bulk.

        ``corrwith`` dispatches to ``Series.corr`` per column, which re-ranks the target for every
        column in turn. Instead the feature ranks are taken in one pass, and the target is ranked
        once per *distinct null pattern*: a pairwise-complete correlation only depends on the
        column's null mask, and generated features inherit their parents' nulls, so wide frames
        carry far fewer distinct masks than columns.

        Matches ``corrwith`` for ordinary numeric input, while preserving large integer ranks.
        Non-numeric input falls back to it.
        """
        numeric = X.select_dtypes(include=np.number)
        if numeric.shape[1] != X.shape[1]:
            return X.corrwith(y, method="spearman").abs()

        target = y.to_numpy(copy=False)
        valid = numeric.notna().to_numpy()
        dtypes = numeric.dtypes
        unique_dtypes = dtypes.unique()
        if len(unique_dtypes) == 1 and isinstance(unique_dtypes[0], np.dtype):
            # Avoid allocating a second full rank matrix on homogeneous numpy frames.
            ranks = numeric.rank().to_numpy(dtype=float, copy=False)
        else:
            ranks = np.full(numeric.shape, np.nan)
            for dtype in unique_dtypes:
                columns = (dtypes == dtype).to_numpy()
                if isinstance(dtype, np.dtype):
                    # Ranking a mixed int/float frame converts it to float64 first, merging large
                    # integer values. Homogeneous numpy blocks retain their native precision.
                    block = numeric.loc[:, columns]
                    ranks[:, columns] = block.rank().to_numpy(dtype=float, copy=False)
                else:
                    # Nullable arrays need their missing values removed before conversion; converting
                    # an Int64 column with pd.NA directly to numpy can also coerce it to float64.
                    for column in np.flatnonzero(columns):
                        mask = valid[:, column]
                        ranks[mask, column] = rankdata(numeric.iloc[:, column].dropna().to_numpy(copy=False))
        # Each feature's non-null ranks are its ranks within the pairwise-complete subset, since
        # target labels have no nulls. Rank the target below in its original dtype for the same reason.

        # Group columns by null pattern. Bit-packing and grouping the packed rows is ~10x faster
        # than `np.unique(valid.T, axis=0)`, which lexsorts the raw boolean matrix.
        packed = np.ascontiguousarray(np.packbits(valid, axis=0).T)
        keys = packed.view([("", packed.dtype)] * packed.shape[1]).ravel()
        _, inverse = np.unique(keys, return_inverse=True)

        out = np.full(numeric.shape[1], np.nan)
        for group in np.unique(inverse):
            columns = np.flatnonzero(inverse == group)
            mask = valid[:, columns[0]]
            if mask.sum() < 2:
                continue
            y_ranks = rankdata(target[mask])
            y_ranks = y_ranks - y_ranks.mean()
            y_ss = float((y_ranks * y_ranks).sum())
            x_ranks = ranks[np.ix_(mask, columns)]
            x_ranks = x_ranks - x_ranks.mean(axis=0)
            x_ss = (x_ranks * x_ranks).sum(axis=0)
            with np.errstate(invalid="ignore", divide="ignore"):
                out[columns] = (x_ranks * y_ranks[:, None]).sum(axis=0) / np.sqrt(x_ss * y_ss)
        return pd.Series(np.abs(out), index=numeric.columns)

    def _transform(self, X):
        return X[self.selected_features_]

    def _fit_transform(self, X, y, **kwargs):
        self._fit(X, y)
        return self._transform(X), dict()

    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return dict()
