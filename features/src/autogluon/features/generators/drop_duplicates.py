from __future__ import annotations

import hashlib
import logging
import warnings
from collections import defaultdict
from typing import Union

import numpy as np
import pandas as pd
from pandas import DataFrame

from autogluon.common.features.types import R_BOOL, R_CATEGORY, R_FLOAT, R_INT

from .abstract import AbstractFeatureGenerator

logger = logging.getLogger(__name__)


# TODO: Not necessary to exist after fitting, can just update outer context feature_out/feature_in and then delete this
class DropDuplicatesFeatureGenerator(AbstractFeatureGenerator):
    """
    Drops features which are exact duplicates of other features, leaving only one instance of the data.

    Parameters
    ----------
    sample_size_init : int, default 1000
        The number of rows to sample when doing an initial filter of duplicate feature candidates.
        Usually, the majority of features can be filtered out using this smaller amount of rows which greatly speeds up the computation of the final check.
        If None or greater than the number of rows, no initial filter will occur. This may increase the time to fit immensely for large datasets.
    sample_size_final : int, default 5000
        The number of rows to sample when doing the final filter to determine duplicate features.
        This theoretically can lead to features that are very nearly duplicates but not exact duplicates being removed,
        but should be near impossible in practice.
        If None or greater than the number of rows, will perform exact duplicate detection (most expensive).
        It is recommended to keep this value below 100000 to maintain reasonable fit times.
    **kwargs :
        Refer to :class:`AbstractFeatureGenerator` documentation for details on valid key word arguments.
    """

    def __init__(self, sample_size_init=1000, sample_size_final=5000, **kwargs):
        super().__init__(**kwargs)
        self.sample_size_init = sample_size_init
        self.sample_size_final = sample_size_final

    def _fit_transform(self, X: DataFrame, **kwargs) -> (DataFrame, dict):
        if self.sample_size_init is not None and len(X) > self.sample_size_init:
            features_to_check = self._drop_duplicate_features(
                X, self.feature_metadata_in, keep=False, sample_size=self.sample_size_init
            )
            X_candidates = X[features_to_check]
        else:
            X_candidates = X
        features_to_drop = self._drop_duplicate_features(
            X_candidates, self.feature_metadata_in, sample_size=self.sample_size_final
        )
        self._remove_features_in(features_to_drop)
        if features_to_drop:
            self._log(15, f"\t{len(features_to_drop)} duplicate columns removed: {features_to_drop}")
        # Avoid creating an unnecessary copy with X[self.features_in], if possible
        if self.features_in != X.columns.to_list():
            X = X[self.features_in]
        return X, self.feature_metadata_in.type_group_map_special

    def _transform(self, X: DataFrame) -> DataFrame:
        return X

    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return dict()

    @classmethod
    def _drop_duplicate_features(
        cls, X: DataFrame, feature_metadata_in, keep: Union[str, bool] = "first", sample_size=None
    ) -> list:
        if sample_size is not None and len(X) > sample_size:
            # Sampling with replacement is much faster than without replacement
            X = X.sample(sample_size, random_state=0, replace=True)
        features_to_remove = []

        X_columns = set(X.columns)
        features_to_check_numeric = feature_metadata_in.get_features(valid_raw_types=[R_INT, R_FLOAT])
        features_to_check_numeric = [feature for feature in features_to_check_numeric if feature in X_columns]
        if features_to_check_numeric:
            features_to_remove += cls._drop_duplicate_features_numeric(X=X[features_to_check_numeric], keep=keep)
            X = X.drop(columns=features_to_check_numeric)

        X_columns = set(X.columns)
        features_to_check_categorical = feature_metadata_in.get_features(valid_raw_types=[R_CATEGORY, R_BOOL])
        features_to_check_categorical = [feature for feature in features_to_check_categorical if feature in X_columns]
        if features_to_check_categorical:
            features_to_remove += cls._drop_duplicate_features_categorical(
                X=X[features_to_check_categorical], keep=keep
            )
            X = X.drop(columns=features_to_check_categorical)

        if len(X.columns) > 0:
            features_to_remove += cls._drop_duplicate_features_generic(X=X, keep=keep)

        return features_to_remove

    @classmethod
    def _drop_duplicate_features_generic(cls, X: DataFrame, keep: Union[str, bool] = "first"):
        """Generic duplication dropping method. Much slower than optimized variants, but can handle all data types."""
        X_columns = list(X.columns)
        features_to_keep = set(X.T.drop_duplicates(keep=keep).T.columns)
        features_to_remove = [column for column in X_columns if column not in features_to_keep]
        return features_to_remove

    @staticmethod
    def _fingerprint_numeric_series_full(s: pd.Series) -> bytes:
        """
        Deterministic full-column fingerprint for numeric-like Series.

        - Includes an isna mask, so NaN/NA positions matter.
        - Normalizes missing values by zeroing them in the value buffer (mask carries the info).
        - Normalizes -0.0 to +0.0 for float types.
        - Uses blake2b (fast, stable) to produce a 16-byte digest.
        """
        if isinstance(s.dtype, np.dtype):
            vals = np.asarray(s.to_numpy(copy=False), dtype=np.float64).copy()
        else:
            # Nullable numeric arrays may contain pd.NA, which cannot be cast by np.asarray.
            vals = s.to_numpy(dtype=np.float64, na_value=np.nan, copy=True)
        mask = np.isnan(vals)
        # Normalize signed zero and NaN payload differences; the mask preserves NA positions.
        vals[vals == 0.0] = 0.0
        vals[mask] = 0.0

        # Hash both: value bytes + mask bytes
        h = hashlib.blake2b(digest_size=16)
        h.update(np.ascontiguousarray(vals).view(np.uint8))
        h.update(np.ascontiguousarray(mask).view(np.uint8))
        return h.digest()

    @staticmethod
    def _numeric_series_equal(left: pd.Series, right: pd.Series) -> bool:
        if left.dtype != right.dtype:
            # Python numeric equality preserves integer precision across dtypes, unlike numpy's
            # int64/float64 comparison, which rounds integers above 2**53 before comparing.
            left = pd.Series(left.to_numpy(dtype=object, na_value=np.nan), dtype=object)
            right = pd.Series(right.to_numpy(dtype=object, na_value=np.nan), dtype=object)
        return left.equals(right)

    @classmethod
    def _drop_duplicate_features_numeric(
        cls,
        X: DataFrame,
        keep: Union[str, bool] = "first",
    ) -> list[str]:
        """
        >100x faster than pandas drop_duplicates

        Numeric duplicate detection using:
          1) summary-stat bucketing (sum, std, min, max)
          2) full-column fingerprint hash (values + missing mask)
          3) exact comparison of the original values within fingerprint buckets

        Never calls pandas drop_duplicates.

        keep semantics:
          - "first": keep earliest column in X.columns order
          - "last":  keep latest column in X.columns order
          - False:   drop all columns in duplicate groups
          - True:    treated like "first" (for compatibility)
        """
        if X.empty or X.shape[1] <= 1:
            return []

        # Normalize keep
        if keep is True:
            keep_mode = "first"
        elif keep in ("first", "last") or keep is False:
            keep_mode = keep
        else:
            raise ValueError(f"Invalid keep={keep!r}. Expected 'first', 'last', False (or True).")

        cols = list(X.columns)

        # ---- Vectorized stats pass (cheap) ----
        # On one float64 array: a frame of many single-column blocks reduces column by column.
        # Missing values are skipped, as pandas' reductions skip them; an all-missing column gets
        # NaN statistics and never shares a bucket, as before.
        values = X.to_numpy(dtype=np.float64, na_value=np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            stats = np.round(
                np.column_stack(
                    [
                        np.nansum(values, axis=0),
                        np.nanstd(values, axis=0),
                        np.nanmin(values, axis=0),
                        np.nanmax(values, axis=0),
                    ]
                ),
                6,
            )

        # ---- Bucket by stats ----
        bucket_map: dict[tuple[float, float, float, float], list[str]] = defaultdict(list)
        needs_exact_comparison = set()
        for c, dtype, key in zip(cols, X.dtypes, map(tuple, stats.tolist())):
            bucket_map[key].append(c)
            if (dtype.kind in "iu" and (key[2] <= -(2**53) or key[3] >= 2**53)) or (
                dtype.kind == "f" and dtype.itemsize > np.dtype(np.float64).itemsize
            ):
                needs_exact_comparison.add(c)

        # ---- Within each stats bucket, bucket by full fingerprint ----
        to_remove: list[str] = []

        for bucket_cols in bucket_map.values():
            if len(bucket_cols) <= 1:
                continue

            fp_map: dict[bytes, list[str]] = defaultdict(list)
            for c in bucket_cols:
                fp = cls._fingerprint_numeric_series_full(X[c])
                fp_map[fp].append(c)

            for candidates in fp_map.values():
                if len(candidates) <= 1:
                    continue

                duplicate_groups = [candidates]
                if any(c in needs_exact_comparison for c in candidates):
                    # Distinct int64 values above 2**53 can share a float64 fingerprint.
                    # Only lossy conversions need an exact check of the original values.
                    duplicate_groups = []
                    for c in candidates:
                        for group in duplicate_groups:
                            if cls._numeric_series_equal(X[c], X[group[0]]):
                                group.append(c)
                                break
                        else:
                            duplicate_groups.append([c])

                # All buckets retain X.columns order; no sorting (or quadratic cols.index) is needed.
                for group in duplicate_groups:
                    if len(group) <= 1:
                        continue
                    if keep_mode == "first":
                        to_remove.extend(group[1:])
                    elif keep_mode == "last":
                        to_remove.extend(group[:-1])
                    else:  # keep_mode is False
                        to_remove.extend(group)

        return to_remove

    @classmethod
    def _drop_duplicate_features_categorical(cls, X: DataFrame, keep: Union[str, bool] = "first"):
        """
        Drops duplicate features if they contain the same information, ignoring the actual values in the features.
        For example, ['a', 'b', 'b'] is considered a duplicate of ['b', 'a', 'a'], but not ['a', 'b', 'a'].
        """
        # Codes in order of first appearance, missing values included as a value of their own:
        # ['a', 'd', 'f', 'a'] and [5, 'a', np.nan, 5] both become [0, 1, 2, 0], so they are duplicates.
        features_by_unique_count = defaultdict(list)
        for feature in X.columns:
            codes, uniques = pd.factorize(X[feature].to_numpy(dtype=object), use_na_sentinel=False)
            features_by_unique_count[len(uniques)].append((feature, codes))

        features_to_remove = []
        for features_to_check in features_by_unique_count.values():
            # Only need to check features that have same amount of unique values.
            if len(features_to_check) <= 1:
                continue
            X_cur = DataFrame({feature: codes.astype(np.int64) for feature, codes in features_to_check}, index=X.index)
            features_to_remove += cls._drop_duplicate_features_numeric(X=X_cur, keep=keep)

        return features_to_remove

    def _more_tags(self):
        return {"feature_interactions": False}
