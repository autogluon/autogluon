from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter
from typing import Literal, Optional, Sequence

import numpy as np
import pandas as pd

from autogluon.common.features.types import (
    S_DATETIME_AS_INT,
    S_IMAGE_BYTEARRAY,
    S_IMAGE_PATH,
    S_TEXT,
    S_TEXT_EMBEDDING,
    S_TEXT_EMBEDDING_DR,
    S_TEXT_NGRAM,
    S_TEXT_SPECIAL,
)

from .abstract import AbstractFeatureGenerator
from .oof_target_encoder import OOFTargetEncodingFeatureGenerator


class TimerLog:
    # TODO: Mainly used for debugging and tracking runtimes during development. Not needed for preprocessing logic. Better remove?
    # TODO: Copied from arithmetic preprocessors, might move it somewhere else if we want to reuse it.
    def __init__(self):
        self.times = {}

    @contextmanager
    def block(self, name: str):
        t0 = perf_counter()
        try:
            yield
        finally:
            dt = perf_counter() - t0
            self.times[name] = self.times.get(name, 0) + dt

    def summary(self, verbose: bool = False) -> dict:
        if verbose:
            print("\n--- Timing Summary (in order) ---")
            for name, total in self.times.items():
                print(f"{name:<20} {total:.3f}s")
        return dict(self.times)


class RandomSubsetFeatureCompressionGenerator(AbstractFeatureGenerator):
    """
    Random Subset Feature Compression (RSFC) with target-awareness via OOF-TE.

    Runs target-aware random subset compression on:
      - full feature set (always attempted)
      - additional random subsets of features

    Output columns:
      RSFC_0_*  -> full-feature set compression
      RSFC_1_*  -> first random subset
      ...
    """

    def __init__(
        self,
        only_cat: bool = False,
        binary_as_cat: bool = True,
        max_cardinality: Optional[int] = None,
        round_numerical: Optional[int] = 2,
        n_subsets: int = 50,
        subset_size: Optional[int] = None,
        min_subset_size: int = 2,
        max_subset_size: Optional[int] = None,
        max_base_feats_to_consider: Optional[int] = 150,
        select_for_multiclass: bool = False,
        random_state: int = 42,
        **kwargs,
    ):
        super().__init__(random_state=random_state, **kwargs)

        assert self.target_type is not None, f"Must specify target_type for {self.__class__.__name__}"

        self.n_subsets = int(n_subsets)
        self.subset_size = subset_size
        self.min_subset_size = int(min_subset_size)
        self.max_subset_size = max_subset_size
        self.only_cat = bool(only_cat)
        self.binary_as_cat = bool(binary_as_cat)
        self.max_cardinality = int(max_cardinality) if max_cardinality is not None else None
        self.round_numerical = int(round_numerical) if round_numerical is not None else None
        self.select_for_multiclass = bool(select_for_multiclass)

        self.max_base_feats_to_consider = (
            int(max_base_feats_to_consider) if max_base_feats_to_consider is not None else None
        )

        # fitted state
        self.base_features_: Optional[tuple[str, ...]] = None

        self.timelog = TimerLog()

    @staticmethod
    def _select_top_mode_features(X: pd.DataFrame, k: Optional[int]) -> list[str]:
        """
        Pick top-k columns by mode strength:
          max(value_count(col)) / n_rows   (dropna=False)
        """
        if k is None or k <= 0 or X.shape[1] == 0:
            return list(X.columns)

        k = min(k, X.shape[1])
        n = len(X)
        if n == 0:
            return list(X.columns)[:k]

        scores = {}
        for c in X.columns:
            vc = X[c].value_counts(dropna=False)
            top = int(vc.iloc[0]) if len(vc) else 0
            scores[c] = top / n

        return sorted(scores.keys(), key=lambda c: (-scores[c], str(c)))[:k]

    @staticmethod
    def _select_features_by_dtype_and_cardinality(X: pd.DataFrame, k: Optional[int]) -> list[str]:
        if k is not None and k <= 0:
            return []

        ordered = (
            list(X.select_dtypes(include="category").nunique().sort_values().index)
            + list(X.select_dtypes(include="object").nunique().sort_values().index)
            + list(X.columns[X.nunique() == 2])
            + list(X.select_dtypes(include="integer").nunique().sort_values().index)  # Note: Just int not captured
            + list(X.select_dtypes(include="float").nunique().sort_values().index)
        )

        # remove duplicates, keep order
        ordered = list(dict.fromkeys(ordered))

        return ordered if k is None else ordered[:k]

    @staticmethod
    def _sample_unique_subsets(
        features: Sequence[str],
        rng: np.random.Generator,
        n: int,
        subset_size: Optional[int],
        min_k: int,
        max_k: Optional[int],
        max_tries_multiplier: int = 50,
    ) -> list[tuple[str, ...]]:
        feats = np.asarray(list(features), dtype=object)
        p = len(feats)
        if p <= 1 or n <= 0:
            return []

        max_k_eff = min(max_k if max_k is not None else (p - 1), p - 1)
        min_k_eff = max(min_k, 1)

        if subset_size is not None:
            k_min = k_max = int(subset_size)
            if not (1 <= k_min <= p - 1):
                return []
        else:
            k_min, k_max = min_k_eff, max_k_eff
            if k_min > k_max:
                return []

        selected: list[tuple[str, ...]] = []
        seen: set[tuple[str, ...]] = set()
        max_tries = max_tries_multiplier * n

        for _ in range(max_tries):
            if len(selected) >= n:
                break

            k = k_min if subset_size is not None else int(rng.integers(k_min, k_max + 1))
            subset = tuple(sorted(rng.choice(feats, size=k, replace=False).tolist()))
            if subset in seen:
                continue
            seen.add(subset)
            selected.append(subset)

        return selected

    def _prepare_X(self, X: pd.DataFrame) -> pd.DataFrame:
        if X.shape[1] == 0:
            # Degenerate: no columns; hash empty rows
            return pd.Series(pd.util.hash_pandas_object(X, index=False).astype("uint64").astype(str), index=X.index)

        # Select columns
        if self.only_cat:
            cols = X.select_dtypes(include=["object", "category"]).columns
            numeric_cols = X.select_dtypes(include=["number"]).columns
            if self.binary_as_cat:
                binary_cols = X.select_dtypes(include=["number"]).columns[
                    X[numeric_cols].nunique() <= 2
                ]  # NOTE: uniform may occur at test, hence <=, should generally make train/test prepare versions
                cols = cols.union(binary_cols)
        else:
            cols = X.columns

        if self.max_cardinality is not None and len(cols) > 0:
            nunique = X[cols].nunique(dropna=False)
            cols = nunique[nunique < self.max_cardinality].index

        if len(cols) == 0:
            cols = X.columns

        X_candidates = X.loc[:, cols]

        # Round numeric columns (copy only when needed)
        num_cols = X_candidates.select_dtypes(include=["number"]).columns
        if len(num_cols) > 0 and self.round_numerical is not None:
            X_candidates = X_candidates.copy()
            X_candidates.loc[:, num_cols] = X_candidates.loc[:, num_cols].round(self.round_numerical)

        return X_candidates

    def _make_key(self, X: pd.DataFrame) -> pd.Series:
        """
        Build a deterministic key per row by hashing selected columns.
        """
        # hash rows -> uint64 -> str (OOF-TE expects single column input)
        return pd.util.hash_pandas_object(X, index=False).astype("uint64").astype(str)

    @staticmethod
    def collapse_singletons(s, threshold=1, label="__single__"):
        vc = s.value_counts()
        return s.where(s.map(vc) > threshold, label)

    def _fit_transform(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        # Prepare X
        with self.timelog.block("prepare_input"):
            X_local = self._prepare_X(X)

        # Restrict base feature space if requested
        with self.timelog.block("select_base_features"):
            if self.max_base_feats_to_consider is not None and X.shape[1] > 0:
                selected = self._select_features_by_dtype_and_cardinality(X_local, self.max_base_feats_to_consider)
                X_local = X_local[selected]
                self.base_features_ = list(selected)
            else:
                self.base_features_ = list(X.columns)

        features = list(X_local.columns)
        rng = np.random.default_rng(self.random_state)

        # ---- 1..n) Random subsets ----
        with self.timelog.block("select_random_subsets"):
            self.selected_subsets = [tuple(features)]  # always include full set
            n_random = max(self.n_subsets - 1, 0)
            self.selected_subsets += self._sample_unique_subsets(
                features=features,
                rng=rng,
                n=n_random,
                subset_size=self.subset_size,
                min_k=self.min_subset_size,
                max_k=self.max_subset_size,
            )

        with self.timelog.block("make_key"):
            X_str = pd.concat([self._make_key(X_local[list(i)]) for i in self.selected_subsets], axis=1)

        # with self.timelog.block("filter_uninformative_keys"): # Improves efficiency but hurts performance
        #     X_str = X_str.apply(self.collapse_singletons)

        with self.timelog.block("oof-te"):
            self.subset_oof = OOFTargetEncodingFeatureGenerator(
                target_type=self.target_type, verbosity=0, alpha=0, random_state=self.random_state
            )
            X_oof = self.subset_oof.fit_transform(X_str, y)

        self.col_names = [f"RSFC_{i}" for i in range(X_oof.shape[1])]
        X_oof.columns = self.col_names

        if self.select_for_multiclass and self.target_type == "multiclass":
            y_corrs = pd.get_dummies(y).apply(lambda y_: X_oof.corrwith(y_))
            best_corr_rank = y_corrs.abs().rank().min(axis=1)
            self.selected_cols = best_corr_rank.index[best_corr_rank < self.n_subsets]
            X_oof = X_oof[self.selected_cols]

        return X_oof, {}

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # Prepare X
        with self.timelog.block("transform_prepare_input"):
            X_local = self._prepare_X(X[self.base_features_])
        with self.timelog.block("transform_make_key"):
            X_str = pd.concat([self._make_key(X_local[list(i)]) for i in self.selected_subsets], axis=1)
        with self.timelog.block("transform_oof-transform"):
            out = self.subset_oof.transform(X_str)
            out.columns = self.col_names
        if self.select_for_multiclass and self.target_type == "multiclass":
            out = out[self.selected_cols]
        return out

    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return dict(
            invalid_special_types=[
                S_DATETIME_AS_INT,
                S_IMAGE_BYTEARRAY,
                S_IMAGE_PATH,
                S_TEXT,
                S_TEXT_EMBEDDING,
                S_TEXT_EMBEDDING_DR,
                S_TEXT_NGRAM,
                S_TEXT_SPECIAL,
            ],
        )
