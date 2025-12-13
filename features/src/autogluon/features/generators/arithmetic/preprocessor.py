import logging
import warnings

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from ..abstract import AbstractFeatureGenerator
from ..cat_as_num import CatAsNumFeatureGenerator

logger = logging.getLogger(__name__)  # TODO: Unsure what this does, copied it since its also in other preprocessors

import re
from contextlib import contextmanager
from math import comb
from time import perf_counter
from typing import Literal, Tuple

from pandas.api.types import is_numeric_dtype

from .combinations import (
    add_higher_interaction,
    get_all_bivariate_interactions,
    estimate_no_higher_interaction_features,
)
from .combinations_lite import (
    add_higher_interaction as add_higher_interaction_lite,
    get_all_bivariate_interactions as get_all_bivariate_interactions_lite,
    Operation,
)
from .filtering import basic_filter, filter_by_cross_correlation, filter_by_spearman
from .memory import reduce_memory_usage
import operator


class TimerLog:
    # TODO: Mainly used for debugging and tracking runtimes during development. Not needed for preprocessing logic. Better remove?
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


# Compact op codes for numba
OP_CODE = {
    "+": 0,
    "-": 1,
    "*": 2,
    "/": 3,
}


class ArithmeticFeatureGenerator(AbstractFeatureGenerator):
    """
    Converts category features to one-hot boolean features by mapping to the category codes.

    Parameters
    ----------
    target_type: str
        The type of the target variable ('regression', 'classification', 'binary').
    selection_method : str, default = 'random'
        Method to select features for interaction generation. Options are 'spearman' or 'random'.
    max_order : int, default = 3
        The maximum number of features considered to use for arithmetic interaction feature generation.
    max_base_feats : int, default = 150
        The maximum number of (randomly selected) base features to consider for arithmetic interaction generation.
    max_new_feats : int, default = 2000
        The maximum number of new arithmetic interaction features to generate.
    cat_as_num: bool, default = False
        Whether to transform categorical features to numeric and use them as base features in arithmetic interaction generation.
    min_cardinality : int, default = 3
        Minimum cardinality for a feature to be considered as a base feature in arithmetic interaction generation.
    random_state : int, default = 42
        Random seed for reproducibility.
    interaction_types : list of str, default = ['/', '*', '-', '+']
        List of arithmetic interaction types to consider.
    data_cleaning : bool, default = True
        Whether to remove features that are nearly constant or mostly NaN after generation.
    nan_threshold : float, default = 0.95
        Threshold for removing features with too many NaN values.
    mode_imbalance_threshold : float, default = 0.95
        Threshold for mode imbalance to remove features.
    subsample : int, default = 100000
        Number of samples to use when selecting new features, used to improve efficiency.
    reduce_memory : bool, default = True
        Whether to reduce memory usage during feature generation. Might affect precision.
    rescale_avoid_overflow : bool, default = True
        Whether to normalize all features to the same constant to deal with large integer values, mainly to avoid overflow during arithmetic operations.
    corr_threshold : float, default = 0.95
        Correlation threshold for feature selection with selection_method=='spearman'.
    use_cross_corr : bool, default = False
        Whether to use cross-correlation with the base features for feature selection when selection_method=='spearman'.
    cross_corr_n_block_size : int, default = 5000
        Block size for cross-correlation computation. Used to limit memory usage.
    max_accept_for_pairwise : int, default = 10000
        Maximum number of accepted features for generating pairwise interactions when selection_method=='spearman'. At most max_accept_for_pairwise feature are generated and then filtered to max_new_feats using spearman correlation.
    verbose : bool, default = False
        Whether to print logs.

    """

    def __init__(
        self,
        target_type: Literal[
            "regression", "multiclass", "binary"
        ],  # TODO: Currently not used, but required for the overall structure how preprocessors are used
        selection_method: Literal["spearman", "random"] = "random",
        max_order: int = 3,
        max_base_feats: int = 150,  # TODO: Need to implement a better heuristic than choosing randomly
        max_new_feats: int = 2000,  # FIXME: 2000 originally
        cat_as_num: bool = False,
        min_cardinality: int = 3,
        random_state: int = 42,
        interaction_types: list[str] = ["/", "*", "-", "+"],
        data_cleaning: bool = True,
        nan_threshold: float = 0.95,
        mode_imbalance_threshold: float = 0.9,
        subsample: int = 100000,  # TODO: Need to implement
        reduce_memory: bool = False,
        rescale_avoid_overflow: bool = True,
        corr_threshold: float = 0.95,
        use_cross_corr: bool = False,
        cross_corr_n_block_size: int = 5000,
        max_accept_for_pairwise: int = 10000,
        inference_mode: Literal["dag"] = "dag",
        inner_dtype=np.float32,
        out_dtype=np.float32,
        verbose: bool = False,
        passthrough: bool = True,
        **kwargs,
    ):
        super().__init__(passthrough=passthrough, **kwargs)
        self.target_type = target_type  # TODO: Clarify if and how problem_type generally is used in AG preprocessors
        self.max_order = max_order
        self.cat_as_num = cat_as_num
        self.min_cardinality = min_cardinality
        self.max_base_feats = max_base_feats
        self.max_new_feats = max_new_feats
        self.selection_method = selection_method
        self.subsample = subsample
        self.reduce_memory = reduce_memory
        self.rescale_avoid_overflow = rescale_avoid_overflow
        self.corr_threshold = corr_threshold
        self.use_cross_corr = use_cross_corr
        self.cross_corr_n_block_size = cross_corr_n_block_size
        self.max_accept_for_pairwise = max_accept_for_pairwise
        self.inference_mode = inference_mode
        self.verbose = verbose
        self.inner_dtype = inner_dtype
        self.out_dtype = out_dtype
        self.interaction_types = interaction_types
        self.data_cleaning = data_cleaning
        self.nan_threshold = nan_threshold
        self.mode_imbalance_threshold = mode_imbalance_threshold

        for i in self.interaction_types:
            if i not in OP_CODE:
                raise ValueError(f"Unsupported interaction type: {i}")

        self.rng = np.random.default_rng(random_state)

        self.used_base_cols = []

        self.timelog = TimerLog()
        self.new_feats = []

    def estimate_new_dtypes(self, n_numeric, n_categorical, n_binary, **kwargs) -> int:
        num_base_feats = n_numeric
        if self.min_cardinality < 2:
            num_base_feats += n_binary
        if self.cat_as_num:
            num_base_feats += n_categorical

        # 2. Estimate the no. of new arithmetic features per order
        no_interaction_types = len(self.interaction_types)
        num_new_feats = 0
        for order in range(2, self.max_order + 1):
            if order > num_base_feats:
                break
            if order == 2:
                if "/" in self.interaction_types:
                    no_interaction_types += 1
                num_new_feats = (
                    comb(num_base_feats, 2) * no_interaction_types
                )  # num_base_feats*(num_base_feats-1)/2*no_interaction_types
                if "/" in self.interaction_types:
                    no_interaction_types -= 1
            else:
                # num_new_feats += ((num_base_feats - 2) * (num_new_feats)) * no_interaction_types
                num_new_feats += estimate_no_higher_interaction_features(num_base_feats, num_new_feats)
            if num_new_feats > self.max_new_feats:
                num_new_feats = self.max_new_feats
                break
        return n_numeric + int(num_new_feats), n_categorical, n_binary

    def estimate_no_of_new_features(self, X: pd.DataFrame, **kwargs) -> int:
        if self.selection_method != "random":
            warnings.warn(
                "Estimation of new features is only implemented for selection_method='random'. Returning max_new_feats.",
                UserWarning,
            )
            return self.max_new_feats

        # 1. Determine the no. of base features
        pass_cardinality_filter = X.nunique().values >= self.min_cardinality
        pass_cat_filter = X.apply(is_numeric_dtype).values if not self.cat_as_num else np.array([True] * X.shape[1])
        base_feat_mask = pass_cardinality_filter & pass_cat_filter
        num_base_feats = min(np.sum(base_feat_mask), self.max_base_feats)

        # 2. Estimate the no. of new arithmetic features per order
        no_interaction_types = len(self.interaction_types)
        num_new_feats = 0
        for order in range(2, self.max_order + 1):
            if order > num_base_feats:
                break
            if order == 2:
                if "/" in self.interaction_types:
                    no_interaction_types += 1
                num_new_feats = (
                    comb(num_base_feats, 2) * no_interaction_types
                )  # num_base_feats*(num_base_feats-1)/2*no_interaction_types
                if "/" in self.interaction_types:
                    no_interaction_types -= 1
            else:
                # num_new_feats += ((num_base_feats - 2) * (num_new_feats)) * no_interaction_types
                num_new_feats += estimate_no_higher_interaction_features(num_base_feats, num_new_feats)
            if num_new_feats > self.max_new_feats:
                num_new_feats = self.max_new_feats
                break
        return int(num_new_feats), X.columns[base_feat_mask].tolist()

    def spearman_selection(self, X: pd.DataFrame, y: pd.Series):
        # FIXME: Currently heavily relies on polars for performance, make sure a numpy and a polars version exists for each function
        ### Apply advanced filtering steps (spearman correlation thresholding)
        # TODO: Might skip that and instead add the corr based filter to basic + use a max_base_features parameter
        with self.timelog.block("advanced_filter_base"):
            use_cols = filter_by_spearman(X, corr_threshold=self.corr_threshold)

        if self.verbose:
            print(f"Using {len(use_cols)}/{X.shape[1]} features after advanced filtering")
        X = X[use_cols]

        if X.shape[1] == 0:
            if self.verbose:
                print("No features left after filtering. Exiting.")
            return self

        X_dict = {1: X}
        for order in range(2, self.max_order + 1):
            if order > X.shape[1]:
                break
            if self.verbose:
                print("---" * 20)
                print(f"Generating order {order} interaction features")

            # 6. Generate higher-order interaction features
            with self.timelog.block(f"get_interactions_{order}-order"):
                if order == 2:
                    X_dict[2] = get_all_bivariate_interactions(
                        X,
                        max_feats=int(self.max_accept_for_pairwise / 5),
                        random_state=self.rng,
                        interaction_types=self.interaction_types,
                    )
                else:
                    X_dict[order] = add_higher_interaction(
                        X,
                        X_dict[order - 1],
                        max_feats=int(self.max_accept_for_pairwise / 5),
                        random_state=self.rng,
                        interaction_types=self.interaction_types,
                    )

            if self.reduce_memory:
                with self.timelog.block(f"reduce_memory_{order}-order"):
                    X_dict[order] = reduce_memory_usage(X_dict[order], rescale=False, verbose=self.verbose)
            if self.verbose:
                print(f"Generated {X_dict[order].shape[1]} {order}-order interaction features")

            # 7. Filter higher-order interaction features
            n_feats_start = X_dict[order].shape[1]
            # basic
            with self.timelog.block(f"basic_filter_{order}-order"):
                X_dict[order] = basic_filter(
                    X_dict[order],
                    use_polars=False,
                    min_cardinality=self.min_cardinality,
                    data_cleaning=self.data_cleaning,
                    nan_threshold=self.nan_threshold,
                    mode_imbalance_threshold=self.mode_imbalance_threshold,
                )
            if self.verbose:
                print(f"Using {len(X_dict[order].columns)}/{n_feats_start} features after basic filtering")

            # based on correlations among interaction features
            if X_dict[order].shape[1] > self.max_accept_for_pairwise:
                if self.verbose:
                    print(
                        f"Limiting interaction features to {self.max_accept_for_pairwise} (from {X_dict[order].shape[1]})"
                    )
                X_dict[order] = X_dict[order].sample(n=self.max_accept_for_pairwise, random_state=42, axis=1)

            # TODO: Implement filtering in chunks. Currently it is too slow and too memory intensive
            n_feats_start = X_dict[order].shape[1]
            with self.timelog.block(f"spearman_int_filter_{order}-order"):
                use_cols = filter_by_spearman(X_dict[order], corr_threshold=self.corr_threshold)
            X_dict[order] = X_dict[order][use_cols]
            if self.verbose:
                print(f"Using {len(use_cols)}/{n_feats_start} features after spearman filtering")

            # based on cross-correlation with base features
            if self.use_cross_corr:
                n_feats_start = X_dict[order].shape[1]
                with self.timelog.block(f"cross_correlation_{order}-order"):
                    use_cols, novelty_scores = filter_by_cross_correlation(
                        X_dict[order - 1], X_dict[order], corr_threshold=self.corr_threshold
                    )
                X_dict[order] = X_dict[order][use_cols]
                if self.verbose:
                    print(f"Using {len(use_cols)}/{n_feats_start} features after cross-correlation filtering")

            if len(self.new_feats) + X_dict[order].shape[1] >= self.max_new_feats:
                if self.use_cross_corr:
                    max_new = self.max_new_feats - len(self.new_feats)
                    self.new_feats.extend(novelty_scores.sort_values(ascending=False).index[:max_new].tolist())
                else:
                    self.new_feats.extend(X_dict[order].columns.tolist()[: self.max_new_feats - len(self.new_feats)])
                if self.verbose:
                    print(f"Reached max new features limit of {self.max_new_feats}. Stopping.")
                break
            else:
                self.new_feats.extend(X_dict[order].columns.tolist())

    def random_selection(self, X: pd.DataFrame, y: pd.Series):
        # TODO: Improve memory efficiency for max_order > 3 by deleting unneeded intermediate results
        X_columns = list(X.columns)
        X_dict = {1: X_columns}

        for order in range(2, self.max_order + 1):
            if order > X.shape[1]:
                break
            if self.verbose:
                print("---" * 20)
                print(f"Generating order {order} interaction features")
            remaining_new_feats = self.max_new_feats - len(self.new_feats)

            if remaining_new_feats <= 0:
                self.new_feats = self.new_feats[: self.max_new_feats]
                break

            # 6. Generate higher-order interaction features
            with self.timelog.block(f"get_interactions_{order}-order"):
                if order == 2:
                    X_dict[2] = get_all_bivariate_interactions_lite(
                        X_columns,
                        max_feats=remaining_new_feats,
                        random_state=self.rng,
                        interaction_types=self.interaction_types,
                    )
                else:
                    X_dict[order] = add_higher_interaction_lite(
                        X_columns,
                        X_dict[order - 1],
                        max_feats=remaining_new_feats,
                        random_state=self.rng,
                        interaction_types=self.interaction_types,
                    )

            if self.verbose:
                print(f"Generated {len(X_dict[order])} {order}-order interaction features")

            self.new_feats.extend(X_dict[order])

            if len(self.new_feats) >= self.max_new_feats:
                self.new_feats = self.new_feats[: self.max_new_feats]
                if self.verbose:
                    print(f"Reached max new features limit of {self.max_new_feats}. Stopping.")
                break

    def _fit(self, X: pd.DataFrame, y: pd.Series | None, **kwargs):
        # TODO: Add a check that the original features names don't contain arithmetic operators to avoid issues in transform
        use_y = (self.selection_method != "random") and (y is not None)

        # ------------------------------------------------------
        # 0) Optional row subsampling
        # ------------------------------------------------------
        with self.timelog.block("row_subsample"):
            if self.subsample and self.subsample < X.shape[0]:
                X = X.sample(n=self.subsample, random_state=self.rng, axis=0)
            sample_index = X.index

        # ------------------------------------------------------
        # 1) Prepare y only if needed
        # ------------------------------------------------------
        with self.timelog.block("prepare_y"):
            if use_y:
                y = y.loc[sample_index]
            else:
                y = None

        # ------------------------------------------------------
        # 2) Detect numeric columns early (non-cat_as_num)
        # ------------------------------------------------------
        with self.timelog.block("detect_numeric_cols"):
            if not self.cat_as_num:
                num_cols = X.select_dtypes(include=[np.number]).columns
                if self.verbose:
                    print(f"Using {len(num_cols)}/{X.shape[1]} base features (filtering to numeric)")
                if len(num_cols) == 0:
                    if self.verbose:
                        print("No numeric features available. Exiting.")
                    self.new_feats = []
                    self.time_logs = {}
                    return self
                X = X[num_cols]

        # ------------------------------------------------------
        # 3) Basic filtering
        # ------------------------------------------------------
        with self.timelog.block("basic_filter_base"):
            n_start = X.shape[1]
            X = basic_filter(
                X,
                use_polars=False,
                min_cardinality=self.min_cardinality,
                data_cleaning=self.data_cleaning,
                nan_threshold=self.nan_threshold,
                mode_imbalance_threshold=self.mode_imbalance_threshold,
            )

        if self.verbose:
            print(f"Using {X.shape[1]}/{n_start} base features after basic filtering")

        if X.shape[1] == 0:
            if self.verbose:
                print("All base features removed after basic filtering.")
            self.new_feats = []
            self.time_logs = {}
            return self

        # ------------------------------------------------------
        # 4) Column subsampling for base features and clean column names
        # ------------------------------------------------------
        with self.timelog.block("column_subsample"):
            if X.shape[1] > self.max_base_feats:
                if self.verbose:
                    print(f"Limiting base features to {self.max_base_feats} (from {X.shape[1]})")
                sampled_cols = set(self.rng.choice(
                    X.columns,
                    size=self.max_base_feats,
                    replace=False,
                ))

                X = X[[c for c in X.columns if c in sampled_cols]]

        self.used_base_cols = X.columns.tolist()
        self._keep_features_in(features=X.columns.tolist())

        # ------------------------------------------------------
        # 5) Cat-as-num or numeric selection
        # ------------------------------------------------------
        with self.timelog.block("apply_cat_as_num_or_select_numeric"):
            if self.cat_as_num:
                self.cat_as_num_preprocessor = CatAsNumFeatureGenerator(
                    target_type=self.target_type,
                    keep_original=False,
                )
                X = self.cat_as_num_preprocessor.fit_transform(X)
                if X.shape[1] == 0:
                    if self.verbose:
                        print("No features left after CatAsNum conversion.")
                    self.new_feats = []
                    self.time_logs = {}
                    return self

        # ------------------------------------------------------
        # 6) Memory reduction
        # ------------------------------------------------------
        if self.reduce_memory and X.shape[1] > 0:
            with self.timelog.block("reduce_memory_usage_base"):
                X = reduce_memory_usage(X, rescale=self.rescale_avoid_overflow, verbose=self.verbose)

        # ------------------------------------------------------
        # 7) Interaction generation
        # ------------------------------------------------------
        if self.selection_method == "random":
            with self.timelog.block("random_selection"):
                self.random_selection(X, y)
        else:
            with self.timelog.block("spearman_selection"):
                self.spearman_selection(X, y)

        # ------------------------------------------------------
        # 9) Collect timing logs
        # ------------------------------------------------------
        with self.timelog.block("collect_time_logs"):
            self.time_logs = self.timelog.summary(verbose=False)

        # ------------------------------------------------------
        # 11) Print summary at end
        # ------------------------------------------------------
        if self.verbose:
            print("\n=== ArithmeticFeatureGenerator Timing Summary ===")
            for k, v in self.time_logs.items():
                print(f"{k:<32} {v:.4f} sec")
            print("=" * 52)
        self.time_logs = {}

        return self

    def _fit_transform(self, X: DataFrame, y: Series, **kwargs) -> Tuple[DataFrame, dict]:
        self._fit(X, y, **kwargs)
        X_out = self._transform(X[self.features_in])

        # features_out = list(X_out.columns)
        # type_group_map_special = {R_FLOAT: features_out}
        return X_out, dict()  # TODO: Unsure whether we need to return anything special here

    def _add_arithmetic_dag(
            self,
            X: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Fast evaluator with DAG optimization.
        Computes common sub-expressions only once.
        expressions: list[Operation]
        """

        # ---------------------------------------------------------------------
        # 1) Prepare input array
        # ---------------------------------------------------------------------
        if self.inner_dtype != np.float64:
            arr = X.to_numpy(dtype=self.inner_dtype)
        else:
            arr = X.to_numpy()
            if not np.issubdtype(arr.dtype, np.floating):
                arr = arr.astype(float)
        col_idx = {c: i for i, c in enumerate(self.used_base_cols)}
        opmap = {
            "+": operator.add,
            "-": operator.sub,
            "*": operator.mul,
            "/": operator.truediv,
        }

        # ---------------------------------------------------------------------
        # 2) Compile DAG (only if expressions changed)
        # ---------------------------------------------------------------------
        compile_needed = not hasattr(self, "_compiled_dag")

        # FIXME: Add support for recompiling in case we pruned features
        if compile_needed:
            expressions: list[Operation] = self.new_feats


            nodes = {}  # key -> None (placeholder)
            expr_roots = {}  # Operation -> node key or int

            def lower(node):
                """Recursively lower Operation → DAG node or base index"""
                if isinstance(node, Operation):
                    left = lower(node.left)
                    right = lower(node.right)
                    key = (left, node.op, right)
                    if key not in nodes:
                        nodes[key] = None
                    return key
                else:
                    # base feature name → column index
                    return col_idx[node]

            for expr in expressions:
                expr_roots[expr] = lower(expr)

            self._compiled_dag = {
                "nodes": list(nodes.keys()),
                "expr_roots": expr_roots,
                "exprs": tuple(expressions),
                "names": tuple([expression.name() for expression in expressions])
            }

        # ---------------------------------------------------------------------
        # 3) Evaluate DAG
        # ---------------------------------------------------------------------
        nodes = self._compiled_dag["nodes"]
        expr_roots = self._compiled_dag["expr_roots"]
        exprs = self._compiled_dag["exprs"]
        names = self._compiled_dag["names"]

        base_values = {i: arr[:, i] for i in range(arr.shape[1])}
        results = {}

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            for key in nodes:
                left, op, right = key

                left_arr = (
                    base_values[left] if isinstance(left, int) else results[left]
                )
                right_arr = (
                    base_values[right] if isinstance(right, int) else results[right]
                )

                results[key] = opmap[op](left_arr, right_arr)

        # ---------------------------------------------------------------------
        # 4) Build output DataFrame
        # ---------------------------------------------------------------------
        output = {}
        for expr, name in zip(exprs, names):
            root = expr_roots[expr]
            if isinstance(root, int):
                out = base_values[root]
            else:
                out = results[root]

            out[np.isinf(out)] = np.nan
            output[name] = out

        return pd.DataFrame(output, index=X.index)

    def _transform(self, X: DataFrame) -> DataFrame:
        # Note: It is important that X have the same order as `self.features_in` when entering this method
        if not self.new_feats:
            return pd.DataFrame(index=X.index)

        X = X[self.used_base_cols]  # TODO: Remove this for a slight speedup, but need to be careful

        if self.cat_as_num:
            X = self.cat_as_num_preprocessor.transform(X)

        if self.inference_mode == "dag":
            X_new = self._add_arithmetic_dag(X)

        # All columns in X_new are numeric floats created from numpy, so
        # cast the whole block at once instead of per-column astype.
        if self.inner_dtype != self.out_dtype:
            if np.issubdtype(self.out_dtype, np.floating):
                with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                    arr = X_new.to_numpy(dtype=self.out_dtype, copy=False)
                X_new = pd.DataFrame(arr, index=X_new.index, columns=X_new.columns)

        # Combine original and interaction features
        return X_new

    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return dict()  # TODO: Unsure what to include here
