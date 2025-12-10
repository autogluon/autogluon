import logging
import warnings

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from autogluon.common.features.types import R_BOOL, R_CATEGORY, R_FLOAT, R_INT, R_OBJECT, S_BINNED, S_BOOL

from ..abstract import AbstractFeatureGenerator
from ..cat_as_num import CatAsNumFeatureGenerator

logger = logging.getLogger(__name__)  # TODO: Unsure what this does, copied it since its also in other preprocessors

import re
from contextlib import contextmanager
from math import comb
from time import perf_counter
from typing import Literal, Tuple

from numba import njit, prange
from pandas.api.types import is_numeric_dtype

from .combinations import add_higher_interaction, get_all_bivariate_interactions, estimate_no_higher_interaction_features
from .filtering import basic_filter, filter_by_cross_correlation, filter_by_spearman, clean_column_names
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


# Map from name-encoded tokens to actual ops
OP_TOKENS = {
    "_+_": "+",
    "_-_": "-",
    "_*_": "*",
    "_/_": "/",
}

# Compact op codes for numba
OP_CODE = {
    "+": 0,
    "-": 1,
    "*": 2,
    "/": 3,
}


def parse_feature_expr(name: str, base_idx: dict) -> tuple[list[int] | None, list[int] | None]:
    """
    Parse a feature name like 'colA_*_colB_*_colC' into:
      - indices: list[int] of base column indices
      - ops:     list[int] of op codes between them (len = order-1)
    Returns (indices, op_codes) or (None, None) if unparsable.
    """
    expr = name
    for tok in OP_TOKENS.keys():
        expr = expr.replace(tok, f" {tok} ")

    parts = expr.split()
    if not parts:
        return None, None

    operands = parts[0::2]  # col names
    op_tokens = parts[1::2]  # '_+_', '_*_', ...

    if len(op_tokens) != max(0, len(operands) - 1):
        return None, None

    try:
        indices = [base_idx[col] for col in operands]
    except KeyError:
        return None, None

    ops = []
    for tok in op_tokens:
        op_char = OP_TOKENS.get(tok)
        if op_char is None or op_char not in OP_CODE:
            return None, None
        ops.append(OP_CODE[op_char])

    return indices, ops


@njit(parallel=True, fastmath=True)
def eval_order_fused(X_base: np.ndarray, idx_mat: np.ndarray, op_mat: np.ndarray) -> np.ndarray:
    n_rows, n_base = X_base.shape
    n_feats, order = idx_mat.shape

    out = np.empty((n_rows, n_feats), dtype=X_base.dtype)

    for i in prange(n_rows):
        for f in range(n_feats):
            idx_row = idx_mat[f]  # 1D view: length = order
            ops_row = op_mat[f]  # 1D view: length = order-1

            v = X_base[i, idx_row[0]]

            for k in range(1, order):
                b = X_base[i, idx_row[k]]
                op = ops_row[k - 1]

                if op == 0:  # +
                    v += b
                elif op == 1:  # -
                    v -= b
                elif op == 2:  # *
                    v *= b
                else:  # /
                    if b == 0.0:
                        v = np.nan
                    else:
                        v /= b

            out[i, f] = v

    return out


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
    remove_constant_mostlynan : bool, default = True
        Whether to remove features that are nearly constant or mostly NaN after generation.
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
        max_new_feats: int = 1000,  # FIXME: 2000 originally
        cat_as_num: bool = False,
        min_cardinality: int = 3,
        random_state: int = 42,
        interaction_types: list[str] = ["/", "*", "-", "+"],
        remove_constant_mostlynan: bool = True,
        subsample: int = 100000,  # TODO: Need to implement
        reduce_memory: bool = False,
        rescale_avoid_overflow: bool = True,
        corr_threshold: float = 0.95,
        use_cross_corr: bool = False,
        cross_corr_n_block_size: int = 5000,
        max_accept_for_pairwise: int = 10000,
        inference_mode: Literal["compiled_numba", "dag"] = "compiled_numba",
        out_dtype=np.float32,
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
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
        self.out_dtype = out_dtype
        self.interaction_types = interaction_types
        self.remove_constant_mostlynan = remove_constant_mostlynan

        for i in self.interaction_types:
            if i not in OP_CODE:
                raise ValueError(f"Unsupported interaction type: {i}")

        self.rng = np.random.default_rng(random_state)

        self.timelog = TimerLog()
        self.new_feats = []
        self.order_batches = {}  # order -> {'idx': np.ndarray, 'ops': np.ndarray, 'names': list[str]}

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
                    remove_constant_mostlynan=self.remove_constant_mostlynan,
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
                        max_feats=self.max_new_feats,
                        random_state=self.rng,
                        interaction_types=self.interaction_types,
                    )
                    # X_dict[order] = add_higher_interaction(X, X, max_feats=self.max_new_feats, random_state=self.rng, interaction_types=self.interaction_types)
                else:
                    X_dict[order] = add_higher_interaction(
                        X,
                        X_dict[order - 1],
                        max_feats=self.max_new_feats - X_dict[order - 1].shape[1],
                        random_state=self.rng,
                        interaction_types=self.interaction_types,
                    )

            if self.reduce_memory:
                with self.timelog.block(f"reduce_memory_{order}-order"):
                    X_dict[order] = reduce_memory_usage(X_dict[order], rescale=False, verbose=self.verbose)
                # X_dict[order] = basic_filter(X_dict[order], use_polars=False, min_cardinality=self.min_cardinality, remove_constant_mostlynan=self.remove_constant_mostlynan)
            if self.verbose:
                print(f"Generated {X_dict[order].shape[1]} {order}-order interaction features")

            self.new_feats.extend(X_dict[order].columns.tolist())

            if len(self.new_feats) >= self.max_new_feats:
                self.new_feats = self.new_feats[: self.max_new_feats]
                if self.verbose:
                    print(f"Reached max new features limit of {self.max_new_feats}. Stopping.")
                break

    def _post_hoc_adjust_new_feats(self, new_feats: list[str]):
        # In case some new features were removed after fit (e.g., because they were found to be not predictive after modeling), adjust the new_feats list
        self.new_feats = [f for f in self.new_feats if f in new_feats]
        self._prepare_order_batches()
        self._warmup_fused_orders()

    def _warmup_fused_orders(self):
        """
        Trigger Numba JIT for each order group during fit(),
        so transform() is consistently fast.
        """
        if not self.order_batches:
            return

        # tiny dummy data: 2 rows, same number of base cols
        dummy_X_T = np.zeros((len(self.used_base_cols), 2), dtype=np.float64)

        for order, batch in self.order_batches.items():
            idx_mat = batch["idx"]
            ops_mat = batch["ops"]
            if idx_mat.size == 0:
                continue
            eval_order_fused(dummy_X_T, idx_mat, ops_mat)

    def _prepare_order_batches(self):
        """
        Build per-order fused execution plans from self.new_feats.

        For each interaction feature, we parse:
        - its base column indices
        - the sequence of op codes between operands

        and group them by interaction order.
        """
        self.order_batches = {}

        base_idx = {col: i for i, col in enumerate(self.used_base_cols)}

        for name in self.new_feats:
            indices, ops = parse_feature_expr(name, base_idx)
            if indices is None:
                if self.verbose:
                    print(f"[ArithmeticPreprocessor] Skipping unparsable feature name: {name}")
                continue

            order = len(indices)
            if order < 2:
                # Interactions are typically order>=2; skip or handle separately if needed
                continue

            batch = self.order_batches.setdefault(
                order,
                {
                    "idx": [],
                    "ops": [],
                    "names": [],
                },
            )
            batch["idx"].append(indices)
            batch["ops"].append(ops)
            batch["names"].append(name)

        # Convert lists to numpy arrays for numba
        for order, batch in self.order_batches.items():
            idx_mat = np.asarray(batch["idx"], dtype=np.int32)
            ops_mat = np.zeros((idx_mat.shape[0], order - 1), dtype=np.int8)
            for i, ops in enumerate(batch["ops"]):
                if len(ops) != order - 1:
                    # shouldn't happen if parsing is consistent
                    raise ValueError(f"Feature with order {order} has wrong ops length: {len(ops)}")
                for k, op_code in enumerate(ops):
                    ops_mat[i, k] = op_code

            batch["idx"] = idx_mat
            batch["ops"] = ops_mat

    def _fit(self, X_in: pd.DataFrame, y_in: pd.Series | None, **kwargs):
        # TODO: Add a check that the original features names don't contain arithmetic operators to avoid issues in transform
        use_y = (self.selection_method != "random") and (y_in is not None)

        # ------------------------------------------------------
        # 0) Detect numeric columns early (non-cat_as_num)
        # ------------------------------------------------------
        with self.timelog.block("detect_numeric_cols"):
            if not self.cat_as_num:
                num_cols = X_in.select_dtypes(include=[np.number]).columns
                if len(num_cols) == 0:
                    if self.verbose:
                        print("No numeric features available. Exiting.")
                    self.used_base_cols = []
                    self.new_feats = []
                    self.order_batches = {}
                    self.time_logs = {}
                    return self
            else:
                num_cols = None

        # ------------------------------------------------------
        # 1) Optional row subsampling
        # ------------------------------------------------------
        with self.timelog.block("row_subsample"):
            X = X_in
            if self.subsample and self.subsample < X_in.shape[0]:
                X = X.sample(n=self.subsample, random_state=self.rng, axis=0)
            sample_index = X.index

        # ------------------------------------------------------
        # 2) Prepare y only if needed
        # ------------------------------------------------------
        with self.timelog.block("prepare_y"):
            if use_y:
                y = y_in.loc[sample_index].reset_index(drop=True)
            else:
                y = None

        # ------------------------------------------------------
        # 3) Reset index
        # ------------------------------------------------------
        with self.timelog.block("reset_index"):
            X = X.reset_index(drop=True)

        # ------------------------------------------------------
        # 4) Cat-as-num or numeric selection
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
                    self.used_base_cols = []
                    self.new_feats = []
                    self.order_batches = {}
                    self.time_logs = {}
                    return self
            else:
                X = X[num_cols]

        # ------------------------------------------------------
        # 5) Column subsampling for base features and clean column names
        # ------------------------------------------------------
        with self.timelog.block("column_subsample"):
            if X.shape[1] > self.max_base_feats:
                if self.verbose:
                    print(f"Limiting base features to {self.max_base_feats} (from {X.shape[1]})")
                X = X.sample(n=self.max_base_feats, random_state=42, axis=1)

            X, self.columns_name_map = clean_column_names(X)

        # ------------------------------------------------------
        # 6) Memory reduction
        # ------------------------------------------------------
        if self.reduce_memory and X.shape[1] > 0:
            with self.timelog.block("reduce_memory_usage_base"):
                X = reduce_memory_usage(X, rescale=self.rescale_avoid_overflow, verbose=self.verbose)

        # ------------------------------------------------------
        # 7) Basic filtering
        # ------------------------------------------------------
        with self.timelog.block("basic_filter_base"):
            n_start = X.shape[1]
            X = basic_filter(
                X,
                use_polars=False,
                min_cardinality=self.min_cardinality,
                remove_constant_mostlynan=self.remove_constant_mostlynan,
            )

        if self.verbose:
            print(f"Using {X.shape[1]}/{n_start} base features after basic filtering")

        if X.shape[1] == 0:
            if self.verbose:
                print("All base features removed after basic filtering.")
            self.used_base_cols = []
            self.new_feats = []
            self.order_batches = {}
            self.time_logs = {}
            return self

        self.used_base_cols = X.columns.tolist()

        # ------------------------------------------------------
        # 8) Interaction generation
        # ------------------------------------------------------
        if self.selection_method == "random":
            with self.timelog.block("random_selection"):
                self.random_selection(X, y)
        else:
            with self.timelog.block("spearman_selection"):
                self.spearman_selection(X, y)

        # ------------------------------------------------------
        # 9) Build execution plan + numba warmup
        # ------------------------------------------------------
        with self.timelog.block("prepare_order_batches"):
            if self.inference_mode == "compiled_numba":
                self._prepare_order_batches()

        with self.timelog.block("numba_warmup"):
            if self.inference_mode == "compiled_numba":
                self._warmup_fused_orders()

        # ------------------------------------------------------
        # 10) Collect timing logs
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

        return self

    def _fit_transform(self, X: DataFrame, y: Series, **kwargs) -> Tuple[DataFrame, dict]:
        self._fit(X, y, **kwargs)
        X_out = self._transform(X)

        # features_out = list(X_out.columns)
        # type_group_map_special = {R_FLOAT: features_out}
        return X_out, dict()  # TODO: Unsure whether we need to return anything special here

    def _add_arithmetic(self, X_in, **kwargs):
        X = X_in  # we only read from X, no inplace mods

        if not self.order_batches:
            return pd.DataFrame(index=X.index)

        # Same preprocessing as in fit
        if self.cat_as_num:
            X = self.cat_as_num_preprocessor.transform(X)
        X = X.rename(columns=self.columns_name_map, errors="ignore")
        X = X[self.used_base_cols]

        # Base matrix and its transpose for better locality
        X_base = X.to_numpy(dtype="float64", copy=False)

        blocks = []
        col_names = []

        # Evaluate one fused batch per order
        for order in sorted(self.order_batches.keys()):
            batch = self.order_batches[order]
            idx_mat = batch["idx"]  # (n_feats_order, order)
            ops_mat = batch["ops"]  # (n_feats_order, order-1)

            if idx_mat.size == 0:
                continue

            block = eval_order_fused(X_base, idx_mat, ops_mat)  # (n_rows, n_feats_order)
            blocks.append(block)
            col_names.extend(batch["names"])

        if not blocks:
            return pd.DataFrame(index=X.index)

        out_mat = np.hstack(blocks)
        out_mat[np.isinf(out_mat)] = np.nan
        X_out = pd.DataFrame(out_mat, columns=col_names, index=X.index)
        # X_out = X_out.replace([np.inf, -np.inf], np.nan)

        return X_out

    def _add_arithmetic_dag(self, df_in, expressions):
        """
        Fast evaluator with DAG optimization.
        Fully compatible with the original function signature.
        Computes common sub-expressions only once.
        """
        # -----------------------------
        # 1) Preprocessing (unchanged)
        # -----------------------------
        df = df_in
        if self.cat_as_num:
            df = self.cat_as_num_preprocessor.transform(df)
        df = df.rename(columns=self.columns_name_map, errors="ignore")
        df = df[self.used_base_cols]

        # Ensure float (original behavior)
        arr = df.to_numpy()
        if not np.issubdtype(arr.dtype, np.floating):
            arr = arr.astype(float)

        # Column index lookup
        col_idx = {c: i for i, c in enumerate(self.used_base_cols)}

        # Map symbol → operator
        opmap = {
            "+": operator.add,
            "-": operator.sub,
            "*": operator.mul,
            "/": operator.truediv,
        }

        # Regex for splitting: same as original
        OP_SPLIT = re.compile(r"_(\+|\-|\*|/)_")

        # ---------------------------------------------------------------------
        # 2) Compile the expressions to a DAG, but only if they have changed
        # ---------------------------------------------------------------------
        compile_needed = (
            not hasattr(self, "_compiled_dag") or
            self._compiled_dag.get("exprs") != tuple(expressions)
        )

        if compile_needed:
            nodes = {}        # DAG nodes: key = (left, op, right)
            expr_roots = {}   # Final nodes per expression
            base_cols = {i for i in range(arr.shape[1])}

            for expr in expressions:
                tokens = OP_SPLIT.split(expr)
                # Example tokens: [colA, "/", colB, "-", colC]

                # Start with first column
                current = col_idx[tokens[0]]

                i = 1
                while i < len(tokens):
                    op = tokens[i]
                    right_col = col_idx[tokens[i+1]]

                    key = (current, op, right_col)
                    if key not in nodes:
                        nodes[key] = None  # placeholder

                    current = key
                    i += 2

                expr_roots[expr] = current

            # Store the DAG for future evaluations
            self._compiled_dag = {
                "nodes": nodes,
                "expr_roots": expr_roots,
                "exprs": tuple(expressions),
            }

        # ---------------------------------------------------------------------
        # 3) Evaluate DAG (this is now the fast part)
        # ---------------------------------------------------------------------
        nodes = self._compiled_dag["nodes"]
        expr_roots = self._compiled_dag["expr_roots"]

        # Base column arrays
        base_values = {i: arr[:, i] for i in range(arr.shape[1])}

        results = {}

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            # Evaluate all unique nodes
            for key in nodes:
                left, op, right = key

                left_arr = base_values[left] if isinstance(left, int) else results[left]
                right_arr = base_values[right] if isinstance(right, int) else results[right]

                results[key] = opmap[op](left_arr, right_arr)

        # ---------------------------------------------------------------------
        # 4) Build output DataFrame
        # ---------------------------------------------------------------------
        output = {}
        for expr in expressions:
            root = expr_roots[expr]
            if isinstance(root, int):
                output[expr] = base_values[root]
                output[expr][np.isinf(output[expr])] = np.nan
            else:
                output[expr] = results[root]
                output[expr][np.isinf(output[expr])] = np.nan
        return pd.DataFrame(output, index=df.index)


    def _transform(self, X: DataFrame) -> DataFrame:
        # Fast path: if no interaction plan exists, just return X unchanged
        if self.inference_mode == 'compiled_numba':
            if not self.order_batches:
                return X
            X_new = self._add_arithmetic(X)
        elif self.inference_mode == 'dag':
            if len(self.new_feats) == 0:
                return X
            X_new = self._add_arithmetic_dag(X, self.new_feats)

        # If nothing was generated (e.g. all pruned), avoid concat overhead
        if X_new.empty:
            return X

        # All columns in X_new are numeric floats created from numpy, so
        # cast the whole block at once instead of per-column astype.
        if np.issubdtype(self.out_dtype, np.floating):
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                arr = X_new.to_numpy(dtype=self.out_dtype, copy=False)
            X_new = pd.DataFrame(arr, index=X_new.index, columns=X_new.columns)

        # Combine original and interaction features
        return pd.concat([X, X_new], axis=1)

    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return dict()  # TODO: Unsure what to include here
