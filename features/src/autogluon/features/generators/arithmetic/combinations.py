from __future__ import annotations

from itertools import combinations, product
from typing import Dict, Hashable, Iterable, List, Tuple

import numpy as np
import pandas as pd

# Precomputed translation table to strip parentheses quickly
_PAREN_TRANS = str.maketrans("", "", "()")


def estimate_no_higher_interaction_features(num_base_feats, num_new_feats):
    n_combinations = (num_base_feats - 2) * num_new_feats
    unique_div_add_sub = n_combinations * 0.8 * 3
    unique_prod = n_combinations * 0.4666666666666667
    return int(unique_div_add_sub + unique_prod)


def _expr_to_canonical_key(expr: str) -> Tuple:
    """
    Convert an interaction expression like 'A_/_B_*_C' or '(A_*_B)_-_C'
    into a hashable canonical key that is invariant to:

    - Commutativity of * and +
    - Associativity of * and +
    - Interaction of * and / (e.g. (A/B)*C == (C*A)/B)
    - Superfluous parentheses introduced by the generator

    The expression format is assumed to be:
        var op var op var ...
    with '_' separating tokens, and optional parentheses.
    """
    # Remove parentheses and split by the '_' tokens
    tokens = expr.translate(_PAREN_TRANS).split("_")
    if not tokens:
        return ()

    # Each monomial is represented by:
    #   frozenset({var: exponent, ...}.items()) -> coefficient (int)
    # The overall expression is a sum of such monomials.
    monoms: Dict[frozenset, int] = {frozenset({tokens[0]: 1}.items()): 1}

    # Process operator / variable pairs: (op, var), (op, var), ...
    # tokens = [v0, op1, v1, op2, v2, ...]
    it = iter(tokens[1:])
    for op, var in zip(it, it):
        if op in ("*", "/"):
            # Scale EVERY monomial by *var or /var
            delta = 1 if op == "*" else -1
            new_monoms: Dict[frozenset, int] = {}
            for exp_fs, coeff in monoms.items():
                exp_dict = dict(exp_fs)
                exp_dict[var] = exp_dict.get(var, 0) + delta
                if exp_dict[var] == 0:
                    del exp_dict[var]

                key = frozenset(exp_dict.items())
                new_monoms[key] = new_monoms.get(key, 0) + coeff
            monoms = new_monoms

        elif op in ("+", "-"):
            # Add a new monomial ± var (does NOT touch previous monoms)
            sign = 1 if op == "+" else -1
            key = frozenset({var: 1}.items())
            monoms[key] = monoms.get(key, 0) + sign

        else:
            raise ValueError(f"Unknown operator {op!r} in expression {expr!r}")

    # Build a canonical, hashable representation:
    #   key = tuple(sorted( (coeff, tuple(sorted((var, exp), ...))) ))
    canonical_items: List[Tuple[int, Tuple[Tuple[str, int], ...]]] = []
    for exp_fs, coeff in monoms.items():
        if coeff == 0:
            continue
        vars_exps = tuple(sorted(exp_fs))  # (var, exponent)
        canonical_items.append((coeff, vars_exps))

    canonical_items.sort()
    return tuple(canonical_items)


def filter_canonical_expressions(exprs: Iterable[str]) -> np.ndarray:
    """
    Given an iterable of expression strings, return indices of the
    *first* occurrence of each canonical expression.

    This is a drop-in replacement for the original Sympy-based
    implementation, but much faster and with no Sympy dependency.

    Parameters
    ----------
    exprs : Iterable[str]
        Expressions like 'A_*_B_-_C', '(A_/_B)_*_C', etc.

    Returns
    -------
    np.ndarray
        Indices of non-duplicate expressions (keep='first' semantics).
    """
    seen: Dict[Hashable, int] = {}
    keep_indices: List[int] = []

    for i, expr in enumerate(exprs):
        key = _expr_to_canonical_key(str(expr))
        if key not in seen:
            seen[key] = i
            keep_indices.append(i)

    out = np.asarray(keep_indices, dtype=int)
    return out


def get_all_bivariate_interactions(
    X_num,
    max_feats=2000,
    interaction_types=["/", "*", "-", "+"],  # TODO: make inverse_div an own op
    random_state=None,
):
    """
    Generate all bivariate interactions of numeric features up to a maximum number of features.
    Parameters
    ----------
    X_num : pd.DataFrame
        Input numeric DataFrame.
    max_feats : int, default=2000
        Maximum number of interaction features to generate.
    interaction_types : list of str, default=['/', '*', '-', '+']
        Types of interactions to generate.
    random_state : int or None, default=None
        Random state for reproducibility.
    Returns
    -------
    pd.DataFrame
        DataFrame containing the generated interaction features.
    """
    if random_state is None or isinstance(random_state, int):
        rng = np.random.default_rng(random_state)
    else:
        rng = random_state

    cols = X_num.columns.to_numpy()
    combs = np.array(list(combinations(cols, 2)))

    # Sample combinations directly instead of shuffling entire array
    combs = combs[rng.choice(len(combs), np.min([len(combs), max_feats]), replace=False)]

    feat0, feat1 = combs.T

    # Pre-cache arrays for speed
    arr = X_num.to_numpy()
    col_idx = {c: i for i, c in enumerate(cols)}

    # Helper to quickly extract columns
    def get_pair_arrays(f0, f1):
        return arr[:, [col_idx[c] for c in f0]], arr[:, [col_idx[c] for c in f1]]

    new_data = {}

    A, B = get_pair_arrays(feat0, feat1)
    # Avoid division by zero with masking instead of replace()
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if "/" in interaction_types:
            div1 = A / np.where(B == 0, np.nan, B)
            names = [f"{a}_/_{b}" for a, b in zip(feat0, feat1)]
            new_data.update(dict(zip(names, div1.T)))
        if "*" in interaction_types:
            names = [f"{a}_*_{b}" for a, b in zip(feat0, feat1)]
            new_data.update(dict(zip(names, (A * B).T)))
        if "-" in interaction_types:
            names = [f"{a}_-_{b}" for a, b in zip(feat0, feat1)]
            new_data.update(dict(zip(names, (A - B).T)))
        if "+" in interaction_types:
            names = [f"{a}_+_{b}" for a, b in zip(feat0, feat1)]
            new_data.update(dict(zip(names, (A + B).T)))
        if "/" in interaction_types:
            div2 = B / np.where(A == 0, np.nan, A)
            names = [f"{b}_/_{a}" for a, b in zip(feat0, feat1)]
            new_data.update(dict(zip(names, div2.T)))

    return pd.DataFrame(new_data, index=X_num.index)


def add_higher_interaction(
    X_base,
    X_interact,
    max_feats=2000,
    interaction_types=[
        "/",
        "*",
        "-",
        "+",
    ],  # FIXME: Might need to fix bug if one of these operators occurs in feature names
    random_state=None,
):
    """
    Generate higher-order interaction features between two sets of numeric features.
    Parameters
    ----------
    X_base : pd.DataFrame
        Base numeric features DataFrame.
    X_interact : pd.DataFrame
        Numeric features DataFrame to interact with base features.
    max_feats : int, default=2000
        Maximum number of interaction features to generate.
    interaction_types : list of str, default=['/', '*', '-', '+']
        Types of interactions to generate.
    random_state : int or None, default=None
        Random state for reproducibility.
    Returns
    -------
    pd.DataFrame
        DataFrame containing the generated higher-order interaction features.
    """
    if random_state is None or isinstance(random_state, int):
        rng = np.random.default_rng(random_state)
    else:
        rng = random_state

    # Generate valid column pairs (avoid j inside i for safety)
    all_pairs = [(i, j) for i, j in product(X_interact.columns, X_base.columns) if j not in i]
    all_pairs = np.array(all_pairs)
    all_pairs = all_pairs[rng.choice(len(all_pairs), np.min([len(all_pairs), max_feats]), replace=False)]
    feat0, feat1 = all_pairs.T

    # Convert to numpy arrays once for speed
    X_interact_vals = X_interact[feat0].to_numpy(dtype=float)
    X_base_vals = X_base[feat1].to_numpy(dtype=float)

    new_data = {}

    # --- Division (/)
    if "/" in interaction_types:
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            # Forward A/B
            res_arr = X_interact_vals / np.where(X_base_vals == 0, np.nan, X_base_vals)
            name_arr = np.array([f"{a}_/_{b}" for a, b in zip(feat0, feat1)])
            canonical_expr_deduplicated = filter_canonical_expressions(name_arr)
            new_data.update(dict(zip(name_arr[canonical_expr_deduplicated], res_arr.T[canonical_expr_deduplicated])))

    # --- Multiplication (*), commutative: remove duplicates
    if "*" in interaction_types:
        # Identify unique sorted pairs
        unique_pairs = {}
        for a, b in zip(feat0, feat1):
            key = tuple(sorted((a, b)))
            if key not in unique_pairs:
                unique_pairs[key] = (a, b)

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            # Compute all multiplications at once
            res_list = [(X_interact[a].values * X_base[b].values).astype(float) for (a, b) in unique_pairs.values()]
        name_arr = np.array([f"{a}_*_{b}" for (a, b) in unique_pairs.values()])

        if res_list:
            res_arr = np.column_stack(res_list)
            canonical_expr_deduplicated = filter_canonical_expressions(name_arr)
            new_data.update(dict(zip(name_arr[canonical_expr_deduplicated], res_arr.T[canonical_expr_deduplicated])))

    # --- Addition (+), commutative: remove duplicates
    if "+" in interaction_types:
        unique_pairs = {}
        for a, b in zip(feat0, feat1):
            key = tuple(sorted((a, b)))
            if key not in unique_pairs:
                unique_pairs[key] = (a, b)

        res_list = [(X_interact[a].values + X_base[b].values).astype(float) for (a, b) in unique_pairs.values()]
        name_arr = np.array([f"{a}_+_{b}" for (a, b) in unique_pairs.values()])

        if res_list:
            res_arr = np.column_stack(res_list)
            canonical_expr_deduplicated = filter_canonical_expressions(name_arr)
            new_data.update(dict(zip(name_arr[canonical_expr_deduplicated], res_arr.T[canonical_expr_deduplicated])))

    # --- Subtraction (−), non-commutative
    if "-" in interaction_types:
        res_arr = X_interact_vals - X_base_vals
        name_arr = np.array([f"{a}_-_{b}" for a, b in zip(feat0, feat1)])
        canonical_expr_deduplicated = filter_canonical_expressions(name_arr)
        new_data.update(dict(zip(name_arr[canonical_expr_deduplicated], res_arr.T[canonical_expr_deduplicated])))

    # Build the new DataFrame once (fast)
    X_int_new = pd.DataFrame(new_data, index=X_interact.index)

    # Filter canonical expressions to remove duplicates
    canonical_expr_deduplicated = filter_canonical_expressions(X_int_new.columns.tolist())
    X_int_new = X_int_new.iloc[:, canonical_expr_deduplicated]

    # Return combined features
    return X_int_new  # pd.concat([X_interact, X_int_new], axis=1)
