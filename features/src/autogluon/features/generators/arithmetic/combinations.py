import numpy as np
import pandas as pd
from itertools import combinations, product


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
    rng = np.random.default_rng(random_state)

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
    rng = np.random.default_rng(random_state)

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
    if "/" in interaction_types and len(new_data) < max_feats:
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            # Forward A/B
            res1 = X_interact_vals / np.where(X_base_vals == 0, np.nan, X_base_vals)
            names1 = [f"{a}_/_{b}" for a, b in zip(feat0, feat1)]
            new_data.update(dict(zip(names1, res1.T)))

    # --- Multiplication (*), commutative: remove duplicates
    if "*" in interaction_types and len(new_data) < max_feats:
        # Identify unique sorted pairs
        unique_pairs = {}
        for a, b in zip(feat0, feat1):
            key = tuple(sorted((a, b)))
            if key not in unique_pairs:
                unique_pairs[key] = (a, b)

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            # Compute all multiplications at once
            res_list = [(X_interact[a].values * X_base[b].values).astype(float) for (a, b) in unique_pairs.values()]
        name_list = [f"{a}_*_{b}" for (a, b) in unique_pairs.values()]

        if res_list:
            res_arr = np.column_stack(res_list)
            new_data.update(dict(zip(name_list, res_arr.T)))

    # --- Addition (+), commutative: remove duplicates
    if "+" in interaction_types and len(new_data) < max_feats:
        unique_pairs = {}
        for a, b in zip(feat0, feat1):
            key = tuple(sorted((a, b)))
            if key not in unique_pairs:
                unique_pairs[key] = (a, b)

        res_list = [(X_interact[a].values + X_base[b].values).astype(float) for (a, b) in unique_pairs.values()]
        name_list = [f"{a}_+_{b}" for (a, b) in unique_pairs.values()]

        if res_list:
            res_arr = np.column_stack(res_list)
            new_data.update(dict(zip(name_list, res_arr.T)))

    # --- Subtraction (−), non-commutative
    if "-" in interaction_types and len(new_data) < max_feats:
        res = X_interact_vals - X_base_vals
        names = [f"{a}_-_{b}" for a, b in zip(feat0, feat1)]
        new_data.update(dict(zip(names, res.T)))

    # if '/' in interaction_types and len(new_data) < max_feats:
    #     with np.errstate(divide='ignore', invalid='ignore', over="ignore"):
    #         # Reverse B/A
    #         res2 = X_base_vals / np.where(X_interact_vals == 0, np.nan, X_interact_vals)
    #         names2 = [f"{b}_/_{a}" for a, b in zip(feat0, feat1)]
    #         new_data.update(dict(zip(names2, res2.T)))

    # Build the new DataFrame once (fast)
    X_int_new = pd.DataFrame(new_data, index=X_interact.index)

    # Return combined features
    return X_int_new  # pd.concat([X_interact, X_int_new], axis=1)
