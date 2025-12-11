from itertools import combinations, product

import numpy as np

from .combinations import filter_canonical_expressions


# FIXME: What if `_` in feature name???
# FIXME: What if `*/-+` in feature name???
def get_all_bivariate_interactions(
    base_feats: list[str],
    max_feats=2000,
    interaction_types=["/", "*", "-", "+"],  # TODO: make inverse_div an own op
    random_state=None,
):
    """
    Generate all bivariate interactions of numeric features up to a maximum number of features.
    Parameters
    ----------
    base_feats : list[str]
        Input feature names.
    max_feats : int, default=2000
        Maximum number of interaction features to generate.
    interaction_types : list of str, default=['/', '*', '-', '+']
        Types of interactions to generate.
    random_state : int or None, default=None
        Random state for reproducibility.
    Returns
    -------
    list[str]
        List containing the generated interaction feature names.
    """
    if random_state is None or isinstance(random_state, int):
        rng = np.random.default_rng(random_state)
    else:
        rng = random_state

    cols = base_feats
    combs = np.array(list(combinations(cols, 2)))

    # Sample combinations directly instead of shuffling entire array
    combs = combs[rng.choice(len(combs), np.min([len(combs), max_feats]), replace=False)]

    feat0, feat1 = combs.T

    new_data = []

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if "/" in interaction_types:
            names = [f"{a}_/_{b}" for a, b in zip(feat0, feat1)]
            new_data += names
        if "*" in interaction_types:
            names = [f"{a}_*_{b}" for a, b in zip(feat0, feat1)]
            new_data += names
        if "-" in interaction_types:
            names = [f"{a}_-_{b}" for a, b in zip(feat0, feat1)]
            new_data += names
        if "+" in interaction_types:
            names = [f"{a}_+_{b}" for a, b in zip(feat0, feat1)]
            new_data += names
        if "/" in interaction_types:
            names = [f"{b}_/_{a}" for a, b in zip(feat0, feat1)]
            new_data += names

    return new_data


def add_higher_interaction(
    base_feats: list[str],
    interact_feats: list[str],
    max_feats=2000,
    interaction_types=[
        "/",
        "*",
        "-",
        "+",
    ],  # FIXME: Might need to fix bug if one of these operators occurs in feature names
    random_state=None,
) -> list[str]:
    """
    Generate higher-order interaction features between two sets of numeric features.
    Parameters
    ----------
    base_feats : list[str]
        Base numeric features.
    interact_feats : list[str]
        Numeric features to interact with base features.
    max_feats : int, default=2000
        Maximum number of interaction features to generate.
    interaction_types : list of str, default=['/', '*', '-', '+']
        Types of interactions to generate.
    random_state : int or None, default=None
        Random state for reproducibility.
    Returns
    -------
    list[str]
        List containing the generated higher-order interaction feature names.
    """
    if random_state is None or isinstance(random_state, int):
        rng = np.random.default_rng(random_state)
    else:
        rng = random_state

    # Generate valid column pairs (avoid j inside i for safety)
    all_pairs = [(i, j) for i, j in product(interact_feats, base_feats) if j not in i]
    all_pairs = np.array(all_pairs)
    all_pairs = all_pairs[rng.choice(len(all_pairs), np.min([len(all_pairs), max_feats]), replace=False)]
    feat0, feat1 = all_pairs.T

    new_data = []

    # --- Division (/)
    if "/" in interaction_types:
        name_arr = [f"{a}_/_{b}" for a, b in zip(feat0, feat1)]
        new_data += name_arr

    # --- Multiplication (*), commutative: remove duplicates
    if "*" in interaction_types:
        # Identify unique sorted pairs
        unique_pairs = {}
        for a, b in zip(feat0, feat1):
            key = tuple(sorted((a, b)))
            if key not in unique_pairs:
                unique_pairs[key] = (a, b)

        name_arr = [f"{a}_*_{b}" for (a, b) in unique_pairs.values()]
        new_data += name_arr

    # --- Addition (+), commutative: remove duplicates
    if "+" in interaction_types:
        unique_pairs = {}
        for a, b in zip(feat0, feat1):
            key = tuple(sorted((a, b)))
            if key not in unique_pairs:
                unique_pairs[key] = (a, b)

        name_arr = [f"{a}_+_{b}" for (a, b) in unique_pairs.values()]
        new_data += name_arr

    # --- Subtraction (−), non-commutative
    if "-" in interaction_types:
        name_arr = [f"{a}_-_{b}" for a, b in zip(feat0, feat1)]
        new_data += name_arr

    # Filter canonical expressions to remove duplicates
    new_data = np.array(new_data)
    canonical_expr_deduplicated = filter_canonical_expressions(new_data)
    new_data = new_data[canonical_expr_deduplicated]

    return new_data
