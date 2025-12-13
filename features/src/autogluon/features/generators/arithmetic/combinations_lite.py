from __future__ import annotations

from itertools import combinations, product

import numpy as np

from .combinations import filter_canonical_expressions


class Operation:
    def __init__(self, left, right, op: str):
        self.left = left
        self.right = right
        self.op = op

    def _render(self, x):
        if isinstance(x, Operation):
            return f"({x.name()})"
        return str(x)

    def name(self) -> str:
        left = self._render(self.left)
        right = self._render(self.right)
        return f"{left}{self.op}{right}"

    def __str__(self):
        return self.name()


def get_all_bivariate_interactions(
    base_feats: list[str],
    max_feats=2000,
    interaction_types=["/", "*", "-", "+"],
    random_state=None,
) -> list[Operation]:
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
            names = [Operation(a, b, "/") for a, b in zip(feat0, feat1)]
            new_data += names
        if "*" in interaction_types:
            names = [Operation(a, b, "*") for a, b in zip(feat0, feat1)]
            new_data += names
        if "-" in interaction_types:
            names = [Operation(a, b, "-") for a, b in zip(feat0, feat1)]
            new_data += names
        if "+" in interaction_types:
            names = [Operation(a, b, "+") for a, b in zip(feat0, feat1)]
            new_data += names
        if "/" in interaction_types:
            names = [Operation(b, a, "/") for a, b in zip(feat0, feat1)]
            new_data += names

    return new_data


def add_higher_interaction(
    base_feats: list[str],
    interact_feats: list[Operation],
    max_feats=2000,
    interaction_types=("/", "*", "-", "+"),
    random_state=None,
) -> list[Operation]:
    """
    Generate higher-order interaction features between two sets of numeric features.
    """
    if random_state is None or isinstance(random_state, int):
        rng = np.random.default_rng(random_state)
    else:
        rng = random_state

    # Generate valid column pairs (FIXME: flawed logic retained)
    all_pairs = [
        (i, j)
        for i, j in product(interact_feats, base_feats)
        if j not in i.name()
    ]

    if not all_pairs:
        return []

    all_pairs = np.array(all_pairs, dtype=object)
    all_pairs = all_pairs[
        rng.choice(len(all_pairs), min(len(all_pairs), max_feats), replace=False)
    ]

    feat0, feat1 = all_pairs.T

    new_ops: list[Operation] = []

    # --- Division (/), non-commutative
    if "/" in interaction_types:
        new_ops.extend(Operation(a, b, "/") for a, b in zip(feat0, feat1))

    # --- Subtraction (-), non-commutative
    if "-" in interaction_types:
        new_ops.extend(Operation(a, b, "-") for a, b in zip(feat0, feat1))

    # --- Multiplication (*), commutative
    if "*" in interaction_types:
        seen = {}
        for a, b in zip(feat0, feat1):
            key = tuple(sorted((a.name(), str(b))))
            if key not in seen:
                seen[key] = Operation(a, b, "*")
        new_ops.extend(seen.values())

    # --- Addition (+), commutative
    if "+" in interaction_types:
        seen = {}
        for a, b in zip(feat0, feat1):
            key = tuple(sorted((a.name(), str(b))))
            if key not in seen:
                seen[key] = Operation(a, b, "+")
        new_ops.extend(seen.values())

    # --- Canonical deduplication (string-based, unchanged)
    names = np.array([op.name() for op in new_ops])
    mask = filter_canonical_expressions(names)

    return [op for op, keep in zip(new_ops, mask) if keep]
