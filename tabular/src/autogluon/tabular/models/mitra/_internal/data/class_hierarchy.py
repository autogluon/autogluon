"""Balanced label decomposition for Mitra's fixed-width classification head."""

from __future__ import annotations

import numpy as np


def hierarchy_node_count(n_classes, max_classes=10):
    """Return the number of classifier nodes required by the hierarchy."""
    if n_classes < 1:
        raise ValueError("n_classes must be positive.")
    if max_classes < 2:
        raise ValueError("max_classes must be at least 2.")
    if n_classes <= max_classes:
        return 1
    n_groups = min(int(np.ceil(n_classes / max_classes)), max_classes)
    quotient, remainder = divmod(n_classes, n_groups)
    group_sizes = [quotient + (index < remainder) for index in range(n_groups)]
    return 1 + sum(hierarchy_node_count(size, max_classes=max_classes) for size in group_sizes if size > 1)


def _normalized_probabilities(probabilities, *, n_rows, n_classes):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    expected_shape = (n_rows, n_classes)
    if probabilities.shape != expected_shape:
        raise ValueError(f"Expected node probabilities shaped {expected_shape}, got {probabilities.shape}")
    if not np.isfinite(probabilities).all():
        raise ValueError("Hierarchy node returned non-finite probabilities")
    if (probabilities < -1e-7).any():
        raise ValueError("Hierarchy node returned negative probabilities")
    probabilities = np.maximum(probabilities, 0.0)
    row_sums = probabilities.sum(axis=1, keepdims=True)
    if (row_sums <= 0.0).any():
        raise ValueError("Hierarchy node returned a zero-sum probability row")
    return probabilities / row_sums


class ClassNode:
    def __init__(self, classes, groups, n_classes):
        self.classes = classes
        self.groups = groups
        self.estimator = None
        self.children = []
        self.mapping = np.full(n_classes, -1, dtype=np.int64)
        for i, group in enumerate(groups):
            self.mapping[group] = i

    def mask(self, y):
        return self.mapping[y] >= 0

    def encode(self, y):
        return self.mapping[y]


class ClassHierarchy:
    def __init__(self, n_classes, max_classes=10, seed=0):
        self.n_classes = n_classes
        self.nodes = []
        ordered = np.random.default_rng(seed).permutation(n_classes)
        self.root = self._build(ordered, max_classes)

    def _build(self, classes, max_classes):
        if len(classes) <= max_classes:
            groups = [np.asarray([value]) for value in classes]
        else:
            n_groups = min(int(np.ceil(len(classes) / max_classes)), max_classes)
            groups = list(np.array_split(classes, n_groups))
        node = ClassNode(classes, groups, self.n_classes)
        self.nodes.append(node)
        node.children = [self._build(group, max_classes) if len(group) > 1 else None for group in groups]
        return node

    def predict_proba(self, X):
        result = np.zeros((len(X), self.n_classes), dtype=np.float64)

        def visit(node, mass):
            probabilities = _normalized_probabilities(
                node.estimator.predict_proba(X),
                n_rows=len(X),
                n_classes=len(node.groups),
            )
            for i, (group, child) in enumerate(zip(node.groups, node.children)):
                group_mass = mass * probabilities[:, i]
                if child is None:
                    result[:, group[0]] = group_mass
                else:
                    visit(child, group_mass)

        visit(self.root, np.ones(len(X)))
        return _normalized_probabilities(result, n_rows=len(X), n_classes=self.n_classes)
