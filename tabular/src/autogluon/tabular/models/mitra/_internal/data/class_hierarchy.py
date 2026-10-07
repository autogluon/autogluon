"""Balanced label decomposition for Mitra's fixed-width classification head."""

from __future__ import annotations

import numpy as np


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
            probabilities = node.estimator.predict_proba(X)
            for i, (group, child) in enumerate(zip(node.groups, node.children)):
                group_mass = mass * probabilities[:, i]
                if child is None:
                    result[:, group[0]] = group_mass
                else:
                    visit(child, group_mass)

        visit(self.root, np.ones(len(X)))
        return result / result.sum(axis=1, keepdims=True)
