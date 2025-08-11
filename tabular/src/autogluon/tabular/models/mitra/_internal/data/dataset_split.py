from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

from ..._internal.config.enums import Task


def make_dataset_split(x: np.ndarray, y: np.ndarray, task: Task, seed: int) -> tuple[np.ndarray, ...]:
    # Splits the dataset into train and validation sets with ratio 80/20

    if task == Task.REGRESSION:
        return make_standard_dataset_split(x, y, seed=seed)

    size_of_smallest_class = np.min(np.bincount(y))

    if size_of_smallest_class >= 5:
        # stratification needs have at least 5 samples in each class if split is 80/20
        return make_stratified_dataset_split(x, y, seed=seed)
    else:
        return make_standard_dataset_split(x, y, seed=seed)


def make_stratified_dataset_split(x, y, n_splits=5, seed=0):
    if isinstance(seed, int):
        seed = np.random.RandomState(seed)

    # Stratify doesn't shuffle the data, so we shuffle it first
    permutation = seed.permutation(len(y))
    x, y = x[permutation], y[permutation]

    min_samples_per_class = np.min(np.bincount(y))

    # Adjust n_splits based on both total samples and minimum samples per class
    n_samples = len(y)
    max_possible_splits = min(n_samples - 1, min_samples_per_class)
    n_splits = min(n_splits, max_possible_splits)

    # Ensure we have at least 2 splits if possible
    if n_samples >= 2 and min_samples_per_class >= 2:
        n_splits = max(2, n_splits)
    else:
        # If we can't do stratified splitting, fall back to standard split
        return make_standard_dataset_split(x, y, seed)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    indices = next(skf.split(x, y))
    x_t_train, x_t_valid = x[indices[0]], x[indices[1]]  # 80%, 20%
    y_t_train, y_t_valid = y[indices[0]], y[indices[1]]

    return x_t_train, x_t_valid, y_t_train, y_t_valid


def make_standard_dataset_split(x, y, seed):
    return train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=seed,
    )
