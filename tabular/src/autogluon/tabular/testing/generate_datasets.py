from __future__ import annotations

import pandas as pd
from sklearn.datasets import make_blobs

from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION, QUANTILE


def generate_toy_binary_dataset():
    label = "label"
    dummy_dataset = {
        "int": [0, 1, 2, 3],
        label: [0, 0, 1, 1],
    }

    dataset_info = {
        "problem_type": BINARY,
        "label": label,
    }

    train_data = pd.DataFrame(dummy_dataset)
    test_data = train_data
    return train_data, test_data, dataset_info


def generate_toy_multiclass_dataset():
    label = "label"
    dummy_dataset = {
        "int": [0, 1, 2, 3, 4, 5],
        label: [0, 0, 1, 1, 2, 2],
    }

    dataset_info = {
        "problem_type": MULTICLASS,
        "label": label,
    }

    train_data = pd.DataFrame(dummy_dataset)
    test_data = train_data
    return train_data, test_data, dataset_info


def generate_toy_regression_dataset():
    label = "label"
    dummy_dataset = {
        "int": [0, 1, 2, 3],
        label: [0.1, 0.9, 1.1, 1.9],
    }

    dataset_info = {
        "problem_type": REGRESSION,
        "label": label,
    }

    train_data = pd.DataFrame(dummy_dataset)
    test_data = train_data
    return train_data, test_data, dataset_info


def generate_toy_quantile_dataset():
    train_data, test_data, dataset_info = generate_toy_regression_dataset()
    dataset_info["problem_type"] = QUANTILE
    dataset_info["init_kwargs"] = {"quantile_levels": [0.25, 0.5, 0.75]}
    return train_data, test_data, dataset_info


def generate_toy_quantile_single_level_dataset():
    train_data, test_data, dataset_info = generate_toy_regression_dataset()
    dataset_info["problem_type"] = QUANTILE
    dataset_info["init_kwargs"] = {"quantile_levels": [0.71]}
    return train_data, test_data, dataset_info


def generate_toy_binary_10_dataset():
    label = "label"
    dummy_dataset = {
        "int": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        label: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
    }

    dataset_info = {
        "problem_type": BINARY,
        "label": label,
    }

    train_data = pd.DataFrame(dummy_dataset)
    test_data = train_data
    return train_data, test_data, dataset_info


def generate_toy_multiclass_10_dataset():
    label = "label"
    dummy_dataset = {
        "int": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        label: [0, 0, 1, 1, 2, 2, 0, 0, 1, 1],
    }

    dataset_info = {
        "problem_type": MULTICLASS,
        "label": label,
    }

    train_data = pd.DataFrame(dummy_dataset)
    test_data = train_data
    return train_data, test_data, dataset_info


def generate_toy_regression_10_dataset():
    label = "label"
    dummy_dataset = {
        "int": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        label: [0.1, 0.9, 1.1, 1.9, 0.2, 0.8, 1.2, 1.8, -0.1, 0.7],
    }

    dataset_info = {
        "problem_type": REGRESSION,
        "label": label,
    }

    train_data = pd.DataFrame(dummy_dataset)
    test_data = train_data
    return train_data, test_data, dataset_info


def generate_toy_quantile_10_dataset():
    train_data, test_data, dataset_info = generate_toy_regression_10_dataset()
    dataset_info["problem_type"] = QUANTILE
    dataset_info["init_kwargs"] = {"quantile_levels": [0.25, 0.5, 0.75]}
    return train_data, test_data, dataset_info


def generate_toy_multiclass_30_dataset():
    label = "label"
    train_data = generate_toy_multiclass_n_dataset(n_samples=30, n_features=2, n_classes=3)
    test_data = train_data

    dataset_info = {
        "problem_type": MULTICLASS,
        "label": label,
    }
    return train_data, test_data, dataset_info


def generate_toy_multiclass_n_dataset(n_samples, n_features, n_classes) -> pd.DataFrame:
    X, y = make_blobs(centers=n_classes, n_samples=n_samples, n_features=n_features, cluster_std=0.5, random_state=0)
    data = pd.DataFrame(X)
    data["label"] = y
    return data
