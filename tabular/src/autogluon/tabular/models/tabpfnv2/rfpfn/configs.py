#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class TabPFNRFConfig:
    min_samples_split: int = 1000
    min_samples_leaf: int = 5
    max_depth: int = 5
    splitter: Literal["best", "random"] = "best"
    n_estimators: int = 16
    max_features: Literal["sqrt", "auto"] = "sqrt"
    criterion: Literal[
        "gini",
        "entropy",
        "log_loss",
        "squared_error",
        "friedman_mse",
        "poisson",
    ] = "gini"
    preprocess_X: bool = False
    preprocess_X_once: bool = False
    adaptive_tree: bool = True
    fit_nodes: bool = True
    adaptive_tree_overwrite_metric: Literal["logloss", "roc"] = None
    adaptive_tree_test_size: float = 0.2
    adaptive_tree_min_train_samples: int = 100
    adaptive_tree_min_valid_samples_fraction_of_train: int = 0.2
    adaptive_tree_max_train_samples: int = 5000
    adaptive_tree_skip_class_missing: bool = True
    max_predict_time: float = -1

    bootstrap: bool = True
    rf_average_logits: bool = False
    dt_average_logits: bool = True
