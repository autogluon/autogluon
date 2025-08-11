"""Random Forest implementation that uses TabPFN at the leaf nodes."""

#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0

from __future__ import annotations

import logging
import time

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.utils.multiclass import unique_labels

from .sklearn_based_decision_tree_tabpfn import (
    DecisionTreeTabPFNClassifier,
    DecisionTreeTabPFNRegressor,
)
from .sklearn_compat import validate_data

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("RF-PFN")


def softmax_numpy(logits: np.ndarray) -> np.ndarray:
    """Apply softmax to numpy array of logits.

    Args:
        logits: Input logits array

    Returns:
        Probabilities after softmax
    """
    exp_logits = np.exp(logits)  # Apply exponential to each logit
    sum_exp_logits = np.sum(
        exp_logits,
        axis=-1,
        keepdims=True,
    )  # Sum of exponentials across classes
    return exp_logits / sum_exp_logits  # Normalize to get probabilities


class RandomForestTabPFNBase:
    """Base Class for common functionalities."""

    def get_n_estimators(self, X: np.ndarray) -> int:
        """Get the number of estimators to use.

        Args:
            X: Input features

        Returns:
            Number of estimators
        """
        return self.n_estimators

    def _validate_tabpfn(self):
        """Validate that tabpfn is not None and is of the correct type.

        Raises:
            ValueError: If tabpfn is None
            TypeError: If tabpfn is not of the expected type
        """
        if self.tabpfn is None:
            raise ValueError(
                f"The tabpfn parameter cannot be None. Please provide a TabPFN{'Classifier' if self.task_type == 'multiclass' else 'Regressor'} instance.",
            )

        if self.task_type == "multiclass":
            # For classifier, check for predict_proba method
            if not hasattr(self.tabpfn, "predict_proba"):
                raise TypeError(
                    f"Expected a TabPFNClassifier instance with predict_proba method, but got {type(self.tabpfn).__name__}",
                )
        else:
            # For regressor, check for predict method but no predict_proba
            if not hasattr(self.tabpfn, "predict"):
                raise TypeError(
                    f"Expected a TabPFNRegressor instance with predict method, but got {type(self.tabpfn).__name__}",
                )
            if hasattr(self.tabpfn, "predict_proba"):
                raise TypeError(
                    "Expected a TabPFNRegressor instance, but got a classifier with predict_proba method. "
                    "Please use TabPFNRegressor with RandomForestTabPFNRegressor.",
                )

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray = None):
        """Fits RandomForestTabPFN.

        Args:
            X: Feature training data
            y: Label training data
            sample_weight: Weights of each sample

        Returns:
            Fitted model

        Raises:
            ValueError: If n_estimators is not positive
            ValueError: If tabpfn is None
            TypeError: If tabpfn is not of the expected type
        """
        # Validate tabpfn parameter
        self._validate_tabpfn()

        self.estimator = self.init_base_estimator()
        self.X = X
        self.n_estimators = self.get_n_estimators(X)

        X, y = validate_data(
            self,
            X,
            y,
            ensure_all_finite=False,
        )

        if self.task_type == "multiclass":
            self.classes_ = unique_labels(y)
            self.n_classes_ = len(self.classes_)

        # Special case for depth 0 - just use TabPFN directly
        if self.max_depth == 0:
            self.tabpfn.fit(X, y)
            return self

        # Initialize the tree estimators - convert to Python int to ensure client compatibility
        n_estimators = int(self.n_estimators) if hasattr(self.n_estimators, "item") else self.n_estimators
        if n_estimators <= 0:
            raise ValueError(
                f"n_estimators must be greater than zero, got {n_estimators}",
            )

        # Initialize estimators list
        self.estimators_ = []

        # Generate bootstrapped datasets and fit trees
        for i in range(n_estimators):
            # Clone the base estimator
            tree = self.init_base_estimator()

            # Bootstrap sample if requested (like in RandomForest)
            if self.bootstrap:
                n_samples = X.shape[0]

                # Convert max_samples to Python int if needed for client compatibility
                max_samples = self.max_samples
                if max_samples is not None and hasattr(max_samples, "item"):
                    max_samples = int(max_samples)

                # Calculate sample size (convert to Python int for client compatibility)
                sample_size = n_samples if max_samples is None else int(max_samples * n_samples)
                sample_size = int(sample_size) if hasattr(sample_size, "item") else sample_size

                # Generate random indices for bootstrapping
                indices = np.random.choice(
                    n_samples,
                    size=sample_size,
                    replace=True,
                )

                # Handle pandas DataFrame properly by converting to numpy or using iloc
                if hasattr(X, "iloc") and hasattr(
                    X,
                    "values",
                ):  # It's a pandas DataFrame
                    X_boot = X.iloc[indices].values if hasattr(X, "values") else X.iloc[indices]
                    y_boot = (
                        y[indices]
                        if isinstance(y, np.ndarray)
                        else y.iloc[indices]
                        if hasattr(y, "iloc")
                        else np.array(y)[indices]
                    )
                else:  # It's a numpy array or similar
                    X_boot = X[indices]
                    y_boot = y[indices]
            else:
                X_boot = X
                y_boot = y

            # Fit the tree on bootstrapped data
            tree.fit(X_boot, y_boot)
            self.estimators_.append(tree)

        # Track features seen during fit
        self.n_features_in_ = X.shape[1]

        # Set flag to indicate successful fit
        self._fitted = True

        return self


class RandomForestTabPFNClassifier(RandomForestTabPFNBase, RandomForestClassifier):
    """RandomForestTabPFNClassifier implements Random Forest using TabPFN at leaf nodes.

    This classifier combines decision trees with TabPFN models at the leaf nodes for
    improved performance on tabular data. It extends scikit-learn's RandomForestClassifier
    with TabPFN's neural network capabilities.

    Parameters:
        tabpfn: TabPFNClassifier instance to use at leaf nodes
        n_jobs: Number of parallel jobs
        categorical_features: List of categorical feature indices
        show_progress: Whether to display progress during fitting
        verbose: Verbosity level (0=quiet, >0=verbose)
        adaptive_tree: Whether to use adaptive tree-based method
        fit_nodes: Whether to fit the leaf node models
        adaptive_tree_overwrite_metric: Metric used for adaptive node fitting
        adaptive_tree_test_size: Test size for adaptive node fitting
        adaptive_tree_min_train_samples: Minimum samples for training leaf nodes
        adaptive_tree_max_train_samples: Maximum samples for training leaf nodes
        adaptive_tree_min_valid_samples_fraction_of_train: Min fraction of validation samples
        preprocess_X_once: Whether to preprocess X only once
        max_predict_time: Maximum time allowed for prediction (seconds)
        rf_average_logits: Whether to average logits instead of probabilities
        dt_average_logits: Whether to average logits in decision trees
        adaptive_tree_skip_class_missing: Whether to skip classes missing in nodes
        n_estimators: Number of trees in the forest
        criterion: Function to measure split quality
        max_depth: Maximum depth of the trees
        min_samples_split: Minimum samples required to split a node
        min_samples_leaf: Minimum samples required at a leaf node
        min_weight_fraction_leaf: Minimum weighted fraction of sum total
        max_features: Number of features to consider for best split
        max_leaf_nodes: Maximum number of leaf nodes
        min_impurity_decrease: Minimum impurity decrease required for split
        bootstrap: Whether to use bootstrap samples
        oob_score: Whether to use out-of-bag samples
        random_state: Controls randomness of the estimator
        warm_start: Whether to reuse previous solution
        class_weight: Weights associated with classes
        ccp_alpha: Complexity parameter for minimal cost-complexity pruning
        max_samples: Number of samples to draw to train each tree
    """

    task_type = "multiclass"

    def __init__(
        self,
        tabpfn=None,
        n_jobs=1,
        categorical_features=None,
        show_progress=False,
        verbose=0,
        adaptive_tree=True,
        fit_nodes=True,
        adaptive_tree_overwrite_metric="log_loss",
        adaptive_tree_test_size=0.2,
        adaptive_tree_min_train_samples=100,
        adaptive_tree_max_train_samples=5000,
        adaptive_tree_min_valid_samples_fraction_of_train=0.2,
        preprocess_X_once=False,
        max_predict_time=60,
        rf_average_logits=True,
        dt_average_logits=True,
        adaptive_tree_skip_class_missing=True,
        # Added to make cloneable.
        n_estimators=100,
        criterion="gini",
        max_depth=5,
        min_samples_split=1000,
        min_samples_leaf=5,
        min_weight_fraction_leaf=0.0,
        max_features="sqrt",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        random_state=None,
        warm_start=False,
        class_weight=None,
        ccp_alpha=0.0,
        max_samples=None,
    ):
        super().__init__(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            verbose=verbose,
            n_jobs=n_jobs,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap,
            oob_score=oob_score,
            random_state=random_state,
            warm_start=warm_start,
            class_weight=class_weight,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples,
        )

        if tabpfn is None:
            raise ValueError(
                "The tabpfn parameter cannot be None. Please provide a TabPFNClassifier instance.",
            )

        # Check if tabpfn is a classifier instance
        if not hasattr(tabpfn, "predict_proba"):
            raise TypeError(
                f"Expected a TabPFNClassifier instance with predict_proba method, but got {type(tabpfn).__name__}",
            )

        self.tabpfn = tabpfn

        self.categorical_features = categorical_features
        self.show_progress = show_progress
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.adaptive_tree = adaptive_tree
        self.fit_nodes = fit_nodes
        self.adaptive_tree_overwrite_metric = adaptive_tree_overwrite_metric
        self.adaptive_tree_test_size = adaptive_tree_test_size
        self.adaptive_tree_min_train_samples = adaptive_tree_min_train_samples
        self.adaptive_tree_max_train_samples = adaptive_tree_max_train_samples
        self.adaptive_tree_min_valid_samples_fraction_of_train = adaptive_tree_min_valid_samples_fraction_of_train
        self.preprocess_X_once = preprocess_X_once
        self.max_predict_time = max_predict_time
        self.rf_average_logits = rf_average_logits
        self.dt_average_logits = dt_average_logits
        self.adaptive_tree_skip_class_missing = adaptive_tree_skip_class_missing
        self.n_estimators = n_estimators

    def _more_tags(self):
        return {
            "allow_nan": True,
        }

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        tags.estimator_type = "regressor"
        if self.task_type == "multiclass":
            tags.estimator_type = "classifier"
        else:
            tags.estimator_type = "regressor"
        return tags

    def init_base_estimator(self):
        """Initialize a base decision tree estimator.

        Returns:
            A new DecisionTreeTabPFNClassifier instance
        """
        return DecisionTreeTabPFNClassifier(
            tabpfn=self.tabpfn,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
            categorical_features=self.categorical_features,
            max_depth=self.max_depth,
            show_progress=self.show_progress,
            adaptive_tree=self.adaptive_tree,
            fit_nodes=self.fit_nodes,
            verbose=self.verbose,
            adaptive_tree_test_size=self.adaptive_tree_test_size,
            adaptive_tree_overwrite_metric=self.adaptive_tree_overwrite_metric,
            adaptive_tree_min_train_samples=self.adaptive_tree_min_train_samples,
            adaptive_tree_max_train_samples=self.adaptive_tree_max_train_samples,
            adaptive_tree_min_valid_samples_fraction_of_train=self.adaptive_tree_min_valid_samples_fraction_of_train,
            average_logits=self.dt_average_logits,
            adaptive_tree_skip_class_missing=self.adaptive_tree_skip_class_missing,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class for X.

        The predicted class of an input sample is a vote by the trees in
        the forest, weighted by their probability estimates. That is,
        the predicted class is the one with highest mean probability
        estimate across the trees.

        Parameters:
            X: {array-like, sparse matrix} of shape (n_samples, n_features)
               The input samples.

        Returns:
            y: ndarray of shape (n_samples,)
               The predicted classes.

        Raises:
            ValueError: If model is not fitted
        """
        # Get class probabilities
        proba = self.predict_proba(X)

        # Return class with highest probability
        if hasattr(self, "classes_"):
            return self.classes_.take(np.argmax(proba, axis=1), axis=0)
        else:
            return np.argmax(proba, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for X.

        The predicted class probabilities of an input sample are computed as
        the mean predicted class probabilities of the trees in the forest.

        Parameters:
            X: {array-like, sparse matrix} of shape (n_samples, n_features)
               The input samples.

        Returns:
            p: ndarray of shape (n_samples, n_classes)
               The class probabilities of the input samples.

        Raises:
            ValueError: If model is not fitted
        """
        # Check if fitted
        if not hasattr(self, "_fitted") or not self._fitted:
            raise ValueError(
                "This RandomForestTabPFNClassifier instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this estimator.",
            )

        # Convert input if needed
        if torch.is_tensor(X):
            X = X.numpy()

        # Special case for depth 0 - TabPFN can handle missing values directly
        if self.max_depth == 0:
            # No need for preprocessing - TabPFN handles NaN values
            return self.tabpfn.predict_proba(X)

        # First collect all the classes from all estimators to ensure we handle all possible classes
        if not hasattr(self, "classes_"):
            all_classes_sets = [set(np.unique(estimator.classes_)) for estimator in self.estimators_]
            all_classes = sorted(set().union(*all_classes_sets))
            self.classes_ = np.array(all_classes)
            self.n_classes_ = len(self.classes_)

        # Initialize probabilities array
        n_samples = X.shape[0]
        all_proba = np.zeros((n_samples, self.n_classes_), dtype=np.float64)

        # Accumulate predictions from trees
        start_time = time.time()
        evaluated_estimators = 0

        for estimator in self.estimators_:
            # Get predictions from this tree
            proba = estimator.predict_proba(X)

            # If this estimator has fewer classes than the overall set, expand it
            if proba.shape[1] < self.n_classes_:
                expanded_proba = np.zeros(
                    (n_samples, self.n_classes_),
                    dtype=np.float64,
                )
                for i, class_val in enumerate(estimator.classes_):
                    # Find the index of this class in the overall classes array
                    idx = np.where(self.classes_ == class_val)[0][0]
                    expanded_proba[:, idx] = proba[:, i]
                proba = expanded_proba

            # Convert to logits if needed
            if self.rf_average_logits:
                proba = np.log(proba + 1e-10)  # Add small constant to avoid log(0)

            # Accumulate
            all_proba += proba

            # Check timeout
            evaluated_estimators += 1
            time_elapsed = time.time() - start_time
            if time_elapsed > self.max_predict_time and self.max_predict_time > 0:
                break

        # Average probabilities
        all_proba /= evaluated_estimators

        # Convert back from logits if needed
        if self.rf_average_logits:
            all_proba = softmax_numpy(all_proba)

        return all_proba


class RandomForestTabPFNRegressor(RandomForestTabPFNBase, RandomForestRegressor):
    """RandomForestTabPFNRegressor implements a Random Forest using TabPFN at leaf nodes.

    This regressor combines decision trees with TabPFN models at the leaf nodes for
    improved regression performance on tabular data. It extends scikit-learn's
    RandomForestRegressor with TabPFN's neural network capabilities.

    Parameters:
        tabpfn: TabPFNRegressor instance to use at leaf nodes
        n_jobs: Number of parallel jobs
        categorical_features: List of categorical feature indices
        show_progress: Whether to display progress during fitting
        verbose: Verbosity level (0=quiet, >0=verbose)
        adaptive_tree: Whether to use adaptive tree-based method
        fit_nodes: Whether to fit the leaf node models
        adaptive_tree_overwrite_metric: Metric used for adaptive node fitting
        adaptive_tree_test_size: Test size for adaptive node fitting
        adaptive_tree_min_train_samples: Minimum samples for training leaf nodes
        adaptive_tree_max_train_samples: Maximum samples for training leaf nodes
        adaptive_tree_min_valid_samples_fraction_of_train: Min fraction of validation samples
        preprocess_X_once: Whether to preprocess X only once
        max_predict_time: Maximum time allowed for prediction (seconds)
        rf_average_logits: Whether to average logits instead of raw predictions
        n_estimators: Number of trees in the forest
        criterion: Function to measure split quality
        max_depth: Maximum depth of the trees
        min_samples_split: Minimum samples required to split a node
        min_samples_leaf: Minimum samples required at a leaf node
        min_weight_fraction_leaf: Minimum weighted fraction of sum total
        max_features: Number of features to consider for best split
        max_leaf_nodes: Maximum number of leaf nodes
        min_impurity_decrease: Minimum impurity decrease required for split
        bootstrap: Whether to use bootstrap samples
        oob_score: Whether to use out-of-bag samples
        random_state: Controls randomness of the estimator
        warm_start: Whether to reuse previous solution
        ccp_alpha: Complexity parameter for minimal cost-complexity pruning
        max_samples: Number of samples to draw to train each tree
    """

    task_type = "regression"

    def _more_tags(self):
        return {
            "allow_nan": True,
        }

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        tags.estimator_type = "regressor"
        return tags

    def __init__(
        self,
        tabpfn=None,
        n_jobs=1,
        categorical_features=None,
        show_progress=False,
        verbose=0,
        adaptive_tree=True,
        fit_nodes=True,
        adaptive_tree_overwrite_metric="rmse",
        adaptive_tree_test_size=0.2,
        adaptive_tree_min_train_samples=100,
        adaptive_tree_max_train_samples=5000,
        adaptive_tree_min_valid_samples_fraction_of_train=0.2,
        preprocess_X_once=False,
        max_predict_time=-1,
        rf_average_logits=False,
        # Added to make cloneable.
        n_estimators=16,
        criterion="friedman_mse",
        max_depth=5,
        min_samples_split=300,
        min_samples_leaf=5,
        min_weight_fraction_leaf=0.0,
        max_features="sqrt",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        random_state=None,
        warm_start=False,
        ccp_alpha=0.0,
        max_samples=None,
    ):
        super().__init__(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples,
        )

        self.tabpfn = tabpfn

        self.categorical_features = categorical_features
        self.show_progress = show_progress
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.adaptive_tree = adaptive_tree
        self.fit_nodes = fit_nodes
        self.adaptive_tree_overwrite_metric = adaptive_tree_overwrite_metric
        self.adaptive_tree_test_size = adaptive_tree_test_size
        self.adaptive_tree_min_train_samples = adaptive_tree_min_train_samples
        self.adaptive_tree_max_train_samples = adaptive_tree_max_train_samples
        self.adaptive_tree_min_valid_samples_fraction_of_train = adaptive_tree_min_valid_samples_fraction_of_train
        self.preprocess_X_once = preprocess_X_once
        self.max_predict_time = max_predict_time
        self.rf_average_logits = rf_average_logits

    def init_base_estimator(self):
        """Initialize a base decision tree estimator.

        Returns:
            A new DecisionTreeTabPFNRegressor instance
        """
        return DecisionTreeTabPFNRegressor(
            tabpfn=self.tabpfn,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
            categorical_features=self.categorical_features,
            max_depth=self.max_depth,
            show_progress=self.show_progress,
            adaptive_tree=self.adaptive_tree,
            fit_nodes=self.fit_nodes,
            verbose=self.verbose,
            adaptive_tree_test_size=self.adaptive_tree_test_size,
            adaptive_tree_overwrite_metric=self.adaptive_tree_overwrite_metric,
            adaptive_tree_min_train_samples=self.adaptive_tree_min_train_samples,
            adaptive_tree_max_train_samples=self.adaptive_tree_max_train_samples,
            adaptive_tree_min_valid_samples_fraction_of_train=self.adaptive_tree_min_valid_samples_fraction_of_train,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict regression target for X.

        The predicted regression target of an input sample is computed as the
        mean predicted regression targets of the trees in the forest.

        Parameters:
            X: {array-like, sparse matrix} of shape (n_samples, n_features)
               The input samples.

        Returns:
            y: ndarray of shape (n_samples,) or (n_samples, n_outputs)
               The predicted values.

        Raises:
            ValueError: If model is not fitted
        """
        # Check if fitted
        if not hasattr(self, "_fitted") or not self._fitted:
            raise ValueError(
                "This RandomForestTabPFNRegressor instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this estimator.",
            )

        X = validate_data(
            self,
            X,
            ensure_all_finite=False,
        )

        # Special case for depth 0 - TabPFN can handle missing values directly
        if self.max_depth == 0:
            # No need for preprocessing - TabPFN handles NaN values
            return self.tabpfn.predict(X)

        # Initialize output array
        n_samples = X.shape[0]
        self.n_outputs_ = 1  # Only supporting single output for now
        y_hat = np.zeros(n_samples, dtype=np.float64)

        # Accumulate predictions from trees
        start_time = time.time()
        evaluated_estimators = 0

        for estimator in self.estimators_:
            # Get predictions from this tree
            pred = estimator.predict(X)

            # Accumulate
            y_hat += pred

            # Check timeout
            evaluated_estimators += 1
            time_elapsed = time.time() - start_time
            if time_elapsed > self.max_predict_time and self.max_predict_time > 0:
                break

        # Average predictions
        y_hat /= evaluated_estimators

        return y_hat


def _accumulate_prediction(
    predict,
    X: np.ndarray,
    out: list[np.ndarray],
    accumulate_logits: bool = False,
) -> None:
    """This is a utility function for joblib's Parallel.

    It can't go locally in ForestClassifier or ForestRegressor, because joblib
    complains that it cannot pickle it when placed there.

    Args:
        predict: Prediction function to call
        X: Input data
        out: Output array to accumulate predictions into
        accumulate_logits: Whether to accumulate logits instead of probabilities
    """
    prediction = predict(X, check_input=False)

    if accumulate_logits:
        # convert multiclass probabilities to logits
        prediction = np.log(prediction + 1e-10)  # Add small value to avoid log(0)

    if len(out) == 1:
        out[0] += prediction
    else:
        for i in range(len(out)):
            out[i] += prediction[i]
