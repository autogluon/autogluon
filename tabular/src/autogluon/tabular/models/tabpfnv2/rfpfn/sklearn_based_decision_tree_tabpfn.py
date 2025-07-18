# Copyright (c) Prior Labs GmbH 2025.
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import random
import warnings

# For type checking only
from typing import TYPE_CHECKING, Any
from copy import deepcopy

import numpy as np
import torch

if TYPE_CHECKING:
    from numpy.typing import NDArray
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import (
    BaseDecisionTree,
    DecisionTreeClassifier,
    DecisionTreeRegressor,
)
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import (
    _check_sample_weight,
    check_is_fitted,
)

from .sklearn_compat import validate_data
from .scoring_utils import (
    score_classification,
    score_regression,
)
from .utils import softmax

###############################################################################
#                             BASE DECISION TREE                              #
###############################################################################


class DecisionTreeTabPFNBase(BaseDecisionTree, BaseEstimator):
    """Abstract base class combining a scikit-learn Decision Tree with TabPFN at the leaves.

    This class provides a hybrid approach by combining the standard decision tree
    splitting algorithm from scikit-learn with TabPFN models at the leaves or
    internal nodes. This allows for both interpretable tree-based partitioning
    and high-performance TabPFN prediction.

    Key features:
    -------------
    • Inherits from sklearn's BaseDecisionTree to leverage standard tree splitting algorithms
    • Uses TabPFN (Classifier or Regressor) to fit leaf nodes (or all internal nodes)
    • Provides adaptive pruning logic (optional) that dynamically determines optimal tree depth
    • Supports both classification and regression through specialized subclasses

    Subclasses:
    -----------
    • DecisionTreeTabPFNClassifier - for classification tasks
    • DecisionTreeTabPFNRegressor - for regression tasks

    Parameters
    ----------
    tabpfn : Any
        A TabPFN instance (TabPFNClassifier or TabPFNRegressor) that will be used at tree nodes.
    criterion : str
        The function to measure the quality of a split (from sklearn).
    splitter : str
        The strategy used to choose the split at each node (e.g. "best" or "random").
    max_depth : int, optional
        The maximum depth of the tree (None means unlimited).
    min_samples_split : int
        The minimum number of samples required to split an internal node.
    min_samples_leaf : int
        The minimum number of samples required to be at a leaf node.
    min_weight_fraction_leaf : float
        The minimum weighted fraction of the sum total of weights required to be at a leaf node.
    max_features : Union[int, float, str, None]
        The number of features to consider when looking for the best split.
    random_state : Union[int, np.random.RandomState, None]
        Controls the randomness of the estimator.
    max_leaf_nodes : Optional[int]
        If not None, grow a tree with max_leaf_nodes in best-first fashion.
    min_impurity_decrease : float
        A node will be split if this split induces a decrease of the impurity >= this value.
    class_weight : Optional[Union[Dict[int, float], str]]
        Only used in classification. Dict of class -> weight or “balanced”.
    ccp_alpha : float
        Complexity parameter used for Minimal Cost-Complexity Pruning (non-negative).
    monotonic_cst : Any
        Optional monotonicity constraints (depending on sklearn version).
    categorical_features : Optional[List[int]]
        Indices of categorical features for TabPFN usage (if any).
    verbose : Union[bool, int]
        Verbosity level; higher values produce more output.
    show_progress : bool
        Whether to show progress bars for leaf/node fitting using TabPFN.
    fit_nodes : bool
        Whether to fit TabPFN at internal nodes (True) or only final leaves (False).
    tree_seed : int
        Used to set seeds for TabPFN fitting in each node.
    adaptive_tree : bool
        Whether to do adaptive node-by-node pruning using a hold-out strategy.
    adaptive_tree_min_train_samples : int
        Minimum number of training samples required to fit a TabPFN in a node.
    adaptive_tree_max_train_samples : int
        Maximum number of training samples above which a node might be pruned if not a final leaf.
    adaptive_tree_min_valid_samples_fraction_of_train : float
        Fraction controlling the minimum valid/test points to consider a node for re-fitting.
    adaptive_tree_overwrite_metric : Optional[str]
        If set, overrides the default metric for pruning. E.g., "roc" or "rmse".
    adaptive_tree_test_size : float
        Fraction of data to hold out for adaptive pruning if no separate valid set is provided.
    average_logits : bool
        Whether to average logits (True) or probabilities (False) when combining predictions.
    adaptive_tree_skip_class_missing : bool
        If True, skip re-fitting if the nodes training set does not contain all classes (classification only).
    """

    # Task type set by subclasses: "multiclass" or "regression"
    task_type: str | None = None

    def __init__(
        self,
        *,
        # Decision Tree arguments
        criterion: str = "gini",
        splitter: str = "best",
        max_depth: int | None = None,
        min_samples_split: int = 1000,
        min_samples_leaf: int = 1,
        min_weight_fraction_leaf: float = 0.0,
        max_features: int | float | str | None = None,
        random_state: int | np.random.RandomState | None = None,
        max_leaf_nodes: int | None = None,
        min_impurity_decrease: float = 0.0,
        class_weight: dict[int, float] | str | None = None,
        ccp_alpha: float = 0.0,
        monotonic_cst: Any = None,
        # TabPFN argument
        tabpfn: Any = None,  # TabPFNClassifier or TabPFNRegressor
        categorical_features: list[int] | None = None,
        verbose: bool | int = False,
        show_progress: bool = False,
        fit_nodes: bool = True,
        tree_seed: int = 0,
        adaptive_tree: bool = True,
        adaptive_tree_min_train_samples: int = 50,
        adaptive_tree_max_train_samples: int = 2000,
        adaptive_tree_min_valid_samples_fraction_of_train: float = 0.2,
        adaptive_tree_overwrite_metric: str | None = None,
        adaptive_tree_test_size: float = 0.2,
        average_logits: bool = True,
        adaptive_tree_skip_class_missing: bool = True,
    ):
        # Collect recognized arguments
        self.tabpfn = tabpfn
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.class_weight = class_weight
        self.ccp_alpha = ccp_alpha
        self.monotonic_cst = monotonic_cst

        self.categorical_features = categorical_features
        self.verbose = verbose
        self.show_progress = show_progress
        self.fit_nodes = fit_nodes
        self.tree_seed = tree_seed
        self.adaptive_tree = adaptive_tree
        self.adaptive_tree_min_train_samples = adaptive_tree_min_train_samples
        self.adaptive_tree_max_train_samples = adaptive_tree_max_train_samples
        self.adaptive_tree_min_valid_samples_fraction_of_train = (
            adaptive_tree_min_valid_samples_fraction_of_train
        )
        self.adaptive_tree_overwrite_metric = adaptive_tree_overwrite_metric
        self.adaptive_tree_test_size = adaptive_tree_test_size
        self.average_logits = average_logits
        self.adaptive_tree_skip_class_missing = adaptive_tree_skip_class_missing

        # Initialize internal flags/structures that will be set during fit
        self._need_post_fit: bool = False
        self._decision_tree = None

        # Handling possible differences in sklearn versions, specifically monotonic_cst
        optional_args_filtered = {}
        if BaseDecisionTree.__init__.__code__.co_varnames.__contains__("monotonic_cst"):
            optional_args_filtered["monotonic_cst"] = monotonic_cst

        # Initialize the underlying DecisionTree
        super().__init__(
            criterion=self.criterion,
            splitter=self.splitter,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
            max_leaf_nodes=self.max_leaf_nodes,
            min_impurity_decrease=self.min_impurity_decrease,
            ccp_alpha=self.ccp_alpha,
            **optional_args_filtered,
        )

        # If the user gave a TabPFN, we do not want it to have a random_state forcibly set
        # because we handle seeds ourselves at each node
        if self.tabpfn is not None:
            self.tabpfn.random_state = None

    def _validate_tabpfn_runtime(self) -> None:
        """Validate the TabPFN instance at runtime before using it.

        This ensures the TabPFN instance is still available when needed during
        prediction or fitting operations.

        Raises:
            ValueError: If self.tabpfn is None at runtime
        """
        if self.tabpfn is None:
            raise ValueError("TabPFN was None at runtime - cannot proceed.")

    def _more_tags(self) -> dict[str, Any]:
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

    def fit(
        self,
        X: NDArray[np.float64],
        y: NDArray[Any],
        sample_weight: NDArray[np.float64] | None = None,
        check_input: bool = True,
    ) -> DecisionTreeTabPFNBase:
        """Fit the DecisionTree + TabPFN model.

        This method trains the hybrid model by:
        1. Building a decision tree structure
        2. Fitting TabPFN models at the leaves (or at all nodes if fit_nodes=True)
        3. Optionally performing adaptive pruning if adaptive_tree=True

        Args:
            X: The training input samples, shape (n_samples, n_features).
            y: The target values (class labels for classification, real values for regression),
                shape (n_samples,) or (n_samples, n_outputs).
            sample_weight: Sample weights. If None, then samples are equally weighted.
            check_input: Whether to validate the input data arrays. Default is True.

        Returns:
            self: Fitted estimator.
        """
        return self._fit(X, y, sample_weight=sample_weight, check_input=check_input)

    def _fit(
        self,
        X: NDArray[Any],
        y: NDArray[Any],
        sample_weight: NDArray[Any] | None = None,
        check_input: bool = True,
        missing_values_in_feature_mask: np.ndarray | None = None,  # Unused placeholder
    ) -> DecisionTreeTabPFNBase:
        """Internal method to fit the DecisionTree-TabPFN model on X, y.

        Parameters
        ----------
        X : NDArray
            Training features of shape (n_samples, n_features).
        y : NDArray
            Target labels/values of shape (n_samples,).
        sample_weight : NDArray, optional
            Sample weights for each sample.
        check_input : bool
            Whether to check inputs.
        missing_values_in_feature_mask : np.ndarray, optional
            Unused placeholder for older code or possible expansions.

        Returns:
        -------
        self : DecisionTreeTabPFNBase
            The fitted model.
        """
        # Initialize attributes (per scikit-learn conventions)
        self._leaf_nodes = []
        self._leaf_train_data = {}
        self._label_encoder = LabelEncoder()
        self._need_post_fit = False
        self._node_prediction_type = {}

        # Make sure tabpfn is valid
        self._validate_tabpfn_runtime()

        # Possibly randomize tree_seed if not set
        if self.tree_seed == 0:
            self.tree_seed = random.randint(1, 10000)

        sample_weight = _check_sample_weight(sample_weight, X, dtype=np.float64)
        X, y = validate_data(
            self,
            X,
            y,
            ensure_all_finite=False,  # scikit-learn sets self.n_features_in_ automatically
        )

        if self.task_type == "multiclass":
            self.classes_ = unique_labels(y)
            self.n_classes_ = len(self.classes_)

        # Convert torch tensor -> numpy if needed, handle NaNs
        X_preprocessed = self._preprocess_data_for_tree(X)

        if sample_weight is None:
            sample_weight = np.ones((X_preprocessed.shape[0],), dtype=np.float64)

        # Setup classes_ or n_classes_ if needed
        if self.task_type == "multiclass":
            # Classification
            self.classes_ = np.unique(y)
            self.n_classes_ = len(self.classes_)
        else:
            # Regression
            self.n_classes_ = (
                1  # Not used for numeric tasks, but keep it for consistency
            )

        # Possibly label-encode y for classification if your TabPFN needs it
        # (Here we just rely on uniqueness checks above.)
        y_ = y.copy()

        # If adaptive_tree is on, do a train/validation split
        if self.adaptive_tree:
            stratify = y_ if (self.task_type == "multiclass") else None

            # Basic checks for classification to see if splitting is feasible
            if self.task_type == "multiclass":
                unique_classes, counts = np.unique(y_, return_counts=True)
                # Disable adaptive tree in extreme cases
                if counts.min() == 1 or len(unique_classes) < 2:
                    self.adaptive_tree = False
                elif len(unique_classes) > int(len(y_) * self.adaptive_tree_test_size):
                    self.adaptive_tree_test_size = min(
                        0.5,
                        len(unique_classes) / len(y_) * 1.5,
                    )
            if len(y_) < 10:
                self.adaptive_tree = False

            if self.adaptive_tree:
                (
                    X_train,
                    X_valid,
                    X_preproc_train,
                    X_preproc_valid,
                    y_train,
                    y_valid,
                    sw_train,
                    sw_valid,
                ) = train_test_split(
                    X,
                    X_preprocessed,
                    y_,
                    sample_weight,
                    test_size=self.adaptive_tree_test_size,
                    random_state=self.random_state,
                    stratify=stratify,
                )

                # Safety check - if split is empty, revert
                if len(y_train) == 0 or len(y_valid) == 0:
                    self.adaptive_tree = False
                    X_train, X_preproc_train, y_train, sw_train = (
                        X,
                        X_preprocessed,
                        y_,
                        sample_weight,
                    )
                    X_valid = X_preproc_valid = y_valid = sw_valid = None

                # If classification, also ensure train/valid has same classes
                if (
                    self.task_type == "multiclass"
                    and self.adaptive_tree
                    and (len(np.unique(y_train)) != len(np.unique(y_valid)))
                ):
                    self.adaptive_tree = False
            else:
                # If we were disabled, keep all data as training
                X_train, X_preproc_train, y_train, sw_train = (
                    X,
                    X_preprocessed,
                    y_,
                    sample_weight,
                )
                X_valid = X_preproc_valid = y_valid = sw_valid = None
        else:
            # Not adaptive, everything is train
            X_train, X_preproc_train, y_train, sw_train = (
                X,
                X_preprocessed,
                y_,
                sample_weight,
            )
            X_valid = X_preproc_valid = y_valid = sw_valid = None

        # Build the sklearn decision tree
        self._decision_tree = self._init_decision_tree()
        self._decision_tree.fit(X_preproc_train, y_train, sample_weight=sw_train)
        self._tree = self._decision_tree  # for sklearn compatibility

        # Keep references for potential post-fitting (leaf-level fitting)
        self.X = X
        self.y = y_
        self.train_X = X_train
        self.train_X_preprocessed = X_preproc_train
        self.train_y = y_train
        self.train_sample_weight = sw_train

        if self.adaptive_tree:
            self.valid_X = X_valid
            self.valid_X_preprocessed = X_preproc_valid
            self.valid_y = y_valid
            self.valid_sample_weight = sw_valid

        # We will do a leaf-fitting step on demand (lazy) in predict
        self._need_post_fit = True

        # If verbose, optionally do it right away:
        if self.verbose:
            self._post_fit()

        return self

    def _init_decision_tree(self) -> BaseDecisionTree:
        """Initialize the underlying scikit-learn Decision Tree.

        Overridden by child classes for classifier vs regressor.

        Returns:
        -------
        BaseDecisionTree
            An instance of a scikit-learn DecisionTreeClassifier or DecisionTreeRegressor.
        """
        raise NotImplementedError("Must be implemented in subclass.")

    def _post_fit(self) -> None:
        """Hook after the decision tree is fitted. Can be used for final prints/logs."""
        if self.verbose:
            pass

    def _preprocess_data_for_tree(self, X: np.ndarray) -> np.ndarray:
        """Handle missing data prior to feeding into the decision tree.

        Replaces NaNs with a default value, handles pandas DataFrames and other input types.
        Uses scikit-learn's validation functions for compatibility.

        Parameters
        ----------
        X : array-like
            Input features, possibly containing NaNs.

        Returns:
        -------
        np.ndarray
            A copy of X with NaNs replaced by default value.
        """
        # Use check_array from sklearn_compat to handle different input types
        from .sklearn_compat import check_array

        # Handle torch tensor
        if torch.is_tensor(X):
            X = X.cpu().numpy()

        # Convert to array and handle input validation
        # Don't extract DataFrame values - let check_array handle it
        X = check_array(
            X,
            dtype=np.float64,
            ensure_all_finite=False,  # We'll handle NaNs ourselves
            ensure_2d=True,
            copy=True,  # Make a copy so we don't modify the original
        )

        # Replace NaN with our specific value (-1000.0)
        X = np.nan_to_num(X, nan=-1000.0)
        return X

    def _apply_tree(self, X: np.ndarray) -> np.ndarray:
        """Apply the fitted tree to X, returning a matrix of leaf membership.

        Returns:
        -------
        np.ndarray
            A dense matrix of shape (n_samples, n_nodes, n_estimators),
            though we typically only have 1 estimator.
        """
        X_preprocessed = self._preprocess_data_for_tree(X)
        decision_path = self.get_tree().decision_path(X_preprocessed)
        return np.expand_dims(decision_path.todense(), axis=2)

    def _apply_tree_train(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply the tree for training data, returning leaf membership plus (X, y) unchanged.

        Returns:
        -------
        leaf_matrix : np.ndarray
            Shape (n_samples, n_nodes, n_estimators)
        X_array : np.ndarray
            Same as X input
        y_array : np.ndarray
            Same as y input
        """
        return self._apply_tree(X), X, y

    def get_tree(self) -> BaseDecisionTree:
        """Return the underlying fitted sklearn decision tree.

        Returns:
            DecisionTreeClassifier or DecisionTreeRegressor: The fitted decision tree.

        Raises:
            sklearn.exceptions.NotFittedError: If the model has not been fitted yet.
        """
        # This will raise NotFittedError if the model is not fitted
        check_is_fitted(self, ["_tree", "X", "y"])
        return self._tree

    @property
    def tree_(self):
        """Expose the fitted tree for sklearn compatibility.

        Returns:
        -------
        sklearn.tree._tree.Tree
            Underlying scikit-learn tree object.
        """
        return self.get_tree().tree_

    def fit_leaves(
        self,
        train_X: np.ndarray,
        train_y: np.ndarray,
    ) -> None:
        """Fit a TabPFN model in each leaf node (or each node, if self.fit_nodes=True).

        This populates an internal dictionary of training data for each leaf
        so that TabPFN can make predictions at these leaves.

        Parameters
        ----------
        train_X : np.ndarray
            Training features for all samples.
        train_y : np.ndarray
            Training labels/targets for all samples.
        """
        self._leaf_train_data = {}
        leaf_node_matrix, _, _ = self._apply_tree_train(train_X, train_y)
        self._leaf_nodes = leaf_node_matrix

        n_samples, n_nodes, n_estims = leaf_node_matrix.shape

        for estimator_id in range(n_estims):
            self._leaf_train_data[estimator_id] = {}
            for leaf_id in range(n_nodes):
                indices = np.argwhere(
                    leaf_node_matrix[:, leaf_id, estimator_id],
                ).ravel()
                X_leaf_samples = np.take(train_X, indices, axis=0)
                y_leaf_samples = np.take(train_y, indices, axis=0).ravel()

                self._leaf_train_data[estimator_id][leaf_id] = (
                    X_leaf_samples,
                    y_leaf_samples,
                )

    def _predict_internal(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        check_input: bool = True,
    ) -> np.ndarray:
        """Internal method used to produce probabilities or regression predictions,
        with optional adaptive pruning logic.

        If y is given and we have adaptive_tree=True, node-level pruning is applied.

        Parameters
        ----------
        X : np.ndarray
            Features to predict.
        y : np.ndarray, optional
            Target values, only required if we are in adaptive pruning mode
            and need to compare node performance.
        check_input : bool, default=True
            Whether to validate input arrays.

        Returns:
        -------
        np.ndarray
            The final predictions (probabilities for classification, or continuous values for regression).
        """
        # If we haven't yet done the final leaf fit, do it here
        if self._need_post_fit:
            self._need_post_fit = False
            if self.adaptive_tree:
                # Fit leaves on train data, check performance on valid data if available
                self.fit_leaves(self.train_X, self.train_y)
                if (
                    hasattr(self, "valid_X")
                    and self.valid_X is not None
                    and self.valid_y is not None
                ):
                    # Force a pass to evaluate node performance
                    # so we can prune or decide node updates
                    self._predict_internal(
                        self.valid_X,
                        self.valid_y,
                        check_input=False,
                    )
            # Now fit leaves again using the entire dataset (train + valid, effectively)
            self.fit_leaves(self.X, self.y)

        # Assign TabPFNs categorical features if needed
        if self.tabpfn is not None:
            self.tabpfn.categorical_features_indices = self.categorical_features

        # Find leaf membership in X
        X_leaf_nodes = self._apply_tree(X)
        n_samples, n_nodes, n_estims = X_leaf_nodes.shape

        # Track intermediate predictions
        y_prob: dict[int, dict[int, np.ndarray]] = {}
        y_metric: dict[int, dict[int, float]] = {}

        # If pruning, track how each node is updated
        do_pruning = (y is not None) and self.adaptive_tree
        if do_pruning:
            self._node_prediction_type: dict[int, dict[int, str]] = {}

        for est_id in range(n_estims):
            if do_pruning:
                self._node_prediction_type[est_id] = {}
            y_prob[est_id] = {}
            y_metric[est_id] = {}
            if self.show_progress:
                import tqdm.auto

                node_iter = tqdm.auto.tqdm(range(n_nodes), desc=f"Estimator {est_id}")
            else:
                node_iter = range(n_nodes)

            for leaf_id in node_iter:
                self._pruning_init_node_predictions(
                    leaf_id,
                    est_id,
                    y_prob,
                    y_metric,
                    n_nodes,
                    n_samples,
                )
                if est_id > 0 and leaf_id == 0:
                    # Skip repeated re-initialization if multiple trees
                    continue

                # Gather test-sample indices that belong to this leaf
                test_sample_indices = np.argwhere(
                    X_leaf_nodes[:, leaf_id, est_id],
                ).ravel()

                # Gather training samples that belong to this leaf
                X_train_leaf, y_train_leaf = self._leaf_train_data[est_id][leaf_id]

                # If no training or test samples in this node, skip
                if (X_train_leaf.shape[0] == 0) or (len(test_sample_indices) == 0):
                    if do_pruning:
                        self._node_prediction_type[est_id][leaf_id] = "previous"
                    continue

                # Determine if this is a final leaf
                # If the sum of membership in subsequent nodes is zero, its final
                is_leaf = (
                    X_leaf_nodes[test_sample_indices, leaf_id + 1 :, est_id].sum()
                    == 0.0
                )

                # If it's not a leaf and we are not fitting internal nodes, skip
                # (unless leaf_id==0 and we do a top-level check for adaptive_tree)
                if (
                    (not is_leaf)
                    and (not self.fit_nodes)
                    and not (leaf_id == 0 and self.adaptive_tree)
                ):
                    if do_pruning:
                        self._node_prediction_type[est_id][leaf_id] = "previous"
                    continue

                # Additional adaptive checks
                if self.adaptive_tree and leaf_id != 0:
                    should_skip_previously_pruned = False
                    if y is None:
                        # Safely check if the key exists before accessing
                        node_type = self._node_prediction_type.get(est_id, {}).get(
                            leaf_id,
                        )
                        if node_type == "previous":
                            should_skip_previously_pruned = True

                    if should_skip_previously_pruned:
                        continue

                    # Skip if classification is missing a class
                    if (
                        self.task_type == "multiclass"
                        and len(np.unique(y_train_leaf)) < self.n_classes_
                        and self.adaptive_tree_skip_class_missing
                    ):
                        self._node_prediction_type[est_id][leaf_id] = "previous"
                        continue

                    # Skip if too few or too many training points
                    if (
                        (X_train_leaf.shape[0] < self.adaptive_tree_min_train_samples)
                        or (
                            len(test_sample_indices)
                            < self.adaptive_tree_min_valid_samples_fraction_of_train
                            * self.adaptive_tree_min_train_samples
                        )
                        or (
                            X_train_leaf.shape[0] > self.adaptive_tree_max_train_samples
                            and not is_leaf
                        )
                    ):
                        if do_pruning:
                            self._node_prediction_type[est_id][leaf_id] = "previous"
                        continue

                # Perform leaf-level TabPFN prediction
                leaf_prediction = self._predict_leaf(
                    X_train_leaf,
                    y_train_leaf,
                    leaf_id,
                    X,
                    test_sample_indices,
                )

                # Evaluate “averaging” and “replacement” for pruning
                y_prob_averaging, y_prob_replacement = (
                    self._pruning_get_prediction_type_results(
                        y_prob,
                        leaf_prediction,
                        test_sample_indices,
                        est_id,
                        leaf_id,
                    )
                )

                # Decide best approach if in adaptive mode
                if self.adaptive_tree:
                    # If not adaptive, we simply do replacement
                    y_prob[est_id][leaf_id] = y_prob_replacement
                elif y is not None:
                    self._pruning_set_node_prediction_type(
                        y,
                        y_prob_averaging,
                        y_prob_replacement,
                        y_metric,
                        est_id,
                        leaf_id,
                    )
                    self._pruning_set_predictions(
                        y_prob,
                        y_prob_averaging,
                        y_prob_replacement,
                        est_id,
                        leaf_id,
                    )
                    y_metric[est_id][leaf_id] = self._score(
                        y,
                        y_prob[est_id][leaf_id],
                    )
                else:
                    # If not validating and not adaptive, just use replacement
                    y_prob[est_id][leaf_id] = y_prob_replacement

        # Final predictions come from the last estimators last node
        return y_prob[n_estims - 1][n_nodes - 1]

    def _pruning_init_node_predictions(
        self,
        leaf_id: int,
        estimator_id: int,
        y_prob: dict[int, dict[int, np.ndarray]],
        y_metric: dict[int, dict[int, float]],
        n_nodes: int,
        n_samples: int,
    ) -> None:
        """Initialize node predictions for the pruning logic.

        Parameters
        ----------
        leaf_id : int
            Index of the leaf/node being processed.
        estimator_id : int
            Index of the current estimator (if multiple).
        y_prob : dict
            Nested dictionary of predictions.
        y_metric : dict
            Nested dictionary of scores/metrics.
        n_nodes : int
            Total number of nodes in the tree.
        n_samples : int
            Number of samples in X.
        """
        if estimator_id == 0 and leaf_id == 0:
            y_prob[0][0] = self._init_eval_probability_array(n_samples, to_zero=True)
            y_metric[0][0] = 0.0
        elif leaf_id == 0 and estimator_id > 0:
            # If first leaf of new estimator, carry from last node of previous estimator
            y_prob[estimator_id][leaf_id] = y_prob[estimator_id - 1][n_nodes - 1]
            y_metric[estimator_id][leaf_id] = y_metric[estimator_id - 1][n_nodes - 1]
        else:
            # Use last leaf of the same estimator
            y_prob[estimator_id][leaf_id] = y_prob[estimator_id][leaf_id - 1]
            y_metric[estimator_id][leaf_id] = y_metric[estimator_id][leaf_id - 1]

    def _pruning_get_prediction_type_results(
        self,
        y_eval_prob: dict[int, dict[int, np.ndarray]],
        leaf_prediction: np.ndarray,
        test_sample_indices: np.ndarray,
        estimator_id: int,
        leaf_id: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Produce the “averaging” and “replacement” predictions for pruning decisions.

        Parameters
        ----------
        y_eval_prob : dict
            Nested dictionary of predictions.
        leaf_prediction : np.ndarray
            Predictions from the newly fitted leaf (for relevant samples).
        test_sample_indices : np.ndarray
            Indices of the test samples that fall into this leaf.
        estimator_id : int
            Index of the current estimator.
        leaf_id : int
            Index of the current leaf/node.

        Returns:
        -------
        y_prob_averaging : np.ndarray
            Updated predictions using an “averaging” rule.
        y_prob_replacement : np.ndarray
            Updated predictions using a “replacement” rule.
        """
        y_prob_current = y_eval_prob[estimator_id][leaf_id]
        y_prob_replacement = np.copy(y_prob_current)
        # "replacement" sets the new leaf prediction directly
        y_prob_replacement[test_sample_indices] = leaf_prediction[test_sample_indices]

        if self.task_type == "multiclass":
            # Normalize
            row_sums = y_prob_replacement.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            y_prob_replacement /= row_sums

        # "averaging" -> combine old predictions with new
        y_prob_averaging = np.copy(y_prob_current)

        if self.task_type == "multiclass":
            if self.average_logits:
                # Convert old + new to log, sum them, then softmax
                y_prob_averaging[test_sample_indices] = np.log(
                    y_prob_averaging[test_sample_indices] + 1e-6,
                )
                leaf_pred_log = np.log(leaf_prediction[test_sample_indices] + 1e-6)
                y_prob_averaging[test_sample_indices] += leaf_pred_log
                y_prob_averaging[test_sample_indices] = softmax(
                    y_prob_averaging[test_sample_indices],
                )
            else:
                # Average probabilities directly
                y_prob_averaging[test_sample_indices] += leaf_prediction[
                    test_sample_indices
                ]
                row_sums = y_prob_averaging.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1.0
                y_prob_averaging /= row_sums
        elif self.task_type == "regression":
            # Regression -> simply average
            y_prob_averaging[test_sample_indices] += leaf_prediction[
                test_sample_indices
            ]
            y_prob_averaging[test_sample_indices] /= 2.0

        return y_prob_averaging, y_prob_replacement

    def _pruning_set_node_prediction_type(
        self,
        y_true: np.ndarray,
        y_prob_averaging: np.ndarray,
        y_prob_replacement: np.ndarray,
        y_metric: dict[int, dict[int, float]],
        estimator_id: int,
        leaf_id: int,
    ) -> None:
        """Decide which approach is better: “averaging” vs “replacement” vs “previous,”
        using the nodes previous metric vs new metrics.

        Parameters
        ----------
        y_true : np.ndarray
            Ground-truth labels/targets for pruning comparison.
        y_prob_averaging : np.ndarray
            Predictions if we use averaging.
        y_prob_replacement : np.ndarray
            Predictions if we use replacement.
        y_metric : dict
            Nested dictionary of scores/metrics for each node.
        estimator_id : int
            Index of the current estimator.
        leaf_id : int
            Index of the current leaf/node.
        """
        averaging_score = self._score(y_true, y_prob_averaging)
        replacement_score = self._score(y_true, y_prob_replacement)
        prev_score = y_metric[estimator_id][leaf_id - 1] if (leaf_id > 0) else 0.0

        if (leaf_id == 0) or (max(averaging_score, replacement_score) > prev_score):
            # Pick whichever is better
            if replacement_score > averaging_score:
                prediction_type = "replacement"
            else:
                prediction_type = "averaging"
        else:
            prediction_type = "previous"

        self._node_prediction_type[estimator_id][leaf_id] = prediction_type

    def _pruning_set_predictions(
        self,
        y_prob: dict[int, dict[int, np.ndarray]],
        y_prob_averaging: np.ndarray,
        y_prob_replacement: np.ndarray,
        estimator_id: int,
        leaf_id: int,
    ) -> None:
        """Based on the chosen node_prediction_type, finalize the predictions.

        Parameters
        ----------
        y_prob : dict
            Nested dictionary of predictions.
        y_prob_averaging : np.ndarray
            Predictions if we use averaging.
        y_prob_replacement : np.ndarray
            Predictions if we use replacement.
        estimator_id : int
            Index of the current estimator.
        leaf_id : int
            Index of the current leaf/node.
        """
        node_type = self._node_prediction_type[estimator_id][leaf_id]
        if node_type == "averaging":
            y_prob[estimator_id][leaf_id] = y_prob_averaging
        elif node_type == "replacement":
            y_prob[estimator_id][leaf_id] = y_prob_replacement
        else:
            # “previous”
            y_prob[estimator_id][leaf_id] = y_prob[estimator_id][leaf_id - 1]

    def _init_eval_probability_array(
        self,
        n_samples: int,
        to_zero: bool = False,
    ) -> np.ndarray:
        """Initialize an array of predictions for the entire dataset.

        For classification, this is (n_samples, n_classes).
        For regression, this is (n_samples,).

        Parameters
        ----------
        n_samples : int
            Number of samples to predict.
        to_zero : bool, default=False
            If True, fill with zeros. Otherwise use uniform for classification,
            or zeros for regression.

        Returns:
        -------
        np.ndarray
            An appropriately sized array of initial predictions.
        """
        if self.task_type == "multiclass":
            if to_zero:
                return np.zeros((n_samples, self.n_classes_), dtype=np.float64)
            return (
                np.ones((n_samples, self.n_classes_), dtype=np.float64)
                / self.n_classes_
            )
        else:
            # Regression
            return np.zeros((n_samples,), dtype=np.float64)

    def _score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute a performance score given ground truth and predictions.

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth labels or values.
        y_pred : np.ndarray
            Predictions (probabilities for classification, continuous for regression).

        Returns:
        -------
        float
            The performance score (higher is better for classification,
            or depends on the specific metric).
        """
        metric = self._get_optimize_metric()
        if self.task_type == "multiclass":
            return score_classification(metric, y_true, y_pred)
        elif self.task_type == "regression":
            return score_regression(metric, y_true, y_pred)
        else:
            raise NotImplementedError

    def _get_optimize_metric(self) -> str:
        """Return which metric name to use for scoring.

        Returns:
        -------
        str
            The metric name, e.g. "roc" for classification or "rmse" for regression.
        """
        if self.adaptive_tree_overwrite_metric is not None:
            return self.adaptive_tree_overwrite_metric
        if self.task_type == "multiclass":
            return "roc"
        return "rmse"

    def _predict_leaf(
        self,
        X_train_leaf: np.ndarray,
        y_train_leaf: np.ndarray,
        leaf_id: int,
        X_full: np.ndarray,
        indices: np.ndarray,
    ) -> np.ndarray:
        """Each subclass implements how to call TabPFN for classification or regression.

        Parameters
        ----------
        X_train_leaf : np.ndarray
            Training features for the samples in this leaf/node.
        y_train_leaf : np.ndarray
            Training targets for the samples in this leaf/node.
        leaf_id : int
            Leaf/node index (for seeding or debugging).
        X_full : np.ndarray
            The entire set of features we are predicting on.
        indices : np.ndarray
            The indices in X_full that belong to this leaf.

        Returns:
        -------
        np.ndarray
            Predictions for all n_samples, but only indices are filled meaningfully.
        """
        raise NotImplementedError("Must be implemented in subclass.")


###############################################################################
#                          CLASSIFIER SUBCLASS                                #
###############################################################################


class DecisionTreeTabPFNClassifier(DecisionTreeTabPFNBase, ClassifierMixin):
    """Decision tree that uses TabPFNClassifier at the leaves."""

    task_type: str = "multiclass"

    def _init_decision_tree(self) -> DecisionTreeClassifier:
        """Create a scikit-learn DecisionTreeClassifier with stored parameters."""
        return DecisionTreeClassifier(
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
            max_leaf_nodes=self.max_leaf_nodes,
            min_impurity_decrease=self.min_impurity_decrease,
            class_weight=self.class_weight,
            ccp_alpha=self.ccp_alpha,
            splitter=self.splitter,
        )

    def _predict_leaf(
        self,
        X_train_leaf: np.ndarray,
        y_train_leaf: np.ndarray,
        leaf_id: int,
        X_full: np.ndarray,
        indices: np.ndarray,
    ) -> np.ndarray:
        """Fit a TabPFNClassifier on the leafs train data and predict_proba for the relevant samples.

        Parameters
        ----------
        X_train_leaf : np.ndarray
            Training features for the samples in this leaf/node.
        y_train_leaf : np.ndarray
            Training targets for the samples in this leaf/node.
        leaf_id : int
            Leaf/node index.
        X_full : np.ndarray
            Full feature matrix to predict on.
        indices : np.ndarray
            Indices of X_full that belong to this leaf.

        Returns:
        -------
        np.ndarray
            A (n_samples, n_classes) array of probabilities, with only `indices` updated for this leaf.
        """
        y_eval_prob = self._init_eval_probability_array(X_full.shape[0], to_zero=True)
        classes_in_leaf = [i for i in range(len(np.unique(y_train_leaf)))]

        # If only one class, fill probability 1.0 for that class
        if len(classes_in_leaf) == 1:
            y_eval_prob[indices, classes_in_leaf[0]] = 1.0
            return y_eval_prob

        # Otherwise, fit TabPFN
        leaf_seed = leaf_id + self.tree_seed
        try:
            # Handle pandas DataFrame or numpy array
            if hasattr(X_full, "iloc"):
                # Use .iloc for pandas
                X_subset = X_full.iloc[indices]
            else:
                # Use direct indexing for numpy
                X_subset = X_full[indices]

            try:
                self.tabpfn.random_state = leaf_seed
                self.tabpfn.fit(X_train_leaf, y_train_leaf)
                proba = self.tabpfn.predict_proba(X_subset)
            except Exception as e:
                from tabpfn.preprocessing import default_classifier_preprocessor_configs, \
                    default_regressor_preprocessor_configs
                backup_inf_conf = deepcopy(self.tabpfn.inference_config)
                default_pre = default_classifier_preprocessor_configs if self.task_type == "multiclass" else default_regressor_preprocessor_configs

                # Try to run again without preprocessing which might crash
                self.tabpfn.random_state = leaf_seed
                self.tabpfn.inference_config["PREPROCESS_TRANSFORMS"] = default_pre()
                self.tabpfn.inference_config["REGRESSION_Y_PREPROCESS_TRANSFORMS"] = (None, "safepower")
                print(self.tabpfn.inference_config)
                self.tabpfn.fit(X_train_leaf, y_train_leaf)
                proba = self.tabpfn.predict_proba(X_subset)
                # reset preprocessing
                self.tabpfn.inference_config = backup_inf_conf

            for i, c in enumerate(classes_in_leaf):
                y_eval_prob[indices, c] = proba[:, i]

        except ValueError as e:
            if (
                    not e.args
                    or e.args[0]
                    != "All features are constant and would have been removed! Unable to predict using TabPFN."
            ):
                raise e
            warnings.warn(
                "One node has constant features for TabPFN. Using class-ratio fallback.",
                stacklevel=2,
            )
            _, counts = np.unique(y_train_leaf, return_counts=True)
            ratio = counts / counts.sum()
            for i, c in enumerate(classes_in_leaf):
                y_eval_prob[indices, c] = ratio[i]

        return y_eval_prob

    def predict(self, X: np.ndarray, check_input: bool = True) -> np.ndarray:
        """Predict class labels for X.

        Args:
            X: Input features.
            check_input: Whether to validate input arrays. Default is True.

        Returns:
            np.ndarray: Predicted class labels.
        """
        # Validate the model is fitted
        X = validate_data(
            self,
            X,
            ensure_all_finite=False,
        )
        check_is_fitted(self, ["_tree", "X", "y"])
        proba = self.predict_proba(X, check_input=check_input)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X: np.ndarray, check_input: bool = True) -> np.ndarray:
        """Predict class probabilities for X using the TabPFN leaves.

        Args:
            X: Input features.
            check_input: Whether to validate input arrays. Default is True.

        Returns:
            np.ndarray: Predicted probabilities of shape (n_samples, n_classes).
        """
        # Validate the model is fitted
        X = validate_data(
            self,
            X,
            ensure_all_finite=False,
        )
        check_is_fitted(self, ["_tree", "X", "y"])
        return self._predict_internal(X, check_input=check_input)

    def _post_fit(self) -> None:
        """Optional hook after the decision tree is fitted."""
        if self.verbose:
            pass


###############################################################################
#                           REGRESSOR SUBCLASS                                #
###############################################################################


class DecisionTreeTabPFNRegressor(DecisionTreeTabPFNBase, RegressorMixin):
    """Decision tree that uses TabPFNRegressor at the leaves."""

    task_type: str = "regression"

    def __init__(
        self,
        *,
        criterion="squared_error",
        splitter="best",
        max_depth=None,
        min_samples_split=1000,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,
        random_state=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        ccp_alpha=0.0,
        monotonic_cst=None,
        tabpfn=None,
        categorical_features=None,
        verbose=False,
        show_progress=False,
        fit_nodes=True,
        tree_seed=0,
        adaptive_tree=True,
        adaptive_tree_min_train_samples=50,
        adaptive_tree_max_train_samples=2000,
        adaptive_tree_min_valid_samples_fraction_of_train=0.2,
        adaptive_tree_overwrite_metric=None,
        adaptive_tree_test_size=0.2,
        average_logits=True,
        adaptive_tree_skip_class_missing=True,
    ):
        # Call parent constructor
        super().__init__(
            tabpfn=tabpfn,
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            random_state=random_state,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            ccp_alpha=ccp_alpha,
            monotonic_cst=monotonic_cst,
            categorical_features=categorical_features,
            verbose=verbose,
            show_progress=show_progress,
            fit_nodes=fit_nodes,
            tree_seed=tree_seed,
            adaptive_tree=adaptive_tree,
            adaptive_tree_min_train_samples=adaptive_tree_min_train_samples,
            adaptive_tree_max_train_samples=adaptive_tree_max_train_samples,
            adaptive_tree_min_valid_samples_fraction_of_train=(
                adaptive_tree_min_valid_samples_fraction_of_train
            ),
            adaptive_tree_overwrite_metric=adaptive_tree_overwrite_metric,
            adaptive_tree_test_size=adaptive_tree_test_size,
            average_logits=average_logits,
            adaptive_tree_skip_class_missing=adaptive_tree_skip_class_missing,
        )

    def _init_decision_tree(self) -> DecisionTreeRegressor:
        """Create a scikit-learn DecisionTreeRegressor with stored parameters."""
        return DecisionTreeRegressor(
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
            max_leaf_nodes=self.max_leaf_nodes,
            min_impurity_decrease=self.min_impurity_decrease,
            ccp_alpha=self.ccp_alpha,
            splitter=self.splitter,
        )

    def _predict_leaf(
        self,
        X_train_leaf: np.ndarray,
        y_train_leaf: np.ndarray,
        leaf_id: int,
        X_full: np.ndarray,
        indices: np.ndarray,
    ) -> np.ndarray:
        """Fit a TabPFNRegressor on the nodes train data, then predict for the relevant samples.

        Parameters
        ----------
        X_train_leaf : np.ndarray
            Training features for the samples in this leaf/node.
        y_train_leaf : np.ndarray
            Training targets for the samples in this leaf/node.
        leaf_id : int
            Leaf/node index.
        X_full : np.ndarray
            Full feature matrix to predict on.
        indices : np.ndarray
            Indices of X_full that fall into this leaf.

        Returns:
        -------
        np.ndarray
            An array of shape (n_samples,) with predictions; only `indices` are updated.
        """
        y_eval = np.zeros(X_full.shape[0], dtype=float)

        # If no training data or just 1 sample, fall back to 0 or single value
        if len(X_train_leaf) < 1:
            warnings.warn(
                f"Leaf {leaf_id} has zero training samples. Returning 0.0 predictions.",
                stacklevel=2,
            )
            return y_eval
        elif len(X_train_leaf) == 1:
            y_eval[indices] = y_train_leaf[0]
            return y_eval

        # If all y are identical, return that constant
        if np.all(y_train_leaf == y_train_leaf[0]):
            y_eval[indices] = y_train_leaf[0]
            return y_eval

        # Fit TabPFNRegressor
        leaf_seed = leaf_id + self.tree_seed
        try:
            self.tabpfn.random_state = leaf_seed
            self.tabpfn.fit(X_train_leaf, y_train_leaf)

            # Handle pandas DataFrame or numpy array
            if hasattr(X_full, "iloc"):
                # Use .iloc for pandas
                X_subset = X_full.iloc[indices]
            else:
                # Use direct indexing for numpy
                X_subset = X_full[indices]

            preds = self.tabpfn.predict(X_subset)
            y_eval[indices] = preds
        except (ValueError, RuntimeError, NotImplementedError, AssertionError) as e:
            warnings.warn(
                f"TabPFN fit/predict failed at leaf {leaf_id}: {e}. Using mean fallback.",
                stacklevel=2,
            )
            y_eval[indices] = np.mean(y_train_leaf)

        return y_eval

    def predict(self, X: np.ndarray, check_input: bool = True) -> np.ndarray:
        """Predict regression values using the TabPFN leaves.

        Parameters
        ----------
        X : np.ndarray
            Input features.
        check_input : bool, default=True
            Whether to validate the input arrays.

        Returns:
        -------
        np.ndarray
            Continuous predictions of shape (n_samples,).
        """
        # Validate the model is fitted
        X = validate_data(
            self,
            X,
            ensure_all_finite=False,
        )
        check_is_fitted(self, ["_tree", "X", "y"])
        return self._predict_internal(X, check_input=check_input)

    def predict_full(self, X: np.ndarray) -> np.ndarray:
        """Convenience method to predict with no input checks (optional).

        Parameters
        ----------
        X : np.ndarray
            Input features.

        Returns:
        -------
        np.ndarray
            Continuous predictions of shape (n_samples,).
        """
        # Validate the model is fitted
        X = validate_data(
            self,
            X,
            ensure_all_finite=False,
        )
        check_is_fitted(self, ["_tree", "X", "y"])
        return self._predict_internal(X, check_input=False)

    def _post_fit(self) -> None:
        """Optional hook after the regressor's tree is fitted."""
        if self.verbose:
            pass
