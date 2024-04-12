# The original implementation in this file was based on scikit-garden that comes under the following license.
# The current version of the code has been modified beyond its original version.

# New BSD License

# Copyright (c) 2016 - scikit-garden developers.

# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

#   a. Redistributions of source code must retain the above copyright notice,
#      this list of conditions and the following disclaimer.

#   b. Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.

#   c. Neither the name of the scikit-garden developers nor the names of
#      its contributors may be used to endorse or promote products
#      derived from this software without specific prior written
#      permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
# OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
# DAMAGE.

import logging
from functools import partial

import numpy as np
import pandas as pd
from sklearn.ensemble._forest import ForestRegressor
from sklearn.tree import BaseDecisionTree, DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.utils import check_array, check_random_state, check_X_y

logger = logging.getLogger(__name__)


def weighted_percentile(a, q, weights=None, sorter=None, is_filtered=False):
    """
    Returns the weighted percentile of a at q given weights.

    Parameters
    ----------
    a: array-like, shape=(n_samples,)
        samples at which the quantile.
    q: int
        quantile between 0 and 100.
    weights: array-like, shape=(n_samples,)
        weights[i] is the weight given to point a[i] while computing the
        quantile. If weights[i] is zero, a[i] is simply ignored during the
        percentile computation.
    sorter: array-like, shape=(n_samples,)
        If provided, assume that a[sorter] is sorted.
    is_filtered: bool
        If True, weights is assumed to contain only non-zero values.

    Returns
    -------
    percentile: float
        Weighted percentile of a at q.

    References
    ----------
    1. https://en.wikipedia.org/wiki/Percentile#The_Weighted_Percentile_method

    Notes
    -----
    Note that weighted_percentile(a, q) is not equivalent to
    np.percentile(a, q). This is because in np.percentile
    sorted(a)[i] is assumed to be at quantile 0.0, while here we assume
    sorted(a)[i] is given a weight of 1.0 / len(a), hence it is at the
    1.0 / len(a)th quantile.
    """
    if weights is None:
        weights = np.ones_like(a)
    if q > 100 or q < 0:
        raise ValueError("q should be in-between 0 and 100, " "got %d" % q)

    a = np.asarray(a, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)
    if len(a) != len(weights):
        raise ValueError("a and weights should have the same length.")

    if sorter is not None:
        a = a[sorter]
        weights = weights[sorter]

    if not is_filtered:
        nz = weights != 0
        a = a[nz]
        weights = weights[nz]

    if sorter is None:
        sorted_indices = np.argsort(a)
        sorted_a = a[sorted_indices]
        sorted_weights = weights[sorted_indices]
    else:
        sorted_a = a
        sorted_weights = weights

    # Step 1
    sorted_cum_weights = np.cumsum(sorted_weights)
    total = sorted_cum_weights[-1]

    # Step 2
    partial_sum = 100.0 / total * (sorted_cum_weights - sorted_weights / 2.0)
    start = np.searchsorted(partial_sum, q) - 1
    if start == len(sorted_cum_weights) - 1:
        return sorted_a[-1]
    if start == -1:
        return sorted_a[0]

    # Step 3.
    fraction = (q - partial_sum[start]) / (partial_sum[start + 1] - partial_sum[start])
    return sorted_a[start] + fraction * (sorted_a[start + 1] - sorted_a[start])


class BaseTreeQuantileRegressor(BaseDecisionTree):
    def predict(self, X, quantile=None, check_input=False):
        """
        Predict regression value for X.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        quantile : int, optional
            Value ranging from 0 to 100. By default, the mean is returned.

        check_input : boolean, (default=True)
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        Returns
        -------
        y : array of shape = [n_samples]
            If quantile is set to None, then return E(Y | X). Else return
            y such that F(Y=y | x) = quantile.
        """
        # apply method requires X to be of dtype np.float32
        X = check_array(X, dtype=np.float32, accept_sparse="csc")
        if quantile is None:
            return super(BaseTreeQuantileRegressor, self).predict(X, check_input=check_input)

        quantiles = np.zeros(X.shape[0])
        X_leaves = self.apply(X)
        unique_leaves = np.unique(X_leaves)
        for leaf in unique_leaves:
            quantiles[X_leaves == leaf] = weighted_percentile(self.y_train_[self.y_train_leaves_ == leaf], quantile)
        return quantiles

    def fit(self, X, y, sample_weight=None, check_input=True):
        """
        Build a decision tree classifier from the training set (X, y).

        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]
            The training input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csc_matrix``.

        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            The target values (class labels) as integers or strings.

        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. Splits are also
            ignored if they would result in any single class carrying a
            negative weight in either child node.

        check_input : boolean, (default=True)
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        Returns
        -------
        self : object
            Returns self.
        """
        # y passed from a forest is 2-D. This is to silence the
        # annoying data-conversion warnings.
        y = np.asarray(y)
        if np.ndim(y) == 2 and y.shape[1] == 1:
            y = np.ravel(y)

        # apply method requires X to be of dtype np.float32
        X, y = check_X_y(X, y, accept_sparse="csc", dtype=np.float32, multi_output=False)
        super(BaseTreeQuantileRegressor, self).fit(X, y, sample_weight=sample_weight, check_input=check_input)
        self.y_train_ = y

        # Stores the leaf nodes that the samples lie in.
        self.y_train_leaves_ = self.tree_.apply(X)
        return self


class DecisionTreeQuantileRegressor(BaseTreeQuantileRegressor, DecisionTreeRegressor):
    """A decision tree regressor that provides quantile estimates.

    Parameters
    ----------
    criterion : string, optional (default="squared_error")
        The function to measure the quality of a split. Supported criteria
        are "squared_error" for the mean squared error, which is equal to variance
        reduction as feature selection criterion, and "mae" for the mean
        absolute error.
        .. versionadded:: 0.18
           Mean Absolute Error (MAE) criterion.

    splitter : string, optional (default="best")
        The strategy used to choose the split at each node. Supported
        strategies are "best" to choose the best split and "random" to choose
        the best random split.

    max_features : int, float, string or None, optional (default=None)
        The number of features to consider when looking for the best split:
        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a percentage and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=n_features`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.
        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    max_depth : int or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int, float, optional (default=2)
        The minimum number of samples required to split an internal node:
        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a percentage and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.
        .. versionchanged:: 0.18
           Added float values for percentages.

    min_samples_leaf : int, float, optional (default=1)
        The minimum number of samples required to be at a leaf node:
        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a percentage and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.
        .. versionchanged:: 0.18
           Added float values for percentages.

    max_leaf_nodes : int or None, optional (default=None)
        Grow a tree with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    presort : bool, optional (default=False)
        Whether to presort the data to speed up the finding of best splits in
        fitting. For the default settings of a decision tree on large
        datasets, setting this to true may slow down the training process.
        When using either a smaller dataset or a restricted depth, this may
        speed up the training.

    Attributes
    ----------
    feature_importances_ : array of shape = [n_features]
        The feature importances.
        The higher, the more important the feature.
        The importance of a feature is computed as the
        (normalized) total reduction of the criterion brought
        by that feature. It is also known as the Gini importance [4]_.

    max_features_ : int,
        The inferred value of max_features.

    n_features_ : int
        The number of features when ``fit`` is performed.

    n_outputs_ : int
        The number of outputs when ``fit`` is performed.

    tree_ : Tree object
        The underlying Tree object.

    y_train_ : array-like
        Train target values.

    y_train_leaves_ : array-like.
        Cache the leaf nodes that each training sample falls into.
        y_train_leaves_[i] is the leaf that y_train[i] ends up at.
    """

    def __init__(
        self,
        criterion="squared_error",
        splitter="best",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
        random_state=None,
        max_leaf_nodes=None,
    ):
        super(DecisionTreeQuantileRegressor, self).__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            random_state=random_state,
        )


class ExtraTreeQuantileRegressor(BaseTreeQuantileRegressor, ExtraTreeRegressor):
    def __init__(
        self,
        criterion="squared_error",
        splitter="random",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=1.0,
        random_state=None,
        max_leaf_nodes=None,
    ):
        super(ExtraTreeQuantileRegressor, self).__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            random_state=random_state,
        )


def generate_sample_indices(random_state, n_samples):
    """
    Generates bootstrap indices for each tree fit.

    Parameters
    ----------
    random_state: int, RandomState instance or None
        If int, random_state is the seed used by the random number generator.
        If RandomState instance, random_state is the random number generator.
        If None, the random number generator is the RandomState instance used
        by np.random.
    n_samples: int
        Number of samples to generate from each tree.

    Returns
    -------
    sample_indices: array-like, shape=(n_samples), dtype=np.int32
        Sample indices.
    """
    random_instance = check_random_state(random_state)
    sample_indices = random_instance.randint(0, n_samples, n_samples)
    return sample_indices


def get_weighted_neighbors_dataframe(X_leaves, y_train_leaves, y_train, y_weights):
    """For each test sample, get the list of weighted targets that are assigned to the same leaf by at least one estimator.

    Parameters
    ----------
    X_leaves : array, shape [n_test, n_estimators]
        Index of the leave assigned to each test sample by each estimator.
    y_train_leaves : array, shape [n_estimators, n_train]
        Index of the leave assigned to each training sample by each estimator.
    y_train : array, shape [n_train]
        Values of training samples.
    y_weights : array, shape [n_estimators, n_train]
        Weight assigned to each training sample by each estimator.

    Returns
    -------
    weighted_neighbors_dataframe : pd.DataFrame
        Dataframe that contains weighted neighbors of each item in the test set.
        Columns:
            item_id: ID of each item in the test set
            y: Target value encountered in the same leaf as the item
            weight: Weight assigned to each target value
    """
    assert X_leaves.shape[1] == y_train_leaves.shape[0] == y_weights.shape[0]
    assert y_train_leaves.shape[1] == y_train.shape[0] == y_weights.shape[1]

    num_test, num_trees = X_leaves.shape
    _, num_train = y_train_leaves.shape

    tree_index_x = np.arange(num_trees)[None].repeat([num_test], axis=0)
    item_index_x = np.arange(num_test)[:, None].repeat([num_trees], axis=1)
    df_x = pd.DataFrame(
        {
            "item_id": item_index_x.ravel(),
            "tree_id": tree_index_x.ravel(),
            "leaf": X_leaves.ravel(),
        }
    )
    tree_index_y = np.arange(num_trees)[:, None].repeat([num_train], axis=1)
    target_y = y_train[None].repeat([num_trees], axis=0)
    df_y = pd.DataFrame(
        {
            "tree_id": tree_index_y.ravel(),
            "leaf": y_train_leaves.ravel(),
            "y": target_y.ravel(),
            "weight": y_weights.ravel(),
        }
    )
    samples_with_neighbors = pd.merge(df_x, df_y, on=["tree_id", "leaf"])
    return samples_with_neighbors.groupby(["item_id", "y"]).sum().reset_index()[["item_id", "y", "weight"]]


def get_quantiles(neighbors_df, quantile_levels):
    """Compute predicted quantiles for the given sample.

    Parameters
    ----------
    neighbors_df : pd.DataFrame
        DataFrame with columns y (target values for each sample) and weight (weight assigned to each sample)
    quantile_levels : List[float]
        List of quantiles to predict between 0.0 and 1.0

    Returns
    -------
    quantiles : array, shape [len(quantile_levels)]
        Predicted quantiles.
    """
    result = []
    for q in quantile_levels:
        result.append(
            weighted_percentile(
                neighbors_df.y,
                int(q * 100),
                neighbors_df.weight,
                sorter=slice(None),  # targets are already sorted, so no sorting required
                is_filtered=True,
            )
        )
    return result


class BaseForestQuantileRegressor(ForestRegressor):
    def fit(self, X, y, sample_weight=None):
        """
        Build a forest from the training set (X, y).

        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]
            The training input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csc_matrix``.
        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            The target values (class labels) as integers or strings.
        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. Splits are also
            ignored if they would result in any single class carrying a
            negative weight in either child node.

        Returns
        -------
        self : object
            Returns self.
        """
        logger.warning(
            f"\tWARNING: {self.__class__.__name__} are experimental for quantile regression. "
            f"They may change or be removed without warning in future releases."
        )
        if sample_weight is not None:
            logger.warning(f"\tWARNING: {self.__class__.__name__} ignores sample_weight.")

        # apply method requires X to be of dtype np.float32
        X, y = check_X_y(X, y, accept_sparse="csc", dtype=np.float32, multi_output=False)
        super(BaseForestQuantileRegressor, self).fit(X, y)

        self.y_train_ = y
        self.y_train_leaves_ = -np.ones((self.n_estimators, len(y)), dtype=np.int32)
        self.y_weights_ = np.zeros_like((self.y_train_leaves_), dtype=np.float32)

        for i, est in enumerate(self.estimators_):
            if self.bootstrap:
                bootstrap_indices = generate_sample_indices(est.random_state, len(y))
            else:
                bootstrap_indices = np.arange(len(y))

            est_weights = np.bincount(bootstrap_indices, minlength=len(y))
            # FIXME: When updating from scikit-learn 1.3.2 to 1.4.0, BaseTreeQuantileRegressor.fit is not called
            # Re-calculating y_train_ and y_train_leaves_ to resolve this issue
            est.y_train_ = y
            est.y_train_leaves_ = est.tree_.apply(X)
            y_train_leaves = est.y_train_leaves_
            # Normalize the bootstrap weights such that the total weight of each leaf sums up to 1
            # Relabel leaves starting from zero in order to efficiently count the total sum per leaf with bincount
            leaves_starting_from_zero = np.unique(y_train_leaves, return_inverse=True)[1]
            weight_per_leaf = np.bincount(leaves_starting_from_zero, weights=est_weights)
            self.y_weights_[i] = est_weights / weight_per_leaf[leaves_starting_from_zero]

            self.y_train_leaves_[i, bootstrap_indices] = y_train_leaves[bootstrap_indices]
        return self

    def predict(self, X, quantile_levels=None):
        """
        Predict regression value for X.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.
        quantile_levels : List[float], optional
            List of quantiles (between 0.0 and 1.0) to predict. If not provided, mean is returned.

        Returns
        -------
        y : array
            If quantile_levels is None, then y contains E(Y | X) and has shape [n_samples].
            Otherwise, y contains the predicted quantiles and has shape [n_samples, len(quantile_levels)]
        """
        # apply method requires X to be of dtype np.float32
        X = check_array(X, dtype=np.float32, accept_sparse="csc")
        if quantile_levels is None:
            return super(BaseForestQuantileRegressor, self).predict(X)
        elif isinstance(quantile_levels, float):
            quantile_levels = [quantile_levels]

        X_leaves = self.apply(X)

        samples_with_weighted_neighbors = get_weighted_neighbors_dataframe(
            X_leaves=X_leaves, y_train_leaves=self.y_train_leaves_, y_train=self.y_train_, y_weights=self.y_weights_
        )
        quantile_preds = samples_with_weighted_neighbors.groupby("item_id").apply(partial(get_quantiles, quantile_levels=quantile_levels), include_groups=False)
        return np.stack(quantile_preds.values.tolist())


class RandomForestQuantileRegressor(BaseForestQuantileRegressor):
    """
    A random forest regressor that provides quantile estimates.
    A random forest is a meta estimator that fits a number of classifying
    decision trees on various sub-samples of the dataset and use averaging
    to improve the predictive accuracy and control over-fitting.
    The sub-sample size is always the same as the original
    input sample size but the samples are drawn with replacement if
    `bootstrap=True` (default).

    Parameters
    ----------
    n_estimators : integer, optional (default=10)
        The number of trees in the forest.
    criterion : string, optional (default="squared_error")
        The function to measure the quality of a split. Supported criteria
        are "squared_error" for the mean squared error, which is equal to variance
        reduction as feature selection criterion, and "mae" for the mean
        absolute error.
        .. versionadded:: 0.18
           Mean Absolute Error (MAE) criterion.
    max_features : int, float, string or None, optional (default="auto")
        The number of features to consider when looking for the best split:
        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a percentage and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=n_features`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.
        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.
    max_depth : integer or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.
    min_samples_split : int, float, optional (default=2)
        The minimum number of samples required to split an internal node:
        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a percentage and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.
        .. versionchanged:: 0.18
           Added float values for percentages.
    min_samples_leaf : int, float, optional (default=1)
        The minimum number of samples required to be at a leaf node:
        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a percentage and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.
        .. versionchanged:: 0.18
           Added float values for percentages.
    max_leaf_nodes : int or None, optional (default=None)
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.
    bootstrap : boolean, optional (default=True)
        Whether bootstrap samples are used when building trees.
    oob_score : bool, optional (default=False)
        whether to use out-of-bag samples to estimate
        the R^2 on unseen data.
    n_jobs : integer, optional (default=1)
        The number of jobs to run in parallel for both `fit` and `predict`.
        If -1, then the number of jobs is set to the number of cores.
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    verbose : int, optional (default=0)
        Controls the verbosity of the tree building process.
    warm_start : bool, optional (default=False)
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest.
    Attributes
    ----------
    estimators_ : list of DecisionTreeQuantileRegressor
        The collection of fitted sub-estimators.
    feature_importances_ : array of shape = [n_features]
        The feature importances (the higher, the more important the feature).
    n_features_ : int
        The number of features when ``fit`` is performed.
    n_outputs_ : int
        The number of outputs when ``fit`` is performed.
    oob_score_ : float
        Score of the training dataset obtained using an out-of-bag estimate.
    oob_prediction_ : array of shape = [n_samples]
        Prediction computed with out-of-bag estimate on the training set.
    y_train_ : array-like, shape=(n_samples,)
        Cache the target values at fit time.
    y_weights_ : array-like, shape=(n_estimators, n_samples)
        y_weights_[i, j] is the weight given to sample ``j` while
        estimator ``i`` is fit. If bootstrap is set to True, this
        reduces to a 2-D array of ones.
    y_train_leaves_ : array-like, shape=(n_estimators, n_samples)
        y_train_leaves_[i, j] provides the leaf node that y_train_[i]
        ends up when estimator j is fit. If y_train_[i] is given
        a weight of zero when estimator j is fit, then the value is -1.

    References
    ----------
    .. [1] Nicolai Meinshausen, Quantile Regression Forests
        http://www.jmlr.org/papers/volume7/meinshausen06a/meinshausen06a.pdf
    """

    def __init__(
        self,
        n_estimators=10,
        criterion="squared_error",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=1.0,
        max_leaf_nodes=None,
        bootstrap=True,
        oob_score=False,
        n_jobs=1,
        random_state=None,
        verbose=0,
        warm_start=False,
        max_samples=None,
    ):
        super(RandomForestQuantileRegressor, self).__init__(
            DecisionTreeQuantileRegressor(),
            n_estimators=n_estimators,
            estimator_params=(
                "criterion",
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "max_features",
                "max_leaf_nodes",
                "random_state",
            ),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            max_samples=max_samples,
        )

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes


class ExtraTreesQuantileRegressor(BaseForestQuantileRegressor):
    """
    An extra-trees regressor that provides quantile estimates.
    This class implements a meta estimator that fits a number of
    randomized decision trees (a.k.a. extra-trees) on various sub-samples
    of the dataset and use averaging to improve the predictive accuracy
    and control over-fitting.

    Parameters
    ----------
    n_estimators : integer, optional (default=10)
        The number of trees in the forest.
    criterion : string, optional (default="squared_error")
        The function to measure the quality of a split. Supported criteria
        are "squared_error" for the mean squared error, which is equal to variance
        reduction as feature selection criterion, and "mae" for the mean
        absolute error.
        .. versionadded:: 0.18
           Mean Absolute Error (MAE) criterion.
    max_features : int, float, string or None, optional (default="auto")
        The number of features to consider when looking for the best split:
        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a percentage and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=n_features`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.
        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.
    max_depth : integer or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.
    min_samples_split : int, float, optional (default=2)
        The minimum number of samples required to split an internal node:
        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a percentage and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.
        .. versionchanged:: 0.18
           Added float values for percentages.
    min_samples_leaf : int, float, optional (default=1)
        The minimum number of samples required to be at a leaf node:
        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a percentage and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.
        .. versionchanged:: 0.18
           Added float values for percentages.
    max_leaf_nodes : int or None, optional (default=None)
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.
    bootstrap : boolean, optional (default=False)
        Whether bootstrap samples are used when building trees.
    oob_score : bool, optional (default=False)
        Whether to use out-of-bag samples to estimate the R^2 on unseen data.
    n_jobs : integer, optional (default=1)
        The number of jobs to run in parallel for both `fit` and `predict`.
        If -1, then the number of jobs is set to the number of cores.
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    verbose : int, optional (default=0)
        Controls the verbosity of the tree building process.
    warm_start : bool, optional (default=False)
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest.

    Attributes
    ----------
    estimators_ : list of ExtraTreeQuantileRegressor
        The collection of fitted sub-estimators.
    feature_importances_ : array of shape = [n_features]
        The feature importances (the higher, the more important the feature).
    n_features_ : int
        The number of features when ``fit`` is performed.
    n_outputs_ : int
        The number of outputs when ``fit`` is performed.
    oob_score_ : float
        Score of the training dataset obtained using an out-of-bag estimate.
    oob_prediction_ : array of shape = [n_samples]
        Prediction computed with out-of-bag estimate on the training set.
    y_train_ : array-like, shape=(n_samples,)
        Cache the target values at fit time.
    y_weights_ : array-like, shape=(n_estimators, n_samples)
        y_weights_[i, j] is the weight given to sample ``j` while
        estimator ``i`` is fit. If bootstrap is set to True, this
        reduces to a 2-D array of ones.
    y_train_leaves_ : array-like, shape=(n_estimators, n_samples)
        y_train_leaves_[i, j] provides the leaf node that y_train_[i]
        ends up when estimator j is fit. If y_train_[i] is given
        a weight of zero when estimator j is fit, then the value is -1.

    References
    ----------
    .. [1] Nicolai Meinshausen, Quantile Regression Forests
        http://www.jmlr.org/papers/volume7/meinshausen06a/meinshausen06a.pdf
    """

    def __init__(
        self,
        n_estimators=10,
        criterion="squared_error",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=1.0,
        max_leaf_nodes=None,
        bootstrap=True,
        oob_score=False,
        n_jobs=1,
        random_state=None,
        verbose=0,
        warm_start=False,
        max_samples=None,
    ):
        super(ExtraTreesQuantileRegressor, self).__init__(
            ExtraTreeQuantileRegressor(),
            n_estimators=n_estimators,
            estimator_params=(
                "criterion",
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "max_features",
                "max_leaf_nodes",
                "random_state",
            ),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            max_samples=max_samples,
        )

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
