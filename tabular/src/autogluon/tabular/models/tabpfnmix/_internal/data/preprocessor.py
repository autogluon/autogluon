import logging

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.preprocessing import QuantileTransformer

from ..core.enums import Task

logger = logging.getLogger(__name__)


class Preprocessor(TransformerMixin, BaseEstimator):
    """
    This class is used to preprocess the data before it is pushed through the model.
    The preprocessor assures that the data has the right shape and is normalized,
    This way the model always gets the same input distribution,
    no matter whether the input data is synthetic or real.

    """

    def __init__(
        self,
        max_features: int,
        use_quantile_transformer: bool,
        use_feature_count_scaling: bool,
        task: Task,
    ):
        self.max_features = max_features
        self.use_quantile_transformer = use_quantile_transformer
        self.use_feature_count_scaling = use_feature_count_scaling
        self.task = task

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.compute_pre_nan_mean(X)
        X = self.impute_nan_features_with_mean(X)

        self.determine_which_features_are_singular(X)
        X = self.cutoff_singular_features(X, self.singular_features)

        self.determine_which_features_to_select(X, y)
        X = self.select_features(X)

        if self.use_quantile_transformer:
            n_obs, n_features = X.shape
            n_quantiles = min(n_obs, 1000)
            self.quantile_transformer = QuantileTransformer(n_quantiles=n_quantiles, output_distribution="normal")
            X = self.quantile_transformer.fit_transform(X)

        self.mean, self.std = self.calc_mean_std(X)
        X = self.normalize_by_mean_std(X, self.mean, self.std)

        assert np.isnan(X).sum() == 0, "There are NaNs in the data after preprocessing"

        return self

    def transform(self, X: np.ndarray):
        X = self.cutoff_singular_features(X, self.singular_features)
        X = self.impute_nan_features_with_mean(X)
        X = self.select_features(X)

        if self.use_quantile_transformer:
            X = self.quantile_transformer.transform(X)

        X = self.normalize_by_mean_std(X, self.mean, self.std)

        if self.use_feature_count_scaling:
            X = self.normalize_by_feature_count(X, self.max_features)

        X = self.extend_feature_dim_to_max_features(X, self.max_features)

        assert np.isnan(X).sum() == 0, "There are NaNs in the data after preprocessing"

        return X

    def determine_which_features_are_singular(self, x: np.ndarray) -> None:
        self.singular_features = np.array([len(np.unique(x_col)) for x_col in x.T]) == 1

    def determine_which_features_to_select(self, x: np.ndarray, y: np.ndarray) -> None:
        if x.shape[1] > self.max_features:
            logger.info(
                f"A maximum of {self.max_features} features are allowed, but the dataset has {x.shape[1]} features. A subset of {self.max_features} are selected using SelectKBest"
            )

            if self.task == Task.CLASSIFICATION:
                self.select_k_best = SelectKBest(k=self.max_features, score_func=f_classif)
            else:  # Task.REGRESSION
                self.select_k_best = SelectKBest(k=self.max_features, score_func=f_regression)
            self.select_k_best.fit(x, y)

    def compute_pre_nan_mean(self, x: np.ndarray) -> None:
        """
        Computes the mean of the data before the NaNs are imputed
        """
        self.pre_nan_mean = np.nanmean(x, axis=0)

    def impute_nan_features_with_mean(self, x: np.ndarray) -> np.ndarray:
        inds = np.where(np.isnan(x))
        x[inds] = np.take(self.pre_nan_mean, inds[1])
        return x

    def select_features(self, x: np.ndarray) -> np.ndarray:
        if x.shape[1] > self.max_features:
            x = self.select_k_best.transform(x)

        return x

    def cutoff_singular_features(self, x: np.ndarray, singular_features: np.ndarray) -> np.ndarray:
        if singular_features.any():
            x = x[:, ~singular_features]

        return x

    def calc_mean_std(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculates the mean and std of the training data
        """
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        return mean, std

    def normalize_by_mean_std(self, x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        """
        Normalizes the data by the mean and std
        """

        x = (x - mean) / std
        return x

    def normalize_by_feature_count(self, x: np.ndarray, max_features) -> np.ndarray:
        """
        An interesting way of normalization by the tabPFN paper
        """

        x = x * max_features / x.shape[1]
        return x

    def extend_feature_dim_to_max_features(self, x: np.ndarray, max_features) -> np.ndarray:
        """
        Increases the number of features to the number of features the model has been trained on
        """
        added_zeros = np.zeros((x.shape[0], max_features - x.shape[1]), dtype=np.float32)
        x = np.concatenate([x, added_zeros], axis=1)
        return x
