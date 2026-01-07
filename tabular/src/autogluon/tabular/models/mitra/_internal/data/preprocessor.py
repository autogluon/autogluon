import random
from typing import Optional

import numpy as np
from loguru import logger
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OrdinalEncoder, QuantileTransformer, StandardScaler

from ..._internal.config.enums import Task


class NoneTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


class Preprocessor:
    """
    This class is used to preprocess the data before it is pushed through the model.
    The preprocessor assures that the data has the right shape and is normalized,
    This way the model always gets the same input distribution,
    no matter whether the input data is synthetic or real.

    """

    def __init__(
        self,
        dim_embedding: Optional[
            int
        ],  # Size of the feature embedding. For some models this is None, which means the embedding does not depend on the number of features
        n_classes: int,  # Actual number of classes in the dataset, assumed to be numbered 0, ..., n_classes - 1
        dim_output: int,  # Maximum number of classes the model has been trained on -> size of the output
        use_quantile_transformer: bool,
        use_feature_count_scaling: bool,
        use_random_transforms: bool,
        shuffle_classes: bool,
        shuffle_features: bool,
        random_mirror_regression: bool,
        random_mirror_x: bool,
        task: Task,
    ):
        self.dim_embedding = dim_embedding
        self.n_classes = n_classes
        self.dim_output = dim_output
        self.use_quantile_transformer = use_quantile_transformer
        self.use_feature_count_scaling = use_feature_count_scaling
        self.use_random_transforms = use_random_transforms
        self.shuffle_classes = shuffle_classes
        self.shuffle_features = shuffle_features
        self.random_mirror_regression = random_mirror_regression
        self.random_mirror_x = random_mirror_x
        self.task = task

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Preprocessor":
        """
        X: np.ndarray [n_samples, n_features]
        y: np.ndarray [n_samples]
        """

        if self.task == Task.CLASSIFICATION:
            # We assume that y properly presents classes [0, 1, 2, ...] before passing to the preprocessor
            # If the test set has a class that is not in the training set, we will throw an error

            assert np.all(y < self.n_classes), "y contains class values that are not in the range of n_classes"

        self.compute_pre_nan_mean(X)
        X = self.impute_nan_features_with_mean(X)

        self.determine_which_features_are_singular(X)
        X = self.cutoff_singular_features(X, self.singular_features)

        self.determine_which_features_to_select(X, y)
        X = self.select_features(X)

        if self.use_quantile_transformer:
            # If use quantile transform is off, it means that the preprocessing will happen on the GPU.
            X = self.fit_transform_quantile_transformer(X)

            self.mean, self.std = self.calc_mean_std(X)
            X = self.normalize_by_mean_std(X, self.mean, self.std)

        if self.use_random_transforms:
            X = self.transform_tabpfn(X)

        if self.task == Task.CLASSIFICATION and self.shuffle_classes:
            self.determine_shuffle_class_order()

        if self.shuffle_features:
            self.determine_feature_order(X)

        if self.task == Task.REGRESSION:
            self.determine_mix_max_scale(y)

        if self.task == Task.REGRESSION and self.random_mirror_regression:
            self.determine_regression_mirror()

        if self.random_mirror_x:
            self.determine_mirror(X)

        X[np.isnan(X)] = 0
        X[np.isinf(X)] = 0

        return self

    def transform_X(self, X: np.ndarray):
        X = self.impute_nan_features_with_mean(X)
        X = self.cutoff_singular_features(X, self.singular_features)
        X = self.select_features(X)

        if self.use_quantile_transformer:
            # If use quantile transform is off, it means that the preprocessing will happen on the GPU.

            X = self.quantile_transformer.transform(X)

            X = self.normalize_by_mean_std(X, self.mean, self.std)

            if self.use_feature_count_scaling:
                X = self.normalize_by_feature_count(X)

        if self.use_random_transforms:
            X = self.random_transforms.transform(X)

        if self.shuffle_features:
            X = self.randomize_feature_order(X)

        if self.random_mirror_x:
            X = self.apply_random_mirror_x(X)

        X = X.astype(np.float32)

        X[np.isnan(X)] = 0
        X[np.isinf(X)] = 0

        return X

    def transform_tabpfn(self, X: np.ndarray):
        n_samples = X.shape[0]
        n_features = X.shape[1]

        use_config1 = random.random() < 0.5
        random_state = random.randint(0, 1000000)

        if use_config1:
            self.random_transforms = Pipeline(
                [
                    (
                        "quantile",
                        QuantileTransformer(
                            output_distribution="normal",
                            n_quantiles=max(n_samples // 10, 2),
                            random_state=random_state,
                        ),
                    ),
                    (
                        "svd",
                        FeatureUnion(
                            [
                                ("passthrough", NoneTransformer()),
                                (
                                    "svd",
                                    Pipeline(
                                        [
                                            ("standard", StandardScaler(with_mean=False)),
                                            (
                                                "svd",
                                                TruncatedSVD(
                                                    algorithm="arpack",
                                                    n_components=max(1, min(n_samples // 10 + 1, n_features // 2)),
                                                    random_state=random_state,
                                                ),
                                            ),
                                        ]
                                    ),
                                ),
                            ]
                        ),
                    ),
                ]
            )
        else:
            self.random_transforms = ColumnTransformer(
                [("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan), [])],
                remainder="passthrough",
            )

        return self.random_transforms.fit_transform(X)

    def transform_y(self, y: np.ndarray):
        if self.task == Task.CLASSIFICATION:
            # We assume that y properly presents classes [0, 1, 2, ...] before passing to the preprocessor
            # If the test set has a class that is not in the training set, we will throw an error
            assert np.all(y < self.n_classes), "y contains class values that are not in the range of n_classes"

        if self.task == Task.CLASSIFICATION and self.shuffle_classes:
            y = self.randomize_class_order(y)

        if self.task == Task.REGRESSION:
            y = self.normalize_y(y)

        if self.task == Task.REGRESSION and self.random_mirror_regression:
            y = self.apply_random_mirror_regression(y)

        if self.task == Task.CLASSIFICATION:
            y = y.astype(np.int64)
        elif self.task == Task.REGRESSION:
            y = y.astype(np.float32)

        return y

    def inverse_transform_y(self, y: np.ndarray):
        # Function used during the prediction to transform the model output back to the original space
        # For classification, y is assumed to be logits of shape [n_samples, n_classes]

        if self.task == Task.CLASSIFICATION:
            y = self.extract_correct_classes(y)

            if self.shuffle_classes:
                y = self.undo_randomize_class_order(y)

        elif self.task == Task.REGRESSION:
            if self.random_mirror_regression:
                y = self.apply_random_mirror_regression(y)

            y = self.undo_normalize_y(y)

        return y

    def fit_transform_quantile_transformer(self, X: np.ndarray) -> np.ndarray:
        n_obs, n_features = X.shape
        n_quantiles = min(n_obs, 1000)
        self.quantile_transformer = QuantileTransformer(n_quantiles=n_quantiles, output_distribution="normal")
        X = self.quantile_transformer.fit_transform(X)

        return X

    def determine_which_features_are_singular(self, x: np.ndarray) -> None:
        self.singular_features = np.array([len(np.unique(x_col)) for x_col in x.T]) == 1

    def determine_which_features_to_select(self, x: np.ndarray, y: np.ndarray) -> None:
        if self.dim_embedding is None:
            # All features are selected
            return

        if x.shape[1] > self.dim_embedding:
            logger.info(
                f"Number of features is capped at {self.dim_embedding}, but the dataset has {x.shape[1]} features. A subset of {self.dim_embedding} are selected using SelectKBest"
            )

            self.select_k_best = SelectKBest(k=self.dim_embedding)
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
        if self.dim_embedding is None:
            # All features are selected
            return x

        if x.shape[1] > self.dim_embedding:
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
        std = x.std(axis=0) + 1e-6
        return mean, std

    def normalize_by_mean_std(self, x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        """
        Normalizes the data by the mean and std
        """

        x = (x - mean) / std
        return x

    def normalize_by_feature_count(self, x: np.ndarray) -> np.ndarray:
        """
        An interesting way of normalization by the tabPFN paper
        """

        assert self.dim_embedding is not None, "dim_embedding must be set to use this feature count scaling"

        x = x * self.dim_embedding / x.shape[1]

        return x

    def extend_feature_dim_to_dim_embedding(self, x: np.ndarray, dim_embedding) -> np.ndarray:
        """
        Increases the number of features to the number of features the model has been trained on
        """

        assert self.dim_embedding is not None, "dim_embedding must be set to extend the feature dimension"

        added_zeros = np.zeros((x.shape[0], dim_embedding - x.shape[1]), dtype=np.float32)
        x = np.concatenate([x, added_zeros], axis=1)
        return x

    def determine_mix_max_scale(self, y: np.ndarray) -> None:
        self.y_min = y.min()
        self.y_max = y.max()
        assert self.y_min != self.y_max, "y_min and y_max are the same, cannot normalize, regression makes no sense"

    def normalize_y(self, y: np.ndarray) -> np.ndarray:
        y = (y - self.y_min) / (self.y_max - self.y_min)
        return y

    def undo_normalize_y(self, y: np.ndarray) -> np.ndarray:
        y = y * (self.y_max - self.y_min) + self.y_min
        return y

    def determine_regression_mirror(self) -> None:
        self.regression_mirror = np.random.choice([True, False], size=(1,)).item()

    def apply_random_mirror_regression(self, y: np.ndarray) -> np.ndarray:
        if self.regression_mirror:
            y = 1 - y
        return y

    def determine_mirror(self, x: np.ndarray) -> None:
        n_features = x.shape[1]
        self.mirror = np.random.choice([1, -1], size=(1, n_features))

    def apply_random_mirror_x(self, x: np.ndarray) -> np.ndarray:
        x = x * self.mirror
        return x

    def determine_shuffle_class_order(self) -> None:
        if self.shuffle_classes:
            self.new_shuffle_classes = np.random.permutation(self.n_classes)
        else:
            self.new_shuffle_classes = np.arange(self.n_classes)

    def randomize_class_order(self, y: np.ndarray) -> np.ndarray:
        mapping = {i: self.new_shuffle_classes[i] for i in range(self.n_classes)}
        y = np.array([mapping[i.item()] for i in y], dtype=np.int64)

        return y

    def undo_randomize_class_order(self, y_logits: np.ndarray) -> np.ndarray:
        """
        We assume y_logits has shape [n_samples, n_classes]
        """

        # mapping = {self.new_shuffle_classes[i]: i for i in range(self.n_classes)}
        mapping = {i: self.new_shuffle_classes[i] for i in range(self.n_classes)}
        y = np.concatenate([y_logits[:, mapping[i] : mapping[i] + 1] for i in range(self.n_classes)], axis=1)

        return y

    def extract_correct_classes(self, y_logits: np.ndarray) -> np.ndarray:
        # Even though our network might be able to support 10 classes,
        # If the problem only has three classes, we should give three classes as output.
        # We assume y_logits has shape [n_samples, n_classes]
        y_logits = y_logits[:, : self.n_classes]
        return y_logits

    def determine_feature_order(self, x: np.ndarray) -> None:
        n_features = x.shape[1]
        self.new_feature_order = np.random.permutation(n_features)

    def randomize_feature_order(self, x: np.ndarray) -> np.ndarray:
        x = x[:, self.new_feature_order]

        return x
