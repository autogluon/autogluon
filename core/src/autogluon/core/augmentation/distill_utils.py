import gc
import logging

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from autogluon.common.features.feature_metadata import FeatureMetadata
from autogluon.common.features.types import R_CATEGORY, R_FLOAT, R_INT

from ..constants import BINARY, MULTICLASS, REGRESSION
from ..metrics import mean_squared_error

logger = logging.getLogger(__name__)


def format_distillation_labels(y, problem_type, num_classes=None, eps_labelsmooth=0.01):
    """Transforms train/test label objects (y) to the correct type for distillation (smoothed regression targets for binary, one-hot labels for multiclass).
    eps_labelsmooth : truncates labels to [EPS, 1-EPS], eg. when converting binary problems -> regression
    """
    if problem_type == MULTICLASS:
        y_int = y.to_numpy()
        y = np.zeros((y_int.size, num_classes))
        y[np.arange(y_int.size), y_int] = 1
        y = pd.DataFrame(y)
    elif problem_type == BINARY:
        min_pred = 0.0
        max_pred = 1.0
        y = eps_labelsmooth + ((1 - 2 * eps_labelsmooth) / (max_pred - min_pred)) * (y - min_pred)
    return y


def augment_data(
    X, feature_metadata: FeatureMetadata, augmentation_data=None, augment_method="spunge", augment_args=None
):
    """augment_method options: ['spunge', 'munge']"""
    if augment_args is None:
        augment_args = {}
    if augmentation_data is not None:
        X_aug = augmentation_data
    else:
        if "num_augmented_samples" not in augment_args:
            if "max_size" not in augment_args:
                augment_args["max_size"] = np.inf
            augment_args["num_augmented_samples"] = int(
                min(augment_args["max_size"], augment_args["size_factor"] * len(X))
            )

        if augment_method == "spunge":
            X_aug = spunge_augment(X, feature_metadata, **augment_args)
        elif augment_method == "munge":
            X_aug = munge_augment(X, feature_metadata, **augment_args)
        else:
            raise ValueError(f"unknown augment_method: {augment_method}")

    # return postprocess_augmented(X_aug, X)  # TODO: dropping duplicates is much more efficient, but may skew distribution for entirely-categorical data with few categories.
    logger.log(15, f"Augmented training dataset with {len(X_aug)} extra datapoints")
    return X_aug.reset_index(drop=True)


def postprocess_augmented(X_aug, X):
    """Drops rows from augmented data that are duplicated (including duplicates that appeared in original data X)."""
    X_aug = pd.concat([X, X_aug])
    X_aug.drop_duplicates(keep="first", inplace=True)
    X_aug = X_aug.tail(len(X_aug) - len(X))
    logger.log(15, f"Augmented training dataset with {len(X_aug)} extra datapoints")
    return X_aug.reset_index(drop=True, inplace=False)


def spunge_augment(
    X,
    feature_metadata: FeatureMetadata,
    num_augmented_samples=10000,
    frac_perturb=0.1,
    continuous_feature_noise=0.1,
    **kwargs,
):
    """Generates synthetic datapoints for learning to mimic teacher model in distillation
    via simplified version of MUNGE strategy (that does not require near-neighbor search).

    Args:
        num_augmented_samples: number of additional augmented data points to return
        frac_perturb: fraction of features/examples that are perturbed during augmentation. Set near 0 to ensure augmented sample distribution remains closer to real data.
        continuous_feature_noise: we noise numeric features by this factor times their std-dev. Set near 0 to ensure augmented sample distribution remains closer to real data.
    """
    if frac_perturb > 1.0:
        raise ValueError("frac_perturb must be <= 1")
    logger.log(
        20, f"SPUNGE: Augmenting training data with {num_augmented_samples} synthetic samples for distillation..."
    )

    X = X.copy()
    nan_category = "__NaN__"
    category_featnames = feature_metadata.get_features(valid_raw_types=[R_CATEGORY])
    # Store original categories to restore later
    original_categories = {}
    for feature in category_featnames:
        current_categories = X[feature].cat.categories
        original_categories[feature] = current_categories
        if nan_category in current_categories:
            X[feature] = X[feature].fillna(nan_category)
        else:
            X[feature] = X[feature].cat.add_categories(nan_category).fillna(nan_category)

    continuous_types = [R_FLOAT, R_INT]
    continuous_featnames = feature_metadata.get_features(
        valid_raw_types=continuous_types
    )  # these features will have shuffled values with added noise

    # Rather than loop row-wise, we build a large mask that indicates all cells to be
    # resampled and then loop column-wise, resampling and replacing
    _, y_i = X.shape
    feature_count = int(frac_perturb * y_i)
    # Builds our new frame with much less copying than before
    X_aug = pd.concat([X] * (int(num_augmented_samples / len(X)) + 1)).copy()[:num_augmented_samples]
    x_aug_i, _ = X_aug.shape
    X_aug.reset_index(drop=True, inplace=True)
    arr = np.array([np.random.choice(y_i, replace=False, size=feature_count) for _ in range(x_aug_i)])
    max_col = y_i

    # Create an empty boolean array of appropriate shape
    n_rows = arr.shape[0]
    mask = np.zeros((n_rows, max_col), dtype=bool)

    # Use advanced indexing to set the specified positions to True
    row_indices = np.arange(n_rows)[:, None]
    mask[row_indices, arr] = True
    mask = mask.T
    for i, col in enumerate(mask):
        X_aug.loc[col, X.columns[i]] = X.iloc[:, i].sample(col.sum(), replace=True).values

    for feature in X.columns:
        if feature in continuous_featnames:
            feature_data = X[feature]
            aug_data = X_aug[feature]
            noise = np.random.normal(
                scale=np.nanstd(feature_data) * continuous_feature_noise, size=num_augmented_samples
            )
            mask = np.random.binomial(n=1, p=frac_perturb, size=num_augmented_samples)
            aug_data = aug_data + noise * mask
            X_aug[feature] = pd.Series(aug_data, index=X_aug.index)

    for feature in category_featnames:
        # Properly restore categorical features to their original state
        if nan_category not in original_categories[feature]:
            # Need to convert to string temporarily to handle NaN values correctly
            X_aug[feature] = X_aug[feature].astype(str)
            X_aug.loc[X_aug[feature] == nan_category, feature] = np.nan
            X_aug[feature] = pd.Categorical(X_aug[feature], categories=original_categories[feature])
        else:
            # Keep the NaN category but ensure we have the original categorical structure
            X_aug[feature] = pd.Categorical(X_aug[feature], categories=original_categories[feature])

    return X_aug


# TODO: Remove or fix, likely doesn't work anymore
def munge_augment(
    X, feature_metadata: FeatureMetadata, num_augmented_samples=10000, perturb_prob=0.5, s=1.0, **kwargs
):
    """Uses MUNGE algorithm to generate synthetic datapoints for learning to mimic teacher model in distillation: https://www.cs.cornell.edu/~caruana/compression.kdd06.pdf
    Args:
        num_augmented_samples: number of additional augmented data points to return
        perturb_prob: probability of perturbing each feature during augmentation. Set near 0 to ensure augmented sample distribution remains closer to real data.
        s: We noise numeric features by their std-dev divided by this factor (inverse of continuous_feature_noise). Set large to ensure augmented sample distribution remains closer to real data.
    """
    from autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch import TabularNeuralNetTorchModel

    nn_dummy = TabularNeuralNetTorchModel(
        path="nn_dummy",
        name="nn_dummy",
        problem_type=REGRESSION,
        eval_metric=mean_squared_error,
        hyperparameters={"num_dataloading_workers": 0, "proc.embed_min_categories": np.inf},
        features=list(X.columns),
        feature_metadata=feature_metadata,
    )
    processed_data = nn_dummy._process_train_data(
        df=nn_dummy.preprocess(X),
        labels=pd.Series([1] * len(X)),
        batch_size=nn_dummy.params["batch_size"],
        num_dataloading_workers=0,
        impute_strategy=nn_dummy.params["proc.impute_strategy"],
        max_category_levels=nn_dummy.params["proc.max_category_levels"],
        skew_threshold=nn_dummy.params["proc.skew_threshold"],
        embed_min_categories=nn_dummy.params["proc.embed_min_categories"],
        use_ngram_features=nn_dummy.params["use_ngram_features"],
    )
    X_vector = processed_data.dataset._data[processed_data.vectordata_index].asnumpy()
    processed_data = None
    nn_dummy = None
    gc.collect()

    neighbor_finder = NearestNeighbors(n_neighbors=2)
    neighbor_finder.fit(X_vector)
    neigh_dist, neigh_ind = neighbor_finder.kneighbors(X_vector)
    neigh_ind = neigh_ind[:, 1]  # contains indices of nearest neighbors
    neigh_dist = None
    # neigh_dist = neigh_dist[:,1]  # contains distances to nearest neighbors
    neighbor_finder = None
    gc.collect()

    if perturb_prob > 1.0:
        raise ValueError("frac_perturb must be <= 1")
    logger.log(
        20, f"MUNGE: Augmenting training data with {num_augmented_samples} synthetic samples for distillation..."
    )
    X = X.copy()
    X_aug = pd.concat([X.iloc[[0]].copy()] * num_augmented_samples)
    X_aug.reset_index(drop=True, inplace=True)
    continuous_types = ["float", "int"]
    continuous_featnames = feature_metadata.get_features(
        valid_raw_types=continuous_types
    )  # these features will have shuffled values with added noise

    # Store original categories for categorical features
    category_featnames = feature_metadata.get_features(valid_raw_types=[R_CATEGORY])
    original_categories = {}
    for feature in category_featnames:
        original_categories[feature] = X[feature].cat.categories

    for col in continuous_featnames:
        X_aug[col] = X_aug[col].astype(float)
        X[col] = X[col].astype(float)

    for i in range(num_augmented_samples):
        og_ind = i % len(X)
        augdata_i = X.iloc[og_ind].copy()
        neighbor_i = X.iloc[neigh_ind[og_ind]].copy()
        # dist_i = neigh_dist[og_ind]
        cols_toperturb = np.random.choice(
            list(X.columns), size=np.random.binomial(X.shape[1], p=perturb_prob, size=1)[0], replace=False
        )
        for col in cols_toperturb:
            new_val = neighbor_i[col]
            if col in continuous_featnames:
                new_val += np.random.normal(scale=np.abs(augdata_i[col] - new_val) / s)
            augdata_i[col] = new_val
        X_aug.iloc[i] = augdata_i

    # Properly restore categorical features to their original state
    for feature in category_featnames:
        X_aug[feature] = pd.Categorical(X_aug[feature], categories=original_categories[feature])

    return X_aug
