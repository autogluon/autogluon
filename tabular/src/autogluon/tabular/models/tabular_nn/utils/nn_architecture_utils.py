"""Helper functions related to NN architectures"""

import numpy as np

from autogluon.core.constants import REGRESSION


def get_embed_sizes(train_dataset, params, num_categs_per_feature):
    """Returns list of embedding sizes for each categorical variable.
    Selects this adaptively based on training_dataset.
    Note: Assumes there is at least one embed feature.
    """
    max_embedding_dim = params["max_embedding_dim"]
    embed_exponent = params["embed_exponent"]
    size_factor = params["embedding_size_factor"]
    embed_dims = [
        int(size_factor * max(2, min(max_embedding_dim, 1.6 * num_categs_per_feature[i] ** embed_exponent)))
        for i in range(len(num_categs_per_feature))
    ]
    return embed_dims


def infer_y_range(y_vals, y_range_extend):
    """Infers good output range for neural network when used for regression."""
    min_y = float(y_vals.min())
    max_y = float(y_vals.max())
    std_y = np.std(y_vals)
    y_ext = y_range_extend * std_y
    if min_y >= 0:  # infer y must be nonnegative
        min_y = max(0, min_y - y_ext)
    else:
        min_y = min_y - y_ext
    if max_y <= 0:  # infer y must be non-positive
        max_y = min(0, max_y + y_ext)
    else:
        max_y = max_y + y_ext
    return (min_y, max_y)


def get_default_layers(problem_type, num_net_outputs, max_layer_width):
    """Default sizes for NN layers."""
    if problem_type == REGRESSION:
        default_layer_sizes = [
            256,
            128,
        ]  # overall network will have 4 layers. Input layer, 256-unit hidden layer, 128-unit hidden layer, output layer.
    else:
        default_sizes = [256, 128]  # will be scaled adaptively
        # base_size = max(1, min(num_net_outputs, 20)/2.0) # scale layer width based on number of classes
        base_size = max(
            1, min(num_net_outputs, 100) / 50
        )  # TODO: Updated because it improved model quality and made training far faster
        default_layer_sizes = [defaultsize * base_size for defaultsize in default_sizes]
    layer_expansion_factor = 1  # TODO: consider scaling based on num_rows, eg: layer_expansion_factor = 2-np.exp(-max(0,train_dataset.num_examples-10000))
    return [int(min(max_layer_width, layer_expansion_factor * defaultsize)) for defaultsize in default_layer_sizes]


def default_numeric_embed_dim(train_dataset, max_layer_width, first_layer_width):
    """Default embedding dimensionality for numeric features."""
    vector_dim = train_dataset.dataset._data[train_dataset.vectordata_index].shape[
        1
    ]  # total dimensionality of vector features
    prop_vector_features = train_dataset.num_vector_features() / float(
        train_dataset.num_features
    )  # Fraction of features that are numeric
    min_numeric_embed_dim = 32
    max_numeric_embed_dim = max_layer_width
    return int(
        min(
            max_numeric_embed_dim,
            max(min_numeric_embed_dim, first_layer_width * prop_vector_features * np.log10(vector_dim + 10)),
        )
    )
