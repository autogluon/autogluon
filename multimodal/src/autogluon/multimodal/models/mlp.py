from typing import Optional

import numpy as np
import torch
from torch import nn

ALL_ACT_LAYERS = {
    "leaky_relu": nn.LeakyReLU,
    "gelu": nn.GELU,
    "relu": nn.ReLU,
}


class GhostBatchNorm(nn.Module):
    """
    Ghost Batch Normalization.
    It allows the use of large batch sizes,
    but with batch normalization parameters calculated on smaller sub-batches.

    [1] Train longer, generalize better: closing the generalization gap in large batch training of neural networks : https://arxiv.org/abs/1705.08741
    [2] Simple Modifications to Improve Tabular Neural Networks: https://arxiv.org/pdf/2108.03214
    """

    def __init__(
        self,
        input_dim: int,
        virtual_batch_size: Optional[int] = 64,
        momentum: Optional[float] = 0.01,
    ):
        super(GhostBatchNorm, self).__init__()

        self.input_dim = input_dim
        self.virtual_batch_size = virtual_batch_size
        self.bn = nn.BatchNorm1d(self.input_dim, momentum=momentum)

    def forward(self, x):
        chunks = x.chunk(int(np.ceil(x.shape[0] / self.virtual_batch_size)), 0)
        res = [self.bn(x_) for x_ in chunks]

        return torch.cat(res, dim=0)


class Unit(nn.Module):
    """
    One MLP layer. It orders the operations as: norm -> fc -> act_fn -> dropout
    """

    def __init__(
        self,
        normalization: str,
        in_features: int,
        out_features: int,
        activation: str,
        dropout_prob: float,
    ):
        """
        Parameters
        ----------
        normalization
            Name of activation function.
        in_features
            Dimension of input features.
        out_features
            Dimension of output features.
        activation
            Name of activation function.
        dropout_prob
            Dropout probability.
        """
        super().__init__()
        if normalization == "layer_norm":
            self.norm = nn.LayerNorm(in_features)
        elif normalization == "batch_norm":
            self.norm = nn.BatchNorm1d(in_features)
        elif normalization == "ghost_batch_norm":
            self.norm = GhostBatchNorm(in_features)
        else:
            raise ValueError(f"unknown normalization: {normalization}")
        self.fc = nn.Linear(in_features, out_features)
        self.act_fn = ALL_ACT_LAYERS[activation]()
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        # pre normalization
        x = self.norm(x)
        x = self.fc(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        return x


class MLP(nn.Module):
    """
    Multi-layer perceptron (MLP). If the hidden or output feature dimension is
    not provided, we assign it the input feature dimension.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        num_layers: Optional[int] = 1,
        activation: Optional[str] = "gelu",
        dropout_prob: Optional[float] = 0.5,
        normalization: Optional[str] = "layer_norm",
    ):
        """
        Parameters
        ----------
        in_features
            Dimension of input features.
        hidden_features
            Dimension of hidden features.
        out_features
            Dimension of output features.
        num_layers
            Number of layers.
        activation
            Name of activation function.
        dropout_prob
            Dropout probability.
        normalization
            Name of normalization function.
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        layers = []
        for _ in range(num_layers):
            per_unit = Unit(
                normalization=normalization,
                in_features=in_features,
                out_features=hidden_features,
                activation=activation,
                dropout_prob=dropout_prob,
            )
            in_features = hidden_features
            layers.append(per_unit)
        if out_features != hidden_features:
            self.fc_out = nn.Linear(hidden_features, out_features)
        else:
            self.fc_out = None
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        if self.fc_out is not None:
            return self.fc_out(x)
        else:
            return x
