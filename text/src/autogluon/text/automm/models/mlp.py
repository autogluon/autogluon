from torch import nn
from typing import Optional

ALL_ACT_LAYERS = {
    "leaky_relu": nn.LeakyReLU,
    "gelu": nn.GELU,
    "relu": nn.ReLU,
}


class Unit(nn.Module):
    def __init__(
            self,
            normalization: str,
            in_features: int,
            out_features: int,
            activation: str,
            dropout_prob: float,
    ):
        super().__init__()
        if normalization == "layer_norm":
            self.norm = nn.LayerNorm(in_features)
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
                dropout_prob=dropout_prob
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

