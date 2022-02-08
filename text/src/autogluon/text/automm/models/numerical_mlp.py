import torch
import json
from torch import nn
from .mlp import MLP
from ..constants import NUMERICAL, LABEL, LOGITS, FEATURES
from typing import Optional
from .utils import init_weights


class NumericalMLP(nn.Module):
    def __init__(
            self,
            prefix: str,
            in_features: int,
            hidden_features: Optional[int] = None,
            out_features: Optional[int] = None,
            num_layers: Optional[int] = 1,
            activation: Optional[str] = "leaky_relu",
            dropout_prob: Optional[float] = 0.5,
            normalization: Optional[str] = "layer_norm",
            num_classes: Optional[int] = 0,
    ):
        super().__init__()
        self.out_features = out_features
        self.mlp = MLP(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            num_layers=num_layers,
            activation=activation,
            dropout_prob=dropout_prob,
            normalization=normalization,
        )
        self.head = nn.Linear(out_features, num_classes) if num_classes > 0 else nn.Identity()
        # init weights
        self.apply(init_weights)

        self.numerical_key = f"{prefix}_{NUMERICAL}"
        self.label_key = f"{prefix}_{LABEL}"

        self.name_to_id = self.get_layer_ids()
        self.head_layer_names = [n for n, layer_id in self.name_to_id.items() if layer_id == 0]

    def forward(self, batch):
        features = self.mlp(batch[self.numerical_key])
        logits = self.head(features)

        return {
            LOGITS: logits,
            FEATURES: features,
        }

    def get_layer_ids(self,):
        """
        All layers have the same id since there is no pre-trained transformers used here
        """
        name_to_id = {}
        for n, _ in self.named_parameters():
            name_to_id[n] = 0
        return name_to_id

