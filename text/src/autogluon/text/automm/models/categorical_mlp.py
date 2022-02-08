import torch
import json
from torch import nn
from .mlp import MLP
from ..constants import CATEGORICAL, LABEL, LOGITS, FEATURES
from typing import Optional, List
from .utils import init_weights


class CategoricalMLP(nn.Module):
    def __init__(
            self,
            prefix: str,
            num_categories: List[int],
            out_features: Optional[int] = None,
            num_layers: Optional[int] = 1,
            activation: Optional[str] = "gelu",
            dropout_prob: Optional[float] = 0.5,
            normalization: Optional[str] = "layer_norm",
            num_classes: Optional[int] = 0,
    ):
        super().__init__()
        self.out_features = out_features
        max_embedding_dim = 100
        embed_exponent = 0.56
        size_factor = 1.0
        self.column_embeddings = nn.ModuleList()
        self.column_mlps = nn.ModuleList()
        assert isinstance(num_categories, list)

        for num_categories_per_col in num_categories:
            embedding_dim_per_col = int(
                size_factor * max(2, min(
                    max_embedding_dim,
                    1.6 * num_categories_per_col ** embed_exponent
                ))
            )
            self.column_embeddings.append(
                nn.Embedding(
                    num_embeddings=num_categories_per_col,
                    embedding_dim=embedding_dim_per_col,
                )
            )

            self.column_mlps.append(
                MLP(
                    in_features=embedding_dim_per_col,
                    hidden_features=out_features,
                    out_features=out_features,
                    num_layers=num_layers,
                    activation=activation,
                    dropout_prob=dropout_prob,
                    normalization=normalization,
                )
            )

        self.aggregator_mlp = MLP(
            in_features=out_features * len(num_categories),
            hidden_features=out_features * len(num_categories),
            out_features=out_features,
            num_layers=num_layers,
            activation=activation,
            dropout_prob=dropout_prob,
            normalization=normalization,
        )

        self.head = nn.Linear(out_features, num_classes) if num_classes > 0 else nn.Identity()
        # init weights
        self.apply(init_weights)

        self.categorical_key = f"{prefix}_{CATEGORICAL}"
        self.label_key = f"{prefix}_{LABEL}"

        self.name_to_id = self.get_layer_ids()
        self.head_layer_names = [n for n, layer_id in self.name_to_id.items() if layer_id == 0]

    def forward(self, batch):
        assert len(batch[self.categorical_key]) == len(self.column_embeddings)
        features = []
        for categorical_id, embed, mlp in zip(batch[self.categorical_key], self.column_embeddings, self.column_mlps):
            features.append(mlp(embed(categorical_id)))
        cat_features = torch.cat(features, dim=1)
        features = self.aggregator_mlp(cat_features)
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
