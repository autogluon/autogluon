from torch import nn
from .mlp import MLP
from ..constants import NUMERICAL, LABEL, LOGITS, FEATURES
from typing import Optional, List
from .utils import init_weights
from .numerical_transformer import NumEmbeddings


class NumericalMLP(nn.Module):
    """
    MLP for numerical input.
    """

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
        d_token: Optional[int] = 8,
        embedding_arch: Optional[List[str]] = None,
    ):
        """
        Parameters
        ----------
        prefix
            The model prefix.
        in_features
            Dimension of input features.
        hidden_features
            Dimension of hidden features.
        out_features
            Dimension of output features.
        num_layers
            Number of MLP layers.
        activation
            Name of activation function.
        dropout_prob
            Dropout probability.
        normalization
            Name of normalization function.
        num_classes
            Number of classes. 1 for a regression task.
        d_token
            The size of one token for `NumericalEmbedding`.
        embedding_arch
            A list containing the names of embedding layers.
            Currently support:
            {'linear', 'shared_linear', 'autodis', 'positional', 'relu', 'layernorm'}
        """
        super().__init__()
        self.out_features = out_features

        self.numerical_feature_tokenizer = (
            NumEmbeddings(
                in_features=in_features,
                d_embedding=d_token,
                embedding_arch=embedding_arch,
            )
            if embedding_arch is not None
            else nn.Identity()
        )

        in_features = in_features * d_token if embedding_arch is not None else in_features

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

        self.prefix = prefix

        self.name_to_id = self.get_layer_ids()
        self.head_layer_names = [n for n, layer_id in self.name_to_id.items() if layer_id == 0]

    @property
    def numerical_key(self):
        return f"{self.prefix}_{NUMERICAL}"

    @property
    def label_key(self):
        return f"{self.prefix}_{LABEL}"

    def forward(
        self,
        batch: dict,
    ):
        """

        Parameters
        ----------
        batch
            A dictionary containing the input mini-batch data.
            We need to use the keys with the model prefix to index required data.

        Returns
        -------
            A dictionary with logits and features.
        """
        features = self.numerical_feature_tokenizer(batch[self.numerical_key])
        features = features.flatten(1, 2) if features.ndim == 3 else features
        features = self.mlp(features)
        logits = self.head(features)

        return {
            self.prefix: {
                LOGITS: logits,
                FEATURES: features,
            }
        }

    def get_layer_ids(
        self,
    ):
        """
        All layers have the same id 0 since there is no pre-trained models used here.

        Returns
        -------
        A dictionary mapping the layer names (keys) to their ids (values).
        """
        name_to_id = {}
        for n, _ in self.named_parameters():
            name_to_id[n] = 0
        return name_to_id
