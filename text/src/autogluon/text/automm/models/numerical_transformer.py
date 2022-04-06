from torch import nn
from torch import Tensor
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Union, cast
from .ft_transformer import FT_Transformer,_TokenInitialization,_CLSToken
from ..constants import NUMERICAL, LABEL, LOGITS, FEATURES


class NumericalFeatureTokenizer(nn.Module):
    """
    Reference: 
        1. Github : https://github.com/Yura52/rtdl/blob/3c13a4f18b76b7b25f09eb94075c39ba4d1d7565/rtdl/modules.py#L161 
        2. Paper: "Revisiting Deep Learning Models for Tabular Data"
                   https://arxiv.org/pdf/2106.11959.pdf  

    Transforms continuous features to tokens (embeddings).
    For one feature, the transformation consists of two steps:
        1. the feature is multiplied by a trainable vector i.e., weights,
        2. another trainable vector is added i.e., bias.

    Note that each feature has its separate pair of trainable vectors, 
    i.e. the vectors are not shared between features.
    """

    def __init__(
        self,
        in_features: int,
        d_token: int,
        bias: Optional[bool] = True,
        initialization: Optional[str] = 'normal',
    ):
        """
        Args:
            prefix:
                The model prefix.
            in_features: 
                Dimension of input features i.e. the number of continuous (scalar) features
            d_token: 
                The size of one token
            bias: 
                If `False`, then the transformation will include only multiplication.
                **Warning**: :code:`bias=False` leads to significantly worse results for
                Transformer-like (token-based) architectures.
            initialization: 
                Initialization policy for parameters. Must be one of :code:`['uniform', 'normal']`. 
                Let :code:`s = d ** -0.5`. Then, the
                corresponding distributions are :code:`Uniform(-s, s)` and :code:`Normal(0, s)`.
                In [gorishniy2021revisiting], the 'uniform' initialization was used.

        References:
            * [gorishniy2021revisiting] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko, "Revisiting Deep Learning Models for Tabular Data", 2021
        """
        super().__init__()

        initialization_ = _TokenInitialization.from_str(initialization)
        self.weight = nn.Parameter(Tensor(in_features, d_token))
        self.bias = nn.Parameter(Tensor(in_features, d_token)) if bias else None
        for parameter in [self.weight, self.bias]:
            if parameter is not None:
                initialization_.apply(parameter, d_token)

    @property
    def n_tokens(self) -> int:
        """The number of tokens."""
        return len(self.weight)

    @property
    def d_token(self) -> int:
        """The size of one token."""
        return self.weight.shape[1]

    def forward(
        self, 
        x: Tensor,
    ) -> Tensor:
        x = self.weight[None] * x[..., None]
        if self.bias is not None:
            x = x + self.bias[None]

        return x


class  NumericalTransformer(nn.Module):
    """The FT-Transformer model proposed in [gorishniy2021revisiting].
    References:
        * [gorishniy2021revisiting] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko, "Revisiting Deep Learning Models for Tabular Data", 2021
        * [vaswani2017attention] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin, "Attention Is All You Need", 2017
    """

    def __init__(
        self, 
        prefix: str, 
        in_features: int,
        d_token: int,
        # n_tokens: Optional[int] = None,
        cls_token: Optional[bool] = False,
        out_features: Optional[int] = None,
        num_classes: Optional[int] = 0,
        bias: Optional[bool] = True,
        initialization: Optional[str] = 'normal',
        n_blocks: Optional[int] = 2,
        attention_n_heads: Optional[int] = 8,
        attention_initialization: Optional[str] = 'kaiming',
        attention_normalization: Optional[str] = 'LayerNorm',
        attention_dropout: Optional[str] = 0.2, 
        residual_dropout: Optional[str] = 0.0,
        ffn_activation: Optional[str] = 'ReGLU',
        ffn_normalization: Optional[str] = 'LayerNorm',
        ffn_d_hidden: Optional[str] = 6,
        ffn_dropout: Optional[str] = 0.0,
        prenormalization: Optional[bool] = True,
        first_prenormalization: Optional[bool] =  False,
        last_layer_query_idx: Union[None, List[int], slice] = None,
        kv_compression_ratio: Optional[float] = None,
        kv_compression_sharing: Optional[str] = None,
        head_activation: Optional[str] =  'ReLU',
        head_normalization: Optional[str] = 'LayerNorm',
    ):
        super().__init__()
        self.prefix = prefix
        self.numerical_key = f"{prefix}_{NUMERICAL}"
        self.label_key = f"{prefix}_{LABEL}"

        self.out_features = out_features

        self.numerical_feature_tokenizer = NumericalFeatureTokenizer(
            in_features=in_features,
            d_token=d_token,
            bias=bias,
            initialization=initialization,
        )

        self.cls_token = _CLSToken(
            d_token=d_token, 
            initialization=initialization,
        ) if cls_token else None

        if kv_compression_ratio is not None: 
            if self.cls_token:
                n_tokens = self.numerical_feature_tokenizer.n_tokens + 1
            else:
                n_tokens = self.numerical_feature_tokenizer.n_tokens
        else:
            n_tokens = None

        self.transformer = FT_Transformer(
            d_token=d_token,
            n_blocks=n_blocks,
            attention_n_heads=attention_n_heads,
            attention_dropout=attention_dropout,
            attention_initialization=attention_initialization,
            attention_normalization=attention_normalization,
            ffn_d_hidden=ffn_d_hidden,
            ffn_dropout=ffn_dropout,
            ffn_activation=ffn_activation,
            ffn_normalization=ffn_normalization,
            residual_dropout=residual_dropout,
            prenormalization=prenormalization,
            first_prenormalization=first_prenormalization,
            last_layer_query_idx=last_layer_query_idx,
            n_tokens=n_tokens,
            kv_compression_ratio=kv_compression_ratio,
            kv_compression_sharing=kv_compression_sharing,
            head_activation=head_activation,
            head_normalization=head_normalization,
            d_out=out_features,
        )

        self.head = nn.Linear(out_features, num_classes) if num_classes > 0 else nn.Identity()

        self.name_to_id = self.get_layer_ids()
        # self.head_layer_names = [n for n, layer_id in self.name_to_id.items() if layer_id == 0]

    def forward(
        self, 
        batch: dict
    ):
        x = self.numerical_feature_tokenizer(batch[self.numerical_key])

        if self.cls_token:
            x = self.cls_token(x)

        features = self.transformer(x)
        logits = self.head(features)

        return {
            self.prefix: {
                LOGITS: logits,
                FEATURES: features,
            }
        }

    def get_layer_ids(self,):
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







