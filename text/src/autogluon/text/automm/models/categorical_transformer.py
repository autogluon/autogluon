import torch
from torch import nn
from torch import Tensor
from typing import Any, Dict, List, Optional, Union, cast
from ..constants import CATEGORICAL, LABEL, LOGITS, FEATURES
from .ft_transformer import _TokenInitialization, CLSToken, FT_Transformer


class CategoricalFeatureTokenizer(nn.Module):
    """
    Feature tokenizer for categorical features in tabular data. 
    It transforms the input categorical features to tokens (embeddings).

    The categorical features usually refers to discrete features.
    """

    def __init__(
        self,
        num_categories: List[int],
        d_token: int,
        bias: Optional[bool] = True,
        initialization: Optional[str] = 'normal',
    ) -> None:
        """
        Parameters
        ----------
        num_categories: 
            A list of integers. Each one is the number of categories in one categorical column.
        d_token: 
            The size of one token.
        bias: 
            If `True`, for each feature, an additional trainable vector will be added to the
            embedding regardless of feature value. Notablly, the bias are not shared between features.
        initialization: 
            Initialization policy for parameters. Must be one of `['uniform', 'normal']`. 

        References
        ----------
        Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko, 
        "Revisiting Deep Learning Models for Tabular Data", 2021
        https://arxiv.org/pdf/2106.11959.pdf
        """
        super().__init__()
        
        self.num_categories = num_categories
        category_offsets = torch.tensor([0] + num_categories[:-1]).cumsum(0)

        self.register_buffer('category_offsets', category_offsets, persistent=False)
        self.embeddings = nn.Embedding(sum(num_categories), d_token)
        self.bias = nn.Parameter(Tensor(len(num_categories), d_token)) if bias else None
        initialization_ = _TokenInitialization.from_str(initialization)

        for parameter in [self.embeddings.weight, self.bias]:
            if parameter is not None:
                initialization_.apply(parameter, d_token)
    
    @property
    def n_tokens(self) -> int:
        """The number of tokens."""
        return len(self.num_categories)

    @property
    def d_token(self) -> int:
        """The size of one token."""
        return self.embeddings.embedding_dim

    def forward(self, x: Tensor) -> Tensor:
        x = self.embeddings(x + self.category_offsets[None])

        if self.bias is not None:
            x = x + self.bias[None]

        return x


class CategoricalTransformer(nn.Module):
    """
    FT-Transformer for categorical tabular features. 
    The input dimension is automatically computed based on
    the number of categories in each categorical column.
    """

    def __init__(
        self, 
        prefix: str, 
        num_categories: List[int],
        d_token: int,
        cls_token: Optional[bool] = False,
        out_features: Optional[int] = None,
        num_classes: Optional[int] = 0,
        token_bias: Optional[bool] = True,
        token_initialization: Optional[str] = 'normal',
        n_blocks: Optional[int] = 0,
        attention_n_heads: Optional[int] = 8,
        attention_initialization: Optional[str] = 'kaiming',
        attention_normalization: Optional[str] = 'layer_norm',
        attention_dropout: Optional[str] = 0.2, 
        residual_dropout: Optional[str] = 0.0,
        ffn_activation: Optional[str] = 'reglu',
        ffn_normalization: Optional[str] = 'layer_norm',
        ffn_d_hidden: Optional[str] = 6,
        ffn_dropout: Optional[str] = 0.0,
        prenormalization: Optional[bool] = True,
        first_prenormalization: Optional[bool] =  False,
        kv_compression_ratio: Optional[float] = None,
        kv_compression_sharing: Optional[str] = None,
        head_activation: Optional[str] = 'relu',
        head_normalization: Optional[str] = 'layer_norm',
    ) -> None :
        """
        Parameters
        ----------
        prefix
            The model prefix.
        num_categories
            A list of integers. Each one is the number of categories in one categorical column.
        d_token
            The size of one token for `_CategoricalFeatureTokenizer`.
        cls_token
            If `True`, cls token will be added to the token embeddings.
        out_features
            Dimension of output features.
        num_classes
            Number of classes. 1 for a regression task.
        token_bias
            If `True`, for each feature, an additional trainable vector will be added in `_CategoricalFeatureTokenizer` 
            to the embedding regardless of feature value. Notablly, the bias are not shared between features.
        token_initialization
            Initialization policy for parameters in `_CategoricalFeatureTokenizer` and `_CLSToke`. 
            Must be one of `['uniform', 'normal']`. 
        n_blocks
            Number of the `FT_Transformer` blocks, which should be non-negative.
        attention_n_heads
            Number of attention heads in each `FT_Transformer` block, which should be postive.
        attention_initialization
            Weights initalization scheme for Multi Headed Attention module.
        attention_dropout
            Dropout ratio for the Multi Headed Attention module.
        residual_dropout
            Dropout ratio for the linear layers in FT_Transformer block.
        ffn_activation
            Activation function type for the Feed-Forward Network module.
        ffn_normalization
            Normalization scheme of the Feed-Forward Network module.
        ffn_d_hidden
            Number of the hidden nodes of the linaer layers in the Feed-Forward Network module.
        ffn_dropout
            Dropout ratio of the hidden nodes of the linaer layers in the Feed-Forward Network module.
        prenormalization, first_prenormalization
            Prenormalization to stablize the training.
        kv_compression_ratio
            The compression ration to reduce the input sequence length.
        kv_compression_sharing
            If `true` the projections will share weights.
        head_activation
            Activation function type of the MLP layer.
        head_normalization
            Normalization scheme of the MLP layer.

        References
        ----------
        Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko, 
        "Revisiting Deep Learning Models for Tabular Data", 2021
        https://arxiv.org/pdf/2106.11959.pdf
        """

        super().__init__()

        assert num_categories, 'num_categories must be non-empty'
        assert d_token > 0, 'd_token must be positive'
        assert n_blocks >= 0, 'n_blocks must be non-negative' 
        assert attention_n_heads > 0, 'attention_n_heads must be postive'
        assert token_initialization in ['uniform', 'normal'], 'initialization must be uniform or normal'

        self.num_categories = num_categories

        self.prefix = prefix
        self.out_features = out_features


        self.categorical_feature_tokenizer = CategoricalFeatureTokenizer(
            num_categories=num_categories,
            d_token=d_token,
            bias=token_bias,
            initialization=token_initialization,
        ) 

        self.cls_token = CLSToken(
            d_token=d_token, 
            initialization=token_initialization,
        ) if cls_token else nn.Identity()

        if kv_compression_ratio is not None: 
            n_tokens = self.categorical_feature_tokenizer.n_tokens + 1
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
            last_layer_query_idx=None,
            n_tokens=n_tokens,
            kv_compression_ratio=kv_compression_ratio,
            kv_compression_sharing=kv_compression_sharing,
            head_activation=head_activation,
            head_normalization=head_normalization,
            d_out=out_features,
        )

        self.head = FT_Transformer.Head(
            d_in=d_token,
            d_out=num_classes,
            bias=True,
            activation=head_activation, 
            normalization=head_normalization if prenormalization else 'Identity',
        )

        self.name_to_id = self.get_layer_ids()

    @property
    def categorical_key(self):
        return f"{self.prefix}_{CATEGORICAL}"

    @property
    def label_key(self):
        return f"{self.prefix}_{LABEL}"

    def forward(
        self, 
        batch: dict
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
        
        categorical_features = []
        for categorical_feature in batch[self.categorical_key]:
            categorical_features.append(categorical_feature)
        categorical_features = torch.stack(categorical_features,dim=1)

        features = self.categorical_feature_tokenizer(categorical_features)
        features = self.cls_token(features)
        features = self.transformer(features)
        
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
