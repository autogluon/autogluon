import torch
from torch import nn
from torch import Tensor
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast
from ..constants import CATEGORICAL, LABEL, LOGITS, FEATURES
from .ft_transformer import _TokenInitialization, CLSToken, Transformer


class CategoricalFeatureTokenizer(nn.Module):
    """
        Transforms categorical features to tokens (embeddings).
    """

    category_offsets: Tensor

    def __init__(
        self,
        num_categories: List[int],
        d_token: int,
        bias: Optional[bool] = True,
        initialization: Optional[str] = 'normal',
    ) -> None:
        """
        Args:
            prefix:
                The model prefix.
            num_categories: 
                The number of distinct values for each feature. For example,
                :code:`num_categories=[3, 4]` describes two features: the first one can
                take values in the range :code:`[0, 1, 2]` and the second one can take
                values in the range :code:`[0, 1, 2, 3]`.
            d_token: 
                The size of one token.
            bias: 
                If `True`, for each feature, a trainable vector is added to the
                embedding regardless of feature value. The bias vectors are not shared
                between features.
            initialization: 
                Initialization policy for parameters. Must be one of
                :code:`['uniform', 'normal']`. Let :code:`s = d ** -0.5`. Then, the
                corresponding distributions are :code:`Uniform(-s, s)` and :code:`Normal(0, s)`. In
                the paper [gorishniy2021revisiting], the 'uniform' initialization was
                used.

        References:
            * [gorishniy2021revisiting] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko, "Revisiting Deep Learning Models for Tabular Data", 2021
        """
        super().__init__()
        assert num_categories, 'num_categories must be non-empty'
        assert d_token > 0, 'd_token must be positive'
        
        self.num_categories = num_categories

        initialization_ = _TokenInitialization.from_str(initialization)

        category_offsets = torch.tensor([0] + num_categories[:-1]).cumsum(0)

        self.register_buffer('category_offsets', category_offsets, persistent=False)
        self.embeddings = nn.Embedding(sum(num_categories), d_token)
        self.bias = nn.Parameter(Tensor(len(num_categories), d_token)) if bias else None

        for parameter in [self.embeddings.weight, self.bias]:
            if parameter is not None:
                initialization_.apply(parameter, d_token)

    @property
    def n_tokens(self) -> int:
        """The number of tokens."""
        return len(self.category_offsets)

    @property
    def d_token(self) -> int:
        """The size of one token."""
        return self.embeddings.embedding_dim

    def forward(
            self,
            x: Tensor,
    ) -> Tensor:
        
        x = self.embeddings(x + self.category_offsets[None])

        if self.bias is not None:
            x = x + self.bias[None]

        return x




class  CategoricalTransformer(nn.Module):
    """The FT-Transformer model proposed in [gorishniy2021revisiting].

    Transforms features to tokens with `FeatureTokenizer` and applies `Transformer` [vaswani2017attention]
    to the tokens. The following illustration provides a high-level overview of the
    architecture:

    .. image:: ../images/ft_transformer.png
        :scale: 25%
        :alt: FT-Transformer

    The following illustration demonstrates one Transformer block for :code:`prenormalization=True`:

    .. image:: ../images/transformer_block.png
        :scale: 25%
        :alt: PreNorm Transformer block

    Examples:
        .. testcode::

            x_num = torch.randn(4, 3)
            x_cat = torch.tensor([[0, 1], [1, 0], [0, 2], [1, 1]])

            module = FTTransformer.make_baseline(
                n_num_features=3,
                cat_cardinalities=[2, 3],
                d_token=8,
                n_blocks=2,
                attention_dropout=0.2,
                ffn_d_hidden=6,
                ffn_dropout=0.2,
                residual_dropout=0.0,
                d_out=1,
            )
            x = module(x_num, x_cat)
            assert x.shape == (4, 1)

            module = FTTransformer.make_default(
                n_num_features=3,
                cat_cardinalities=[2, 3],
                d_out=1,
            )
            x = module(x_num, x_cat)
            assert x.shape == (4, 1)

        To learn more about the baseline and default parameters:

        .. testcode::

            baseline_parameters = FTTransformer.get_baseline_transformer_subconfig()
            default_parameters = FTTransformer.get_default_transformer_config()

    References:
        * [gorishniy2021revisiting] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko, "Revisiting Deep Learning Models for Tabular Data", 2021
        * [vaswani2017attention] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin, "Attention Is All You Need", 2017
    """

    def __init__(
        self, 
        prefix: str, 
        num_categories: List[int],
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
        self.num_categories = num_categories
        self.categorical_key = f"{prefix}_{CATEGORICAL}"
        self.label_key = f"{prefix}_{LABEL}"

        self.out_features = out_features

        self.categorical_feature_tokenizer = CategoricalFeatureTokenizer(
            num_categories=num_categories,
            d_token=d_token,
            bias=bias,
            initialization=initialization,
        ) 

        self.cls_token = CLSToken(
            d_token=d_token, 
            initialization=initialization,
        ) if cls_token else None

        if kv_compression_ratio is not None: 
            n_tokens = self.categorical_feature_tokenizer.n_tokens + 1
        else:
            n_tokens = None

        self.transformer = Transformer(
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
        assert len(batch[self.categorical_key]) == len(self.num_categories)
        
        # x = self.categorical_feature_tokenizer(batch[self.categorical_key])
        # if self.cls_token:
        #     x = self.cls_token(x)
        # features = self.transformer(x)

        xs = []
        for x in batch[self.categorical_key]:
            xs.append(x)
        x = torch.stack(xs,dim=1)

        x = self.categorical_feature_tokenizer(x)
        
        if self.cls_token:
            x = self.cls_token(x)

        features = self.transformer(x)
        logits = self.head(x)

        return {
            LOGITS: logits,
            FEATURES: features,
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






