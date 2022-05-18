from torch import nn
from torch import Tensor
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Union, cast, Literal
from .ft_transformer import FT_Transformer,_TokenInitialization,CLSToken
from ..constants import NUMERICAL, LABEL, LOGITS, FEATURES
import torch



class PeriodicOptions:
    n: int  # the output size is 2 * n
    sigma: float
    trainable: bool
    initialization: Literal['log-linear', 'normal']


class Periodic(nn.Module):
    def __init__(self, n_features: int, options: PeriodicOptions) -> None:
        super().__init__()
        if options.initialization == 'log-linear':
            coefficients = options.sigma ** (torch.arange(options.n) / options.n)
            coefficients = coefficients[None].repeat(n_features, 1)
        else:
            assert options.initialization == 'normal'
            coefficients = torch.normal(0.0, options.sigma, (n_features, options.n))
        if options.trainable:
            self.coefficients = nn.Parameter(coefficients)  # type: ignore[code]
        else:
            self.register_buffer('coefficients', coefficients)

    def cos_sin(x: Tensor) -> Tensor:
        return torch.cat([torch.cos(x), torch.sin(x)], -1)

    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim == 2
        return self.cos_sin(2 * torch.pi * self.coefficients[None] * x[..., None])


class NLinear(nn.Module):
    def __init__(
        self, 
        n: int, 
        d_in: int, 
        d_out: int, 
        bias: bool = True
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(Tensor(n, d_in, d_out))
        self.bias = nn.Parameter(Tensor(n, d_out)) if bias else None
        with torch.no_grad():
            for i in range(n):
                layer = nn.Linear(d_in, d_out)
                self.weight[i] = layer.weight.T
                if self.bias is not None:
                    self.bias[i] = layer.bias

    def forward(self, x):
        assert x.ndim == 3, 'Error Dimension, should be 3, but given {}'.format(x.ndim)
        x = x[..., None] * self.weight[None]
        x = x.sum(-2)
        if self.bias is not None:
            x = x + self.bias[None]
        return x


class NLinearMemoryEfficient(nn.Module):
    def __init__(
        self, 
        n: int, 
        d_in: int, 
        d_out: int
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(d_in, d_out) for _ in range(n)])

    def forward(self, x):
        return torch.stack([l(x[:, i]) for i, l in enumerate(self.layers)], 1)


class NumericalFeatureTokenizer(nn.Module):
    """
    Numerical tokenizer for numerical features in tabular data. 
    It transforms the input numerical features to tokens (embeddings).

    The numerical features usually refers to continous features.
    
    It consists of two steps:
        1. each feature is multiplied by a trainable vector i.e., weights,
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
        Parameters
        ----------
        in_features: 
            Dimension of input features i.e. the number of continuous (scalar) features
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


class AutoDis(nn.Module):
    """
    Paper (the version is important): https://arxiv.org/abs/2012.08986v2
    Code: https://github.com/mindspore-ai/models/tree/bdf2d8bcf11fe28e4ad3060cf2ddc818eacd8597/research/recommend/autodis
    The paper is significantly different from the code (it looks like the code
    implements the first version of the paper). We implement the second version
    here. Not all technical details are given for the second version, so what we do
    here can be different from what authors actually did.
    Anyway, AutoDis (v2) is essentially the following sequence of layers (applied from
    left to right): [Linear(no bias), LeakyReLU, Linear(no bias), Softmax, Linear]
    """

    def __init__(
        self, 
        n_features: int, 
        d_embedding: int, 
        n_meta_embeddings: int,
        temperature: float,
    ) -> None:
        super().__init__()
        self.first_layer = NumericalFeatureTokenizer(
            in_features=n_features,
            d_token=n_meta_embeddings,
            bias=False,
            initialization='uniform',
        )
        self.leaky_relu = nn.LeakyReLU()
        self.second_layer = NLinear(
            n_features, n_meta_embeddings, n_meta_embeddings, False
        )
        self.softmax = nn.Softmax(-1)
        self.temperature = temperature
        # "meta embeddings" from the paper are just a linear layer
        self.third_layer = NLinear(
            n_features, n_meta_embeddings, d_embedding, False
        )
        # 0.01 is taken from the source code
        nn.init.uniform_(self.third_layer.weight, 0.01)

    def forward(self, x: Tensor) -> Tensor:
        x = self.first_layer(x)
        x = self.leaky_relu(x)
        x = self.second_layer(x)
        x = self.softmax(x / self.temperature)
        x = self.third_layer(x)
        return x


class NumEmbeddings(nn.Module):
    def __init__(
        self,
        n_features: int,
        d_embedding: Optional[int],
        embedding_arch: list[str],
        periodic_options: Optional[PeriodicOptions],
        d_feature: Optional[int],
        memory_efficient: bool,
        n_meta_embeddings: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> None:
        super().__init__()
        assert embedding_arch
        assert set(embedding_arch) <= {
            'linear',
            'positional',
            'relu',
            'shared_linear',
            'layernorm',
            'autodis',
        }
        if any(x in embedding_arch for x in ['linear', 'shared_linear', 'autodis']):
            assert d_embedding is not None
        else:
            assert d_embedding is None
        assert embedding_arch.count('positional') <= 1
        if 'autodis' in embedding_arch:
            embedding_arch == ['autodis']

        NLinear_ = NLinearMemoryEfficient if memory_efficient else NLinear
        layers: list[nn.Module] = []

        if embedding_arch[0] == 'linear':
            assert periodic_options is None
            assert n_meta_embeddings is None and temperature is None
            assert d_embedding is not None
            layers.append(
                NumericalFeatureTokenizer(n_features, d_embedding, True, 'uniform')
                if d_feature is None
                else NLinear_(n_features, d_feature, d_embedding)
            )
            d_current = d_embedding
        elif embedding_arch[0] == 'positional':
            assert d_feature is None
            assert periodic_options is not None
            assert n_meta_embeddings is None or temperature is None
            layers.append(Periodic(n_features, periodic_options))
            d_current = periodic_options.n * 2
        elif embedding_arch[0] == 'autodis':
            assert d_feature is None
            assert periodic_options is None
            assert n_meta_embeddings is None or temperature is None
            assert d_embedding is not None
            layers.append(AutoDis(n_features, d_embedding, n_meta_embeddings, temperature))
            d_current = d_embedding
        else:
            assert False

        for x in embedding_arch[1:]:
            layers.append(
                nn.ReLU()
                if x == 'relu'
                else NLinear_(n_features, d_current, d_embedding)  # type: ignore[code]
                if x == 'linear'
                else nn.Linear(d_current, d_embedding)  # type: ignore[code]
                if x == 'shared_linear'
                else NLayerNorm(n_features, d_current)  # type: ignore[code]
                if x == 'layernorm'
                else nn.Identity()
            )
            if x in ['linear', 'shared_linear']:
                d_current = d_embedding
            assert not isinstance(layers[-1], nn.Identity)
        self.d_embedding = d_current
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class  NumericalTransformer(nn.Module):
    """
    FT-Transformer for numerical tabular features. 
    """
    def __init__(
        self, 
        prefix: str, 
        in_features: int,
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
        ffn_d_hidden: Optional[str] = 192,
        ffn_dropout: Optional[str] = 0.0,
        prenormalization: Optional[bool] = True,
        first_prenormalization: Optional[bool] =  False,
        kv_compression_ratio: Optional[float] = None,
        kv_compression_sharing: Optional[str] = None,
        head_activation: Optional[str] =  'relu',
        head_normalization: Optional[str] = 'layer_norm',
        numerical_embedding: Optional[bool] = True
    ):
        """
        Parameters
        ----------
        prefix
            The model prefix.
        in_features
            Dimension of input features.
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

        assert d_token > 0, 'd_token must be positive'
        assert n_blocks >= 0, 'n_blocks must be non-negative' 
        assert attention_n_heads > 0, 'attention_n_heads must be postive'
        assert token_initialization in ['uniform', 'normal'], 'initialization must be uniform or normal'

        self.prefix = prefix

        self.out_features = out_features

        if numerical_embedding :
            self.numerical_feature_tokenizer = NumEmbeddings(
                n_features=in_features,
                d_embedding=d_token,
                embedding_arch=['linear'],
                periodic_options=None,
                d_feature=None,
                memory_efficient=False,
                # n_meta_embeddings: Optional[int] = None,
                # temperature: Optional[float] = None,
            )
        else:
            self.numerical_feature_tokenizer = NumericalFeatureTokenizer(
                in_features=in_features,
                d_token=d_token,
                bias=token_bias,
                initialization=token_initialization,
            )

        self.cls_token = CLSToken(
            d_token=d_token, 
            initialization=token_initialization,
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
    def numerical_key(self):
        return f"{self.prefix}_{NUMERICAL}"

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

        features = self.numerical_feature_tokenizer(batch[self.numerical_key])

        if self.cls_token:
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
