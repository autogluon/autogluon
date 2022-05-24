from torch import nn
from torch import Tensor
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Union, cast
from .ft_transformer import FT_Transformer,_TokenInitialization,CLSToken
from ..constants import NUMERICAL, LABEL, LOGITS, FEATURES
import torch


class Periodic(nn.Module):
    def __init__(
        self, 
        in_features: int, 
        d_embedding: int, 
        trainable: Optional[bool] = True,
        initialization: Optional[str] = 'normal',
        sigma: Optional[float] = 1.0,
    ):
        """
        Parameters
        ----------
        in_features
            Input feature size.
        d_embedding
            Output feature size, should be an even number.
        trainable
            Determine whether the coefficients needed to be updated.
        initialization
            Initalization scheme.
        sigma
            Standard deviation used for initalization='normal' 

        Reference:
        ----------
        1. Code: https://github.com/Yura52/tabular-dl-num-embeddings
        2. Paper: On Embeddings for Numerical Features in Tabular Deep Learning, https://arxiv.org/abs/2203.05556
        """
        super().__init__()

        assert d_embedding % 2 == 0 , 'd_embedding mod 2 should be 0, current d_embedding is {}'.format(d_embedding)

        if initialization == 'log-linear':
            coefficients = sigma ** (torch.arange(d_embedding//2) / (d_embedding//2) )
            coefficients = coefficients[None].repeat(in_features, 1)
        elif initialization == 'normal':
            coefficients = torch.normal(0.0, sigma, (in_features, d_embedding//2))

        if trainable:
            self.coefficients = nn.Parameter(coefficients) 
        else:
            self.register_buffer('coefficients', coefficients)

    def cos_sin(self, x: Tensor):
        return torch.cat(
            [torch.cos(x), torch.sin(x)], -1
        )

    def forward(self, x: Tensor):
        assert x.ndim == 2, 'Periodic should only be applied to first layer i.e. ndim==2'
        return self.cos_sin(
            2 * torch.pi * self.coefficients[None] * x[..., None]
        )


class NLinear(nn.Module):
    def __init__(
        self, 
        n: int, 
        d_in: int, 
        d_out: int, 
        bias: bool = True
    ):
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
        assert x.ndim == 3, 'Error input dimension, should be 3, but given {}'.format(x.ndim)
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
    ):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(d_in, d_out) for _ in range(n)])

    def forward(self, x):
        return torch.stack([l(x[:, i]) for i, l in enumerate(self.layers)], 1)


class NLayerNorm(nn.Module):
    def __init__(self, n_features: int, d: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_features, d))
        self.bias = nn.Parameter(torch.zeros(n_features, d))

    def forward(self, x: Tensor):
        assert x.ndim == 3
        x = (x - x.mean(-1, keepdim=True)) / x.std(-1, keepdim=True)
        x = self.weight * x + self.bias
        return x


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
    We borrow the implementations from: https://github.com/Yura52/tabular-dl-num-embeddings/blob/main/bin/train4.py
    The paper is significantly different from the code (it looks like the code
    implements the first version of the paper). We implement the second version
    here. Not all technical details are given for the second version, so what we do
    here can be different from what authors actually did.
    Anyway, AutoDis (v2) is essentially the following sequence of layers (applied from
    left to right): [Linear(no bias), LeakyReLU, Linear(no bias), Softmax, Linear]
    """

    def __init__(
        self, 
        in_features: int, 
        d_embedding: int, 
        n_meta_embeddings: int,
        temperature: Optional[float] = 3.0,
    ):
        super().__init__()
        self.first_layer = NumericalFeatureTokenizer(
            in_features=in_features,
            d_token=n_meta_embeddings,
            bias=False,
            initialization='uniform',
        )
        self.leaky_relu = nn.LeakyReLU()
        self.second_layer = NLinear(
            in_features, n_meta_embeddings, n_meta_embeddings, False
        )
        self.softmax = nn.Softmax(-1)
        self.temperature = temperature
        # "meta embeddings" from the paper are just a linear layer
        self.third_layer = NLinear(
            in_features, n_meta_embeddings, d_embedding, False
        )
        # 0.01 is taken from the source code
        nn.init.uniform_(self.third_layer.weight, 0.01)

    def forward(self, x: Tensor):
        x = self.first_layer(x)
        x = self.leaky_relu(x)
        x = self.second_layer(x)
        x = self.softmax(x/self.temperature)
        x = self.third_layer(x)
        return x


class NumEmbeddings(nn.Module):
    def __init__(
        self,
        in_features: int,
        embedding_arch: List[str],
        d_embedding: Optional[int] = None,
        memory_efficient: Optional[bool] = False,
    ):
        """
        Parameters
        ----------
        in_features
            Input feature size.
        embedding_arch
            A list containing the names of embedding layers.
            Currently support:
                {'linear', 'shared_linear', 'autodis', 'positional', 'relu', 'layernorm'}
            To use the embedding schemes summarized in Table 3 of 'On Embeddings for Numerical Features in Tabular Deep Learning' (https://arxiv.org/abs/2203.05556)
            By setting the embedding_arch as follows:
                1. `L`: ['linear']
                2. `LR`: ['linear', 'relu']
                3. `LRLR`: ['linear', 'relu', 'linear', 'relu']
                4. `P`: ['positional']
                5. `PL`: ['positional', 'linear']
                6. `PLR`: ['positional', 'linear', 'relu']
                7. `PLRLR`: ['positional', 'linear', 'relu', 'linear', 'relu']
                8. `AutoDis`: ['autodis']
            Notably, in `L` (i.e. embedding_arch=['linear']) for numerical transformer, 
            it identical as the original feature_tokenzier in FT_Transformer (c.f. Figure 2.a in https://arxiv.org/pdf/2106.11959.pdf).
        d_embedding:
            Dimension of the embeddings. 
            The output shape should be [batch_size, number_of_numerical_featurs, d_embedding]
        memory_efficient:
            Use efficient linear layer scheme if True. Default is False.   
         
        Reference:
        ----------
        1. Code: https://github.com/Yura52/tabular-dl-num-embeddings
        2. Paper: On Embeddings for Numerical Features in Tabular Deep Learning, https://arxiv.org/abs/2203.05556
        """
        
        super().__init__()
        assert embedding_arch
        assert set(embedding_arch) <= {
            'linear',
            'shared_linear',
            'autodis',
            'positional',
            'relu',
            'layernorm',
        }

        if any(x in embedding_arch for x in ['linear', 'shared_linear', 'autodis']):
            assert d_embedding is not None 
            
        assert embedding_arch.count('positional') <= 1
        
        if 'autodis' in embedding_arch:
            embedding_arch = ['autodis']
            
        NLinear_ = NLinearMemoryEfficient if memory_efficient else NLinear
        layers: list[nn.Module] = []

        if embedding_arch[0] == 'linear':
            layers.append(
                NumericalFeatureTokenizer(
                    in_features=in_features, 
                    d_token=d_embedding, 
                    bias=True, 
                    initialization='normal'
                )
            )
        elif embedding_arch[0] == 'positional':
            layers.append(
                Periodic(
                    in_features=in_features, 
                    d_embedding=d_embedding,
                    trainable=True,
                    initialization='normal',
                    sigma=1.0,
                )
            )
        elif embedding_arch[0] == 'autodis':
            layers.append(
                AutoDis(
                    in_features=in_features, 
                    d_embedding=d_embedding, 
                    n_meta_embeddings=d_embedding,
                    temperature=3.0,
                )
                )
        else:
            layers.append(
                nn.Identity(),
            )

        for x in embedding_arch[1:]:
            
            layers.append(
                nn.ReLU()
                if x == 'relu'
                else NLinear_(in_features, d_embedding, d_embedding)  
                if x == 'linear'
                else nn.Linear(d_embedding, d_embedding) 
                if x == 'shared_linear'
                else NLayerNorm(in_features, d_embedding) 
                if x == 'layernorm'
                else nn.Identity()
            )
            
            assert not isinstance(layers[-1], nn.Identity)
            
        self.d_embedding = d_embedding
        self.in_features = in_features
        self.layers = nn.Sequential(*layers)
        
    @property
    def n_tokens(self) -> int:
        """The number of tokens."""
        y = self.forward(torch.ones(1,self.in_features))
        return y.shape[1]

    @property
    def d_token(self) -> int:
        """The size of one token."""
        y = self.forward(torch.ones(1,self.in_features))
        return y.shape[-1]
    
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
        head_activation: Optional[str] = 'relu',
        head_normalization: Optional[str] = 'layer_norm',
        embedding_arch: Optional[List[str]] = ['linear'],
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
        embedding_arch
            A list containing the names of embedding layers.
            Currently support:
            {'linear', 'shared_linear', 'autodis', 'positional', 'relu', 'layernorm'}
            
        References
        ----------
        1. Paper: Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko, 
        "Revisiting Deep Learning Models for Tabular Data", 2021 https://arxiv.org/pdf/2106.11959.pdf
        2. Paper: On Embeddings for Numerical Features in Tabular Deep Learning, https://arxiv.org/abs/2203.05556
        3. Code: https://github.com/Yura52/tabular-dl-revisiting-models
        4. Code: https://github.com/Yura52/tabular-dl-num-embeddings
        """

        super().__init__()

        assert d_token > 0, 'd_token must be positive'
        assert n_blocks >= 0, 'n_blocks must be non-negative' 
        assert attention_n_heads > 0, 'attention_n_heads must be postive'
        assert token_initialization in ['uniform', 'normal'], 'initialization must be uniform or normal'

        self.prefix = prefix
        self.out_features = out_features

        self.numerical_feature_tokenizer = NumEmbeddings(
            in_features=in_features,
            d_embedding=d_token,
            embedding_arch=embedding_arch,
        )

        self.cls_token = CLSToken(
            d_token=d_token, 
            initialization=token_initialization,
        ) if cls_token else nn.Identity()

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
