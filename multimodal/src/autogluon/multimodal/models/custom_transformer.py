import enum
import math
import warnings
from typing import Callable, Dict, List, Optional, Tuple, Union, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .utils import init_weights

ModuleType = Union[str, Callable[..., nn.Module]]
_INTERNAL_ERROR_MESSAGE = "Internal error. Please, open an issue."


def _is_glu_activation(activation: ModuleType):
    return isinstance(activation, str) and activation.endswith("glu") or activation in [ReGLU, GEGLU]


def _make_nn_module(module_type: ModuleType, *args) -> nn.Module:
    if isinstance(module_type, str):
        if module_type == "reglu":
            return ReGLU()
        elif module_type == "geglu":
            return GEGLU()
        elif module_type == "gelu":
            return nn.GELU()
        elif module_type == "relu":
            return nn.ReLU()
        elif module_type == "leaky_relu":
            return nn.LeakyReLU()
        elif module_type == "layer_norm":
            return nn.LayerNorm(*args)
        else:
            try:
                cls = getattr(nn, module_type)
            except AttributeError as err:
                raise ValueError(f"Failed to construct the module {module_type} with the arguments {args}") from err
            return cls(*args)
    else:
        return module_type(*args)


def _all_or_none(values):
    return all(x is None for x in values) or all(x is not None for x in values)


def reglu(x: Tensor) -> Tensor:
    """The ReGLU activation function from [1].

    References:
    ----------
    [1] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """
    assert x.shape[-1] % 2 == 0
    a, b = x.chunk(2, dim=-1)
    return a * F.relu(b)


def geglu(x: Tensor) -> Tensor:
    """The GEGLU activation function from [1].

    References:
    ----------
    [1] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """
    assert x.shape[-1] % 2 == 0
    a, b = x.chunk(2, dim=-1)
    return a * F.gelu(b)


class ReGLU(nn.Module):
    """
    The ReGLU activation function from [1].

    References:
    ----------
    [1] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """

    def forward(self, x: Tensor) -> Tensor:
        return reglu(x)


class GEGLU(nn.Module):
    """
    The GEGLU activation function from [1].

    References:
    ----------
    [1] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """

    def forward(self, x: Tensor) -> Tensor:
        return geglu(x)


class CLSToken(nn.Module):
    """[CLS]-token for BERT-like inference.

    To learn about the [CLS]-based inference, see [1].

    When used as a module, the [CLS]-token is appended **to the end** of each item in
    the batch.

    References:
    ----------
    [1] Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" 2018
    """

    def __init__(self, token_dim: int, initialization: str) -> None:
        """
        Args:
            token_dim: the size of token
            initialization: initialization policy for parameters. Must be one of
                :code:`['uniform', 'normal']`. Let :code:`s = d ** -0.5`. Then, the
                corresponding distributions are :code:`Uniform(-s, s)` and :code:`Normal(0, s)`. In
                the paper [gorishniy2021revisiting], the 'uniform' initialization was
                used.

        References:
            * [gorishniy2021revisiting] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko "Revisiting Deep Learning Models for Tabular Data", 2021
        """
        super().__init__()
        initialization_ = _TokenInitialization.from_str(initialization)
        self.weight = nn.Parameter(Tensor(token_dim))
        initialization_.apply(self.weight, token_dim)

    def expand(self, *leading_dimensions: int) -> Tensor:
        """Expand (repeat) the underlying [CLS]-token to a tensor with the given leading dimensions.

        A possible use case is building a batch of [CLS]-tokens. See `_CLSToken` for
        examples of usage.

        Note:
            Under the hood, the `torch.Tensor.expand` method is applied to the
            underlying :code:`weight` parameter, so gradients will be propagated as
            expected.

        Args:
            leading_dimensions: the additional new dimensions

        Returns:
            tensor of the shape :code:`(*leading_dimensions, len(self.weight))`
        """
        if not leading_dimensions:
            return self.weight
        new_dims = (1,) * (len(leading_dimensions) - 1)
        return self.weight.view(*new_dims, -1).expand(*leading_dimensions, -1)

    def forward(self, x: Tensor) -> Tensor:
        """Append self **to the end** of each item in the batch (see `_CLSToken`)."""
        return torch.cat([x, self.expand(len(x), 1)], dim=1)


class _TokenInitialization(enum.Enum):
    UNIFORM = "uniform"
    NORMAL = "normal"

    @classmethod
    def from_str(cls, initialization: str) -> "_TokenInitialization":
        try:
            return cls(initialization)
        except ValueError:
            valid_values = [x.value for x in _TokenInitialization]
            raise ValueError(f"initialization must be one of {valid_values}")

    def apply(self, x: Tensor, d: int) -> None:
        d_sqrt_inv = 1 / math.sqrt(d)
        if self == _TokenInitialization.UNIFORM:
            # used in the paper "Revisiting Deep Learning Models for Tabular Data";
            # is equivalent to `nn.init.kaiming_uniform_(x, a=math.sqrt(5))` (which is
            # used by torch to initialize nn.Linear.weight, for example)
            nn.init.uniform_(x, a=-d_sqrt_inv, b=d_sqrt_inv)
        elif self == _TokenInitialization.NORMAL:
            nn.init.normal_(x, std=d_sqrt_inv)


class MultiheadAttention(nn.Module):
    """Multihead Attention (self-/cross-) with optional 'linear' attention.

    To learn more about Multihead Attention, see [1]. See the implementation
    of `Transformer` and the examples below to learn how to use the compression technique
    from [2] to speed up the module when the number of tokens is large.

    References:
    ----------
    [1] Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" 2018
    [2] Sinong Wang, Belinda Z. Li, Madian Khabsa, Han Fang, Hao Ma "Linformer: Self-Attention with Linear Complexity", 2020
    """

    def __init__(
        self,
        *,
        token_dim: int,
        num_heads: int,
        dropout: float,
        bias: bool,
        initialization: str,
    ) -> None:
        """
        Parameters
        ----------
        token_dim:
            the token size. Must be a multiple of :code:`num_heads`.
        num_heads:
            the number of heads. If greater than 1, then the module will have
            an addition output layer (so called "mixing" layer).
        dropout:
            dropout rate for the attention map. The dropout is applied to
            *probabilities* and do not affect logits.
        bias:
            if `True`, then input (and output, if presented) layers also have bias.
            `True` is a reasonable default choice.
        initialization:
            initialization for input projection layers. Must be one of
            :code:`['kaiming', 'xavier']`. `kaiming` is a reasonable default choice.

        Raises
        ----------
            AssertionError: if requirements for the inputs are not met.
        """
        super().__init__()
        if num_heads > 1:
            assert token_dim % num_heads == 0, "token_dim must be a multiple of num_heads"
        assert initialization in ["kaiming", "xavier"]

        self.W_q = nn.Linear(token_dim, token_dim, bias)
        self.W_k = nn.Linear(token_dim, token_dim, bias)
        self.W_v = nn.Linear(token_dim, token_dim, bias)
        self.W_out = nn.Linear(token_dim, token_dim, bias) if num_heads > 1 else None
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout) if dropout else None

        for m in [self.W_q, self.W_k, self.W_v]:
            # the "xavier" branch tries to follow torch.nn.MultiheadAttention;
            # the second condition checks if W_v plays the role of W_out; the latter one
            # is initialized with Kaiming in torch
            if initialization == "xavier" and (m is not self.W_v or self.W_out is not None):
                # gain is needed since W_qkv is represented with 3 separate layers (it
                # implies different fan_out)
                nn.init.xavier_uniform_(m.weight, gain=1 / math.sqrt(2))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        if self.W_out is not None:
            nn.init.zeros_(self.W_out.bias)

    def _reshape(self, x: Tensor) -> Tensor:
        batch_size, num_tokens, d = x.shape
        head_dim = d // self.num_heads
        return (
            x.reshape(batch_size, num_tokens, self.num_heads, head_dim)
            .transpose(1, 2)
            .reshape(batch_size * self.num_heads, num_tokens, head_dim)
        )

    def forward(
        self,
        x_q: Tensor,
        x_kv: Tensor,
        key_compression: Optional[nn.Linear],
        value_compression: Optional[nn.Linear],
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Perform the forward pass.

        Parameters
        ----------
        x_q:
            query tokens
        x_kv:
            key-value tokens
        key_compression:
            Linformer-style compression for keys
        value_compression:
            Linformer-style compression for values

        Returns:
        ----------
            (tokens, attention_stats)
        """
        assert _all_or_none(
            [key_compression, value_compression]
        ), "If key_compression is (not) None, then value_compression must (not) be None"
        q, k, v = self.W_q(x_q), self.W_k(x_kv), self.W_v(x_kv)
        for tensor in [q, k, v]:
            assert tensor.shape[-1] % self.num_heads == 0, _INTERNAL_ERROR_MESSAGE
        if key_compression is not None:
            k = key_compression(k.transpose(1, 2)).transpose(1, 2)
            v = value_compression(v.transpose(1, 2)).transpose(1, 2)  # type: ignore

        batch_size = len(q)
        head_dim_key = k.shape[-1] // self.num_heads
        head_dim_value = v.shape[-1] // self.num_heads
        n_q_tokens = q.shape[1]

        q = self._reshape(q)
        k = self._reshape(k)
        attention_logits = q @ k.transpose(1, 2) / math.sqrt(head_dim_key)
        attention_probs = F.softmax(attention_logits, dim=-1)
        if self.dropout is not None:
            attention_probs = self.dropout(attention_probs)
        x = attention_probs @ self._reshape(v)
        x = (
            x.reshape(batch_size, self.num_heads, n_q_tokens, head_dim_value)
            .transpose(1, 2)
            .reshape(batch_size, n_q_tokens, self.num_heads * head_dim_value)
        )
        if self.W_out is not None:
            x = self.W_out(x)
        return x, {
            "attention_logits": attention_logits,
            "attention_probs": attention_probs,
        }


class AdditiveAttention(nn.Module):
    """Additive Attention with linear complexity to input sequence length.

    Additive attention was proposed and used in FastFormer.
    See Ref. [1] for details.
    This implementation is motivated by: https://github.com/jrzaurin/pytorch-widedeep.git

    References:
    ----------
    [1] Wu, Chuhan, et al. "Fastformer: Additive attention can be all you need." arXiv preprint arXiv:2108.09084 (2021).
    """

    def __init__(
        self,
        *,
        token_dim: int,
        num_heads: int,
        dropout: float,
        bias: bool,
        share_qv_weights: bool,
        initialization: str,
    ) -> None:
        """
        Parameters
        ----------
        token_dim:
            the token size. Must be a multiple of :code:`num_heads`.
        num_heads:
            the number of heads. If greater than 1, then the module will have
            an addition output layer (so called "mixing" layer).
        dropout:
            dropout rate for the attention map. The dropout is applied to
            *probabilities* and do not affect logits.
        bias:
            if `True`, then input (and output, if presented) layers also have bias.
            `True` is a reasonable default choice.
        share_qv_weights:
            if 'True', then value and query transformation parameters are shared.
        initialization:
            initialization for input projection layers. Must be one of
            :code:`['kaiming', 'xavier']`. `kaiming` is a reasonable default choice.
        """
        super().__init__()

        assert token_dim % num_heads == 0, "token_dim must be a multiple of num_heads"
        assert initialization in ["kaiming", "xavier"]

        self.head_dim = token_dim // num_heads
        self.num_heads = num_heads
        self.share_qv_weights = share_qv_weights
        self.dropout = nn.Dropout(dropout)
        trainable = []
        if share_qv_weights:
            self.qv_proj = nn.Linear(token_dim, token_dim, bias=bias)
            trainable.extend([self.qv_proj])
        else:
            self.q_proj = nn.Linear(token_dim, token_dim, bias=bias)
            self.v_proj = nn.Linear(token_dim, token_dim, bias=bias)
            trainable.extend([self.q_proj, self.v_proj])

        self.k_proj = nn.Linear(token_dim, token_dim, bias=bias)
        self.W_q = nn.Linear(token_dim, num_heads)
        self.W_k = nn.Linear(token_dim, num_heads)
        self.r_out = nn.Linear(token_dim, token_dim)
        trainable.extend([self.k_proj, self.W_q, self.W_k, self.r_out])

        if initialization == "xavier":
            self.apply(init_weights)
        else:
            for m in trainable:
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x_q: Tensor,
        x_kv: Tensor,
        *args,  # Not used. just to make the input consistent with MultiheadAttention.
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        batch_size, n_q_tokens, token_dim = x_q.shape
        batch_size, n_k_tokens, token_dim = x_kv.shape

        q = self.qv_proj(x_q) if self.share_qv_weights else self.q_proj(x_q)
        v = self.qv_proj(x_kv) if self.share_qv_weights else self.v_proj(x_kv)
        k = self.k_proj(x_kv)

        alphas = (self.W_q(q) / math.sqrt(self.head_dim)).softmax(dim=1)
        q_r = q.reshape(batch_size, n_q_tokens, self.num_heads, self.head_dim)
        global_query = torch.einsum(" b s h, b s h d -> b h d", alphas, q_r)
        global_query = global_query.reshape(batch_size, self.num_heads * self.head_dim).unsqueeze(1)

        p = k * global_query

        betas = (self.W_k(p) / math.sqrt(self.head_dim)).softmax(dim=1)
        p_r = p.reshape(batch_size, n_k_tokens, self.num_heads, self.head_dim)
        global_key = torch.einsum(" b s h, b s h d -> b h d", betas, p_r)
        global_key = global_key.reshape(batch_size, self.num_heads * self.head_dim).unsqueeze(1)

        u = v * global_key
        output = q + self.dropout(self.r_out(u))

        return output, {
            "query_weight": alphas,
            "key_weight": betas,
        }


class Custom_Transformer(nn.Module):
    """Transformer with extra features.

    This module is the backbone of `FTTransformer`."""

    WARNINGS = {"first_prenormalization": True, "prenormalization": True}

    class FFN(nn.Module):
        """The Feed-Forward Network module used in every `Transformer` block."""

        def __init__(
            self,
            *,
            token_dim: int,
            d_hidden: int,
            bias_first: bool,
            bias_second: bool,
            dropout: float,
            activation: ModuleType,
        ):
            super().__init__()
            self.linear_first = nn.Linear(
                token_dim,
                d_hidden * (2 if _is_glu_activation(activation) else 1),
                bias_first,
            )
            self.activation = _make_nn_module(activation)
            self.dropout = nn.Dropout(dropout)
            self.linear_second = nn.Linear(d_hidden, token_dim, bias_second)

        def forward(self, x: Tensor) -> Tensor:
            x = self.linear_first(x)
            x = self.activation(x)
            x = self.dropout(x)
            x = self.linear_second(x)
            return x

    class Head(nn.Module):
        """The final module of the `Transformer` that performs BERT-like inference."""

        def __init__(
            self,
            *,
            d_in: int,
            bias: bool,
            activation: ModuleType,
            normalization: ModuleType,
            d_out: int,
        ):
            super().__init__()
            self.normalization = _make_nn_module(normalization, d_in)
            self.activation = _make_nn_module(activation)
            self.linear = nn.Linear(d_in, d_out, bias)

        def forward(self, x: Tensor) -> Tensor:
            x = x[:, -1]
            x = self.normalization(x)
            x = self.activation(x)
            x = self.linear(x)
            return x

    def __init__(
        self,
        *,
        token_dim: int,
        num_blocks: int,
        attention_num_heads: int,
        attention_dropout: float,
        attention_initialization: str,
        attention_normalization: str,
        ffn_hidden_size: int,
        ffn_dropout: float,
        ffn_activation: str,
        ffn_normalization: str,
        residual_dropout: float,
        prenormalization: bool,
        first_prenormalization: bool,
        last_layer_query_idx: Union[None, List[int], slice],
        num_tokens: Optional[int],
        kv_compression_ratio: Optional[float],
        kv_compression_sharing: Optional[str],
        head_activation: ModuleType,
        head_normalization: ModuleType,
        d_out: int,
        projection: Optional[bool] = False,
        additive_attention: Optional[bool] = False,
        share_qv_weights: Optional[bool] = False,
    ) -> None:
        """
        Parameters
        ----------
        token_dim
            The size of one token for `_CategoricalFeatureTokenizer`.
        num_blocks
            Number of the `FT_Transformer` blocks, which should be non-negative.
        attention_num_heads
            Number of attention heads in each `FT_Transformer` block, which should be positive.
        attention_dropout
            Dropout ratio for the Multi Headed Attention module.
        attention_initialization
            Weights initialization scheme for Multi Headed Attention module.
        attention_normalization
            Normalization policy for attention layers. "layer_norm" is a good default.
        ffn_hidden_size
            Number of the hidden nodes of the linear layers in the Feed-Forward Network module.
        ffn_dropout
            Dropout ratio of the hidden nodes of the linear layers in the Feed-Forward Network module.
        ffn_activation
            Activation function type for the Feed-Forward Network module.
        ffn_normalization
            Normalization scheme of the Feed-Forward Network module.
        residual_dropout
            Dropout ratio for the linear layers in FT_Transformer block.
        prenormalization, first_prenormalization
            Prenormalization to stabilize the training.
        num_tokens
            Number of tokens of the input sequence.
        kv_compression_ratio
            The compression ration to reduce the input sequence length.
        kv_compression_sharing
            If `true` the projections will share weights.
        head_activation
            Activation function type of the MLP layer.
        head_normalization
            Normalization scheme of the MLP layer.
        d_out
            Output dimension.
        projection
            Whether to use a project head.
        additive_attention
            If 'true' the transformer will use additive attention with linear complexity to sequence length.
        share_qv_weights
            if 'true', then value and query transformation parameters are shared in additive attention.
        """
        super().__init__()
        if isinstance(last_layer_query_idx, int):
            raise ValueError(
                "last_layer_query_idx must be None, list[int] or slice. "
                f"Do you mean last_layer_query_idx=[{last_layer_query_idx}] ?"
            )
        if not prenormalization:
            assert (
                not first_prenormalization
            ), "If `prenormalization` is False, then `first_prenormalization` must be False"
        assert _all_or_none([num_tokens, kv_compression_ratio, kv_compression_sharing]), (
            "If any of the following arguments is (not) None, then all of them must (not) be None: "
            "num_tokens, kv_compression_ratio, kv_compression_sharing"
        )
        assert (
            additive_attention or not share_qv_weights
        ), "If `share_qv_weights` is True, then `additive_attention` must be True"
        assert kv_compression_sharing in [None, "headwise", "key-value", "layerwise"]
        if not prenormalization:
            if self.WARNINGS["prenormalization"]:
                warnings.warn(
                    "prenormalization is set to False. Are you sure about this? "
                    "The training can become less stable. "
                    "You can turn off this warning by tweaking the "
                    "rtdl.Transformer.WARNINGS dictionary.",
                    UserWarning,
                )
            assert (
                not first_prenormalization
            ), "If prenormalization is False, then first_prenormalization is ignored and must be set to False"
        if prenormalization and first_prenormalization and self.WARNINGS["first_prenormalization"]:
            warnings.warn(
                "first_prenormalization is set to True. Are you sure about this? "
                "For example, the vanilla FTTransformer with "
                "first_prenormalization=True performs SIGNIFICANTLY worse. "
                "You can turn off this warning by tweaking the "
                "rtdl.Transformer.WARNINGS dictionary.",
                UserWarning,
            )

        def make_kv_compression():
            assert num_tokens and kv_compression_ratio, _INTERNAL_ERROR_MESSAGE  # for mypy
            # https://github.com/pytorch/fairseq/blob/1bba712622b8ae4efb3eb793a8a40da386fe11d0/examples/linformer/linformer_src/modules/multihead_linear_attention.py#L83
            return nn.Linear(num_tokens, int(num_tokens * kv_compression_ratio), bias=False)

        self.shared_kv_compression = (
            make_kv_compression() if kv_compression_ratio and kv_compression_sharing == "layerwise" else None
        )

        self.prenormalization = prenormalization
        self.last_layer_query_idx = last_layer_query_idx

        self.blocks = nn.ModuleList([])
        for layer_idx in range(num_blocks):
            layer = nn.ModuleDict(
                {
                    "attention": AdditiveAttention(
                        token_dim=token_dim,
                        num_heads=attention_num_heads,
                        dropout=attention_dropout,
                        bias=True,
                        share_qv_weights=share_qv_weights,
                        initialization=attention_initialization,
                    )
                    if additive_attention
                    else MultiheadAttention(
                        token_dim=token_dim,
                        num_heads=attention_num_heads,
                        dropout=attention_dropout,
                        bias=True,
                        initialization=attention_initialization,
                    ),
                    "ffn": Custom_Transformer.FFN(
                        token_dim=token_dim,
                        d_hidden=ffn_hidden_size,
                        bias_first=True,
                        bias_second=True,
                        dropout=ffn_dropout,
                        activation=ffn_activation,
                    ),
                    "attention_residual_dropout": nn.Dropout(residual_dropout),
                    "ffn_residual_dropout": nn.Dropout(residual_dropout),
                    "output": nn.Identity(),  # for hooks-based introspection
                }
            )
            if layer_idx or not prenormalization or first_prenormalization:
                layer["attention_normalization"] = _make_nn_module(attention_normalization, token_dim)
            layer["ffn_normalization"] = _make_nn_module(ffn_normalization, token_dim)
            if kv_compression_ratio and self.shared_kv_compression is None:
                layer["key_compression"] = make_kv_compression()
                if kv_compression_sharing == "headwise":
                    layer["value_compression"] = make_kv_compression()
                else:
                    assert kv_compression_sharing == "key-value", _INTERNAL_ERROR_MESSAGE
            self.blocks.append(layer)

        self.head = (
            Custom_Transformer.Head(
                d_in=token_dim,
                d_out=d_out,
                bias=True,
                activation=head_activation,  # type: ignore
                normalization=head_normalization if prenormalization else "Identity",
            )
            if projection
            else nn.Identity()
        )

    def _get_kv_compressions(self, layer):
        return (
            (self.shared_kv_compression, self.shared_kv_compression)
            if self.shared_kv_compression is not None
            else (layer["key_compression"], layer["value_compression"])
            if "key_compression" in layer and "value_compression" in layer
            else (layer["key_compression"], layer["key_compression"])
            if "key_compression" in layer
            else (None, None)
        )

    def _start_residual(self, layer, stage, x):
        assert stage in ["attention", "ffn"], _INTERNAL_ERROR_MESSAGE
        x_residual = x
        if self.prenormalization:
            norm_key = f"{stage}_normalization"
            if norm_key in layer:
                x_residual = layer[norm_key](x_residual)
        return x_residual

    def _end_residual(self, layer, stage, x, x_residual):
        assert stage in ["attention", "ffn"], _INTERNAL_ERROR_MESSAGE
        x_residual = layer[f"{stage}_residual_dropout"](x_residual)
        x = x + x_residual
        if not self.prenormalization:
            x = layer[f"{stage}_normalization"](x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim == 3, "The input must have 3 dimensions: (n_objects, num_tokens, token_dim)"
        for layer_idx, layer in enumerate(self.blocks):
            layer = cast(nn.ModuleDict, layer)

            query_idx = self.last_layer_query_idx if layer_idx + 1 == len(self.blocks) else None
            x_residual = self._start_residual(layer, "attention", x)
            x_residual, _ = layer["attention"](
                x_residual if query_idx is None else x_residual[:, query_idx],
                x_residual,
                *self._get_kv_compressions(layer),
            )
            if query_idx is not None:
                x = x[:, query_idx]
            x = self._end_residual(layer, "attention", x, x_residual)

            x_residual = self._start_residual(layer, "ffn", x)
            x_residual = layer["ffn"](x_residual)
            x = self._end_residual(layer, "ffn", x, x_residual)
            x = layer["output"](x)

        x = self.head(x)

        return x
