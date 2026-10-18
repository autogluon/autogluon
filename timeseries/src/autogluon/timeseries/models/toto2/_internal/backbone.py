# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2026 Datadog, Inc.

"""Transformer backbone of Toto 2.0: rotary embeddings, attention, feed-forward and the KV cache."""

import functools
import math
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange, repeat

from . import unit_scaling as uu
from .configuration import Toto2ModelConfig


class StaticKVCacheLayer(nn.Module):
    """Pre-allocated key/value cache for a single attention layer, laid out as ``(batch, heads, seq, dim)``."""

    def __init__(self, max_size: int):
        super().__init__()
        self._max_size = max_size
        self._initialized = False
        self.register_buffer("_position", torch.tensor(0, dtype=torch.long), persistent=False)

    def reset(self) -> None:
        if self._initialized:
            self._position.zero_()
            self.keys.zero_()
            self.values.zero_()

    def rewind(self, num_positions: int) -> None:
        """Discard the trailing ``num_positions`` entries, so that they are overwritten by the next write."""
        self._position.sub_(num_positions)

    def forward(self, k: torch.Tensor, v: torch.Tensor) -> None:
        if not self._initialized:
            # The batch/head dimensions are only known at the first forward pass.
            for name, tensor in [("keys", k), ("values", v)]:
                shape = list(tensor.shape)
                shape[2] = self._max_size
                buffer = torch.zeros(shape, dtype=tensor.dtype, device=tensor.device)
                self.register_buffer(name, buffer, persistent=False)
            self._initialized = True

        incoming = k.size(2)
        positions = torch.arange(incoming, device=k.device, dtype=torch.long) + self._position
        self.keys.index_copy_(2, positions, k)
        self.values.index_copy_(2, positions, v)
        self._position.add_(incoming)


class KVCache(nn.Module):
    """Container for the per-layer key/value caches of all time-attention layers.

    ``ephemeral_len`` is the number of trailing entries that each layer discards right after writing them. During
    block decoding, the tokens of the prediction block are regenerated on every iteration (with the medians of the
    previous block filled in), so their cache entries must not persist.
    """

    def __init__(self, num_layers: int, max_size: int):
        super().__init__()
        self.cache_layers = nn.ModuleList([StaticKVCacheLayer(max_size) for _ in range(num_layers)])
        self.max_size = max_size
        self.ephemeral_len = 0

    def reset(self) -> None:
        for layer in self.cache_layers:
            layer.reset()


class RotaryProjection(nn.Module):
    """Rotary positional embedding (RoPE)."""

    def __init__(self, proj_width: int, max_len: int = 512, base: int = 10000):
        super().__init__()
        assert proj_width % 2 == 0, f"proj_width must be even, got {proj_width}"
        self.proj_width = proj_width

        theta = 1.0 / torch.pow(base, torch.arange(0, proj_width, 2, dtype=torch.float) / proj_width)
        m_theta = einsum(torch.arange(max_len, dtype=theta.dtype), theta, "length, width -> length width")
        m_theta = repeat(m_theta, "length width -> length (width 2)")
        self.register_buffer("cos", torch.cos(m_theta), persistent=False)
        self.register_buffer("sin", torch.sin(m_theta), persistent=False)

    @staticmethod
    def _rotate(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = rearrange(x, "... (dim r) -> r ... dim", r=2)
        return rearrange([-x2, x1], "r ... dim -> ... (dim r)", r=2)

    def _prepare_seq_ids(self, x: torch.Tensor, seq_ids: Optional[torch.Tensor]) -> torch.Tensor:
        if seq_ids is None:
            return torch.arange(x.shape[-2], device=x.device, dtype=torch.int32)
        return rearrange(seq_ids, "... seq -> ... 1 seq")

    def forward(self, x: torch.Tensor, seq_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        prepared = self._prepare_seq_ids(x, seq_ids)
        cos = self.cos[prepared].to(x.dtype)
        sin = self.sin[prepared].to(x.dtype)
        return cos * x + sin * self._rotate(x)


class ExtrapolatableRotaryProjection(RotaryProjection):
    """RoPE with xPos scaling for length extrapolation [Sun2022]_.

    .. [Sun2022] Sun, Yutao, Dong, Li et al.
        "A Length-Extrapolatable Transformer." (2022). https://arxiv.org/abs/2212.10554
    """

    def __init__(
        self,
        proj_width: int,
        max_len: int = 512,
        base: int = 10000,
        xpos_scale_base: int = 256,
        xpos_scale_exponent: float = 1.0,
    ):
        super().__init__(proj_width=proj_width, max_len=max_len, base=base)
        self.xpos_scale_base = xpos_scale_base
        self.xpos_scale_exponent = xpos_scale_exponent
        base_scale = (torch.arange(0, proj_width, 2).float() + 0.4 * proj_width) / (1.4 * proj_width)
        self.register_buffer("xpos_base_scale", base_scale, persistent=False)

    def _get_xpos_scale(self, seq_ids: torch.Tensor) -> torch.Tensor:
        center = torch.div(seq_ids.max() + 1, 2, rounding_mode="floor")
        power = (seq_ids.float() - center) / self.xpos_scale_base
        scale = self.xpos_base_scale ** power.unsqueeze(-1)
        return repeat(scale, "... d -> ... (d 2)") ** self.xpos_scale_exponent

    def forward(self, x: torch.Tensor, seq_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        xpos_scale = self._get_xpos_scale(self._prepare_seq_ids(x, seq_ids))
        return super().forward(x, seq_ids) * xpos_scale.to(x.dtype)


class QueryKeyProjection(nn.Module):
    """Apply a positional projection to (a leading slice of) the query and key head dimensions.

    ``partial_factor`` selects the fraction of each head's dimensions that the projection is applied to; the remaining
    dimensions pass through unchanged.
    """

    def __init__(
        self,
        head_dim: int,
        proj_layer: Callable[..., RotaryProjection],
        kwargs: Optional[dict] = None,
        key_proj_layer: Optional[Callable[..., RotaryProjection]] = None,
        partial_factor: Optional[tuple[float, float]] = None,
    ):
        super().__init__()
        if partial_factor is not None:
            assert 0.0 <= partial_factor[0] < partial_factor[1] <= 1.0

        self.head_dim = head_dim
        self.partial_factor = partial_factor
        kwargs = kwargs or {}

        if partial_factor is None:
            proj_width = head_dim
            self.split_sizes = (0, head_dim, 0)
        else:
            proj_width = int(head_dim * (partial_factor[1] - partial_factor[0]))
            self.split_sizes = (
                int(partial_factor[0] * head_dim),
                proj_width,
                int((1.0 - partial_factor[1]) * head_dim),
            )

        self.query_proj = proj_layer(proj_width=proj_width, **kwargs)
        self.key_proj = self.query_proj if key_proj_layer is None else key_proj_layer(proj_width=proj_width, **kwargs)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        query_ids: Optional[torch.Tensor] = None,
        kv_ids: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.partial_factor is None:
            return self.query_proj(query, seq_ids=query_ids), self.key_proj(key, seq_ids=kv_ids)

        queries = list(query.split(self.split_sizes, dim=-1))
        keys = list(key.split(self.split_sizes, dim=-1))
        queries[1] = self.query_proj(queries[1], seq_ids=query_ids)
        keys[1] = self.key_proj(keys[1], seq_ids=kv_ids)
        return torch.cat(queries, dim=-1), torch.cat(keys, dim=-1)


class SelfAttention(nn.Module):
    """Grouped-query self-attention with u-muP scaling, in ``(batch, heads, seq, dim)`` layout."""

    def __init__(
        self,
        config: Toto2ModelConfig,
        qk_proj_layer: Optional[Callable[[int], QueryKeyProjection]] = None,
        is_variate_layer: bool = False,
    ):
        super().__init__()
        assert config.qk_dim is not None and config.v_dim is not None and config.num_groups is not None
        self.config = config
        self.is_variate_layer = is_variate_layer
        self.num_heads = config.num_heads
        self.num_groups = config.num_groups

        if config.qk_norm:
            norm = functools.partial(
                uu.RMSNorm, config.qk_dim, eps=config.norm_eps, include_weight=bool(config.qk_norm_include_weight)
            )
            self.q_norm, self.k_norm = norm(), norm()
        else:
            self.q_norm = self.k_norm = None

        self.split_sizes = [
            config.qk_dim * config.num_heads,
            config.qk_dim * config.num_groups,
            config.v_dim * config.num_groups,
        ]
        self.in_proj = uu.Linear(config.d_model, sum(self.split_sizes), bias=config.attn_bias)
        self.qk_proj = qk_proj_layer(config.qk_dim) if qk_proj_layer is not None else None
        self.out_proj = uu.Linear(config.v_dim * config.num_heads, config.d_model, bias=config.attn_bias)

        # u-muP uses 1/d_k rather than 1/sqrt(d_k) to keep the logits bounded as the width grows.
        self.mult = 1.0 / config.qk_dim
        self._pds = uu.PerDimScale(config.qk_dim) if config.per_dim_scale else None

    def forward(
        self,
        state: torch.Tensor,
        seq_ids: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        kv_cache_layer: Optional[StaticKVCacheLayer] = None,
        kv_read_len: Optional[int] = None,
    ) -> torch.Tensor:
        q, k, v = torch.split(self.in_proj(state), self.split_sizes, dim=-1)
        q = rearrange(q, "b s (h d) -> b h s d", h=self.num_heads)
        k = rearrange(k, "b s (h d) -> b h s d", h=self.num_groups)
        v = rearrange(v, "b s (h d) -> b h s d", h=self.num_groups)

        if self.q_norm is not None:
            assert self.k_norm is not None
            q, k = self.q_norm(q), self.k_norm(k)
        if self._pds is not None:
            q = self._pds(q)
        if self.qk_proj is not None:
            seq = seq_ids[..., -q.size(-2) :] if seq_ids is not None else None
            q, k = self.qk_proj(q, k, query_ids=seq, kv_ids=seq)

        if kv_cache_layer is not None:
            kv_cache_layer(k, v)
            k = kv_cache_layer.keys[:, :, :kv_read_len, :]
            v = kv_cache_layer.values[:, :, :kv_read_len, :]

        # Variate layers attend across all variates; time layers are causal unless an explicit mask is given.
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=not self.is_variate_layer if attn_mask is None else False,
            scale=self.mult,
            enable_gqa=self.config.heads_per_group > 1,
        )
        return self.out_proj(rearrange(out, "b h s d -> b s (h d)"))


class GatedLinearUnitFeedForwardNetwork(nn.Module):
    """SwiGLU feed-forward network [Shazeer2020]_ with u-muP scaling.

    .. [Shazeer2020] Shazeer, Noam. "GLU Variants Improve Transformer." (2020). https://arxiv.org/abs/2002.05202
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.fc1 = uu.Linear(in_dim, 2 * hidden_dim, bias=bias, constraint=None)
        self.fc2 = uu.Linear(hidden_dim, out_dim, bias=bias, constraint=None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, x = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(gate * F.silu(x))


class ResidualMLP(nn.Module):
    """Two-layer MLP with a projected skip connection, combined with the u-muP tau-rule."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, bias: bool = True, is_readout: bool = False):
        super().__init__()
        self.tau = 1.0
        if is_readout:
            self.linear1 = uu.Linear(in_dim, hidden_dim, bias=bias)
            self.linear2 = uu.LinearReadout(hidden_dim, out_dim, bias=bias)
            self.skip_proj = uu.LinearReadout(in_dim, out_dim, bias=bias)
        else:
            self.linear1 = uu.Linear(in_dim, hidden_dim, bias=bias, constraint=None)
            self.linear2 = uu.Linear(hidden_dim, out_dim, bias=bias)
            self.skip_proj = uu.Linear(in_dim, out_dim, bias=bias, constraint=None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_main, x_skip = uu.residual_split(x, self.tau)
        hidden = self.linear2(uu.silu(self.linear1(x_main)))
        return uu.residual_add(hidden, self.skip_proj(x_skip), self.tau)


class SelfAttentionTransformerLayer(nn.Module):
    """Pre-norm transformer layer with u-muP tau-rule residual scaling."""

    def __init__(
        self, config: Toto2ModelConfig, attn: SelfAttention, layer_idx: int, tau_rule: Callable[[int, int], float]
    ):
        super().__init__()
        self.attn = attn
        self.ffn = GatedLinearUnitFeedForwardNetwork(
            in_dim=config.d_model, hidden_dim=config.d_ff, out_dim=config.d_model, bias=config.mlp_bias
        )
        norm = functools.partial(
            uu.RMSNorm, config.d_model, eps=config.norm_eps, include_weight=config.norm_include_weight
        )
        self.norm1, self.norm2 = norm(), norm()

        # The attention and MLP sublayers of every layer form a single stack of depth 2 * num_layers.
        total_depth = 2 * config.num_layers
        self.register_buffer("attn_tau", torch.tensor(tau_rule(2 * layer_idx, total_depth)))
        self.register_buffer("mlp_tau", torch.tensor(tau_rule(2 * layer_idx + 1, total_depth)))

    def forward(self, x: torch.Tensor, seq_ids: Optional[torch.Tensor] = None, **attn_kwargs) -> torch.Tensor:
        x, skip = uu.residual_split(x, self.attn_tau)
        x = uu.residual_add(self.attn(self.norm1(x), seq_ids, **attn_kwargs), skip, self.attn_tau)

        x, skip = uu.residual_split(x, self.mlp_tau)
        return uu.residual_add(self.ffn(self.norm2(x)), skip, self.mlp_tau)


class VariateTimeTransformerDecoder(nn.Module):
    """Decoder-only transformer that alternates between time-wise and variate-wise attention layers."""

    def __init__(self, config: Toto2ModelConfig):
        super().__init__()
        self.config = config

        if config.use_xpos:
            # Queries and keys receive reciprocal xPos scales, so their product only depends on the relative distance.
            qk_proj_layer = functools.partial(
                QueryKeyProjection,
                proj_layer=functools.partial(ExtrapolatableRotaryProjection, xpos_scale_exponent=1.0),
                key_proj_layer=functools.partial(ExtrapolatableRotaryProjection, xpos_scale_exponent=-1.0),
                kwargs={"max_len": 8192},
                partial_factor=(0.0, 0.5),
            )
        else:
            qk_proj_layer = functools.partial(
                QueryKeyProjection,
                proj_layer=RotaryProjection,
                kwargs={"max_len": 8192},
                partial_factor=(0.0, 0.5),
            )

        tau_rule = uu.transformer_residual_scaling_rule(
            residual_mult=config.residual_mult, residual_attn_ratio=config.residual_attn_ratio
        )
        self.layers = nn.ModuleList(
            [
                SelfAttentionTransformerLayer(
                    config,
                    attn=SelfAttention(
                        config,
                        # Variate layers attend across variates, where positional information is meaningless.
                        qk_proj_layer=None if config.is_variate_layer(idx) else qk_proj_layer,
                        is_variate_layer=config.is_variate_layer(idx),
                    ),
                    layer_idx=idx,
                    tau_rule=tau_rule,
                )
                for idx in range(config.num_layers)
            ]
        )
        self.out_norm = uu.RMSNorm(config.d_model, eps=config.norm_eps, include_weight=config.norm_include_weight)

    def _attention_masks(
        self,
        state: torch.Tensor,
        group_ids: Optional[torch.Tensor],
        has_missing_values: bool,
    ) -> tuple[dict, dict]:
        """Build the additive attention masks for the time and the variate layers.

        Both masks exclude patches marked as unobserved (``group_ids == -1``); the time mask is additionally causal.
        When ``has_missing_values`` is False, the time mask is omitted so that attention can use the faster
        ``is_causal`` fast path.
        """
        seq_len = state.shape[-2]
        zero = torch.zeros(1, dtype=state.dtype, device=state.device)
        neg_inf = torch.full((1,), -torch.inf, dtype=state.dtype, device=state.device)
        # `group_ids` may cover more positions than `state`; the trailing ones are the ones being attended to.
        if group_ids is not None:
            group_ids = group_ids[..., -seq_len:]

        if has_missing_values:
            time_mask = torch.where(
                torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=state.device)), zero, neg_inf
            )
            if group_ids is not None and seq_len > 1:
                time_mask = time_mask + torch.where(group_ids[..., :, None] == group_ids[..., None, :], zero, neg_inf)
            time_kwargs = {"attn_mask": rearrange(time_mask, "... s1 s2 -> (...) 1 s1 s2").contiguous()}
        else:
            time_kwargs = {}

        if group_ids is not None and group_ids.shape[-2] > 1:
            variate_mask = torch.where(
                rearrange(group_ids, "... n s -> ... s 1 n 1") == rearrange(group_ids, "... n s -> ... s 1 1 n"),
                zero,
                neg_inf,
            )
            variate_kwargs = {"attn_mask": rearrange(variate_mask, "... 1 n1 n2 -> (...) 1 n1 n2").contiguous()}
        else:
            variate_kwargs = {}

        return time_kwargs, variate_kwargs

    def forward(
        self,
        state: torch.Tensor,
        time_ids: Optional[torch.Tensor] = None,
        group_ids: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        kv_read_len: Optional[int] = None,
        cache_valid: Optional[torch.Tensor] = None,
        has_missing_values: bool = True,
    ) -> torch.Tensor:
        """Run the decoder over ``state`` of shape ``(*batch, num_variates, seq_len, d_model)``.

        ``time_ids`` are the positional indices used by the rotary embedding, and ``group_ids`` identify which variates
        belong to the same series (with ``-1`` marking fully unobserved patches). When ``kv_cache`` is given, only the
        keys/values of the incoming ``state`` are computed and the first ``kv_read_len`` cached entries are attended to;
        ``cache_valid`` then marks which cached positions hold observed data.
        """
        time_kwargs, variate_kwargs = self._attention_masks(state, group_ids, has_missing_values)

        if kv_cache is not None and cache_valid is not None:
            # Decode step: attend causally over the cached keys, skipping fully unobserved context patches.
            assert kv_read_len is not None
            q_len = state.shape[-2]
            causal = torch.tril(
                torch.ones(q_len, kv_read_len, dtype=torch.bool, device=state.device), diagonal=kv_read_len - q_len
            )
            time_kwargs = {
                "attn_mask": torch.where(
                    causal[None, None, :, :] & cache_valid[:, None, None, :kv_read_len],
                    torch.zeros(1, dtype=state.dtype, device=state.device),
                    torch.full((1,), -torch.inf, dtype=state.dtype, device=state.device),
                )
            }

        num_variates, seq_len = state.shape[-3], state.shape[-2]
        if time_ids is not None and time_ids.dim() > 1:
            time_ids = time_ids.expand(*state.shape[:-1]).flatten(0, -2)

        leading_shape = state.shape[:-2]
        state = rearrange(state, "... seq_len dim -> (...) seq_len dim")

        time_layer_idx = 0
        for idx, layer in enumerate(self.layers):
            if self.config.is_variate_layer(idx):
                # Attend across variates by moving the time dimension into the batch dimension.
                state = rearrange(state, "(b n) s d -> (b s) n d", n=num_variates)
                state = layer(state, **variate_kwargs)
                state = rearrange(state, "(b s) n d -> (b n) s d", s=seq_len)
            else:
                cache_layer = kv_cache.cache_layers[time_layer_idx] if kv_cache is not None else None
                state = layer(
                    state, seq_ids=time_ids, kv_cache_layer=cache_layer, kv_read_len=kv_read_len, **time_kwargs
                )
                if cache_layer is not None and kv_cache.ephemeral_len > 0:
                    cache_layer.rewind(kv_cache.ephemeral_len)
                time_layer_idx += 1

        return self.out_norm(state.unflatten(0, leading_shape))


class QuantileKnotsOutputHead(nn.Module):
    """Project the final embeddings to one quantile forecast per knot, for every step of the predicted patch."""

    def __init__(self, knots: list[float], embeds_dim: int, hidden_dim: int, patch_size: int):
        super().__init__()
        self.knots = knots
        self.param_projection = FusedPatchedParamProjection(
            embeds_dim=embeds_dim, hidden_dim=hidden_dim, patch_size=patch_size, num_params=len(knots)
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return rearrange(self.param_projection(embeddings), "... q -> q ...")


class FusedPatchedParamProjection(nn.Module):
    """Single projection producing all ``patch_size * num_params`` output parameters at once."""

    def __init__(self, embeds_dim: int, hidden_dim: int, patch_size: int, num_params: int):
        super().__init__()
        self.output_shape = (patch_size, num_params)
        self.proj = ResidualMLP(
            in_dim=embeds_dim, hidden_dim=hidden_dim, out_dim=math.prod(self.output_shape), is_readout=True
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.proj(inputs).unflatten(-1, self.output_shape)
