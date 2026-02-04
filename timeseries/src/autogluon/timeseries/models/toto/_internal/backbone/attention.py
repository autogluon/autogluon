# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2025 Datadog, Inc.

import logging
from enum import Enum

import torch
from einops import rearrange
from torch.nn.functional import scaled_dot_product_attention

from .rope import TimeAwareRotaryEmbedding

log = logging.getLogger(__name__)


class AttentionAxis(Enum):
    TIME = 1
    SPACE = 2


class BaseMultiheadAttention(torch.nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float,
        rotary_emb: TimeAwareRotaryEmbedding | None,
        use_memory_efficient_attention: bool,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads."
        self.head_dim = embed_dim // num_heads
        self.rotary_emb = rotary_emb

        # We allocate a single tensor for the q, k, and v projection matrices,
        # multiply them with the inputs, and then split the projected tensors into q, k, and v using unbind.
        # This reduces overhead a bit vs. having multiple separate Linear layers,
        # which need to be initialized, tracked by the optimizer, etc.
        self.wQKV = torch.nn.Linear(embed_dim, embed_dim * 3)
        self.dropout = dropout
        self.use_memory_efficient_attention = use_memory_efficient_attention
        self.wO = torch.nn.Linear(embed_dim, embed_dim)

        assert not self.use_memory_efficient_attention, (
            "xformers is not available, so use_memory_efficient_attention must be False"
        )

        if not hasattr(self, "attention_axis") or self.attention_axis not in (AttentionAxis.TIME, AttentionAxis.SPACE):
            raise ValueError("Child class must define attention_axis as AttentionAxis.TIME or AttentionAxis.SPACE.")

    def rearrange_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        pattern = (
            "batch variate seq_len embed_dim -> (batch variate) seq_len embed_dim"
            if self.attention_axis == AttentionAxis.TIME
            else "batch variate seq_len embed_dim -> (batch seq_len) variate embed_dim"
        )

        return rearrange(inputs, pattern)

    def get_qkv(
        self,
        inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        pattern: str = ""
        if self.attention_axis == AttentionAxis.TIME and self.use_memory_efficient_attention:
            pattern = "batch_X_variate seq_len (qkv head_dim n_heads) -> qkv batch_X_variate seq_len n_heads head_dim"
        elif self.attention_axis == AttentionAxis.TIME and not self.use_memory_efficient_attention:
            pattern = "batch_X_variate seq_len (qkv head_dim n_heads) -> qkv batch_X_variate n_heads seq_len head_dim"
        elif self.attention_axis == AttentionAxis.SPACE and self.use_memory_efficient_attention:
            pattern = "batch_X_seq_len variate (qkv head_dim n_heads) -> qkv batch_X_seq_len variate n_heads head_dim"
        elif self.attention_axis == AttentionAxis.SPACE and not self.use_memory_efficient_attention:
            pattern = "batch_X_seq_len variate (qkv head_dim n_heads) -> qkv batch_X_seq_len n_heads variate head_dim"

        assert pattern
        qkv = self.wQKV(inputs.contiguous())
        return rearrange(qkv, pattern, qkv=3, head_dim=self.head_dim, n_heads=self.num_heads).unbind(dim=0)

    def positional_embedding(self, q, k, v, kv_cache, layer_idx):
        # Apply the rotary embeddings
        seq_pos_offset = 0
        if self.rotary_emb is not None and self.attention_axis == AttentionAxis.TIME:
            if kv_cache is not None:
                seq_pos_offset = kv_cache.seq_len(layer_idx)

            # We need to permute because rotary embeddings expect the sequence dimension to be the second-to-last dimension
            q, k = self.rotary_emb.rotate_queries_and_keys(q, k, seq_pos_offset=seq_pos_offset)

        if kv_cache is not None and self.attention_axis == AttentionAxis.TIME:
            # First, we append the current input key and value tensors to the cache.
            # This concatenates the current key and value tensors to the existing key and value tensors
            kv_cache.append(layer_idx, (k, v))
            # Then, we retrieve the key and value tensors from the cache.
            # This includes all the key and value tensors from previous time steps
            # as well as the current time step.
            k, v = kv_cache[layer_idx]

        q = q.contiguous()
        k = k.contiguous().to(q.dtype)  # Ensure k is the same dtype as q; this is necessary when using mixed precision
        v = v.contiguous().to(q.dtype)  # Ensure v is the same dtype as q; this is necessary when using mixed precision

        return q, k, v, seq_pos_offset

    def rearrange_output(self, output: torch.Tensor, batch: int, variate: int, seq_len: int) -> torch.Tensor:
        if self.attention_axis == AttentionAxis.TIME and self.use_memory_efficient_attention:
            pattern = "(batch variate) seq_len n_heads head_dim -> batch variate seq_len (n_heads head_dim)"
        elif self.attention_axis == AttentionAxis.TIME and not self.use_memory_efficient_attention:
            pattern = "(batch variate) n_heads seq_len head_dim -> batch variate seq_len (n_heads head_dim)"
        elif self.attention_axis == AttentionAxis.SPACE and self.use_memory_efficient_attention:
            pattern = "(batch seq_len) variate n_heads head_dim -> batch variate seq_len (n_heads head_dim)"
        elif self.attention_axis == AttentionAxis.SPACE and not self.use_memory_efficient_attention:
            pattern = "(batch seq_len) n_heads variate head_dim -> batch variate seq_len (n_heads head_dim)"

        return rearrange(output, pattern, batch=batch, variate=variate, seq_len=seq_len)  # type: ignore

    def run_attention(self, attention_mask, q, k, v, seq_pos_offset, dropout, seq_len, variate):
        # Determine dimension ranges for attention
        # Ensure the last query vector index is used from the cache
        q_dim_start, q_dim_end = seq_pos_offset, seq_pos_offset + seq_len
        kv_dim_start, kv_dim_end = 0, v.shape[1] if self.use_memory_efficient_attention else v.shape[2]
        if self.attention_axis == AttentionAxis.TIME:
            attention_mask = (
                attention_mask[..., q_dim_start:q_dim_end, kv_dim_start:kv_dim_end]
                if torch.is_tensor(attention_mask)
                else None
            )
            return scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attention_mask,
                dropout_p=dropout,
                is_causal=(attention_mask is None and seq_pos_offset == 0),
            )
        elif self.attention_axis == AttentionAxis.SPACE:
            # We don't use causal masking for space-wise attention
            attention_mask = (
                attention_mask[..., kv_dim_start:kv_dim_end, kv_dim_start:kv_dim_end]
                if torch.is_tensor(attention_mask)
                else None
            )
            return scaled_dot_product_attention(q, k, v, attn_mask=attention_mask, dropout_p=dropout, is_causal=False)
        else:
            raise ValueError("Invalid attention axis")

    def forward(
        self,
        layer_idx: int,
        inputs: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        kv_cache=None,
    ) -> torch.Tensor:
        batch_size, variate, seq_len, _ = inputs.shape
        dropout = self.dropout if self.training else 0.0

        rearranged_inputs = self.rearrange_inputs(inputs)
        q, k, v = self.get_qkv(rearranged_inputs)

        q, k, v, seq_pos_offset = self.positional_embedding(q, k, v, kv_cache, layer_idx)

        output = self.run_attention(attention_mask, q, k, v, seq_pos_offset, dropout, seq_len, variate)

        output = self.rearrange_output(output, batch_size, variate, seq_len)
        return self.wO(output)


class TimeWiseMultiheadAttention(BaseMultiheadAttention):
    """
    Computes standard multihead causal attention over the time axis.
    It does this by flattening out the variates along the batch dimension.
    It also applies rotary position embeddings to the query and key matrices
    in order to incorporate relative positional information.
    """

    attention_axis = AttentionAxis.TIME


class SpaceWiseMultiheadAttention(BaseMultiheadAttention):
    """
    Computes bidirectional multihead attention over the space axis (i.e. across variates within
    a multi-variate time series). This is done by flattening out the time axis along the batch dimension.
    This allows the model to attend to different variates at the same time point. By alternating
    between time-wise and space-wise attention, the model can learn both temporal and cross-variate
    dependencies in the data.

    Unlike with time-wise attention, don't apply rotary embeddings here
    because we want cross-variate attention to be invariant to the order of the variates.
    """

    attention_axis = AttentionAxis.SPACE


MultiHeadAttention = TimeWiseMultiheadAttention | SpaceWiseMultiheadAttention
